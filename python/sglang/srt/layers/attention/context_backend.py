"""Ragged native-attention metadata for Context System occurrence windows.

Subsequences partition the query tensor; they never split a model forward.
Their prefixes may read KV written by earlier queries in this same forward,
after the layer's birth KV store and required copy-on-write RoPE operations.
There is no dense attention mask or per-layer CPU planning in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.context_system.request_storage import request_row

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.context_system.occurrence import OccurrenceWindow


@dataclass(frozen=True)
class ContextSequence:
    occurrence_count: int
    query_lengths: tuple[int, ...]
    prefix_offsets: np.ndarray
    prefix_occurrences: np.ndarray
    query_positions: np.ndarray
    prefix_positions: np.ndarray

    @classmethod
    def from_window(cls, window: OccurrenceWindow) -> ContextSequence:
        """Convert one validated request window, retaining original query order."""
        starts = window.segment_query_starts.numpy()
        ends = window.segment_query_ends.numpy()
        if len(starts) == 0 or np.any(ends <= starts):
            raise ValueError("Context attention needs nonempty query segments")
        if not np.array_equal(ends[:-1], starts[1:]):
            raise ValueError("Context attention query segments must be contiguous")
        raw = window.occurrence_raw_tokens.numpy()
        positions = window.occurrence_positions.numpy()
        birth = window.birth_occurrences.numpy()
        keys = window.segment_key_occurrences.numpy()
        key_offsets = window.segment_key_offsets.numpy()
        prefixes, offsets = [], [0]
        for start, end, a, b in zip(starts, ends, key_offsets[:-1], key_offsets[1:]):
            selected = keys[a:b]
            count = int(end - start)
            if len(selected) < count or not np.array_equal(
                selected[-count:], birth[start:end]
            ):
                raise ValueError("Context segment must end in its ordered birth KV")
            prefix = selected[:-count]
            if np.any(raw[prefix] >= start):
                raise ValueError("Context prefix cannot read an uncomputed query")
            prefixes.append(prefix)
            offsets.append(offsets[-1] + len(prefix))
        prefix = np.concatenate(prefixes).astype(np.int64, copy=False)
        return cls(
            len(raw),
            tuple(int(n) for n in ends - starts),
            np.asarray(offsets, dtype=np.int64),
            prefix,
            positions[birth[starts[0] : ends[-1]]].astype(np.int64),
            positions[prefix].astype(np.int64),
        )

    @classmethod
    def ordinary(cls, prefix_length: int, query_length: int) -> ContextSequence:
        """An ordinary request in a mixed batch needs no Context IR compilation.

        The caller supplies its native ordered prefix slots followed by query
        write slots in the batch occurrence binding. Pure ordinary batches
        bypass this module entirely.
        """
        if prefix_length < 0 or query_length < 1:
            raise ValueError("Invalid ordinary prefix/query lengths")
        prefix = np.arange(prefix_length, dtype=np.int64)
        return cls(
            prefix_length + query_length,
            (query_length,),
            np.asarray([0, prefix_length], dtype=np.int64),
            prefix,
            np.arange(prefix_length, prefix_length + query_length, dtype=np.int64),
            prefix,
        )


@dataclass(frozen=True)
class ContextAttentionPlan:
    # One packed transfer for all segment metadata, prepared once per forward.
    packed: torch.Tensor
    field_ranges: tuple[tuple[int, int], ...]
    query_lengths: tuple[int, ...]
    occurrence_count: int

    @classmethod
    def merge(cls, sequences: Sequence[ContextSequence]) -> ContextAttentionPlan:
        if not sequences:
            raise ValueError("Context attention batch cannot be empty")
        q_lengths, kv_lengths, occurrences, q_pos, kv_pos = [], [], [], [], []
        base = 0
        for seq in sequences:
            q_lengths.extend(seq.query_lengths)
            kv_lengths.append(np.diff(seq.prefix_offsets))
            occurrences.append(seq.prefix_occurrences + base)
            q_pos.append(seq.query_positions)
            kv_pos.append(seq.prefix_positions)
            base += seq.occurrence_count
        fields = (
            np.r_[0, np.cumsum(q_lengths, dtype=np.int64)],
            np.r_[0, np.cumsum(np.concatenate(kv_lengths), dtype=np.int64)],
            np.concatenate(occurrences),
            np.concatenate(q_pos),
            np.concatenate(kv_pos),
        )
        # Triton specializes pointer alignment. Unpadded variable-length fields
        # otherwise produce eight alignment combinations for the three dynamic
        # attention pointers, with repeated JIT stalls during long trajectories.
        # Keep every int64 field 16-byte aligned within the single transfer;
        # ranges exclude padding so attention sees exactly the original values.
        offsets = np.r_[0, np.cumsum([(len(field) + 1) // 2 * 2 for field in fields])]
        packed = np.empty(int(offsets[-1]), dtype=np.int64)
        ranges = tuple(
            (int(start), int(start) + len(field))
            for start, field in zip(offsets, fields)
        )
        for field, (start, end), padded_end in zip(fields, ranges, offsets[1:]):
            packed[start:end] = field
            packed[end:padded_end] = 0
        return cls(
            torch.from_numpy(packed),
            ranges,
            tuple(q_lengths),
            base,
        )

    def bind(self, occurrence_slots: torch.Tensor) -> ContextAttentionMetadata:
        """Bind *physical* slots after native full/SWA index translation.

        Unused occurrences may have no allocation. Every referenced occurrence
        must own a live slot: the scheduler validates that on its CPU residency
        map before dispatch, rather than reading device indices back here.
        Full and SWA pools bind separately to their own native physical slots.
        The resulting metadata and pool ownership must live through GPU use.
        """
        if (
            occurrence_slots.ndim != 1
            or occurrence_slots.numel() != self.occurrence_count
            or occurrence_slots.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Physical slot binding must cover all batch occurrences")
        packed = self.packed.to(device=occurrence_slots.device, non_blocking=True)
        fields = [packed[a:b] for a, b in self.field_ranges]
        qo, kv, occurrences, q_positions, kv_positions = fields
        return ContextAttentionMetadata(
            qo,
            kv,
            occurrence_slots.index_select(0, occurrences),
            q_positions,
            kv_positions,
            self.query_lengths,
            sum(self.query_lengths),
            max(self.query_lengths),
        )


@dataclass(frozen=True)
class ContextAttentionMetadata:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    query_positions: torch.Tensor
    kv_positions: torch.Tensor
    query_lengths: tuple[int, ...]
    query_count: int
    max_query_length: int

    def forward(self, kernel, q, k, v, output, k_pool, v_pool, **kwargs):
        """One native extend launch, after this layer's KV writes and COW."""
        if kwargs.get("page_size", 1) != 1:
            raise ValueError("Context attention requires page_size=1")
        if self.query_count != q.shape[0]:
            raise ValueError("Context query metadata does not cover model output")
        kernel(
            q,
            k,
            v,
            output,
            k_pool,
            v_pool,
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            None,
            True,
            None,
            self.max_query_length,
            1.0,
            1.0,
            context_q_positions=self.query_positions,
            context_kv_positions=self.kv_positions,
            extend_seq_lens_cpu=self.query_lengths,
            **kwargs,
        )
        return output


@dataclass(frozen=True)
class ContextLayerCopy:
    """One native layer's fresh destinations, bound before model execution.

    Cached-source copies may be done for all layers before the forward. Copies
    whose source is a query in this forward must wait for that layer's KV store.
    Sources always refer to the resident/birth version, never another copy in
    this launch. Destination page ownership is held by the forward lifecycle.
    """

    k_data_ptrs: torch.Tensor
    v_data_ptrs: torch.Tensor
    k_layout: torch.Tensor
    v_layout: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    position_pairs: torch.Tensor
    cos_sin_cache: torch.Tensor
    is_neox_style: bool
    skip_unmapped: bool = False
    rotary_dim: int | None = None

    def apply(self):
        from sglang.kernels.ops.attention.context_reposition import reposition_kv_layers

        if len(self.k_data_ptrs) != 1 or len(self.v_data_ptrs) != 1:
            raise ValueError("Per-layer birth copies must address exactly one layer")
        reposition_kv_layers(
            self.k_data_ptrs,
            self.v_data_ptrs,
            self.k_layout,
            self.v_layout,
            self.source_slots,
            self.destination_slots,
            self.position_pairs,
            self.cos_sin_cache,
            is_neox_style=self.is_neox_style,
            skip_unmapped=self.skip_unmapped,
            rotary_dim=self.rotary_dim,
        )


@dataclass(frozen=True)
class ContextForwardMetadata:
    full: ContextAttentionMetadata
    sliding_window: ContextAttentionMetadata | None
    # Native global layer IDs; the bindings already select each pool's local
    # pointer-table slice. No model traversal or device allocation per layer.
    layer_copies: dict[int, ContextLayerCopy]

    def for_layer(self, layer) -> ContextAttentionMetadata:
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            if self.sliding_window is None:
                raise ValueError("Context SWA metadata was not bound")
            return self.sliding_window
        return self.full


@dataclass(frozen=True)
class ContextPoolLayer:
    layer_id: int
    sliding_window: bool
    k_ptr: torch.Tensor
    v_ptr: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cos_sin_cache: torch.Tensor
    is_neox_style: bool
    rotary_dim: int


class ContextModelBinding:
    """Engine-lifetime native pool/RoPE bindings, built before Context dispatch.

    Uses the model's existing rotary modules, including YaRN scaling and style.
    Global layer IDs go through the native pool's mapping (including SWA), never
    through an assumption about alternating layers or local TP head counts.
    """

    def __init__(self, model, pool, translator, *, page_size: int):
        if page_size != 1:
            raise ValueError("Context model binding requires page_size=1")
        layers = []
        for module in model.modules():
            attn = getattr(module, "attn", None)
            rotary = getattr(module, "rotary_emb", None)
            if attn is None or rotary is None or not hasattr(attn, "layer_id"):
                continue
            layer_id = attn.layer_id
            k, v = pool.get_kv_buffer(layer_id)
            rope = getattr(rotary, "cos_sin_cache", None)
            rotary_dim = getattr(rotary, "rotary_dim", k.shape[-1])
            if (
                k.ndim != 3
                or v.shape != k.shape
                or k.dtype not in (torch.float16, torch.bfloat16)
                or v.dtype != k.dtype
                or rope is None
                or rope.ndim != 2
                or type(rotary_dim) is not int
                or not 0 < rotary_dim <= k.shape[-1]
                or rotary_dim % 2
                or rope.shape[1] != rotary_dim
                or k.device != v.device
                or k.device != rope.device
                or k.stride(-1) != 1
                or v.stride(-1) != 1
            ):
                raise ValueError("Context needs native NHD KV and valid native rotary_dim")
            layers.append(
                ContextPoolLayer(
                    layer_id,
                    attn.sliding_window_size is not None
                    and attn.sliding_window_size > -1,
                    torch.tensor([k.data_ptr()], dtype=torch.uint64, device=k.device),
                    torch.tensor([v.data_ptr()], dtype=torch.uint64, device=v.device),
                    k,
                    v,
                    rope,
                    rotary.is_neox_style,
                    rotary_dim,
                )
            )
        if not layers or len({layer.layer_id for layer in layers}) != len(layers):
            raise ValueError("Context model must expose unique native attention layers")
        self.layers = tuple(layers)
        self.translator = translator
        self.has_swa = any(layer.sliding_window for layer in layers)
        self._existing_copy_groups = None

    def reposition_existing(self, source, destination, positions):
        """Completed D-cache versions have no per-layer birth dependency.

        Group compatible native pools/RoPE tables once, then copy all their
        layers together. Shared Full/SWA storage preserves mini's semantics.
        """
        from sglang.kernels.ops.attention.context_reposition import reposition_kv_layers

        if self.translator.sliding_window_write_loc_for(source) is not None:
            raise ValueError("Context decode Radix requires shared Full/SWA KV")
        if self._existing_copy_groups is None:
            groups = {}
            for layer in self.layers:
                key = (
                    layer.k_buffer.shape, layer.k_buffer.stride(),
                    layer.v_buffer.stride(), layer.k_buffer.dtype,
                    layer.cos_sin_cache.data_ptr(), layer.is_neox_style,
                    layer.rotary_dim,
                )
                groups.setdefault(key, []).append(layer)
            self._existing_copy_groups = tuple(
                (items[0], torch.cat([x.k_ptr for x in items]),
                 torch.cat([x.v_ptr for x in items]))
                for items in groups.values()
            )
        source = self.translator.translate_full_attn_ids(source).to(torch.int32)
        destination = self.translator.translate_full_attn_ids(destination).to(torch.int32)
        for layer, k_ptrs, v_ptrs in self._existing_copy_groups:
            reposition_kv_layers(
                k_ptrs, v_ptrs, layer.k_buffer, layer.v_buffer,
                source, destination, positions, layer.cos_sin_cache,
                is_neox_style=layer.is_neox_style,
                rotary_dim=layer.rotary_dim,
            )

    def bind(self, inputs: ContextPrefillInput) -> ContextForwardMetadata:
        # Translate virtual FULL ids once, then derive SWA ids using the same
        # native contract as KV writes. Unused -1 entries are padding only;
        # admission must have proved every referenced source is resident.
        full_slots = self.translator.translate_full_attn_ids(
            inputs.occurrence_slots.clamp(min=0)
        )
        copy_source = self.translator.translate_full_attn_ids(inputs.copy_sources)
        copy_destination = self.translator.translate_full_attn_ids(
            inputs.copy_destinations
        )
        full = inputs.attention_plan.bind(full_slots)
        swa = None
        separate_swa_pool = False
        if self.has_swa:
            swa_slots = self.translator.sliding_window_write_loc_for(full_slots)
            if swa_slots is None:
                # Native --disable-hybrid-swa-memory keeps SWA layers in the
                # same token id space as Full layers, as mini-sglang does.
                # Reuse the binding; sliding visibility is still applied by
                # the layer's attention kernel using the occurrence positions.
                swa = full
                swa_source, swa_destination = copy_source, copy_destination
            else:
                separate_swa_pool = True
                swa_source = self.translator.sliding_window_write_loc_for(copy_source)
                swa_destination = self.translator.sliding_window_write_loc_for(
                    copy_destination
                )
                if swa_source is None or swa_destination is None:
                    raise ValueError(
                        "Context SWA pool is missing its native index mapping"
                    )
                swa = inputs.attention_plan.bind(swa_slots)
        copies = {}
        if len(inputs.copy_sources):
            for layer in self.layers:
                source = swa_source if layer.sliding_window else copy_source
                destination = (
                    swa_destination if layer.sliding_window else copy_destination
                )
                copies[layer.layer_id] = ContextLayerCopy(
                    layer.k_ptr,
                    layer.v_ptr,
                    layer.k_buffer,
                    layer.v_buffer,
                    source.to(torch.int32),
                    destination.to(torch.int32),
                    inputs.copy_positions,
                    layer.cos_sin_cache,
                    layer.is_neox_style,
                    skip_unmapped=layer.sliding_window and separate_swa_pool,
                    rotary_dim=layer.rotary_dim,
                )
        return ContextForwardMetadata(full, swa, copies)


@dataclass(frozen=True)
class ContextPrefillInput:
    """Immutable forward snapshot; all pages remain leased through completion."""

    attention_plan: ContextAttentionPlan
    occurrence_slots: torch.Tensor
    copy_sources: torch.Tensor
    copy_destinations: torch.Tensor
    copy_positions: torch.Tensor
    model_binding: ContextModelBinding | None

    @classmethod
    def from_prepared_batch(cls, batch, *, decode: bool):
        """Wrap native rows when an ordinary prefill mixes with Context decode."""
        sequences, slots = [], []
        for i, (req, raw_len) in enumerate(
            zip(batch.reqs, batch.seq_lens_cpu.tolist())
        ):
            query_len = 1 if decode else batch.extend_lens[i]
            raw_row = request_row(batch.req_to_token_pool, req.kv.req_pool_idx)[:raw_len]
            if req.context_program is None:
                sequences.append(
                    ContextSequence.ordinary(raw_len - query_len, query_len)
                )
                slots.append(raw_row)
                continue
            if not decode:
                raise ValueError(
                    "Context prefill must use its admitted occurrence plan"
                )
            view = req.context_decode_layout
            if view is None:
                from sglang.srt.context_system.occurrence import ContextDecodeLayout

                view = ContextDecodeLayout.from_layout(
                    req.context_program.layout, raw_row.device
                )
                req.context_decode_layout = view
            generated = raw_len - view.prompt_length
            if generated < 1:
                raise ValueError(
                    "Mixed Context decode requires its native allocated query"
                )
            raw = np.concatenate(
                (view.raw_indices, np.arange(view.prompt_length, raw_len))
            )
            positions = np.concatenate(
                (
                    view.positions,
                    np.arange(view.next_position, view.next_position + generated),
                )
            )
            n = len(raw)
            sequences.append(
                ContextSequence(
                    n,
                    (1,),
                    np.asarray([0, n - 1], dtype=np.int64),
                    np.arange(n - 1, dtype=np.int64),
                    positions[-1:],
                    positions[:-1],
                )
            )
            indices = torch.from_numpy(raw).to(raw_row.device, non_blocking=True)
            slots.append(raw_row[indices])
        empty = batch.out_cache_loc[:0].to(torch.int32)
        return cls(
            ContextAttentionPlan.merge(sequences),
            torch.cat(slots),
            empty,
            empty,
            empty.reshape(0, 2),
            None,
        )

    @classmethod
    def concatenate(cls, inputs):
        sequences = []
        for item in inputs:
            plan = item.attention_plan
            fields = [
                plan.packed[a:b].numpy()
                for a, b in plan.field_ranges
            ]
            _, offsets, occurrences, query_positions, prefix_positions = fields
            sequences.append(
                ContextSequence(
                    plan.occurrence_count,
                    plan.query_lengths,
                    offsets,
                    occurrences,
                    query_positions,
                    prefix_positions,
                )
            )
        return cls(
            ContextAttentionPlan.merge(sequences),
            torch.cat([item.occurrence_slots for item in inputs]),
            torch.cat([item.copy_sources for item in inputs]),
            torch.cat([item.copy_destinations for item in inputs]),
            torch.cat([item.copy_positions for item in inputs]),
            None,
        )

    def bind(self, model_runner=None) -> ContextForwardMetadata:
        if (
            self.copy_sources.ndim != 1
            or self.copy_destinations.shape != self.copy_sources.shape
            or self.copy_positions.shape != (len(self.copy_sources), 2)
            or any(
                value.dtype != torch.int32
                for value in (
                    self.copy_sources,
                    self.copy_destinations,
                    self.copy_positions,
                )
            )
        ):
            raise ValueError("Context copies require aligned int32 metadata")
        binding = self.model_binding
        if binding is None:
            if model_runner is None:
                raise ValueError("Context prefill needs a native model binding")
            binding = getattr(model_runner, "context_model_binding", None)
            if binding is None:
                binding = ContextModelBinding(
                    model_runner.model,
                    model_runner.token_to_kv_pool,
                    model_runner.kv_index_translator,
                    page_size=1,
                )
                model_runner.context_model_binding = binding
        return binding.bind(self)


class ContextDecodeRegistry:
    """Capture-stable request-slot registry, allocated on first Context decode.

    The raw request table retains native allocation/cache ownership. Only read
    indices are compacted. Pool translation is fused with that existing gather;
    no historical KV is copied and no per-layer Context work is added to decode.
    """

    def __init__(self, translator, pool, request_pool=None):
        if translator.page_size != 1 or translator.defer_read_translate:
            raise ValueError("Context decode requires page_size=1 without DCP")
        self.translator = translator
        self.pool = pool
        self.request_pool = request_pool
        self.rows = torch.zeros(
            (translator.req_to_token.shape[0], 6),
            dtype=torch.int64,
            device=translator.req_to_token.device,
        )

    def bind_batch(self, reqs, raw_lengths, row_ids):
        from sglang.srt.context_system.occurrence import ContextDecodeLayout

        entries, lengths, positions, refs = [], [], [], []
        for req, raw_length in zip(reqs, raw_lengths):
            raw_length = int(raw_length)
            if req.context_program is None:
                entries.append((0, 0, 0, 0, 0, 0))
                lengths.append(raw_length)
                positions.append(raw_length - 1)
                continue
            layout = req.context_decode_layout
            if layout is None:
                layout = ContextDecodeLayout.from_layout(
                    req.context_program.layout, row_ids.device
                )
                req.context_decode_layout = layout
            generated = raw_length - layout.prompt_length
            if generated < 1:
                raise ValueError("Context decode requires a computed output query")
            count = len(layout.raw_indices)
            row_pointer = 0
            if self.request_pool is not None:
                raw_row = request_row(self.request_pool, req.kv.req_pool_idx)
                row_pointer = raw_row.data_ptr()
                if raw_row.is_cuda:
                    raw_row.record_stream(torch.cuda.current_stream(raw_row.device))
                refs.append(raw_row)
            entries.append(
                (
                    layout.device_indices.data_ptr(),
                    layout.device_positions.data_ptr(),
                    layout.prompt_length,
                    count,
                    layout.next_position,
                    row_pointer,
                )
            )
            lengths.append(count + generated)
            positions.append(layout.next_position + generated - 1)
            refs.append(layout)
        packed = torch.tensor(
            [(*entry, n, p) for entry, n, p in zip(entries, lengths, positions)],
            dtype=torch.int64,
        ).to(row_ids.device, non_blocking=True)
        self.rows[row_ids] = packed[:, :6]
        return (
            torch.tensor(lengths, dtype=torch.int64),
            packed[:, 6].to(torch.int32),
            packed[:, 7],
            tuple(refs),
        )

    def fill(self, row_ids, lengths, indptr, output, *, starts=None, swa=False):
        from sglang.kernels.ops.attention.context_page_table import (
            context_decode_indices,
        )

        translator = self.translator
        mapping, multiplier = None, 1
        if translator.is_translating:
            mapping = translator._swa_v2p_table if swa else translator._full_v2p_table
            multiplier = (
                translator._swa_page_multiplier
                if swa
                else translator._full_page_multiplier
            )
        elif swa:
            mapping = getattr(self.pool, "full_to_swa_index_mapping", None)
        context_decode_indices[(len(lengths),)](
            self.rows,
            translator.req_to_token,
            row_ids,
            lengths,
            indptr,
            output,
            starts,
            mapping,
            ROW_STRIDE=translator.req_to_token.stride(0),
            TRANSLATE=mapping is not None,
            MULTIPLIER=multiplier,
            HAS_START=starts is not None,
        )

    def fill_window(self, row_ids, seq_lens, indptr, window, output):
        from sglang.kernels.ops.attention.context_page_table import (
            context_window_lengths,
        )

        bs = len(seq_lens)
        lengths = torch.empty_like(seq_lens)
        context_window_lengths[(bs,)](self.rows, row_ids, seq_lens, lengths, window)
        indptr = indptr[: bs + 1]
        indptr[1:] = lengths.cumsum(0)
        starts = seq_lens - lengths
        self.fill(row_ids, lengths, indptr, output, starts=starts, swa=True)
        return indptr, output, lengths, starts
