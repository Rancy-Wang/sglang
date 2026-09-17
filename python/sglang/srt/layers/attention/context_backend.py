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
    field_offsets: tuple[int, ...]
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
        offsets = (0, *np.cumsum([len(field) for field in fields]).tolist())
        return cls(
            torch.from_numpy(np.concatenate(fields)),
            offsets,
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
        fields = [
            packed[a:b] for a, b in zip(self.field_offsets[:-1], self.field_offsets[1:])
        ]
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
