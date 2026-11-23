# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Adapted from mscheong01/LoopVQ d5d58c83, identity-only serving.
"""
Allocator for the identity LoopVQ + BF16 mixed KV pool.

Three tiers in one flat slot-id namespace::

    quant     : [0, num_quant_pages * N_Q)
    HP-prefix : [HP_OFFSET, HP_OFFSET + num_hp_prefix_slots)        (paged, shared)
    HP-recent : [HP_OFFSET + num_hp_prefix_slots, ...)              (per-req slab)
    HP_OFFSET = num_quant_pages * N_Q

Quant + HP-prefix are pooled and contribute to ``allocator.size``. HP-recent
slabs are statically reserved per req-slot (released by
``LoopVQKVPool.release_req_slab``) and never enter the radix cache.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class LoopVQKVAllocator(BaseTokenToKVPoolAllocator):
    """Paged quant + HP-prefix free-lists, plus per-req HP-recent indexer."""

    def __init__(
        self,
        num_quant_pages: int,
        quant_tokens_per_page: int,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        hp_recent_ring_size: int,
        max_req_slots: int,
        num_hp_prefix_slots: int,
        dtype,
        hp_dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
        scheduler_size: int | None = None,
    ):
        self.num_quant_pages = int(num_quant_pages)
        self.N_Q = int(quant_tokens_per_page)
        self.max_req_slots = int(max_req_slots)
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.hp_recent_ring_size = int(hp_recent_ring_size)
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        if self.N_Q != 8 or self.hp_recent_tokens < 8:
            raise ValueError("LoopVQ requires eight-token pages and recent >= 8")
        if self.num_hp_prefix_slots % self.N_Q != 0:
            raise ValueError(
                f"num_hp_prefix_slots ({self.num_hp_prefix_slots}) must be "
                f"a multiple of N_Q ({self.N_Q})"
            )

        self._hp_offset = self.num_quant_pages * self.N_Q
        # HP-prefix is paged with ``N_Q`` slots per page; this many pages
        # are pooled in ``hp_prefix_free_pages``.
        self.num_hp_prefix_pages = self.num_hp_prefix_slots // self.N_Q
        # Slot ids for HP-recent start past the shared HP-prefix region.
        self._hp_recent_offset = self._hp_offset + self.num_hp_prefix_slots

        # Scheduler-facing capacity = quant pool + HP-prefix pool, in slot
        # units. Page 0 of the quant arena is reserved for padded writes.
        if scheduler_size is None:
            scheduler_size = (
                (self.num_quant_pages - 1) * self.N_Q + self.num_hp_prefix_slots
            )
        # HP-prefix page 0 is reserved too (see ``clear``). Discount it here
        # rather than in each caller: the idle leak check derives
        # used = size - available_size(), so counting a page ``clear`` never
        # hands out reports a permanent full_leaked=N_Q.
        # Only applies when an HP-prefix pool actually exists (num_hp_prefix_pages
        # >= 1) -- with hp_prefix_tokens=0 there is no page 0 to reserve, and
        # discounting it anyway made available_size exceed total_capacity by
        # N_Q (spurious "memory leak detected" at the idle self-check).
        if self.num_hp_prefix_pages >= 1:
            scheduler_size = int(scheduler_size) - self.N_Q
        else:
            scheduler_size = int(scheduler_size)
        super().__init__(
            int(scheduler_size), self.N_Q, dtype, device, kvcache, need_sort
        )
        self.hp_dtype = hp_dtype
        self.clear()

    # -- Configuration accessors -------------------------------------------

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    # Compatibility alias for callers that referenced ``num_pages`` on the
    # legacy shared-arena allocator. Prefer ``num_quant_pages``.
    @property
    def num_pages(self) -> int:
        return self.num_quant_pages

    # -- Clear / state -----------------------------------------------------

    def clear(self):
        self._kvcache.wait_pending_forward()
        # Quant: reserve page 0 for padded/dummy writes.
        self.free_pages = torch.arange(
            1, self.num_quant_pages, dtype=torch.int64, device=self.device
        )
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        # HP-prefix: shared paged free list. Page 0 is reserved for the same
        # reason quant page 0 is -- ``cuda_graph_runner`` sends padded decode
        # rows to ``hp_global_offset``, which is the HP-PREFIX base, not the
        # quant base, so reserving only the quant page left the real target
        # allocatable and padded writes clobbered a live request's sink tokens.
        self.hp_prefix_free_pages = torch.arange(
            1, max(1, self.num_hp_prefix_pages), dtype=torch.int64, device=self.device
        )
        self.hp_prefix_release_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self.free_group = None
        self.request_state = {}
        self._kvcache._next_slab_offset.zero_()
        self._kvcache._flush_counter.zero_()

    def release_request(self, req_pool_idx):
        """Drop host/ring state after all mapped physical pages are released."""
        self.request_state.pop(int(req_pool_idx), None)
        self._kvcache.release_req_slab(req_pool_idx)

    # -- Availability ------------------------------------------------------

    def _free_page_count(self) -> int:
        # ``.numel()`` is host-known tensor metadata; no D2H sync.
        return self.free_pages.numel() + self.release_pages.numel()

    def _hp_prefix_free_slots(self) -> int:
        return (
            self.hp_prefix_free_pages.numel() + self.hp_prefix_release_pages.numel()
        ) * self.N_Q

    def available_size(self):
        """Pooled-slot capacity (quant + shared HP-prefix), in slot units."""
        return self._free_page_count() * self.N_Q + self._hp_prefix_free_slots()

    def quant_available_size(self) -> int:
        """Capacity servable by ``alloc_quant``, in slot units.

        ``available_size`` sums the quant arena with HP-prefix. That is right
        for the scheduler's global accounting and wrong for any caller sizing a
        QUANT allocation: HP-prefix pages cannot serve a quant flush. A pool
        that is quant-starved but HP-rich reports ample capacity through
        ``available_size`` and then raises inside ``alloc_quant``.
        """
        return self._free_page_count() * self.N_Q

    def debug_print(self) -> str:
        return (
            f"#quant-free-pages: {self.free_pages.numel()}, "
            f"#quant-release-pages: {self.release_pages.numel()}, "
            f"#hp-prefix-free-pages: {self.hp_prefix_free_pages.numel()}, "
            f"#hp-prefix-release-pages: {self.hp_prefix_release_pages.numel()}, "
            f"size: {self.size}, N_Q={self.N_Q}, "
            f"hp_offset={self._hp_offset}, "
            f"hp_recent_offset={self._hp_recent_offset}, "
            f"recent_ring={self.hp_recent_ring_size}, "
            f"max_req_slots={self.max_req_slots}"
        )

    def backup_state(self):
        return (
            self.free_pages.clone(),
            self.release_pages.clone(),
            self.hp_prefix_free_pages.clone(),
            self.hp_prefix_release_pages.clone(),
        )

    def restore_state(self, state):
        (
            self.free_pages,
            self.release_pages,
            self.hp_prefix_free_pages,
            self.hp_prefix_release_pages,
        ) = state

    # -- Default alloc path (extend / generic uses quant tier) -------------

    def alloc(self, need_size: int):
        return self.alloc_quant(need_size)

    # -- Per-req recent-slab base ----------------------------------------

    def _recent_slab_base_slots(
        self, req_pool_indices: torch.Tensor
    ) -> torch.Tensor:
        """Global HP slot id of recent_slab[0] for each given req_pool_idx."""
        return (
            req_pool_indices.to(self.device).to(torch.int64)
            * self.hp_recent_ring_size
            + self._hp_recent_offset
        )

    # -- HP-prefix allocation (shared, paged) ----------------------------

    def alloc_hp_prefix(
        self,
        req_pool_indices: torch.Tensor,
        per_req_counts: Sequence[int],
    ) -> torch.Tensor:
        # Pooled allocation; req_pool_indices kept for API symmetry with
        # alloc_hp_recent (only used to drive iteration order). Total must be
        # N_Q-aligned so the result lands on whole HP-prefix pages.
        per_req_counts = list(per_req_counts)
        total = int(sum(per_req_counts))
        if total <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        for c in per_req_counts:
            if c < 0 or c > self.hp_prefix_tokens:
                raise ValueError(
                    f"alloc_hp_prefix per-req count {c} out of range "
                    f"[0, {self.hp_prefix_tokens}]"
                )
        if total % self.N_Q != 0:
            raise ValueError(
                f"alloc_hp_prefix total ({total}) must be a multiple of "
                f"N_Q={self.N_Q}; round per-req counts up at the caller."
            )

        num_pages = total // self.N_Q
        if (
            self.need_sort
            and num_pages > self.hp_prefix_free_pages.numel()
        ):
            self.merge_and_sort_free()
        if num_pages > self.hp_prefix_free_pages.numel():
            raise RuntimeError(
                "Mixed KV: HP-prefix pool exhausted; consider raising "
                "SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS or evicting cache. "
                f"{self.debug_print()}"
            )

        pages = self.hp_prefix_free_pages[:num_pages]
        self.hp_prefix_free_pages = self.hp_prefix_free_pages[num_pages:]
        # Each HP-prefix page contributes N_Q consecutive slots.
        # Layout: page p → local indices [p * N_Q, p * N_Q + N_Q).
        slots_local = (
            pages[:, None] * self.N_Q
            + torch.arange(self.N_Q, dtype=torch.int64, device=self.device)
        ).reshape(-1)
        # Translate to global HP slot ids.
        return (slots_local + self._hp_offset).contiguous()

    def alloc_hp_recent(
        self,
        req_pool_indices: torch.Tensor,
        per_req_counts: Sequence[int],
    ) -> torch.Tensor:
        # Per-req ring cursor on the pool's ``_next_slab_offset``. Decode hot
        # path (uniform count=1) takes the vectorised sync-free branch.
        if isinstance(per_req_counts, torch.Tensor):
            counts_list = per_req_counts.tolist()
        else:
            counts_list = list(per_req_counts)
        bs = len(counts_list)
        total = int(sum(counts_list))
        if total <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        for c in counts_list:
            if c < 0 or c > self.hp_recent_ring_size:
                raise ValueError(
                    f"alloc_hp_recent per-req count {c} out of range "
                    f"[0, {self.hp_recent_ring_size}]"
                )

        cursor_buf = self._kvcache._next_slab_offset
        ring = self.hp_recent_ring_size

        idx_dev = req_pool_indices.to(self.device).to(torch.int64)
        bases = idx_dev * ring + self._hp_recent_offset  # [bs]
        old_cursors = cursor_buf[idx_dev].to(torch.int64)  # [bs]

        # Decode hot path: every request asks for one HP-recent slot.
        if total == bs and all(c == 1 for c in counts_list):
            slots = bases + old_cursors
            new_cursors = ((old_cursors + 1) % ring).to(torch.int32)
            cursor_buf[idx_dev] = new_cursors
            return slots.contiguous()

        # General path (extend, ragged counts).
        bases_h = bases.tolist()
        cursors_h = old_cursors.tolist()
        chunks = []
        new_cursors_list = []
        for base, count, cur in zip(bases_h, counts_list, cursors_h):
            if count == 0:
                new_cursors_list.append(cur)
                continue
            offsets = [(cur + j) % ring for j in range(count)]
            new_cursors_list.append((cur + count) % ring)
            offsets_t = torch.tensor(offsets, dtype=torch.int64, device=self.device)
            chunks.append(int(base) + offsets_t)

        # Single host→device write to update cursors.
        new_cursor_t = torch.tensor(
            new_cursors_list, dtype=torch.int32, device=self.device
        )
        cursor_buf[idx_dev] = new_cursor_t

        if not chunks:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(chunks).contiguous()

    # -- Quant allocation --------------------------------------------------

    def alloc_quant(self, need_size: int):
        # Allocates whole quant pages; need_size must be N_Q-aligned.
        if need_size <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if need_size % self.N_Q != 0:
            raise ValueError(
                f"alloc_quant requires a multiple of N_Q={self.N_Q}, got {need_size}"
            )
        num_pages = need_size // self.N_Q
        if self.need_sort and num_pages > self.free_pages.numel():
            self.merge_and_sort_free()
        if num_pages > self.free_pages.numel():
            return None

        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        slots = (
            pages[:, None] * self.N_Q
            + torch.arange(self.N_Q, dtype=torch.int64, device=self.device)
        ).reshape(-1)
        return slots.contiguous()

    # -- Free path (sync-free, GPU-resident) -------------------------------

    def free(self, free_index: torch.Tensor):
        # Quant -> free quant page pool; HP-prefix -> free HP-prefix page
        # pool; HP-recent -> no-op (reset by release_req_slab). Masked-select
        # branches host-sync, matching the surrounding flush path.
        if free_index.numel() == 0:
            return
        if self.free_group is not None:
            self.free_group.append(self._copy_for_free_group(free_index))
            return

        self._kvcache.wait_pending_forward()
        idx = free_index.to(torch.int64)
        # Quant slot ids: aggregate to whole quant pages.
        is_quant = (idx >= self.N_Q) & (idx < self._hp_offset)
        quant_index = idx[is_quant]
        if quant_index.numel() > 0:
            quant_pages = torch.unique((quant_index // self.N_Q).to(torch.int64))
            if self.need_sort:
                self.release_pages = torch.cat([quant_pages, self.release_pages])
            else:
                self.free_pages = torch.cat([quant_pages, self.free_pages])

        # HP-prefix slot ids: aggregate to whole HP-prefix pages.
        if self.num_hp_prefix_slots > 0:
            is_hp_prefix = (idx >= self._hp_offset + self.N_Q) & (idx < self._hp_recent_offset)
            hp_prefix_index = idx[is_hp_prefix]
            if hp_prefix_index.numel() > 0:
                local = hp_prefix_index - self._hp_offset
                hp_pages = torch.unique((local // self.N_Q).to(torch.int64))
                if self.need_sort:
                    self.hp_prefix_release_pages = torch.cat(
                        [hp_pages, self.hp_prefix_release_pages]
                    )
                else:
                    self.hp_prefix_free_pages = torch.cat(
                        [hp_pages, self.hp_prefix_free_pages]
                    )

        # HP-recent slot ids: no-op (per-req slab).

    def merge_and_sort_free(self):
        if self.release_pages.numel() > 0:
            self.free_pages = torch.cat([self.free_pages, self.release_pages])
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )
        if self.hp_prefix_release_pages.numel() > 0:
            self.hp_prefix_free_pages = torch.cat(
                [self.hp_prefix_free_pages, self.hp_prefix_release_pages]
            )
            self.hp_prefix_free_pages, _ = torch.sort(self.hp_prefix_free_pages)
            self.hp_prefix_release_pages = torch.empty(
                (0,), dtype=self.hp_prefix_release_pages.dtype, device=self.device
            )

    # -- Passthroughs ------------------------------------------------------

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

    def check_decode_capacity(self, *, num_tokens, tree_cache, requests=None,
                              spec_algorithm=None):
        # Conservative: one whole quant page per active request may be needed.
        count = len(requests) if requests is not None else num_tokens
        return self.quant_available_size() >= count * self.N_Q

    def free_segments(self, segments):
        # Logical pages can straddle tiers. No prefix sharing is supported.
        chunks = [indices for indices, _ in segments if indices.numel()]
        if chunks:
            self.free(torch.cat(chunks))

    def free_full_segments(self, segments):
        self.free_segments(segments)


def alloc_for_extend(batch):
    """Fresh full-prompt admission; never import a cached prefix."""
    from sglang.srt.mem_cache.allocation import alloc_req_slots

    allocator = batch.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    pool.wait_pending_forward()
    if any(int(x) != 0 for x in batch.prefix_lens):
        raise ValueError("LoopVQ does not support cached/chunked prefill prefixes")
    lengths = [int(x) for x in batch.seq_lens_cpu.tolist()]
    if any(n != int(e) for n, e in zip(lengths, batch.extend_lens)):
        raise ValueError("LoopVQ requires an entire prompt in each extend")
    prefix_counts = [min(n, pool.hp_prefix_tokens) for n in lengths]
    recent_counts = [min(n - p, pool.hp_recent_tokens) for n, p in zip(lengths, prefix_counts)]
    quant_counts = [n - p - r for n, p, r in zip(lengths, prefix_counts, recent_counts)]
    align = lambda n: (n + 7) // 8 * 8
    prefixes = [align(n) for n in prefix_counts]
    quantized = [align(n) for n in quant_counts]
    if sum(quantized) > allocator.quant_available_size() or sum(prefixes) > allocator._hp_prefix_free_slots():
        raise RuntimeError("LoopVQ prefill capacity exhausted before allocation")
    ids = alloc_req_slots(batch.req_to_token_pool, batch.reqs, batch.tree_cache)
    ids_cpu = torch.tensor(ids, dtype=torch.int64)
    ids_device = ids_cpu.to(batch.device, non_blocking=True)
    pool.release_req_slab(ids_device)
    pslots = allocator.alloc_hp_prefix(ids_device, prefixes)
    qslots = allocator.alloc_quant(sum(quantized))
    rslots = allocator.alloc_hp_recent(ids_device, recent_counts)
    if qslots is None:
        raise RuntimeError("LoopVQ quant page allocation failed")
    parts, po, qo, ro = [], 0, 0, 0
    for i, req_idx in enumerate(ids):
        p, q, r = prefix_counts[i], quant_counts[i], recent_counts[i]
        loc = torch.cat((pslots[po:po + p], qslots[qo:qo + q], rslots[ro:ro + r]))
        batch.req_to_token_pool.write((req_idx, slice(0, lengths[i])), loc.to(torch.int32))
        allocator.request_state[int(req_idx)] = {
            "recent": r,
            "prefix_tail": pslots[po + p:po + prefixes[i]],
        }
        parts.append(loc)
        po += prefixes[i]
        qo += quantized[i]
        ro += r
    out = torch.cat(parts) if parts else torch.empty(0, device=batch.device, dtype=torch.int64)
    batch.req_to_token_pool.alloc_aux_to_lengths(req_pool_indices_cpu=ids_cpu,
                                                target_seq_lens_cpu=batch.seq_lens_cpu)
    for req, seq_len in zip(batch.reqs, lengths):
        req.kv.kv_allocated_len = seq_len
        req.kv.kv_committed_len = seq_len
    return out, ids_device, ids_cpu


def alloc_for_decode(batch, token_per_req):
    """Append BF16 tokens; encode both tiers before publishing a demoted page.

    Allocation runs outside CUDA graphs. The previous forward is ordered before
    remapping its readable slots. No quant row is visible until both K/V exist.
    """
    from types import SimpleNamespace

    if token_per_req != 1:
        raise ValueError("LoopVQ supports ordinary one-token AR decode only")
    allocator = batch.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    pool.wait_pending_forward()
    req_ids = batch.req_pool_indices_cpu.tolist()
    lengths = batch.seq_lens_cpu.tolist()
    locations = []
    flush_src, flush_dst, flush_updates = [], [], []
    for rid, n in zip(req_ids, lengths):
        state = allocator.request_state[int(rid)]
        one_req = torch.tensor([rid], device=batch.device, dtype=torch.int64)
        if n < pool.hp_prefix_tokens:
            if not state["prefix_tail"].numel():
                state["prefix_tail"] = allocator.alloc_hp_prefix(one_req, [8])
            loc = state["prefix_tail"][:1]
            state["prefix_tail"] = state["prefix_tail"][1:]
        else:
            loc = allocator.alloc_hp_recent(one_req, [1])
            state["recent"] += 1
            if state["recent"] >= pool.hp_recent_tokens + 8:
                start = n + 1 - state["recent"]
                src = batch.req_to_token_pool.req_to_token[rid, start:start + 8].to(torch.int64)
                dst = allocator.alloc_quant(8)
                if dst is None:
                    raise RuntimeError("LoopVQ quant pool exhausted at decode flush")
                flush_src.append(src - pool.hp_global_offset)
                flush_dst.append(dst)
                flush_updates.append((rid, start, dst))
                state["recent"] -= 8
        locations.append(loc)
    if flush_src:
        src, dst = torch.cat(flush_src), torch.cat(flush_dst)
        plan = SimpleNamespace(src_hp_slot=src, dst_quant_slots=dst,
                               valid_mask=torch.ones_like(src, dtype=torch.bool))
        pool.vq_flush_k(plan)
        pool.vq_flush_v(plan)
        for rid, start, target in flush_updates:
            batch.req_to_token_pool.write((rid, slice(start, start + 8)), target.to(torch.int32))
    out = torch.cat(locations)
    batch.req_to_token_pool.write((batch.req_pool_indices, batch.seq_lens), out.to(torch.int32))
    batch.req_to_token_pool.alloc_aux_to_lengths(req_pool_indices_cpu=batch.req_pool_indices_cpu,
                                                target_seq_lens_cpu=batch.seq_lens_cpu + 1)
    for req in batch.reqs:
        req.kv.kv_allocated_len += 1
        req.kv.kv_committed_len += 1
    return out
