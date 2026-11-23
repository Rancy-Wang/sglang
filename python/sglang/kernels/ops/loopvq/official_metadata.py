# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Capture-safe packed tier lists for the canonical prefix/quant/recent layout.

All addresses are allocated before graph capture. Slot zero refreshes these
buffers on-device once per forward; the other 191 attention slots reuse them.
The allocator guarantees a contiguous logical quant interval, even when its
physical pages are fragmented. Padding row zero is represented by an empty list.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tier_metadata(PTR, LOC, QPTR, HPTR, QLEN, HLEN, FULL, QSPLITS, HSPLITS,
                   B: tl.constexpr, BB: tl.constexpr, PREFIX: tl.constexpr,
                   OFFSET: tl.constexpr, CAP: tl.constexpr, HP_CAP: tl.constexpr,
                   SEARCH: tl.constexpr, GROUP: tl.constexpr, OCC: tl.constexpr):
    row = tl.arange(0, BB)
    mask = row < B
    begin = tl.load(PTR + row, mask, 0)
    end = tl.load(PTR + row + 1, mask, 0)
    n = end - begin
    tl.device_assert((~mask) | ((n >= 0) & (n <= CAP)), "LoopVQ context exceeds official metadata capacity")
    first = tl.load(LOC + begin, mask & (n > 0), 0)
    tl.device_assert((first == 0) | (first >= OFFSET), "LoopVQ row must begin with an HP prefix")
    n = tl.where(mask & (first != 0), n, 0)
    prefix = tl.minimum(n, PREFIX)
    low, high = prefix, n
    for _ in tl.static_range(SEARCH):
        mid = (low + high) // 2
        slot = tl.load(LOC + begin + mid, mask & (mid < n), OFFSET)
        is_quant = slot < OFFSET
        low = tl.where(is_quant, mid + 1, low)
        high = tl.where(is_quant, high, mid)
    qlen = low - prefix
    hlen = n - qlen
    tl.device_assert(hlen <= HP_CAP, "LoopVQ HP window exceeds metadata capacity")
    tl.store(QLEN + row, qlen, mask)
    tl.store(HLEN + row, hlen, mask)
    tl.store(FULL + row, n, mask)
    tl.store(QPTR, 0)
    tl.store(HPTR, 0)
    tl.store(QPTR + row + 1, tl.cumsum(qlen), mask)
    tl.store(HPTR + row + 1, tl.cumsum(hlen), mask)
    # Official _fix_splits_group: capacity floor plus two SM waves of work.
    group_max = tl.maximum(qlen, tl.gather(qlen, row ^ 1, 0)) if GROUP == 2 else qlen
    want = tl.minimum(48, tl.maximum(1, tl.cdiv(group_max, 2048 // GROUP)))
    if OCC > 1:
        want = tl.maximum(want, tl.minimum(OCC, tl.maximum(1, tl.cdiv(group_max, 256))))
    tl.store(QSPLITS + row, tl.minimum(want, 48), mask)
    tl.store(HSPLITS + row, 12, mask)


@triton.jit
def _tier_indices(PTR, LOC, QPTR, HPTR, QLEN, HLEN, FULL, QLOC, HLOC,
                  PREFIX: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(PTR + row)
    qlen, hlen, n = tl.load(QLEN + row), tl.load(HLEN + row), tl.load(FULL + row)
    prefix = tl.minimum(n, PREFIX)
    qbase, hbase = tl.load(QPTR + row), tl.load(HPTR + row)
    quant = tl.load(LOC + begin + prefix + x, x < qlen, 0)
    hp_position = tl.where(x < prefix, x, qlen + x)
    hp = tl.load(LOC + begin + hp_position, x < hlen, OFFSET) - OFFSET
    tl.store(QLOC + qbase + x, quant, x < qlen)
    tl.store(HLOC + hbase + x, hp, x < hlen)


class OfficialMetadata:
    def __init__(self, *, max_batch, max_context_len, prefix_tokens, recent_tokens,
                 hp_offset, device, sm_count):
        if not 1 <= max_context_len <= 40960 or max_batch < 1:
            raise ValueError("Official LoopVQ metadata requires context <= 40960 and positive batch")
        if min(prefix_tokens, recent_tokens) < 8 or prefix_tokens % 8 or recent_tokens % 8:
            raise ValueError("Official LoopVQ requires positive eight-aligned windows")
        self.max_batch, self.max_context_len = max_batch, max_context_len
        self.prefix, self.hp_offset, self.sm_count = prefix_tokens, hp_offset, sm_count
        self.hp_capacity = min(max_context_len, prefix_tokens + recent_tokens + 8)
        allocate = lambda shape: torch.empty(shape, dtype=torch.int32, device=device)
        self.quant_indptr, self.hp_indptr = allocate(max_batch + 1), allocate(max_batch + 1)
        self.quant_lens, self.hp_lens, self.full_seq_len = (allocate(max_batch) for _ in range(3))
        self.quant_splits, self.hp_splits = allocate(max_batch), allocate(max_batch)
        self.quant_indices = allocate(max_batch * max_context_len)
        self.hp_indices = allocate(max_batch * self.hp_capacity)

    def update(self, kv_indptr, kv_indices, batch_size):
        if not 1 <= batch_size <= self.max_batch or kv_indptr.numel() < batch_size + 1:
            raise ValueError("Official LoopVQ metadata batch exceeds its persistent capacity")
        group = 1 if batch_size == 1 else 2
        occupancy = 2 * self.sm_count // (triton.cdiv(batch_size, group) * 16)
        _tier_metadata[(1,)](
            kv_indptr, kv_indices, self.quant_indptr, self.hp_indptr,
            self.quant_lens, self.hp_lens, self.full_seq_len, self.quant_splits, self.hp_splits,
            batch_size, triton.next_power_of_2(batch_size), self.prefix, self.hp_offset,
            self.max_context_len, self.hp_capacity, self.max_context_len.bit_length(),
            group, occupancy, num_warps=1, debug=True,
        )
        _tier_indices[(batch_size, triton.cdiv(self.max_context_len, 512))](
            kv_indptr, kv_indices, self.quant_indptr, self.hp_indptr,
            self.quant_lens, self.hp_lens, self.full_seq_len, self.quant_indices, self.hp_indices,
            self.prefix, self.hp_offset, 512, num_warps=4,
        )

