"""Decode read indices over native raw KV rows, without moving cached K/V."""

import triton
import triton.language as tl


@triton.jit
def context_window_lengths(Registry, Rows, Lengths, Output, WINDOW: tl.constexpr):
    b = tl.program_id(0)
    row = tl.load(Rows + b)
    length = tl.load(Lengths + b)
    entry = Registry + row * 7
    pointer = tl.load(entry)
    result = tl.minimum(length, WINDOW + 1)
    if pointer != 0:
        positions = tl.load(entry + 1).to(tl.pointer_type(tl.int32))
        active = tl.load(entry + 3).to(tl.int32)
        next_pos = tl.load(entry + 4).to(tl.int32)
        generated = length - active
        # Native RadixAttention's window is the inclusive distance q_pos-k_pos.
        lower = next_pos + generated - 1 - WINDOW
        unordered = tl.load(entry + 6) != 0
        kept = tl.full((), 0, tl.int32)
        if unordered:
            for block in range(tl.cdiv(active, 256)):
                ids = block * 256 + tl.arange(0, 256)
                pos = tl.load(positions + ids, mask=ids < active, other=0)
                kept += tl.sum(((ids < active) & (pos >= lower)).to(tl.int32))
        else:
            lo = tl.full((), 0, tl.int32)
            hi = active
            while lo < hi:
                mid = (lo + hi) // 2
                pos = tl.load(positions + mid)
                lo = tl.where(pos < lower, mid + 1, lo)
                hi = tl.where(pos < lower, hi, mid)
            kept = active - lo
        skipped_outputs = tl.maximum(0, lower - next_pos)
        result = kept + generated - skipped_outputs
    tl.store(Output + b, result)


@triton.jit
def context_decode_indices(
    Registry,
    Table,
    Rows,
    Lengths,
    Indptr,
    Output,
    Starts,
    Mapping,
    FullLengths,
    ROW_STRIDE: tl.constexpr,
    TRANSLATE: tl.constexpr,
    MULTIPLIER: tl.constexpr,
    HAS_START: tl.constexpr,
    WINDOW: tl.constexpr = -1,
    BLOCK: tl.constexpr = 256,
):
    b = tl.program_id(0)
    row = tl.load(Rows + b)
    length = tl.load(Lengths + b)
    offset = tl.load(Indptr + b)
    start = tl.load(Starts + b) if HAS_START else 0
    entry = Registry + row * 7
    pointer = tl.load(entry)
    active = tl.load(entry + 3).to(tl.int32)
    prompt = tl.load(entry + 2).to(tl.int32)
    raw_ids = pointer.to(tl.pointer_type(tl.int32))
    row_pointer = tl.load(entry + 5)
    raw_row = Table + row * ROW_STRIDE
    if row_pointer != 0:
        raw_row = row_pointer.to(tl.pointer_type(tl.int32))
    unordered = tl.load(entry + 6) != 0
    if WINDOW >= 0 and unordered:
        full_length = tl.load(FullLengths + b)
        positions = tl.load(entry + 1).to(tl.pointer_type(tl.int32))
        next_pos = tl.load(entry + 4).to(tl.int32)
        lower = next_pos + full_length - active - 1 - WINDOW
        written = tl.full((), 0, tl.int32)
        for block in range(tl.cdiv(full_length, BLOCK)):
            logical = block * BLOCK + tl.arange(0, BLOCK)
            valid = logical < full_length
            in_prompt = logical < active
            p = tl.load(positions + logical, mask=valid & in_prompt, other=0)
            p = tl.where(in_prompt, p, next_pos + logical - active)
            keep = valid & (p >= lower)
            raw = tl.load(raw_ids + logical, mask=keep & in_prompt, other=0)
            raw = tl.where(in_prompt, raw, prompt + logical - active)
            slot = tl.load(raw_row + raw, mask=keep, other=0)
            if TRANSLATE:
                slot = tl.load(Mapping + slot, mask=keep, other=0) * MULTIPLIER
            rank = tl.cumsum(keep.to(tl.int32)) - 1
            tl.store(Output + offset + written + rank, slot, mask=keep)
            written += tl.sum(keep.to(tl.int32))
    else:
        for block in range(tl.cdiv(length, BLOCK)):
            i = block * BLOCK + tl.arange(0, BLOCK)
            valid = i < length
            logical = start + i
            raw = logical
            if pointer != 0:
                in_prompt = logical < active
                raw = tl.load(raw_ids + logical, mask=valid & in_prompt, other=0)
                raw = tl.where(in_prompt, raw, prompt + logical - active)
            slot = tl.load(raw_row + raw, mask=valid, other=0)
            if TRANSLATE:
                slot = tl.load(Mapping + slot, mask=valid, other=0) * MULTIPLIER
            tl.store(Output + offset + i, slot, mask=valid)
