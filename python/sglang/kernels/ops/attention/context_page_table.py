"""Decode read indices over native raw KV rows, without moving cached K/V."""

import triton
import triton.language as tl


@triton.jit
def _previous_window_run(Tree, size, end, lower):
    """Rightmost qualifying run strictly before end, in O(log runs)."""
    node = size + end - 1
    found = tl.full((), -1, tl.int32)
    searching = end > 0
    while searching:
        if tl.load(Tree + node) >= lower:
            while node < size:
                right = node * 2 + 1
                node = tl.where(tl.load(Tree + right) >= lower, right, right-1)
            found = node-size
            searching = False
        else:
            while (node > 1) & (node % 2 == 0):
                node = node // 2
            if node == 1:
                searching = False
            else:
                node -= 1
    return found


@triton.jit
def _prompt_window_count(Index, lower, budget):
    runs = tl.load(Index)
    size = tl.load(Index + 1)
    tree = Index + 2 + 3*runs
    end, count = runs, tl.full((), 0, tl.int32)
    while (end > 0) & (count < budget):
        run = _previous_window_run(tree, size, end, lower)
        if run < 0:
            end = 0
        else:
            start = tl.load(Index + 2 + run)
            stop = tl.load(Index + 2 + runs + run)
            position = tl.load(Index + 2 + 2*runs + run)
            eligible = stop - tl.maximum(start, start + lower - position)
            count += tl.minimum(eligible, budget-count)
            end = run
    return count


@triton.jit
def context_window_lengths(Registry, Rows, Lengths, Output, WINDOW: tl.constexpr):
    b = tl.program_id(0)
    row = tl.load(Rows + b)
    # Eager batches carry int64 lengths; graph buffers use int32. Registry
    # positions and supported token capacities are int32 in both paths.
    length = tl.load(Lengths + b).to(tl.int32)
    entry = Registry + row * 8
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
            index = tl.load(entry + 7).to(tl.pointer_type(tl.int32))
            kept = _prompt_window_count(index, lower, tl.maximum(0, WINDOW + 1 - generated))
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
        result = tl.minimum(WINDOW + 1, kept + generated - skipped_outputs)
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
    length = tl.load(Lengths + b).to(tl.int32)
    offset = tl.load(Indptr + b)
    start = tl.load(Starts + b) if HAS_START else 0
    entry = Registry + row * 8
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
        full_length = tl.load(FullLengths + b).to(tl.int32)
        next_pos = tl.load(entry + 4).to(tl.int32)
        generated = full_length - active
        lower = next_pos + generated - 1 - WINDOW
        generated_count = tl.minimum(generated, WINDOW + 1)
        remaining = length - generated_count
        index = tl.load(entry + 7).to(tl.pointer_type(tl.int32))
        runs = tl.load(index)
        size = tl.load(index + 1)
        tree = index + 2 + 3*runs
        end = runs
        while (remaining > 0) & (end > 0):
            run = _previous_window_run(tree, size, end, lower)
            if run < 0:
                end = 0
            else:
                begin = tl.load(index + 2 + run)
                stop = tl.load(index + 2 + runs + run)
                position = tl.load(index + 2 + 2*runs + run)
                first = tl.maximum(begin, begin + lower - position)
                take = tl.minimum(stop-first, remaining)
                for block in range(tl.cdiv(take, BLOCK)):
                    i = block*BLOCK + tl.arange(0, BLOCK)
                    valid = i < take
                    logical = stop-take+i
                    raw = tl.load(raw_ids + logical, mask=valid, other=0)
                    slot = tl.load(raw_row + raw, mask=valid, other=0)
                    if TRANSLATE:
                        slot = tl.load(Mapping + slot, mask=valid, other=0) * MULTIPLIER
                    tl.store(Output + offset + remaining-take+i, slot, mask=valid)
                remaining -= take
                end = run
        for block in range(tl.cdiv(generated_count, BLOCK)):
            i = block*BLOCK + tl.arange(0, BLOCK)
            valid = i < generated_count
            raw = prompt + generated-generated_count+i
            slot = tl.load(raw_row + raw, mask=valid, other=0)
            if TRANSLATE:
                slot = tl.load(Mapping + slot, mask=valid, other=0) * MULTIPLIER
            tl.store(Output + offset + length-generated_count+i, slot, mask=valid)
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
