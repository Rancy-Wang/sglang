"""Per-request raw index storage, independent of model positions and KV size.

Only overflowing Context requests allocate a side row. Native graph-visible
request tables never resize; the side row has its final decode capacity at
admission and contains page IDs, not extra model KV.
"""

import torch


def request_row(pool, index):
    rows = getattr(pool, "_context_rows", None)
    if rows and index in rows:
        return rows[index]
    return pool.req_to_token[index]


def prepare_request_row(pool, req):
    if getattr(req, "context_program", None) is None:
        return
    # Native overlap may allocate a discarded decode step past the final token.
    capacity = len(req.origin_input_ids) + req.sampling_params.max_new_tokens + 4
    if capacity <= pool.req_to_token.shape[1]:
        return
    rows = getattr(pool, "_context_rows", None)
    if rows is None:
        rows = pool._context_rows = {}
    index = req.kv.req_pool_idx
    if index in rows:
        if len(rows[index]) < capacity:
            raise RuntimeError("Context raw row cannot grow while a request is live")
        return
    row = torch.full((capacity,), -1, dtype=torch.int32, device=pool.device)
    rows[index] = row
    if getattr(pool, "_context_row_pointers", None) is None:
        table = pool.req_to_token
        pool._context_row_pointers = (
            torch.arange(table.shape[0], dtype=torch.int64, device=pool.device)
            * table.stride(0)
            * table.element_size()
            + table.data_ptr()
        )
    pool._context_row_pointers[index] = row.data_ptr()


def release_request_row(pool, index):
    rows = getattr(pool, "_context_rows", None)
    if rows and index in rows:
        del rows[index]
        pool._context_row_pointers[index] = pool.req_to_token[index].data_ptr()


def row_pointers(pool):
    return pool._context_row_pointers if getattr(pool, "_context_rows", None) else None


def write_request_slots(pool, indices, values):
    rows = getattr(pool, "_context_rows", None)
    if not rows:
        pool.req_to_token[indices] = values
        return
    row, columns = indices
    if isinstance(row, int):
        request_row(pool, row)[columns] = values
    elif pool.req_to_token.is_cuda:
        import triton

        from sglang.kernels.ops.memory.context_rows import write_row_slots

        write_row_slots[(1,)](
            pool._context_row_pointers,
            row,
            columns,
            values,
            row.numel(),
            triton.next_power_of_2(row.numel()),
        )
    else:
        for index, column, value in zip(row.tolist(), columns.tolist(), values):
            request_row(pool, index)[column] = value


def validate_positions(program, limit):
    """Match mini's occurrence admission, including intermediate RoPE positions."""
    layout = program.layout
    for positions in (
        layout.birth_positions,
        layout.transition_old_positions,
        layout.transition_new_positions,
        layout.positions,
    ):
        if len(positions) and (
            int(positions.min()) < 0 or int(positions.max()) >= limit
        ):
            raise ValueError(
                "An occurrence execution position exceeds the model/RoPE limit"
            )
    active = int(layout.keep_mask.count_nonzero())
    if not active or layout.next_position <= int(
        layout.positions[layout.keep_mask].max()
    ):
        raise ValueError("Context next position must cover a nonempty active prompt")
    if layout.next_position >= limit:
        raise ValueError("Context has no room for an output token")
    return active
