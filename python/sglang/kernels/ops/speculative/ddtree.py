from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_ddtree_heap_kernel(
    top_log_probs,
    top_token_ids,
    out_node_token_ids,
    out_node_depths,
    out_parents,
    out_visibility,
    out_actual_tree_sizes,
    heap_scores,
    heap_parents,
    heap_depths,
    heap_ranks,
    BUDGET: tl.constexpr,
    MAX_NODES: tl.constexpr,
    DEPTH_LIMIT: tl.constexpr,
    TOPK: tl.constexpr,
    HEAP_CAP: tl.constexpr,
    NODE_BLOCK: tl.constexpr,
    VIS_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    heap_base = pid * HEAP_CAP
    heap_offsets = tl.arange(0, HEAP_CAP)
    tl.store(heap_scores + heap_base + heap_offsets, -float("inf"))
    tl.store(heap_parents + heap_base + heap_offsets, 0)
    tl.store(heap_depths + heap_base + heap_offsets, 0)
    tl.store(heap_ranks + heap_base + heap_offsets, 0)

    node_offsets = tl.arange(0, NODE_BLOCK)
    node_mask = node_offsets < BUDGET
    tl.store(out_node_token_ids + pid * BUDGET + node_offsets, 0, mask=node_mask)
    tl.store(out_node_depths + pid * BUDGET + node_offsets, 0, mask=node_mask)

    parent_mask = node_offsets < MAX_NODES
    tl.store(out_parents + pid * MAX_NODES + node_offsets, -1, mask=parent_mask)

    vis_offsets = tl.arange(0, VIS_BLOCK)
    vis_mask = vis_offsets < (MAX_NODES * MAX_NODES)
    vis_base = pid * MAX_NODES * MAX_NODES
    tl.store(out_visibility + vis_base + vis_offsets, False, mask=vis_mask)
    tl.store(out_visibility + vis_base, True)

    first_score = tl.load(top_log_probs + pid * DEPTH_LIMIT * TOPK)
    tl.store(heap_scores + heap_base, first_score)
    tl.store(heap_parents + heap_base, 0)
    tl.store(heap_depths + heap_base, 1)
    tl.store(heap_ranks + heap_base, 0)

    cols = tl.arange(0, NODE_BLOCK)
    col_mask = cols < MAX_NODES
    for node_i in range(0, BUDGET):
        scores = tl.load(heap_scores + heap_base + heap_offsets)
        best_score = tl.max(scores, axis=0)
        best_pos = tl.min(tl.where(scores == best_score, heap_offsets, HEAP_CAP), axis=0)

        parent_idx = tl.load(heap_parents + heap_base + best_pos)
        depth = tl.load(heap_depths + heap_base + best_pos)
        rank = tl.load(heap_ranks + heap_base + best_pos)

        current_idx = node_i + 1
        top_offset = pid * DEPTH_LIMIT * TOPK + (depth - 1) * TOPK + rank
        token_id = tl.load(top_token_ids + top_offset)

        tl.store(out_node_token_ids + pid * BUDGET + node_i, token_id)
        tl.store(out_node_depths + pid * BUDGET + node_i, depth)
        tl.store(out_parents + pid * MAX_NODES + current_idx, parent_idx)

        parent_vis = tl.load(
            out_visibility + vis_base + parent_idx * MAX_NODES + cols,
            mask=col_mask,
            other=False,
        )
        row_vis = (cols == current_idx) | ((cols < current_idx) & parent_vis)
        tl.store(
            out_visibility + vis_base + current_idx * MAX_NODES + cols,
            row_vis,
            mask=col_mask,
        )

        tl.store(heap_scores + heap_base + best_pos, -float("inf"))

        sibling_slot = 2 * node_i + 1
        child_slot = 2 * node_i + 2
        if TOPK > 1:
            if rank + 1 < TOPK:
                old_lp = tl.load(top_log_probs + top_offset)
                new_lp = tl.load(top_log_probs + top_offset + 1)
                sibling_score = best_score - old_lp + new_lp
                tl.store(heap_scores + heap_base + sibling_slot, sibling_score)
                tl.store(heap_parents + heap_base + sibling_slot, parent_idx)
                tl.store(heap_depths + heap_base + sibling_slot, depth)
                tl.store(heap_ranks + heap_base + sibling_slot, rank + 1)

        if depth < DEPTH_LIMIT:
            child_score = best_score + tl.load(
                top_log_probs + pid * DEPTH_LIMIT * TOPK + depth * TOPK
            )
            tl.store(heap_scores + heap_base + child_slot, child_score)
            tl.store(heap_parents + heap_base + child_slot, current_idx)
            tl.store(heap_depths + heap_base + child_slot, depth + 1)
            tl.store(heap_ranks + heap_base + child_slot, 0)

        tl.debug_barrier()

    tl.store(out_actual_tree_sizes + pid, MAX_NODES)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _compile_ddtree_verify_inputs_kernel(
    root_token_ids,
    node_token_ids,
    node_depths,
    start_positions,
    verify_input_ids,
    verify_position_ids,
    Q_LEN: tl.constexpr,
    NODE_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    req_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_Q)
    mask = cols < Q_LEN
    is_root = cols == 0
    node_cols = cols - 1

    root_token = tl.load(root_token_ids + req_idx)
    node_token = tl.load(
        node_token_ids + req_idx * NODE_STRIDE + node_cols,
        mask=mask & ~is_root,
        other=0,
    )
    start_position = tl.load(start_positions + req_idx)
    node_depth = tl.load(
        node_depths + req_idx * NODE_STRIDE + node_cols,
        mask=mask & ~is_root,
        other=0,
    )

    out_offset = req_idx * Q_LEN + cols
    tl.store(
        verify_input_ids + out_offset,
        tl.where(is_root, root_token, node_token),
        mask=mask,
    )
    tl.store(
        verify_position_ids + out_offset,
        tl.where(is_root, start_position, start_position + node_depth),
        mask=mask,
    )


@triton.jit
def _pad_ddtree_visibility_kernel(
    visibility,
    actual_tree_sizes,
    Q_LEN: tl.constexpr,
    VIS_B_STRIDE: tl.constexpr,
    VIS_ROW_STRIDE: tl.constexpr,
    VIS_COL_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row = tl.program_id(1)
    actual_size = tl.load(actual_tree_sizes + req_idx)
    cols = tl.arange(0, BLOCK_Q)
    mask = (row >= actual_size) & (cols < Q_LEN)
    tl.store(
        visibility
        + req_idx * VIS_B_STRIDE
        + row * VIS_ROW_STRIDE
        + cols * VIS_COL_STRIDE,
        cols == row,
        mask=mask,
    )


@triton.jit
def _compile_ddtree_packed_mask_kernel(
    visibility,
    actual_tree_sizes,
    past_lengths,
    tree_attention_mask,
    Q_LEN: tl.constexpr,
    BS: tl.constexpr,
    VIS_B_STRIDE: tl.constexpr,
    VIS_ROW_STRIDE: tl.constexpr,
    VIS_COL_STRIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_pid = tl.program_id(0)
    kv_block = tl.program_id(1)
    req_idx = row_pid // Q_LEN
    query_idx = row_pid - req_idx * Q_LEN

    request_offset = tl.zeros((1,), tl.int64)
    for prior_req in range(0, BS):
        prior_len = tl.load(past_lengths + prior_req)
        request_offset += tl.where(
            prior_req < req_idx, Q_LEN * (prior_len + Q_LEN), 0
        )

    past_len = tl.load(past_lengths + req_idx)
    actual_size = tl.load(actual_tree_sizes + req_idx)
    kv_len = past_len + Q_LEN
    cols = kv_block * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = cols < kv_len
    real_query = query_idx < actual_size
    in_prefix = cols < past_len
    tree_col = cols - past_len
    in_tree = (tree_col >= 0) & (tree_col < actual_size)
    visible = tl.load(
        visibility
        + req_idx * VIS_B_STRIDE
        + query_idx * VIS_ROW_STRIDE
        + tree_col * VIS_COL_STRIDE,
        mask=col_mask & real_query & in_tree,
        other=False,
    )
    dummy_self = (~real_query) & (tree_col == query_idx)
    allow = (real_query & (in_prefix | (in_tree & visible))) | dummy_self
    tl.store(
        tree_attention_mask + request_offset + query_idx * kv_len + cols,
        allow,
        mask=col_mask,
    )


def compile_ddtree_tree_triton(
    *,
    root_token_ids: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility: torch.Tensor,
    start_positions: torch.Tensor,
    past_lengths: torch.Tensor,
    actual_tree_sizes: torch.Tensor,
    verify_input_ids: torch.Tensor,
    verify_position_ids: torch.Tensor,
    tree_attention_mask: torch.Tensor | None,
    q_len: int,
    max_kv_len: int | None = None,
    pad_visibility: bool = False,
) -> None:
    """Compile DDTree verify inputs and optional packed allow-mask on GPU."""

    q_len = int(q_len)
    if q_len <= 0:
        raise ValueError(f"DDTree q_len must be positive, got {q_len}.")
    if not root_token_ids.is_cuda:
        raise ValueError("DDTree compile Triton kernels require CUDA tensors.")
    bs = int(root_token_ids.numel())
    if verify_input_ids.numel() < bs * q_len:
        raise ValueError("DDTree verify_input_ids buffer is too small.")
    if verify_position_ids.numel() < bs * q_len:
        raise ValueError("DDTree verify_position_ids buffer is too small.")

    block_q = _next_power_of_2(q_len)
    _compile_ddtree_verify_inputs_kernel[(bs,)](
        root_token_ids,
        node_token_ids,
        node_depths,
        start_positions,
        verify_input_ids,
        verify_position_ids,
        Q_LEN=q_len,
        NODE_STRIDE=int(node_token_ids.stride(0)),
        BLOCK_Q=block_q,
    )
    if pad_visibility:
        _pad_ddtree_visibility_kernel[(bs, q_len)](
            visibility,
            actual_tree_sizes,
            Q_LEN=q_len,
            VIS_B_STRIDE=int(visibility.stride(0)),
            VIS_ROW_STRIDE=int(visibility.stride(1)),
            VIS_COL_STRIDE=int(visibility.stride(2)),
            BLOCK_Q=block_q,
        )

    if tree_attention_mask is not None:
        if max_kv_len is None:
            raise ValueError(
                "DDTree packed-mask compilation requires max_kv_len from "
                "CPU-resident sequence-length metadata."
            )
        max_kv_len = int(max_kv_len)
        block_k = 256
        _compile_ddtree_packed_mask_kernel[
            (bs * q_len, triton.cdiv(max_kv_len, block_k))
        ](
            visibility,
            actual_tree_sizes,
            past_lengths,
            tree_attention_mask,
            Q_LEN=q_len,
            BS=bs,
            VIS_B_STRIDE=int(visibility.stride(0)),
            VIS_ROW_STRIDE=int(visibility.stride(1)),
            VIS_COL_STRIDE=int(visibility.stride(2)),
            BLOCK_K=block_k,
        )


@triton.jit
def _build_fa_suffix_metadata_kernel(
    visibility,
    req_to_token,
    req_pool_indices,
    seq_lens,
    page_table,
    cache_seqlens,
    Q_LEN: tl.constexpr,
    VIS_B_STRIDE: tl.constexpr,
    VIS_ROW_STRIDE: tl.constexpr,
    VIS_COL_STRIDE: tl.constexpr,
    REQ_TO_TOKEN_B_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    row_pid = tl.program_id(0)
    req_idx_in_batch = row_pid // Q_LEN
    query_idx = row_pid - req_idx_in_batch * Q_LEN

    cols = tl.arange(0, BLOCK_Q)
    col_mask = cols < Q_LEN

    visible = tl.load(
        visibility
        + req_idx_in_batch * VIS_B_STRIDE
        + query_idx * VIS_ROW_STRIDE
        + cols * VIS_COL_STRIDE,
        mask=col_mask,
        other=0,
    ).to(tl.int32)
    packed_offsets = tl.cumsum(visible, 0) - 1
    cache_len = tl.sum(visible, axis=0)

    req_pool_idx = tl.load(req_pool_indices + req_idx_in_batch).to(tl.int64)
    seq_len = tl.load(seq_lens + req_idx_in_batch).to(tl.int64)
    token_locs = tl.load(
        req_to_token + req_pool_idx * REQ_TO_TOKEN_B_STRIDE + seq_len + cols,
        mask=col_mask,
        other=0,
    ).to(tl.int32)

    page_base = row_pid * Q_LEN
    tl.store(page_table + page_base + cols, 0, mask=col_mask)
    tl.store(
        page_table + page_base + packed_offsets,
        token_locs,
        mask=col_mask & (visible != 0),
    )
    tl.store(cache_seqlens + row_pid, cache_len)


def build_ddtree_fa_suffix_metadata_triton(
    *,
    visibility: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q_len: int,
) -> None:
    """Pack DDTree ancestor visibility into FA suffix metadata on GPU.

    Each output row corresponds to one (request, tree query).  The first
    cache_seqlens[row] columns of page_table[row] are the KV locations of root,
    ancestors, and self in tree order.  Remaining columns are zero-filled.
    """

    q_len = int(q_len)
    if q_len <= 0:
        raise ValueError(f"DDTree FA suffix q_len must be positive, got {q_len}.")
    if not visibility.is_cuda:
        raise ValueError("DDTree FA suffix metadata Triton kernel requires CUDA visibility.")
    if visibility.ndim != 3:
        raise ValueError(
            f"DDTree visibility must be [bs, q, q], got {tuple(visibility.shape)}."
        )
    bs = int(seq_lens.numel())
    if page_table.shape[0] < bs * q_len or page_table.shape[1] < q_len:
        raise ValueError(
            "DDTree FA suffix page_table buffer is too small: "
            f"shape={tuple(page_table.shape)}, required rows={bs * q_len}, cols={q_len}."
        )
    if cache_seqlens.shape[0] < bs * q_len:
        raise ValueError(
            "DDTree FA suffix cache_seqlens buffer is too small: "
            f"shape={tuple(cache_seqlens.shape)}, required={bs * q_len}."
        )

    block_q = _next_power_of_2(q_len)
    _build_fa_suffix_metadata_kernel[(bs * q_len,)](
        visibility,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        cache_seqlens,
        Q_LEN=q_len,
        VIS_B_STRIDE=int(visibility.stride(0)),
        VIS_ROW_STRIDE=int(visibility.stride(1)),
        VIS_COL_STRIDE=int(visibility.stride(2)),
        REQ_TO_TOKEN_B_STRIDE=int(req_to_token.stride(0)),
        BLOCK_Q=block_q,
    )


@triton.jit
def _copy_fa_full_prefix_metadata_kernel(
    req_to_token,
    req_pool_indices,
    seq_lens,
    page_table,
    Q_LEN: tl.constexpr,
    PAGE_TABLE_ROW_STRIDE: tl.constexpr,
    REQ_TO_TOKEN_B_STRIDE: tl.constexpr,
    PREFIX_CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_pid = tl.program_id(0)
    block_pid = tl.program_id(1)
    req_idx_in_batch = row_pid // Q_LEN
    cols = block_pid * BLOCK_N + tl.arange(0, BLOCK_N)

    req_pool_idx = tl.load(req_pool_indices + req_idx_in_batch).to(tl.int64)
    seq_len = tl.load(seq_lens + req_idx_in_batch).to(tl.int64)
    mask = (cols < seq_len) & (cols < PREFIX_CAPACITY)
    token_locs = tl.load(
        req_to_token + req_pool_idx * REQ_TO_TOKEN_B_STRIDE + cols,
        mask=mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table + row_pid * PAGE_TABLE_ROW_STRIDE + cols,
        token_locs,
        mask=mask,
    )


@triton.jit
def _append_fa_visible_suffix_metadata_kernel(
    visibility,
    req_to_token,
    req_pool_indices,
    seq_lens,
    page_table,
    cache_seqlens,
    Q_LEN: tl.constexpr,
    VIS_B_STRIDE: tl.constexpr,
    VIS_ROW_STRIDE: tl.constexpr,
    VIS_COL_STRIDE: tl.constexpr,
    REQ_TO_TOKEN_B_STRIDE: tl.constexpr,
    PAGE_TABLE_ROW_STRIDE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    row_pid = tl.program_id(0)
    req_idx_in_batch = row_pid // Q_LEN
    query_idx = row_pid % Q_LEN
    cols = tl.arange(0, BLOCK_Q)
    col_mask = cols < Q_LEN

    visible = tl.load(
        visibility
        + req_idx_in_batch * VIS_B_STRIDE
        + query_idx * VIS_ROW_STRIDE
        + cols * VIS_COL_STRIDE,
        mask=col_mask,
        other=0,
    ).to(tl.int32)
    packed_offsets = tl.cumsum(visible, 0) - 1
    visible_count = tl.sum(visible, axis=0)

    req_pool_idx = tl.load(req_pool_indices + req_idx_in_batch).to(tl.int64)
    seq_len = tl.load(seq_lens + req_idx_in_batch).to(tl.int64)
    token_locs = tl.load(
        req_to_token + req_pool_idx * REQ_TO_TOKEN_B_STRIDE + seq_len + cols,
        mask=col_mask,
        other=0,
    ).to(tl.int32)

    row_base = row_pid * PAGE_TABLE_ROW_STRIDE
    tl.store(
        page_table + row_base + seq_len + packed_offsets,
        token_locs,
        mask=col_mask & (visible != 0),
    )
    tl.store(cache_seqlens + row_pid, seq_len + visible_count)


def build_ddtree_fa_full_metadata_triton(
    *,
    visibility: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    q_len: int,
) -> None:
    """Build one-pass FA metadata for every DDTree query.

    Each row contains the complete committed prefix followed by only the tree
    nodes visible to that query. This represents arbitrary tree ancestry with
    one non-causal FlashAttention call per layer.
    """

    q_len = int(q_len)
    if q_len <= 0:
        raise ValueError(f"DDTree FA full q_len must be positive, got {q_len}.")
    if not visibility.is_cuda:
        raise ValueError(
            "DDTree FA full metadata Triton kernels require CUDA visibility."
        )
    if visibility.ndim != 3:
        raise ValueError(
            f"DDTree visibility must be [bs, q, q], got {tuple(visibility.shape)}."
        )
    bs = int(seq_lens.numel())
    rows = bs * q_len
    prefix_capacity = int(page_table.shape[1]) - q_len
    if page_table.shape[0] < rows or prefix_capacity <= 0:
        raise ValueError(
            "DDTree FA full page_table buffer is too small: "
            f"shape={tuple(page_table.shape)}, required rows={rows}, "
            f"suffix_cols={q_len}."
        )
    if cache_seqlens.shape[0] < rows:
        raise ValueError(
            "DDTree FA full cache_seqlens buffer is too small: "
            f"shape={tuple(cache_seqlens.shape)}, required={rows}."
        )

    block_n = 256
    _copy_fa_full_prefix_metadata_kernel[
        (rows, triton.cdiv(prefix_capacity, block_n))
    ](
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        Q_LEN=q_len,
        PAGE_TABLE_ROW_STRIDE=int(page_table.stride(0)),
        REQ_TO_TOKEN_B_STRIDE=int(req_to_token.stride(0)),
        PREFIX_CAPACITY=prefix_capacity,
        BLOCK_N=block_n,
    )
    _append_fa_visible_suffix_metadata_kernel[(rows,)](
        visibility,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        cache_seqlens,
        Q_LEN=q_len,
        VIS_B_STRIDE=int(visibility.stride(0)),
        VIS_ROW_STRIDE=int(visibility.stride(1)),
        VIS_COL_STRIDE=int(visibility.stride(2)),
        REQ_TO_TOKEN_B_STRIDE=int(req_to_token.stride(0)),
        PAGE_TABLE_ROW_STRIDE=int(page_table.stride(0)),
        BLOCK_Q=_next_power_of_2(q_len),
    )


@triton.jit
def _follow_verified_tree_kernel(
    draft_tokens,
    target_predict,
    parents,
    actual_tree_sizes,
    accepted_indices,
    accepted_token_ids,
    accepted_lens,
    next_tokens,
    Q_LEN: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_Q)
    mask = offsets < Q_LEN
    row_base = pid * Q_LEN

    tl.store(accepted_indices + row_base + offsets, -1, mask=mask)
    tl.store(accepted_token_ids + row_base + offsets, 0, mask=mask)

    actual_size = tl.load(actual_tree_sizes + pid).to(tl.int32)
    actual_size = tl.minimum(tl.maximum(actual_size, 1), Q_LEN)

    current_idx = tl.full((), 0, dtype=tl.int32)
    accepted_len = tl.full((), 1, dtype=tl.int32)
    active = tl.full((), True, dtype=tl.int1)

    root_token = tl.load(draft_tokens + row_base)
    next_token = tl.load(target_predict + row_base)
    tl.store(accepted_indices + row_base, 0)
    tl.store(accepted_token_ids + row_base, root_token)

    for _ in range(0, Q_LEN - 1):
        parent_vals = tl.load(
            parents + row_base + offsets,
            mask=mask & (offsets < actual_size),
            other=-2,
        ).to(tl.int32)
        draft_vals = tl.load(
            draft_tokens + row_base + offsets,
            mask=mask & (offsets < actual_size),
            other=-1,
        )
        matches = (
            active
            & (offsets > 0)
            & (offsets < actual_size)
            & (parent_vals == current_idx)
            & (draft_vals == next_token)
        )
        child_idx = tl.min(tl.where(matches, offsets, Q_LEN), axis=0).to(tl.int32)
        found = child_idx < Q_LEN
        write_pos = accepted_len

        child_token = tl.load(
            draft_tokens + row_base + child_idx,
            mask=found,
            other=0,
        )
        tl.store(
            accepted_indices + row_base + write_pos,
            child_idx,
            mask=found & (write_pos < Q_LEN),
        )
        tl.store(
            accepted_token_ids + row_base + write_pos,
            child_token,
            mask=found & (write_pos < Q_LEN),
        )

        next_from_child = tl.load(
            target_predict + row_base + child_idx,
            mask=found,
            other=next_token,
        )
        current_idx = tl.where(found, child_idx, current_idx)
        next_token = tl.where(found, next_from_child, next_token)
        accepted_len += found.to(tl.int32)
        active = active & found

    tl.store(accepted_lens + pid, accepted_len)
    tl.store(next_tokens + pid, next_token)


def follow_ddtree_verified_path_triton(
    *,
    draft_tokens: torch.Tensor,
    target_predict: torch.Tensor,
    parents: torch.Tensor,
    actual_tree_sizes: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_token_ids: torch.Tensor,
    accepted_lens: torch.Tensor,
    next_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Follow DDTree target predictions over parent metadata on GPU."""

    if not draft_tokens.is_cuda:
        raise ValueError("DDTree GPU follow requires CUDA draft_tokens.")
    if draft_tokens.ndim != 2 or target_predict.shape != draft_tokens.shape:
        raise ValueError(
            "DDTree GPU follow expects draft_tokens and target_predict to be [bs, q], "
            f"got {tuple(draft_tokens.shape)} and {tuple(target_predict.shape)}."
        )
    if parents.shape != draft_tokens.shape:
        raise ValueError(
            "DDTree GPU follow expects parents to match draft_tokens, got "
            f"{tuple(parents.shape)} vs {tuple(draft_tokens.shape)}."
        )

    bs, q_len = draft_tokens.shape
    if q_len <= 0:
        raise ValueError(f"DDTree GPU follow q_len must be positive, got {q_len}.")
    if actual_tree_sizes.shape[0] < bs:
        raise ValueError(
            "DDTree GPU follow actual_tree_sizes is too small: "
            f"shape={tuple(actual_tree_sizes.shape)}, bs={bs}."
        )
    if accepted_indices.shape[0] < bs or accepted_indices.shape[1] < q_len:
        raise ValueError(
            "DDTree GPU follow accepted_indices buffer is too small: "
            f"shape={tuple(accepted_indices.shape)}, required=({bs}, {q_len})."
        )
    if accepted_token_ids.shape[0] < bs or accepted_token_ids.shape[1] < q_len:
        raise ValueError(
            "DDTree GPU follow accepted_token_ids buffer is too small: "
            f"shape={tuple(accepted_token_ids.shape)}, required=({bs}, {q_len})."
        )
    if accepted_lens.shape[0] < bs or next_tokens.shape[0] < bs:
        raise ValueError("DDTree GPU follow scalar output buffers are too small.")

    block_q = _next_power_of_2(q_len)
    _follow_verified_tree_kernel[(bs,)](
        draft_tokens,
        target_predict,
        parents,
        actual_tree_sizes,
        accepted_indices,
        accepted_token_ids,
        accepted_lens,
        next_tokens,
        Q_LEN=int(q_len),
        BLOCK_Q=block_q,
    )
    return (
        accepted_indices[:bs, :q_len],
        accepted_token_ids[:bs, :q_len],
        accepted_lens[:bs],
        next_tokens[:bs],
    )


@triton.jit
def _sample_ddtree_accept_path_kernel(
    target_probs,
    draft_tokens,
    parents,
    actual_tree_sizes,
    uniform_samples,
    accepted_indices,
    accepted_token_ids,
    accepted_lens,
    next_tokens,
    reject_indices,
    reject_child_tokens,
    reject_child_counts,
    Q_LEN: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_Q)
    mask = offsets < Q_LEN
    row_base = pid * Q_LEN

    tl.store(accepted_indices + row_base + offsets, -1, mask=mask)
    tl.store(accepted_token_ids + row_base + offsets, 0, mask=mask)
    tl.store(reject_child_tokens + row_base + offsets, 0, mask=mask)
    tl.store(reject_indices + pid, 0)
    tl.store(reject_child_counts + pid, 0)

    actual_size = tl.load(actual_tree_sizes + pid).to(tl.int32)
    actual_size = tl.minimum(tl.maximum(actual_size, 1), Q_LEN)

    current_idx = tl.full((), 0, dtype=tl.int32)
    accepted_len = tl.full((), 1, dtype=tl.int32)
    active = tl.full((), True, dtype=tl.int1)

    root_token = tl.load(draft_tokens + row_base)
    tl.store(accepted_indices + row_base, 0)
    tl.store(accepted_token_ids + row_base, root_token)

    for _ in range(0, Q_LEN - 1):
        parent_vals = tl.load(
            parents + row_base + offsets,
            mask=mask & (offsets < actual_size),
            other=-2,
        ).to(tl.int32)
        draft_vals = tl.load(
            draft_tokens + row_base + offsets,
            mask=mask & (offsets < actual_size),
            other=0,
        )
        child_mask = (
            active
            & (offsets > 0)
            & (offsets < actual_size)
            & (parent_vals == current_idx)
        )
        child_i32 = child_mask.to(tl.int32)
        child_count = tl.sum(child_i32, axis=0)
        child_pos = tl.cumsum(child_i32, 0) - 1
        tl.store(
            reject_child_tokens + row_base + child_pos,
            draft_vals,
            mask=child_mask & (child_pos >= 0) & (child_pos < Q_LEN),
        )

        probs = tl.load(
            target_probs + (row_base + current_idx) * VOCAB_SIZE + draft_vals,
            mask=child_mask,
            other=0.0,
        )
        prefix = tl.cumsum(tl.where(child_mask, probs, 0.0), 0)
        sample_u = tl.load(uniform_samples + row_base + current_idx)
        select_mask = child_mask & (sample_u < prefix)
        child_idx = tl.min(tl.where(select_mask, offsets, Q_LEN), axis=0).to(tl.int32)
        found = active & (child_idx < Q_LEN)
        reject_now = active & (~found)

        tl.store(reject_indices + pid, current_idx, mask=reject_now)
        tl.store(reject_child_counts + pid, child_count, mask=reject_now)

        write_pos = accepted_len
        child_token = tl.load(
            draft_tokens + row_base + child_idx,
            mask=found,
            other=0,
        )
        tl.store(
            accepted_indices + row_base + write_pos,
            child_idx,
            mask=found & (write_pos < Q_LEN),
        )
        tl.store(
            accepted_token_ids + row_base + write_pos,
            child_token,
            mask=found & (write_pos < Q_LEN),
        )

        current_idx = tl.where(found, child_idx, current_idx)
        accepted_len += found.to(tl.int32)
        active = active & found

    tl.store(reject_indices + pid, current_idx, mask=active)
    tl.store(reject_child_counts + pid, 0, mask=active)
    tl.store(accepted_lens + pid, accepted_len)
    tl.store(next_tokens + pid, 0)


def sample_ddtree_target_probs_triton(
    *,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    parents: torch.Tensor,
    actual_tree_sizes: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_final: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_token_ids: torch.Tensor,
    accepted_lens: torch.Tensor,
    next_tokens: torch.Tensor,
    reject_indices: torch.Tensor,
    reject_child_tokens: torch.Tensor,
    reject_child_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample DDTree target verification on GPU for unfiltered non-greedy sampling.

    The Triton kernel samples only along the visited tree path.  Once a node
    rejects, Python/Torch samples the final bonus from that node's target
    distribution with child token probabilities masked out.
    """

    if not target_probs.is_cuda:
        raise ValueError("DDTree native target sampler requires CUDA target_probs.")
    if target_probs.ndim != 3:
        raise ValueError(
            "DDTree native target sampler expects target_probs [bs, q, vocab], got "
            f"{tuple(target_probs.shape)}."
        )
    bs, q_len, vocab_size = target_probs.shape
    if draft_tokens.shape != (bs, q_len) or parents.shape != (bs, q_len):
        raise ValueError(
            "DDTree native target sampler expects draft_tokens/parents [bs, q], got "
            f"{tuple(draft_tokens.shape)} and {tuple(parents.shape)} for {(bs, q_len)}."
        )
    if q_len <= 0:
        raise ValueError(f"DDTree native target sampler q_len must be positive, got {q_len}.")

    target_probs = target_probs.contiguous()
    draft_tokens = draft_tokens.contiguous()
    parents = parents.contiguous()
    actual_tree_sizes = actual_tree_sizes[:bs].contiguous()
    uniform_samples = uniform_samples[:bs, :q_len]
    uniform_final = uniform_final[:bs]
    accepted_indices = accepted_indices[:bs, :q_len]
    accepted_token_ids = accepted_token_ids[:bs, :q_len]
    accepted_lens = accepted_lens[:bs]
    next_tokens = next_tokens[:bs]
    reject_indices = reject_indices[:bs]
    reject_child_tokens = reject_child_tokens[:bs, :q_len]
    reject_child_counts = reject_child_counts[:bs]

    uniform_samples.uniform_()
    uniform_final.uniform_()

    block_q = _next_power_of_2(q_len)
    _sample_ddtree_accept_path_kernel[(bs,)](
        target_probs,
        draft_tokens,
        parents,
        actual_tree_sizes,
        uniform_samples,
        accepted_indices,
        accepted_token_ids,
        accepted_lens,
        next_tokens,
        reject_indices,
        reject_child_tokens,
        reject_child_counts,
        Q_LEN=int(q_len),
        VOCAB_SIZE=int(vocab_size),
        BLOCK_Q=block_q,
    )

    batch_idx = torch.arange(bs, device=target_probs.device)
    reject_rows = target_probs[batch_idx, reject_indices.to(torch.long)].clone()
    child_offsets = torch.arange(q_len, device=target_probs.device)
    child_counts = reject_child_counts.to(torch.long)
    child_mask = child_offsets.unsqueeze(0) < child_counts.unsqueeze(1)
    row_idx = batch_idx.repeat_interleave(child_counts)
    child_ids = reject_child_tokens[child_mask].to(torch.long)
    reject_rows[row_idx, child_ids] = 0.0

    reject_cdf = torch.cumsum(reject_rows, dim=-1)
    reject_totals = reject_cdf[:, -1]
    thresholds = uniform_final * reject_totals.clamp_min(0.0)
    sampled = torch.searchsorted(reject_cdf, thresholds[:, None], right=False).squeeze(1)
    sampled = sampled.clamp_(max=vocab_size - 1).to(next_tokens.dtype)
    next_tokens.copy_(sampled)

    return accepted_indices, accepted_token_ids, accepted_lens, next_tokens


@triton.jit
def _prune_deepest_chains_kernel(
    in_node_token_ids,
    in_node_depths,
    in_parents,
    in_visibility,
    out_node_token_ids,
    out_node_depths,
    out_parents,
    out_visibility,
    out_actual_tree_sizes,
    scratch,
    BUDGET: tl.constexpr,
    MAX_NODES: tl.constexpr,
    NODE_BLOCK: tl.constexpr,
    VIS_BLOCK: tl.constexpr,
    SCRATCH_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    node_offsets = tl.arange(0, NODE_BLOCK)
    node_mask = node_offsets < MAX_NODES
    draft_mask = node_offsets < BUDGET

    tl.store(out_node_token_ids + pid * BUDGET + node_offsets, 0, mask=draft_mask)
    tl.store(out_node_depths + pid * BUDGET + node_offsets, 0, mask=draft_mask)
    tl.store(out_parents + pid * MAX_NODES + node_offsets, -1, mask=node_mask)

    vis_offsets = tl.arange(0, VIS_BLOCK)
    vis_mask = vis_offsets < (MAX_NODES * MAX_NODES)
    vis_rows = vis_offsets // MAX_NODES
    vis_cols = vis_offsets - vis_rows * MAX_NODES
    out_vis_base = pid * MAX_NODES * MAX_NODES
    tl.store(
        out_visibility + out_vis_base + vis_offsets,
        vis_rows == vis_cols,
        mask=vis_mask,
    )

    scratch_base = pid * SCRATCH_STRIDE
    depths = tl.load(
        in_node_depths + pid * BUDGET + node_offsets - 1,
        mask=(node_offsets > 0) & node_mask,
        other=0,
    ).to(tl.int32)
    depths = tl.where(node_offsets == 0, 0, depths)
    tl.store(scratch + scratch_base + node_offsets, depths, mask=node_mask)
    tl.store(scratch + scratch_base + MAX_NODES + node_offsets, -1, mask=node_mask)

    max_depth = tl.max(tl.where(node_mask, depths, 0), axis=0)
    for old_idx in range(BUDGET, 0, -1):
        parent_idx = tl.load(in_parents + pid * MAX_NODES + old_idx).to(tl.int32)
        child_depth = tl.load(scratch + scratch_base + old_idx)
        parent_depth = tl.load(
            scratch + scratch_base + parent_idx,
            mask=parent_idx >= 0,
            other=0,
        )
        tl.store(
            scratch + scratch_base + parent_idx,
            tl.maximum(parent_depth, child_depth),
            mask=parent_idx >= 0,
        )

    subtree_depths = tl.load(scratch + scratch_base + node_offsets, mask=node_mask, other=-1)
    keep = node_mask & ((node_offsets == 0) | (subtree_depths == max_depth))
    keep_i32 = keep.to(tl.int32)
    new_indices = tl.cumsum(keep_i32, 0) - 1
    actual_size = tl.sum(keep_i32, axis=0)
    tl.store(out_actual_tree_sizes + pid, actual_size)
    tl.store(
        scratch + scratch_base + MAX_NODES + node_offsets,
        new_indices,
        mask=keep,
    )

    token_ids = tl.load(
        in_node_token_ids + pid * BUDGET + node_offsets - 1,
        mask=(node_offsets > 0) & node_mask,
        other=0,
    )
    node_depths = tl.load(
        in_node_depths + pid * BUDGET + node_offsets - 1,
        mask=(node_offsets > 0) & node_mask,
        other=0,
    )
    tl.store(
        out_node_token_ids + pid * BUDGET + new_indices - 1,
        token_ids,
        mask=keep & (node_offsets > 0),
    )
    tl.store(
        out_node_depths + pid * BUDGET + new_indices - 1,
        node_depths,
        mask=keep & (node_offsets > 0),
    )

    old_parent_indices = tl.load(
        in_parents + pid * MAX_NODES + node_offsets,
        mask=keep & (node_offsets > 0),
        other=-1,
    ).to(tl.int32)
    new_parent_indices = tl.load(
        scratch + scratch_base + MAX_NODES + old_parent_indices,
        mask=keep & (node_offsets > 0) & (old_parent_indices >= 0),
        other=-1,
    )
    tl.store(
        out_parents + pid * MAX_NODES + new_indices,
        new_parent_indices,
        mask=keep & (node_offsets > 0),
    )

    in_vis_base = pid * MAX_NODES * MAX_NODES
    cols = node_offsets
    col_new_indices = tl.load(
        scratch + scratch_base + MAX_NODES + cols,
        mask=node_mask,
        other=-1,
    )
    col_keep = node_mask & (col_new_indices >= 0)
    for old_row in range(0, MAX_NODES):
        row_new_idx = tl.load(scratch + scratch_base + MAX_NODES + old_row)
        row_keep = row_new_idx >= 0
        row_vis = tl.load(
            in_visibility + in_vis_base + old_row * MAX_NODES + cols,
            mask=row_keep & node_mask,
            other=False,
        )
        tl.store(
            out_visibility + out_vis_base + row_new_idx * MAX_NODES + col_new_indices,
            row_vis,
            mask=row_keep & col_keep,
        )


def prune_ddtree_deepest_chains_triton(
    *,
    in_node_token_ids: torch.Tensor,
    in_node_depths: torch.Tensor,
    in_parents: torch.Tensor,
    in_visibility: torch.Tensor,
    out_node_token_ids: torch.Tensor,
    out_node_depths: torch.Tensor,
    out_parents: torch.Tensor,
    out_visibility: torch.Tensor,
    out_actual_tree_sizes: torch.Tensor,
    scratch: torch.Tensor,
    tree_budget: int,
) -> torch.Tensor:
    """Prune a full GPU-built DDTree to deepest-reaching chains and compact it."""

    if not in_node_token_ids.is_cuda:
        raise ValueError("DDTree prune Triton kernel requires CUDA tensors.")
    bs = int(in_node_token_ids.shape[0])
    budget = int(tree_budget)
    max_nodes = budget + 1
    if scratch.shape[1] < 2 * max_nodes:
        raise ValueError(
            "DDTree prune scratch buffer is too small: "
            f"shape={tuple(scratch.shape)}, required width={2 * max_nodes}."
        )

    node_block = _next_power_of_2(max_nodes)
    vis_block = _next_power_of_2(max_nodes * max_nodes)
    _prune_deepest_chains_kernel[(bs,)](
        in_node_token_ids,
        in_node_depths,
        in_parents,
        in_visibility,
        out_node_token_ids,
        out_node_depths,
        out_parents,
        out_visibility,
        out_actual_tree_sizes,
        scratch,
        BUDGET=budget,
        MAX_NODES=max_nodes,
        NODE_BLOCK=node_block,
        VIS_BLOCK=vis_block,
        SCRATCH_STRIDE=int(scratch.stride(0)),
    )
    return out_actual_tree_sizes[:bs]


def build_ddtree_tree_triton(
    *,
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    tree_budget: int,
    out_node_token_ids: torch.Tensor,
    out_node_depths: torch.Tensor,
    out_parents: torch.Tensor,
    out_visibility: torch.Tensor,
    out_actual_tree_sizes: torch.Tensor,
    heap_scores: torch.Tensor,
    heap_parents: torch.Tensor,
    heap_depths: torch.Tensor,
    heap_ranks: torch.Tensor,
) -> torch.Tensor:
    """Build an exact DDTree best-first tree on GPU for the no-prune path."""

    if not top_log_probs.is_cuda:
        raise ValueError("DDTree Triton builder requires CUDA top_log_probs.")
    if top_log_probs.ndim != 3 or top_token_ids.shape != top_log_probs.shape:
        raise ValueError(
            "DDTree Triton builder expected matching [bs, L, topk] tensors, got "
            f"{tuple(top_log_probs.shape)} and {tuple(top_token_ids.shape)}."
        )

    bs, depth_limit, topk = top_log_probs.shape
    budget = int(tree_budget)
    if budget <= 0:
        raise ValueError(f"DDTree tree_budget must be positive, got {budget}.")
    if int(topk) <= 0:
        raise ValueError(f"DDTree topk must be positive, got {topk}.")

    heap_cap = _next_power_of_2(2 * budget + 4)
    if heap_scores.shape[0] < bs or heap_scores.shape[1] < heap_cap:
        raise ValueError(
            "DDTree heap score scratch buffer is too small: "
            f"shape={tuple(heap_scores.shape)}, required=({bs}, {heap_cap})."
        )
    if out_actual_tree_sizes.shape[0] < bs:
        raise ValueError(
            "DDTree actual_tree_sizes buffer is too small: "
            f"shape={tuple(out_actual_tree_sizes.shape)}, bs={bs}."
        )

    top_log_probs = top_log_probs.contiguous()
    top_token_ids = top_token_ids.contiguous()
    out_node_token_ids = out_node_token_ids[:bs, :budget]
    out_node_depths = out_node_depths[:bs, :budget]
    out_parents = out_parents[:bs, : budget + 1]
    out_visibility = out_visibility[:bs, : budget + 1, : budget + 1]
    out_actual_tree_sizes = out_actual_tree_sizes[:bs]

    node_block = _next_power_of_2(budget + 1)
    vis_block = _next_power_of_2((budget + 1) * (budget + 1))

    _build_ddtree_heap_kernel[(bs,)](
        top_log_probs,
        top_token_ids,
        out_node_token_ids,
        out_node_depths,
        out_parents,
        out_visibility,
        out_actual_tree_sizes,
        heap_scores,
        heap_parents,
        heap_depths,
        heap_ranks,
        BUDGET=budget,
        MAX_NODES=budget + 1,
        DEPTH_LIMIT=int(depth_limit),
        TOPK=int(topk),
        HEAP_CAP=heap_cap,
        NODE_BLOCK=node_block,
        VIS_BLOCK=vis_block,
    )
    return out_actual_tree_sizes
