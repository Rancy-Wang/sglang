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

