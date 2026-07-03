import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def build_ddtree_tree(
    draft_logits: torch.Tensor | None,  # [bs, L, vocab_size]
    tree_budget: int,            # 节点预算 B
    device: torch.device,
    _out_node_token_ids: torch.Tensor | None = None,
    _out_node_depths: torch.Tensor | None = None,
    _out_parents: torch.Tensor | None = None,
    _out_visibility: torch.Tensor | None = None,
    draft_top_log_probs: torch.Tensor | None = None,  # [bs, L, topk]
    draft_top_token_ids: torch.Tensor | None = None,  # [bs, L, topk]
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Dict[int, Dict[int, int]]],
    torch.Tensor,
    torch.Tensor,
    List[int],
]:
    if draft_logits is None:
        if draft_top_log_probs is None or draft_top_token_ids is None:
            raise ValueError(
                "build_ddtree_tree requires either draft_logits or "
                "draft_top_log_probs/draft_top_token_ids."
            )
        bs, L, topk = draft_top_log_probs.shape
        if draft_top_token_ids.shape != draft_top_log_probs.shape:
            raise ValueError(
                "draft_top_token_ids must match draft_top_log_probs shape, got "
                f"{tuple(draft_top_token_ids.shape)} vs "
                f"{tuple(draft_top_log_probs.shape)}."
            )
        logits = None
    else:
        bs, L, V = draft_logits.shape
        logits = draft_logits.float()
        topk = -1
    max_nodes = tree_budget + 1

    # --- General tree path: best-first DDTree expansion with branching ---
    if draft_top_log_probs is None or draft_top_token_ids is None:
        topk = min(tree_budget, V)
        top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_log_probs = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
        top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    else:
        top_log_probs = draft_top_log_probs.to(device="cpu", dtype=torch.float32)
        top_token_ids_cpu = draft_top_token_ids.to(device="cpu", dtype=torch.long)
        topk = int(top_log_probs.shape[-1])

    all_node_token_ids = []
    all_node_depths = []
    all_parents = []
    all_child_maps = []
    all_visibility = []
    actual_sizes = []

    for b in range(bs):
        node_ids, depths, parents, child_map, vis, actual = _build_single_tree(
            top_log_probs[b], top_token_ids_cpu[b], topk, L, tree_budget
        )
        all_node_token_ids.append(node_ids)
        all_node_depths.append(depths)
        all_parents.append(parents)
        all_child_maps.append(child_map)
        all_visibility.append(vis)
        actual_sizes.append(actual)

    # Reuse or allocate padded output buffers.
    if _out_node_token_ids is None or _out_node_token_ids.shape[0] < bs:
        padded_node_token_ids = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    else:
        padded_node_token_ids = _out_node_token_ids[:bs]
        padded_node_token_ids.zero_()

    if _out_node_depths is None or _out_node_depths.shape[0] < bs:
        padded_node_depths = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    else:
        padded_node_depths = _out_node_depths[:bs]
        padded_node_depths.zero_()

    if _out_parents is None or _out_parents.shape[0] < bs:
        padded_parents = torch.full((bs, max_nodes), -1, dtype=torch.long, device=device)
    else:
        padded_parents = _out_parents[:bs]
        padded_parents.fill_(-1)

    if _out_visibility is None or _out_visibility.shape[0] < bs:
        padded_visibility = torch.zeros(bs, max_nodes, max_nodes, dtype=torch.bool, device=device)
    else:
        padded_visibility = _out_visibility[:bs]
        padded_visibility.zero_()

    for b in range(bs):
        n = actual_sizes[b] - 1
        if n > 0:
            padded_node_token_ids[b, :n] = torch.from_numpy(all_node_token_ids[b]).to(device)
            padded_node_depths[b, :n] = torch.from_numpy(all_node_depths[b]).to(device)
        padded_parents[b, :actual_sizes[b]] = torch.from_numpy(all_parents[b]).to(device)
        vis = all_visibility[b]
        padded_visibility[b, :actual_sizes[b], :actual_sizes[b]] = (
            torch.from_numpy(vis).to(device=device, dtype=torch.bool)
        )

    actual_tree_sizes_t = torch.tensor(actual_sizes, dtype=torch.long, device=device)
    return (
        padded_node_token_ids,
        padded_node_depths,
        padded_parents,
        all_child_maps,
        padded_visibility,
        actual_tree_sizes_t,
        actual_sizes,
    )


def _build_single_tree(
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    topk: int,
    depth_limit: int,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[int, int]], np.ndarray, int]:
    log_probs_np = top_log_probs.numpy().astype(np.float64)
    token_ids_np = top_token_ids.numpy().astype(np.int64)

    node_token_ids = np.zeros(budget, dtype=np.int64)
    node_depths = np.zeros(budget, dtype=np.int32)
    parents = np.full(budget + 1, -1, dtype=np.int32)
    child_maps: Dict[int, Dict[int, int]] = {0: {}}

    first_logw = float(log_probs_np[0, 0])
    heap = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_count = 0
    while heap and node_count < budget:
        _, ranks, parent_idx, depth, rank, logw = heapq.heappop(heap)

        token_id = int(token_ids_np[depth - 1, rank])
        current_idx = node_count + 1

        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents[current_idx] = parent_idx
        child_maps.setdefault(parent_idx, {})[token_id] = current_idx
        child_maps.setdefault(current_idx, {})
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = (
                logw
                - float(log_probs_np[depth - 1, rank])
                + float(log_probs_np[depth - 1, rank + 1])
            )
            sibling_ranks = ranks[:-1] + (rank + 1,)
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_idx, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_logw = logw + float(log_probs_np[depth, 0])
            child_ranks = ranks + (0,)
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_idx, depth + 1, 0, child_logw),
            )

    current_length = node_count + 1
    visibility = np.zeros((current_length, current_length), dtype=bool)
    visibility[0, 0] = True
    for idx in range(1, current_length):
        p = int(parents[idx])
        visibility[idx, :idx] = visibility[p, :idx]
        visibility[idx, idx] = True

    return (
        node_token_ids[:node_count],
        node_depths[:node_count],
        parents[:current_length],
        child_maps,
        visibility,
        current_length,
    )


def compile_ddtree_tree(
    root_token_ids: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility: torch.Tensor,
    start_positions: torch.Tensor,
    past_lengths: torch.Tensor,
    tree_budget: int,
    actual_tree_sizes: torch.Tensor,
    device: torch.device,
    past_lens_cpu: Optional[List[int]] = None,
    actual_sizes_cpu: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = root_token_ids.shape[0]
    max_nodes = tree_budget + 1

    verify_input_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_input_ids[:, 0] = root_token_ids
    verify_input_ids[:, 1:] = node_token_ids

    verify_position_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_position_ids[:, 0] = start_positions
    verify_position_ids[:, 1:] = start_positions.unsqueeze(1) + node_depths

    # Attention backends consume the speculative custom mask as a request-packed
    # boolean allow-mask:
    #   concat(mask_i.reshape(-1)), mask_i.shape == [max_nodes, prefix_i + max_nodes]
    # FlashInfer in particular derives mask_indptr from the per-request q/kv
    # lengths, so retaining max-prefix padding between requests corrupts every
    # request after the first one.  Allocate the final packed representation
    # directly instead of materializing a [bs, max_nodes, max_kv_len] rectangle.
    if past_lens_cpu is None:
        past_lens_cpu = [int(x) for x in past_lengths.detach().cpu().tolist()]
    if actual_sizes_cpu is None:
        actual_sizes_cpu = [
            int(x) for x in actual_tree_sizes.detach().cpu().tolist()
        ]
    mask_numel = sum(
        max_nodes * (past_len + max_nodes) for past_len in past_lens_cpu
    )
    tree_attention_mask = torch.empty(
        (mask_numel,), dtype=torch.bool, device=device
    )

    offset = 0
    for b, (past_len_i, actual_size) in enumerate(
        zip(past_lens_cpu, actual_sizes_cpu, strict=True)
    ):
        kv_len_i = past_len_i + max_nodes
        request_mask = tree_attention_mask[
            offset : offset + max_nodes * kv_len_i
        ].view(max_nodes, kv_len_i)

        # Real tree queries attend to the complete committed prefix.
        request_mask[:actual_size, :past_len_i] = True

        # Within the drafted tree, a query sees only the root, its ancestors,
        # and itself.  Only the tree suffix needs explicit false-fill; the
        # committed prefix is all-true and can avoid the previous full-mask
        # zero_() write.
        request_mask[:actual_size, past_len_i:] = False
        request_mask[
            :actual_size, past_len_i : past_len_i + actual_size
        ].copy_(visibility[b, :actual_size, :actual_size])
        if actual_size < max_nodes:
            request_mask[actual_size:, :].fill_(False)
        offset += max_nodes * kv_len_i

    return verify_input_ids, verify_position_ids, tree_attention_mask, actual_tree_sizes


def follow_verified_tree(
    child_maps: List[Dict[int, Dict[int, int]]],
    posterior_tokens: torch.Tensor,
) -> Tuple[List[List[int]], torch.Tensor, List[int]]:
    bs = len(child_maps)
    accepted_indices = []
    next_tokens_list = []

    for b in range(bs):
        posterior = posterior_tokens[b].tolist()
        accepted = [0]
        current_idx = 0
        next_token = posterior[0]

        cmap = child_maps[b]
        while next_token in cmap.get(current_idx, {}):
            current_idx = cmap[current_idx][next_token]
            accepted.append(current_idx)
            next_token = posterior[current_idx]

        accepted_indices.append(accepted)
        next_tokens_list.append(next_token)

    next_tokens = torch.tensor(
        next_tokens_list, dtype=torch.long, device=posterior_tokens.device
    )
    return accepted_indices, next_tokens, next_tokens_list


def compact_ddtree_kv_cache(
    kv_cache_pool,
    layer,
    cache_locs: torch.Tensor,
    keep_indices: List[List[int]],
    past_lengths: torch.Tensor,
    actual_tree_sizes: torch.Tensor,
):
    """Compact KV cache by moving kept slots to the front.

    Uses batched index_select + index_copy_ to replace per-element
    set_kv_buffer kernel launches with 2 launches per layer.
    """
    k_buffer, v_buffer = kv_cache_pool.get_kv_buffer(layer.layer_id)
    device = cache_locs.device

    src_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []

    for b in range(len(keep_indices)):
        keep = keep_indices[b]
        actual = int(actual_tree_sizes[b].item())

        if len(keep) == actual:
            continue

        # Safety: clamp keep indices to valid range [0, actual).
        keep = [idx for idx in keep if 0 <= idx < actual]
        if not keep or len(keep) == actual:
            continue

        # Fast path: if kept indices are contiguous from 0, no compaction needed.
        if keep == list(range(len(keep))):
            continue

        all_locs = cache_locs[b, :actual]
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        keep_locs = all_locs[keep_t]
        tgt_locs = all_locs[: len(keep)]

        mask = keep_locs != tgt_locs
        if mask.any():
            src_list.append(keep_locs[mask])
            tgt_list.append(tgt_locs[mask])

    if not src_list:
        return

    src_idx = torch.cat(src_list)
    tgt_idx = torch.cat(tgt_list)

    k_selected = k_buffer.index_select(0, src_idx)
    v_selected = v_buffer.index_select(0, src_idx)
    k_buffer.index_copy_(0, tgt_idx, k_selected)
    v_buffer.index_copy_(0, tgt_idx, v_selected)
