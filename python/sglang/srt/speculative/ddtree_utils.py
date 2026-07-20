import heapq
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class DDTreeTargetBackendCapability:
    backend_name: str
    build_attention_mask: bool
    use_visibility: bool
    supports_full_tree: bool
    unsupported_reason: str = ""


def resolve_ddtree_target_backend_capability(
    attn_backend: Any,
    speculative_attention_mode: str = "prefill",
) -> DDTreeTargetBackendCapability:
    """Describe how DDTree full-tree target verify should feed an attention backend."""

    if attn_backend is None:
        return DDTreeTargetBackendCapability(
            backend_name="unknown",
            build_attention_mask=False,
            use_visibility=False,
            supports_full_tree=False,
            unsupported_reason="target model runner has no attention backend",
        )

    selected_backend = attn_backend
    visited = set()
    while selected_backend is not None:
        module = selected_backend.__class__.__module__
        cls_name = selected_backend.__class__.__name__
        if not module.endswith("hybrid_attn_backend"):
            break
        obj_id = id(selected_backend)
        if obj_id in visited:
            break
        visited.add(obj_id)
        selected_backend = (
            getattr(selected_backend, "decode_backend", None)
            if speculative_attention_mode == "decode"
            else getattr(selected_backend, "prefill_backend", None)
        )
        if selected_backend is None:
            return DDTreeTargetBackendCapability(
                backend_name=f"{module}.{cls_name}",
                build_attention_mask=False,
                use_visibility=False,
                supports_full_tree=False,
                unsupported_reason=(
                    "hybrid attention backend did not expose the selected "
                    f"{speculative_attention_mode!r} child backend"
                ),
            )

    module = selected_backend.__class__.__module__
    cls_name = selected_backend.__class__.__name__
    backend_name = f"{module}.{cls_name}"
    if module.endswith("flashattention_backend"):
        return DDTreeTargetBackendCapability(
            backend_name=backend_name,
            build_attention_mask=False,
            use_visibility=True,
            supports_full_tree=True,
        )
    if module.endswith("flashinfer_backend"):
        return DDTreeTargetBackendCapability(
            backend_name=backend_name,
            build_attention_mask=True,
            use_visibility=False,
            supports_full_tree=True,
        )
    if module.endswith("triton_backend"):
        return DDTreeTargetBackendCapability(
            backend_name=backend_name,
            build_attention_mask=True,
            use_visibility=False,
            supports_full_tree=True,
        )
    return DDTreeTargetBackendCapability(
        backend_name=backend_name,
        build_attention_mask=False,
        use_visibility=False,
        supports_full_tree=False,
        unsupported_reason=(
            "DDTree full-tree target verify needs either packed-mask "
            "support or FA visibility metadata support"
        ),
    )


def resolve_ddtree_cuda_graph_buckets(
    *,
    tree_budget: int,
    is_ddtree_prune: bool,
    configured_buckets: Optional[List[int]] = None,
) -> List[int]:
    """Return DDTree target-verify token-per-request CUDA graph buckets."""
    max_nodes = int(tree_budget) + 1
    if max_nodes <= 0:
        raise ValueError(f"DDTree tree_budget must be non-negative, got {tree_budget}.")

    if configured_buckets:
        buckets = [int(x) for x in configured_buckets]
    elif not is_ddtree_prune:
        buckets = [max_nodes]
    elif max_nodes <= 33:
        buckets = [8, 12, 16, 20, 24, 28, max_nodes]
    elif max_nodes <= 65:
        buckets = [8, 12, 16, 24, 32, 40, 48, 56, max_nodes]
    elif max_nodes <= 129:
        buckets = [8, 16, 24, 32, 48, 64, 80, 96, 112, max_nodes]
    else:
        buckets = [8, 16, 24, 32, 48, 64, 96, 128, max_nodes]

    normalized = sorted({b for b in buckets if 0 < b <= max_nodes})
    if max_nodes not in normalized:
        normalized.append(max_nodes)
    if not normalized:
        normalized = [max_nodes]
    return normalized


def select_ddtree_cuda_graph_bucket(actual_tree_size: int, buckets: List[int]) -> int:
    actual = max(1, int(actual_tree_size))
    for bucket in buckets:
        if int(bucket) >= actual:
            return int(bucket)
    return int(buckets[-1])


def build_ddtree_tree(
    draft_logits: torch.Tensor | None,  # [bs, L, vocab_size]
    tree_budget: int,  # 节点预算 B
    device: torch.device,
    _out_node_token_ids: torch.Tensor | None = None,
    _out_node_depths: torch.Tensor | None = None,
    _out_parents: torch.Tensor | None = None,
    _out_visibility: torch.Tensor | None = None,
    draft_top_log_probs: torch.Tensor | None = None,  # [bs, L, topk]
    draft_top_token_ids: torch.Tensor | None = None,  # [bs, L, topk]
    prune_to_deepest_chains: bool = False,
    profiler: Optional[Any] = None,
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
    cpu_ctx = profiler.cpu if profiler is not None else (lambda _stage: nullcontext())

    with cpu_ctx("tree_to_cpu"):
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
    with cpu_ctx("tree_heap_cpu"):
        for b in range(bs):
            node_ids, depths, parents, child_map, vis, actual = _build_single_tree(
                top_log_probs[b], top_token_ids_cpu[b], topk, L, tree_budget
            )
            if prune_to_deepest_chains:
                (
                    node_ids,
                    depths,
                    parents,
                    child_map,
                    vis,
                    actual,
                ) = _prune_tree_to_deepest_chains(
                    node_ids,
                    depths,
                    parents,
                    child_map,
                    vis,
                    actual,
                )
            all_node_token_ids.append(node_ids)
            all_node_depths.append(depths)
            all_parents.append(parents)
            all_child_maps.append(child_map)
            all_visibility.append(vis)
            actual_sizes.append(actual)

    # Reuse or allocate padded output buffers.
    if _out_node_token_ids is None or _out_node_token_ids.shape[0] < bs:
        padded_node_token_ids = torch.zeros(
            bs, tree_budget, dtype=torch.long, device=device
        )
    else:
        padded_node_token_ids = _out_node_token_ids[:bs]
        padded_node_token_ids.zero_()

    if _out_node_depths is None or _out_node_depths.shape[0] < bs:
        padded_node_depths = torch.zeros(
            bs, tree_budget, dtype=torch.long, device=device
        )
    else:
        padded_node_depths = _out_node_depths[:bs]
        padded_node_depths.zero_()

    if _out_parents is None or _out_parents.shape[0] < bs:
        padded_parents = torch.full(
            (bs, max_nodes), -1, dtype=torch.long, device=device
        )
    else:
        padded_parents = _out_parents[:bs]
        padded_parents.fill_(-1)

    if _out_visibility is None or _out_visibility.shape[0] < bs:
        padded_visibility = torch.zeros(
            bs, max_nodes, max_nodes, dtype=torch.bool, device=device
        )
    else:
        padded_visibility = _out_visibility[:bs]
        padded_visibility.zero_()

    with cpu_ctx("tree_to_gpu"):
        for b in range(bs):
            n = actual_sizes[b] - 1
            if n > 0:
                padded_node_token_ids[b, :n] = torch.from_numpy(
                    all_node_token_ids[b]
                ).to(device)
                padded_node_depths[b, :n] = torch.from_numpy(all_node_depths[b]).to(
                    device
                )
            padded_parents[b, : actual_sizes[b]] = torch.from_numpy(all_parents[b]).to(
                device
            )
            vis = all_visibility[b]
            padded_visibility[
                b, : actual_sizes[b], : actual_sizes[b]
            ] = torch.from_numpy(vis).to(device=device, dtype=torch.bool)

        actual_tree_sizes_t = torch.tensor(
            actual_sizes, dtype=torch.long, device=device
        )
    return (
        padded_node_token_ids,
        padded_node_depths,
        padded_parents,
        all_child_maps,
        padded_visibility,
        actual_tree_sizes_t,
        actual_sizes,
    )


def build_ddtree_tree_gpu(
    *,
    draft_top_log_probs: torch.Tensor,
    draft_top_token_ids: torch.Tensor,
    tree_budget: int,
    device: torch.device,
    _out_node_token_ids: torch.Tensor,
    _out_node_depths: torch.Tensor,
    _out_parents: torch.Tensor,
    _out_visibility: torch.Tensor,
    _out_actual_tree_sizes: torch.Tensor,
    _heap_scores: torch.Tensor,
    _heap_parents: torch.Tensor,
    _heap_depths: torch.Tensor,
    _heap_ranks: torch.Tensor,
    prune_to_deepest_chains: bool = False,
    _out_pruned_node_token_ids: torch.Tensor | None = None,
    _out_pruned_node_depths: torch.Tensor | None = None,
    _out_pruned_parents: torch.Tensor | None = None,
    _out_pruned_visibility: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Dict[int, Dict[int, int]]],
    torch.Tensor,
    torch.Tensor,
    List[int],
]:
    """Build a no-prune DDTree on GPU, returning the same tensor metadata shape.

    The Python child_maps are intentionally left empty here. They are only
    needed after target verify, so the GPU path reconstructs them lazily from
    tensor metadata in DDTreeVerifyInput.verify instead of blocking target
    verify with a CPU copy.
    """
    from sglang.kernels.ops.speculative.ddtree import (
        build_ddtree_tree_triton,
        prune_ddtree_deepest_chains_triton,
    )

    if draft_top_log_probs is None or draft_top_token_ids is None:
        raise ValueError("DDTree GPU builder requires top log-probs and token ids.")
    if not draft_top_log_probs.is_cuda:
        raise ValueError("DDTree GPU builder requires CUDA tensors.")

    bs, _, _ = draft_top_log_probs.shape
    actual_tree_sizes = build_ddtree_tree_triton(
        top_log_probs=draft_top_log_probs,
        top_token_ids=draft_top_token_ids,
        tree_budget=tree_budget,
        out_node_token_ids=_out_node_token_ids,
        out_node_depths=_out_node_depths,
        out_parents=_out_parents,
        out_visibility=_out_visibility,
        out_actual_tree_sizes=_out_actual_tree_sizes,
        heap_scores=_heap_scores,
        heap_parents=_heap_parents,
        heap_depths=_heap_depths,
        heap_ranks=_heap_ranks,
    )
    max_nodes = int(tree_budget) + 1
    if prune_to_deepest_chains:
        if (
            _out_pruned_node_token_ids is None
            or _out_pruned_node_depths is None
            or _out_pruned_parents is None
            or _out_pruned_visibility is None
        ):
            raise ValueError("DDTree GPU pruning requires pruned output buffers.")
        actual_tree_sizes = prune_ddtree_deepest_chains_triton(
            in_node_token_ids=_out_node_token_ids[:bs, :tree_budget],
            in_node_depths=_out_node_depths[:bs, :tree_budget],
            in_parents=_out_parents[:bs, :max_nodes],
            in_visibility=_out_visibility[:bs, :max_nodes, :max_nodes],
            out_node_token_ids=_out_pruned_node_token_ids[:bs, :tree_budget],
            out_node_depths=_out_pruned_node_depths[:bs, :tree_budget],
            out_parents=_out_pruned_parents[:bs, :max_nodes],
            out_visibility=_out_pruned_visibility[:bs, :max_nodes, :max_nodes],
            out_actual_tree_sizes=_out_actual_tree_sizes[:bs],
            scratch=_heap_depths[:bs],
            tree_budget=tree_budget,
        )
        actual_sizes_cpu = [int(x) for x in actual_tree_sizes.detach().cpu().tolist()]
        return (
            _out_pruned_node_token_ids[:bs, :tree_budget],
            _out_pruned_node_depths[:bs, :tree_budget],
            _out_pruned_parents[:bs, :max_nodes],
            [],
            _out_pruned_visibility[:bs, :max_nodes, :max_nodes],
            actual_tree_sizes,
            actual_sizes_cpu,
        )

    return (
        _out_node_token_ids[:bs, :tree_budget],
        _out_node_depths[:bs, :tree_budget],
        _out_parents[:bs, :max_nodes],
        [],
        _out_visibility[:bs, :max_nodes, :max_nodes],
        actual_tree_sizes,
        [max_nodes] * int(bs),
    )


def _build_single_tree(
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    topk: int,
    depth_limit: int,
    budget: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[int, int]], np.ndarray, int
]:
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
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_idx,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
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


def build_child_maps_from_parent_metadata(
    draft_tokens_cpu: List[List[int]],
    parents_cpu: List[List[int]],
    actual_sizes_cpu: List[int],
) -> List[Dict[int, Dict[int, int]]]:
    child_maps: List[Dict[int, Dict[int, int]]] = []
    for tokens, parents, actual_size in zip(
        draft_tokens_cpu, parents_cpu, actual_sizes_cpu, strict=True
    ):
        child_map: Dict[int, Dict[int, int]] = {idx: {} for idx in range(actual_size)}
        for idx in range(1, actual_size):
            parent = int(parents[idx])
            if parent < 0 or parent >= actual_size:
                raise ValueError(
                    f"Invalid DDTree parent index {parent} for child {idx}."
                )
            child_map.setdefault(parent, {})[int(tokens[idx])] = idx
            child_map.setdefault(idx, {})
        child_maps.append(child_map)
    return child_maps


def _prune_tree_to_deepest_chains(
    node_token_ids: np.ndarray,
    node_depths: np.ndarray,
    parents: np.ndarray,
    child_map: Dict[int, Dict[int, int]],
    visibility: np.ndarray,
    actual_size: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[int, int]], np.ndarray, int
]:
    """Keep only nodes whose subtree reaches the deepest node depth."""
    if actual_size <= 1:
        return node_token_ids, node_depths, parents, child_map, visibility, actual_size

    depths = np.zeros(actual_size, dtype=np.int32)
    depths[1:] = node_depths[: actual_size - 1]
    max_depth = int(depths.max())

    children: List[List[int]] = [[] for _ in range(actual_size)]
    for child in range(1, actual_size):
        parent = int(parents[child])
        if parent < 0 or parent >= actual_size:
            raise ValueError(
                f"Invalid DDTree parent index {parent} for child {child}."
            )
        children[parent].append(child)

    subtree_max_depth = depths.copy()
    for node in range(actual_size - 1, -1, -1):
        for child in children[node]:
            if subtree_max_depth[child] > subtree_max_depth[node]:
                subtree_max_depth[node] = subtree_max_depth[child]

    keep = subtree_max_depth == max_depth
    keep[0] = True
    if bool(keep.all()):
        return node_token_ids, node_depths, parents, child_map, visibility, actual_size

    kept_indices = [idx for idx in range(actual_size) if bool(keep[idx])]
    old_to_new = np.full(actual_size, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(kept_indices):
        old_to_new[old_idx] = new_idx

    new_actual_size = len(kept_indices)
    new_node_token_ids = np.zeros(new_actual_size - 1, dtype=node_token_ids.dtype)
    new_node_depths = np.zeros(new_actual_size - 1, dtype=node_depths.dtype)
    new_parents = np.full(new_actual_size, -1, dtype=parents.dtype)
    new_child_map: Dict[int, Dict[int, int]] = {
        idx: {} for idx in range(new_actual_size)
    }

    for new_idx, old_idx in enumerate(kept_indices):
        if old_idx == 0:
            continue

        old_parent = int(parents[old_idx])
        new_parent = int(old_to_new[old_parent])
        if new_parent < 0:
            raise ValueError(
                "DDTree pruning attempted to keep a node whose parent was pruned."
            )

        token_id = int(node_token_ids[old_idx - 1])
        new_node_token_ids[new_idx - 1] = token_id
        new_node_depths[new_idx - 1] = depths[old_idx]
        new_parents[new_idx] = new_parent
        new_child_map[new_parent][token_id] = new_idx

    new_visibility = visibility[np.ix_(kept_indices, kept_indices)].copy()
    return (
        new_node_token_ids,
        new_node_depths,
        new_parents,
        new_child_map,
        new_visibility,
        new_actual_size,
    )


def _compile_ddtree_tree_torch(
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
    verify_token_num: Optional[int] = None,
    build_attention_mask: bool = True,
    _out_verify_input_ids: Optional[torch.Tensor] = None,
    _out_verify_position_ids: Optional[torch.Tensor] = None,
    _out_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    bs = root_token_ids.shape[0]
    max_nodes = tree_budget + 1 if verify_token_num is None else int(verify_token_num)
    if max_nodes <= 0:
        raise ValueError(f"DDTree verify_token_num must be positive, got {max_nodes}.")

    if _out_verify_input_ids is None:
        verify_input_ids = torch.empty(
            bs, max_nodes, dtype=torch.long, device=device
        )
    else:
        verify_input_ids = _out_verify_input_ids.reshape(-1)[: bs * max_nodes].view(
            bs, max_nodes
        )
    verify_input_ids[:, 0] = root_token_ids
    if max_nodes > 1:
        verify_input_ids[:, 1:] = node_token_ids[:, : max_nodes - 1]

    if _out_verify_position_ids is None:
        verify_position_ids = torch.empty(
            bs, max_nodes, dtype=torch.long, device=device
        )
    else:
        verify_position_ids = _out_verify_position_ids.reshape(-1)[
            : bs * max_nodes
        ].view(bs, max_nodes)
    verify_position_ids[:, 0] = start_positions
    if max_nodes > 1:
        verify_position_ids[:, 1:] = (
            start_positions.unsqueeze(1) + node_depths[:, : max_nodes - 1]
        )

    if actual_sizes_cpu is None:
        actual_sizes_cpu = [int(x) for x in actual_tree_sizes.detach().cpu().tolist()]
    max_actual_size = max(actual_sizes_cpu) if actual_sizes_cpu else 1
    if max_actual_size > max_nodes:
        raise ValueError(
            "DDTree verify_token_num is smaller than a pruned tree: "
            f"verify_token_num={max_nodes}, max_actual_size={max_actual_size}."
        )

    if not build_attention_mask:
        for b, actual_size in enumerate(actual_sizes_cpu):
            if actual_size < max_nodes:
                visibility[b, actual_size:max_nodes, :max_nodes].fill_(False)
                dummy_rows = torch.arange(actual_size, max_nodes, device=device)
                visibility[b, dummy_rows, dummy_rows] = True
        return verify_input_ids, verify_position_ids, None, actual_tree_sizes

    # Attention backends consume the speculative custom mask as a request-packed
    # boolean allow-mask:
    #   concat(mask_i.reshape(-1)), mask_i.shape == [max_nodes, prefix_i + max_nodes]
    # FlashInfer in particular derives mask_indptr from the per-request q/kv
    # lengths, so retaining max-prefix padding between requests corrupts every
    # request after the first one.  Allocate the final packed representation
    # directly instead of materializing a [bs, max_nodes, max_kv_len] rectangle.
    if past_lens_cpu is None:
        past_lens_cpu = [int(x) for x in past_lengths.detach().cpu().tolist()]
    mask_numel = sum(max_nodes * (past_len + max_nodes) for past_len in past_lens_cpu)
    if _out_attention_mask is None:
        tree_attention_mask = torch.empty(
            (mask_numel,), dtype=torch.bool, device=device
        )
    else:
        if _out_attention_mask.numel() < mask_numel:
            raise ValueError(
                "DDTree attention-mask output buffer is too small: "
                f"capacity={_out_attention_mask.numel()}, required={mask_numel}."
            )
        tree_attention_mask = _out_attention_mask[:mask_numel]

    offset = 0
    for b, (past_len_i, actual_size) in enumerate(
        zip(past_lens_cpu, actual_sizes_cpu, strict=True)
    ):
        kv_len_i = past_len_i + max_nodes
        request_mask = tree_attention_mask[offset : offset + max_nodes * kv_len_i].view(
            max_nodes, kv_len_i
        )

        # Real tree queries attend to the complete committed prefix.
        request_mask[:actual_size, :past_len_i] = True

        # Within the drafted tree, a query sees only the root, its ancestors,
        # and itself.  Only the tree suffix needs explicit false-fill; the
        # committed prefix is all-true and can avoid the previous full-mask
        # zero_() write.
        request_mask[:actual_size, past_len_i:] = False
        request_mask[:actual_size, past_len_i : past_len_i + actual_size].copy_(
            visibility[b, :actual_size, :actual_size]
        )
        if actual_size < max_nodes:
            request_mask[actual_size:, :].fill_(False)
            dummy_rows = torch.arange(actual_size, max_nodes, device=device)
            request_mask[dummy_rows, past_len_i + dummy_rows] = True
        offset += max_nodes * kv_len_i

    return verify_input_ids, verify_position_ids, tree_attention_mask, actual_tree_sizes


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
    verify_token_num: Optional[int] = None,
    build_attention_mask: bool = True,
    _out_verify_input_ids: Optional[torch.Tensor] = None,
    _out_verify_position_ids: Optional[torch.Tensor] = None,
    _out_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Compile DDTree verify tensors, using Triton for the CUDA hot path."""

    bs = int(root_token_ids.shape[0])
    max_nodes = tree_budget + 1 if verify_token_num is None else int(verify_token_num)
    if max_nodes <= 0:
        raise ValueError(f"DDTree verify_token_num must be positive, got {max_nodes}.")

    if actual_sizes_cpu is None:
        actual_sizes_cpu = [int(x) for x in actual_tree_sizes.detach().cpu().tolist()]
    max_actual_size = max(actual_sizes_cpu) if actual_sizes_cpu else 1
    if max_actual_size > max_nodes:
        raise ValueError(
            "DDTree verify_token_num is smaller than a pruned tree: "
            f"verify_token_num={max_nodes}, max_actual_size={max_actual_size}."
        )

    if not root_token_ids.is_cuda:
        return _compile_ddtree_tree_torch(
            root_token_ids=root_token_ids,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility=visibility,
            start_positions=start_positions,
            past_lengths=past_lengths,
            tree_budget=tree_budget,
            actual_tree_sizes=actual_tree_sizes,
            device=device,
            past_lens_cpu=past_lens_cpu,
            actual_sizes_cpu=actual_sizes_cpu,
            verify_token_num=verify_token_num,
            build_attention_mask=build_attention_mask,
            _out_verify_input_ids=_out_verify_input_ids,
            _out_verify_position_ids=_out_verify_position_ids,
            _out_attention_mask=_out_attention_mask,
        )

    if _out_verify_input_ids is None:
        verify_input_ids = torch.empty(
            bs, max_nodes, dtype=torch.long, device=device
        )
    else:
        if _out_verify_input_ids.numel() < bs * max_nodes:
            raise ValueError("DDTree verify-input output buffer is too small.")
        verify_input_ids = _out_verify_input_ids.reshape(-1)[
            : bs * max_nodes
        ].view(bs, max_nodes)

    if _out_verify_position_ids is None:
        verify_position_ids = torch.empty(
            bs, max_nodes, dtype=torch.long, device=device
        )
    else:
        if _out_verify_position_ids.numel() < bs * max_nodes:
            raise ValueError("DDTree position-id output buffer is too small.")
        verify_position_ids = _out_verify_position_ids.reshape(-1)[
            : bs * max_nodes
        ].view(bs, max_nodes)

    tree_attention_mask = None
    if build_attention_mask:
        if past_lens_cpu is None:
            past_lens_cpu = [int(x) for x in past_lengths.detach().cpu().tolist()]
        mask_numel = sum(
            max_nodes * (past_len + max_nodes) for past_len in past_lens_cpu
        )
        if _out_attention_mask is None:
            tree_attention_mask = torch.empty(
                mask_numel, dtype=torch.bool, device=device
            )
        else:
            if _out_attention_mask.numel() < mask_numel:
                raise ValueError(
                    "DDTree attention-mask output buffer is too small: "
                    f"capacity={_out_attention_mask.numel()}, required={mask_numel}."
                )
            tree_attention_mask = _out_attention_mask[:mask_numel]

    from sglang.kernels.ops.speculative.ddtree import (
        compile_ddtree_tree_triton,
    )

    compile_ddtree_tree_triton(
        root_token_ids=root_token_ids,
        node_token_ids=node_token_ids,
        node_depths=node_depths,
        visibility=visibility,
        start_positions=start_positions,
        past_lengths=past_lengths,
        actual_tree_sizes=actual_tree_sizes,
        verify_input_ids=verify_input_ids,
        verify_position_ids=verify_position_ids,
        tree_attention_mask=tree_attention_mask,
        q_len=max_nodes,
        max_kv_len=(max(past_lens_cpu) + max_nodes) if past_lens_cpu else None,
        pad_visibility=(
            not build_attention_mask
            and any(actual_size < max_nodes for actual_size in actual_sizes_cpu)
        ),
    )
    return (
        verify_input_ids,
        verify_position_ids,
        tree_attention_mask,
        actual_tree_sizes,
    )


def sample_ddtree_target_probs_gpu(
    *,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    parents: torch.Tensor,
    actual_tree_sizes: Optional[torch.Tensor] = None,
    uniform_samples: torch.Tensor,
    uniform_final: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_token_ids: torch.Tensor,
    accepted_lens: torch.Tensor,
    next_tokens: torch.Tensor,
    reject_indices: torch.Tensor,
    reject_child_tokens: torch.Tensor,
    reject_child_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDTree target-only sampling fast path for unfiltered sampling."""

    if target_probs.ndim != 3:
        raise ValueError(
            "DDTree native sampler expects target_probs [bs, q, vocab], got "
            f"{tuple(target_probs.shape)}."
        )
    bs, q_len, _ = target_probs.shape
    device = target_probs.device
    if actual_tree_sizes is None:
        actual_tree_sizes = torch.full((bs,), q_len, dtype=torch.long, device=device)

    from sglang.kernels.ops.speculative.ddtree import (
        sample_ddtree_target_probs_triton,
    )

    return sample_ddtree_target_probs_triton(
        target_probs=target_probs,
        draft_tokens=draft_tokens,
        parents=parents,
        actual_tree_sizes=actual_tree_sizes,
        uniform_samples=uniform_samples,
        uniform_final=uniform_final,
        accepted_indices=accepted_indices,
        accepted_token_ids=accepted_token_ids,
        accepted_lens=accepted_lens,
        next_tokens=next_tokens,
        reject_indices=reject_indices,
        reject_child_tokens=reject_child_tokens,
        reject_child_counts=reject_child_counts,
    )


def follow_verified_tree_gpu(
    *,
    draft_tokens: torch.Tensor,
    target_predict: torch.Tensor,
    parents: torch.Tensor,
    actual_tree_sizes: Optional[torch.Tensor] = None,
    accepted_indices: Optional[torch.Tensor] = None,
    accepted_token_ids: Optional[torch.Tensor] = None,
    accepted_lens: Optional[torch.Tensor] = None,
    next_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU equivalent of follow_verified_tree using parent metadata.

    Returns accepted local node indices, accepted token ids in path order,
    accepted path lengths, and final target bonus tokens.
    """

    if draft_tokens.ndim != 2 or target_predict.shape != draft_tokens.shape:
        raise ValueError(
            "DDTree GPU follow expects [bs, q] draft_tokens/target_predict, got "
            f"{tuple(draft_tokens.shape)} and {tuple(target_predict.shape)}."
        )
    if parents is None or parents.shape != draft_tokens.shape:
        raise ValueError("DDTree GPU follow requires parents with the same shape.")
    if not draft_tokens.is_cuda:
        raise ValueError("DDTree GPU follow requires CUDA tensors.")

    bs, q_len = draft_tokens.shape
    device = draft_tokens.device
    if actual_tree_sizes is None:
        actual_tree_sizes = torch.full((bs,), q_len, dtype=torch.long, device=device)
    if accepted_indices is None:
        accepted_indices = torch.empty((bs, q_len), dtype=torch.long, device=device)
    if accepted_token_ids is None:
        accepted_token_ids = torch.empty((bs, q_len), dtype=torch.long, device=device)
    if accepted_lens is None:
        accepted_lens = torch.empty((bs,), dtype=torch.long, device=device)
    if next_tokens is None:
        next_tokens = torch.empty((bs,), dtype=torch.long, device=device)

    from sglang.kernels.ops.speculative.ddtree import (
        follow_ddtree_verified_path_triton,
    )

    return follow_ddtree_verified_path_triton(
        draft_tokens=draft_tokens.contiguous(),
        target_predict=target_predict.contiguous(),
        parents=parents.contiguous(),
        actual_tree_sizes=actual_tree_sizes.contiguous(),
        accepted_indices=accepted_indices,
        accepted_token_ids=accepted_token_ids,
        accepted_lens=accepted_lens,
        next_tokens=next_tokens,
    )


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
