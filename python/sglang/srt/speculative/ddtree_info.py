from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.sampler import (
    apply_custom_logit_processor,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.mem_cache.common import (
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


def _maybe_apply_sampling_logits_processors(
    *,
    logits: torch.Tensor,
    sampling_info,
    bs: int,
    q_len: int,
):
    if sampling_info is None:
        return
    if len(sampling_info) != bs:
        raise RuntimeError(
            "DDTREE verify sampling_info size mismatch: "
            f"len(sampling_info)={len(sampling_info)}, bs={bs}."
        )

    # Keep target-verify token selection consistent with the normal sampling path.
    if sampling_info.has_custom_logit_processor:
        apply_custom_logit_processor(
            logits,
            sampling_info,
            num_tokens_in_batch=q_len,
        )

    if (
        sampling_info.penalizer_orchestrator.is_required
        or sampling_info.logit_bias is not None
    ):
        linear_penalty = torch.zeros(
            (bs, logits.shape[1]),
            dtype=torch.float32,
            device=logits.device,
        )
        sampling_info.apply_logits_bias(linear_penalty)
        logits.add_(torch.repeat_interleave(linear_penalty, q_len, dim=0))


def _can_use_ddtree_native_sampling(sampling_info) -> bool:
    if sampling_info is None or sampling_info.is_all_greedy:
        return False
    if getattr(sampling_info, "sampling_seed", None) is not None:
        return False
    if bool(getattr(sampling_info, "need_min_p_sampling", False)):
        return False
    if bool(getattr(sampling_info, "need_top_p_sampling", False)):
        return False
    if bool(getattr(sampling_info, "need_top_k_sampling", False)):
        return False
    return True


def _sample_ddtree_native_target_tokens(
    *,
    logits: torch.Tensor,
    sampling_info,
    bs: int,
    q_len: int,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.srt.speculative.ddtree_utils import sample_ddtree_target_probs_gpu

    target_probs = F.softmax(
        logits.view(bs, q_len, -1) / sampling_info.temperatures.view(bs, 1, 1),
        dim=-1,
    )
    result = sample_ddtree_target_probs_gpu(
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

    tp_group = get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
    if tp_group.world_size > 1:
        for tensor in result:
            tp_group.broadcast(tensor, src=0)
    return result


def _sample_ddtree_target_tokens(
    *,
    logits: torch.Tensor,
    sampling_info,
    positions: torch.Tensor,
    bs: int,
    q_len: int,
) -> torch.Tensor:
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, q_len, dim=0
    )
    probs = F.softmax(logits / expanded_temperature, dim=-1)

    repeated_seed = None
    if sampling_info.sampling_seed is not None:
        repeated_seed = torch.repeat_interleave(sampling_info.sampling_seed, q_len, dim=0)

    sampled = top_k_top_p_min_p_sampling_from_probs_torch(
        probs,
        torch.repeat_interleave(sampling_info.top_ks, q_len, dim=0),
        torch.repeat_interleave(sampling_info.top_ps, q_len, dim=0),
        torch.repeat_interleave(sampling_info.min_ps, q_len, dim=0),
        sampling_info.need_min_p_sampling,
        repeated_seed,
        positions,
    )
    target_predict = sampled.to(dtype=torch.long).view(bs, q_len)

    # Different TP ranks can see tiny floating-point differences in logits/probs.
    # The sampled verifier walk must be identical on every rank because it mutates
    # request state and KV ownership.
    tp_group = get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
    if tp_group.world_size > 1:
        tp_group.broadcast(target_predict, src=0)
    return target_predict


@dataclass
class DDTreeVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    tree_budget: int
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    child_maps: List[Dict[int, Dict[int, int]]] = field(default_factory=list)
    actual_tree_sizes: Optional[torch.Tensor] = None
    parents: Optional[torch.Tensor] = None
    visibility: Optional[torch.Tensor] = None

    custom_mask: Optional[torch.Tensor] = None
    topk: int = 1

    accepted_indices: List[List[int]] = field(default_factory=list)
    next_tokens: Optional[torch.Tensor] = None

    # When True, the tree is a pure linear chain (no branching siblings).
    # In this mode, cascade attention is unnecessary and a standard causal
    # mask suffices, matching DFLASH's verify pattern exactly.
    tree_is_spine: bool = False
    use_tree_attention: bool = False
    raw_tree_size: Optional[int] = None
    cuda_graph_bucket_size: Optional[int] = None
    force_cpu_follow: bool = False
    follow_accepted_indices: Optional[torch.Tensor] = None
    follow_accepted_token_ids: Optional[torch.Tensor] = None
    follow_accepted_lens: Optional[torch.Tensor] = None
    follow_next_tokens: Optional[torch.Tensor] = None
    native_sampling_uniform: Optional[torch.Tensor] = None
    native_sampling_uniform_final: Optional[torch.Tensor] = None
    native_reject_indices: Optional[torch.Tensor] = None
    native_reject_child_tokens: Optional[torch.Tensor] = None
    native_reject_child_counts: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)

    def get_spec_adjust_token_coefficient(self):
        return (self.draft_token_num, self.draft_token_num)

    def prepare_for_verify(self, batch, page_size):
        bs = len(batch.reqs)
        q_len = self.draft_token_num

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + q_len
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset_cpu = [pl + q_len for pl in prefix_lens_cpu.tolist()]
            from sglang.srt.mem_cache.common import (
                alloc_paged_token_slots_extend,
            )

            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens_cpu.tolist(),
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            end_offset = torch.tensor(
                end_offset_cpu, dtype=prefix_lens.dtype, device=prefix_lens.device
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        from sglang.srt.layers.attention.utils import (
            create_flashinfer_kv_indices_triton,
        )

        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            mask = mask.contiguous().view(-1).to(dtype=torch.bool)
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            else:
                mask = mask[:mask_numel]
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self, *, batch, logits_output, page_size, model_runner=None, profiler=None
    ):
        from sglang.srt.speculative.ddtree_utils import (
            follow_verified_tree,
            follow_verified_tree_gpu,
        )
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_correct_drafts_and_bonus,
            is_dflash_sampling_verify_available,
        )

        bs = len(batch.reqs)
        device = batch.device
        cpu_ctx = (
            profiler.cpu if profiler is not None else (lambda _stage: nullcontext())
        )
        gpu_ctx = (
            profiler.gpu if profiler is not None else (lambda _stage: nullcontext())
        )

        sampling_info = batch.sampling_info
        sampling_requested = (
            sampling_info is not None and not sampling_info.is_all_greedy
        )
        use_dflash_chain_sampling = (
            sampling_requested
            and self.tree_is_spine
            and is_dflash_sampling_verify_available()
        )

        with gpu_ctx("sampling_logits_processors"):
            _maybe_apply_sampling_logits_processors(
                logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                bs=bs,
                q_len=self.draft_token_num,
            )

        target_predict = None
        target_native_sample_result = None
        can_use_native_sampling = (
            sampling_requested
            and not self.tree_is_spine
            and not self.force_cpu_follow
            and self.parents is not None
            and self.actual_tree_sizes is not None
            and self.draft_token.is_cuda
            and self.follow_accepted_indices is not None
            and self.follow_accepted_token_ids is not None
            and self.follow_accepted_lens is not None
            and self.follow_next_tokens is not None
            and self.native_sampling_uniform is not None
            and self.native_sampling_uniform_final is not None
            and self.native_reject_indices is not None
            and self.native_reject_child_tokens is not None
            and self.native_reject_child_counts is not None
            and _can_use_ddtree_native_sampling(sampling_info)
        )
        if not use_dflash_chain_sampling:
            if sampling_requested:
                if can_use_native_sampling:
                    with gpu_ctx("target_native_sample"):
                        target_native_sample_result = _sample_ddtree_native_target_tokens(
                            logits=logits_output.next_token_logits,
                            sampling_info=sampling_info,
                            bs=bs,
                            q_len=self.draft_token_num,
                            draft_tokens=self.draft_token.view(bs, self.draft_token_num),
                            parents=self.parents.view(bs, self.draft_token_num),
                            actual_tree_sizes=self.actual_tree_sizes,
                            uniform_samples=self.native_sampling_uniform,
                            uniform_final=self.native_sampling_uniform_final,
                            accepted_indices=self.follow_accepted_indices,
                            accepted_token_ids=self.follow_accepted_token_ids,
                            accepted_lens=self.follow_accepted_lens,
                            next_tokens=self.follow_next_tokens,
                            reject_indices=self.native_reject_indices,
                            reject_child_tokens=self.native_reject_child_tokens,
                            reject_child_counts=self.native_reject_child_counts,
                        )
                else:
                    with gpu_ctx("target_sample"):
                        target_predict = _sample_ddtree_target_tokens(
                            logits=logits_output.next_token_logits,
                            sampling_info=sampling_info,
                            positions=self.positions,
                            bs=bs,
                            q_len=self.draft_token_num,
                        )
            else:
                with gpu_ctx("accept_argmax"):
                    target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
                    target_predict = target_predict.reshape(bs, self.draft_token_num)

        gpu_follow_indices = None
        gpu_follow_token_ids = None

        # --- 1) Acceptance ---
        if self.tree_is_spine:
            candidates = self.draft_token.view(bs, self.draft_token_num)
            if use_dflash_chain_sampling:
                # Chain-based sampling verification via sgl_kernel.
                # candidates must include ALL N tokens (bonus + drafts).
                from sglang.srt.speculative.dflash_utils import (
                    compute_dflash_sampling_correct_drafts_and_bonus,
                )

                correct_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                )
                # correct_len = number of accepted candidates (includes bonus at pos 0).
                # commit_len = correct_len + 1 (includes the final bonus token).
                commit_lens = (correct_len + 1).tolist()
                num_correct_drafts_per_req = [max(0, cl - 1) for cl in commit_lens]
                self.accepted_indices = [
                    list(range(cl)) for cl in (correct_len + 1).tolist()
                ]
                self.next_tokens = bonus
            else:
                correct, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = (correct + 1).tolist()
                num_correct_drafts_per_req = [max(0, cl - 1) for cl in commit_lens]
                self.accepted_indices = [list(range(cl)) for cl in commit_lens]
                self.next_tokens = bonus
        else:
            # Full-tree verification follows the target model's decoding rule:
            # greedy uses argmax, while non-greedy uses a sampled target token per
            # tree node and then walks the draft tree by child-token matches.
            can_use_gpu_follow = (
                target_native_sample_result is not None
                or (
                    not self.force_cpu_follow
                    and self.parents is not None
                    and self.draft_token.is_cuda
                    and target_predict is not None
                    and target_predict.is_cuda
                )
            )
            if can_use_gpu_follow:
                if target_native_sample_result is not None:
                    (
                        gpu_follow_indices,
                        gpu_follow_token_ids,
                        gpu_follow_lens,
                        gpu_next_tokens,
                    ) = target_native_sample_result
                    self.next_tokens = gpu_next_tokens
                else:
                    with gpu_ctx("follow_tree_gpu"):
                        draft_tokens_2d = self.draft_token.view(bs, self.draft_token_num)
                        parents_2d = self.parents.view(bs, self.draft_token_num)
                        (
                            gpu_follow_indices,
                            gpu_follow_token_ids,
                            gpu_follow_lens,
                            gpu_next_tokens,
                        ) = follow_verified_tree_gpu(
                            draft_tokens=draft_tokens_2d,
                            target_predict=target_predict,
                            parents=parents_2d,
                            actual_tree_sizes=self.actual_tree_sizes,
                            accepted_indices=self.follow_accepted_indices,
                            accepted_token_ids=self.follow_accepted_token_ids,
                            accepted_lens=self.follow_accepted_lens,
                            next_tokens=self.follow_next_tokens,
                        )
                        self.next_tokens = gpu_next_tokens

                commit_lens = []
                num_correct_drafts_per_req = []
                with cpu_ctx("commit_output_cpu"):
                    accepted_lens_cpu = gpu_follow_lens.detach().cpu().tolist()
                    accepted_indices_cpu = gpu_follow_indices.detach().cpu().tolist()
                    accepted_token_ids_cpu = (
                        gpu_follow_token_ids.detach().cpu().tolist()
                    )
                    next_tokens_cpu = gpu_next_tokens.detach().cpu().tolist()
                    self.accepted_indices = []
                    for i, req in enumerate(batch.reqs):
                        accepted_len = int(accepted_lens_cpu[i])
                        accepted = [
                            int(idx)
                            for idx in accepted_indices_cpu[i][:accepted_len]
                        ]
                        self.accepted_indices.append(accepted)
                        appended = 0
                        for pos in range(1, accepted_len):
                            token_id = int(accepted_token_ids_cpu[i][pos])
                            req.output_ids.append(token_id)
                            appended += 1
                            req.update_finish_state()
                            if req.finished():
                                break
                        if not req.finished():
                            bonus = int(next_tokens_cpu[i])
                            req.output_ids.append(bonus)
                            appended += 1
                            req.update_finish_state()
                        num_correct_drafts = max(0, appended - 1)
                        commit_lens.append(appended)
                        num_correct_drafts_per_req.append(num_correct_drafts)
                        req.spec_verify_ct += 1
                        req.spec_num_correct_drafts += num_correct_drafts
                        req.update_spec_correct_drafts_histogram(num_correct_drafts)
            else:
                with cpu_ctx("follow_tree_cpu"):
                    draft_tokens_cpu = (
                        self.draft_token.view(bs, self.draft_token_num).cpu().tolist()
                    )
                    child_maps = self.child_maps
                    if not child_maps:
                        if self.parents is None:
                            raise RuntimeError(
                                "DDTree verify needs child_maps or parent metadata."
                            )
                        from sglang.srt.speculative.ddtree_utils import (
                            build_child_maps_from_parent_metadata,
                        )

                        parents_cpu = (
                            self.parents.view(bs, self.draft_token_num).cpu().tolist()
                        )
                        if self.actual_tree_sizes is None:
                            actual_sizes_cpu = [self.draft_token_num] * bs
                        else:
                            actual_sizes_cpu = [
                                int(x) for x in self.actual_tree_sizes.cpu().tolist()
                            ]
                        child_maps = build_child_maps_from_parent_metadata(
                            draft_tokens_cpu, parents_cpu, actual_sizes_cpu
                        )
                        self.child_maps = child_maps
                    (
                        self.accepted_indices,
                        self.next_tokens,
                        next_tokens_cpu,
                    ) = follow_verified_tree(child_maps, target_predict)
                commit_lens = []
                num_correct_drafts_per_req = []
                with cpu_ctx("commit_output_cpu"):
                    for i, req in enumerate(batch.reqs):
                        accepted = self.accepted_indices[i]
                        appended = 0
                        for idx in accepted[1:]:
                            token_id = int(draft_tokens_cpu[i][idx])
                            req.output_ids.append(token_id)
                            appended += 1
                            req.update_finish_state()
                            if req.finished():
                                break
                        if not req.finished():
                            bonus = int(next_tokens_cpu[i])
                            req.output_ids.append(bonus)
                            appended += 1
                            req.update_finish_state()
                        num_correct_drafts = max(0, appended - 1)
                        commit_lens.append(appended)
                        num_correct_drafts_per_req.append(num_correct_drafts)
                        req.spec_verify_ct += 1
                        req.spec_num_correct_drafts += num_correct_drafts
                        req.update_spec_correct_drafts_histogram(num_correct_drafts)

        # --- 2) Commit tokens to output ---
        if self.tree_is_spine:
            for i, req in enumerate(batch.reqs):
                appended = 0
                for idx in range(1, commit_lens[i]):
                    token_id = int(self.draft_token[i * self.draft_token_num + idx])
                    req.output_ids.append(token_id)
                    appended += 1
                    req.update_finish_state()
                    if req.finished():
                        break
                else:
                    bonus = int(self.next_tokens[i].item())
                    req.output_ids.append(bonus)
                    appended += 1
                    req.update_finish_state()
                commit_lens[i] = appended
                num_correct_drafts_per_req[i] = max(0, appended - 1)
                self.accepted_indices[i] = list(range(appended))
                req.spec_verify_ct += 1
                req.spec_num_correct_drafts += num_correct_drafts_per_req[i]
                req.update_spec_correct_drafts_histogram(
                    num_correct_drafts_per_req[i]
                )

        commit_lens_tensor = torch.tensor(commit_lens, dtype=torch.long, device=device)

        # --- KV cache retention ---
        with gpu_ctx("kv_free_update"):
            if model_runner is not None:
                out_cache_loc_2d = batch.out_cache_loc.view(bs, self.draft_token_num)

                # For the token-granular KV pool, SGLang accesses target KV through
                # req_to_token indirection.  A branched DDTree path therefore does
                # not need to physically compact KV slots; keep the accepted cache
                # locations in generation order and free everything else.
                if page_size == 1:
                    kept_locs: List[torch.Tensor] = []
                    free_locs: List[torch.Tensor] = []

                    for i, commit_len in enumerate(commit_lens):
                        if commit_len <= 0:
                            keep_t = torch.empty((0,), dtype=torch.long, device=device)
                            keep_is_contiguous = True
                        elif self.tree_is_spine:
                            keep_t = None
                            keep_is_contiguous = True
                        else:
                            keep = self.accepted_indices[i][:commit_len]
                            if len(keep) != commit_len:
                                raise RuntimeError(
                                    "DDTree accepted path shorter than commit_len: "
                                    f"accepted={self.accepted_indices[i]}, "
                                    f"commit_len={commit_len}."
                                )
                            keep_is_contiguous = keep == list(range(commit_len))
                            keep_t = (
                                None
                                if keep_is_contiguous
                                else torch.tensor(keep, dtype=torch.long, device=device)
                            )

                        row_locs = out_cache_loc_2d[i]
                        if keep_is_contiguous:
                            if commit_len > 0:
                                kept_locs.append(row_locs[:commit_len])
                            free_locs.append(row_locs[commit_len:])
                        else:
                            if keep_t.numel() > 0:
                                kept_locs.append(row_locs.index_select(0, keep_t))

                            free_mask = torch.ones(
                                (self.draft_token_num,),
                                dtype=torch.bool,
                                device=device,
                            )
                            if keep_t.numel() > 0:
                                free_mask[keep_t] = False
                            free_locs.append(row_locs[free_mask])

                    if free_locs:
                        free_loc_tensor = (
                            free_locs[0]
                            if len(free_locs) == 1
                            else torch.cat(free_locs)
                        )
                        if free_loc_tensor.numel() > 0:
                            batch.token_to_kv_pool_allocator.free(free_loc_tensor)
                    batch.out_cache_loc = (
                        kept_locs[0]
                        if len(kept_locs) == 1
                        else torch.cat(kept_locs)
                        if kept_locs
                        else batch.out_cache_loc[:0]
                    )
                else:
                    # Paged KV allocators free at page granularity.  Preserve the
                    # existing physical-compaction path here.
                    need_compaction = False
                    for accepted in self.accepted_indices:
                        if accepted != list(range(len(accepted))):
                            need_compaction = True
                            break

                    if need_compaction:
                        model = model_runner.model
                        model_layers = getattr(model, "model", model)
                        model_layers = getattr(model_layers, "layers", None)
                        if model_layers is None:
                            model_layers = []

                        from sglang.srt.speculative.ddtree_utils import (
                            compact_ddtree_kv_cache,
                        )

                        token_to_kv_pool = model_runner.token_to_kv_pool
                        past_lengths = batch.seq_lens.clone()

                        for layer in model_layers:
                            attn_layer = layer.self_attn.attn
                            compact_ddtree_kv_cache(
                                token_to_kv_pool,
                                attn_layer,
                                out_cache_loc_2d,
                                self.accepted_indices,
                                past_lengths,
                                self.actual_tree_sizes,
                            )

                # Update req-level KV cache accounting.
                for req, commit_len in zip(batch.reqs, commit_lens, strict=True):
                    req.kv_committed_len += commit_len
                    req.kv_allocated_len = req.kv_committed_len

                # Update req_to_token pool mapping for newly committed tokens.
                end_offset = batch.seq_lens + commit_lens_tensor.to(
                    batch.seq_lens.dtype
                )
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    end_offset,
                    batch.out_cache_loc,
                    bs,
                )

                # Update batch seq lens.
                batch.seq_lens.add_(commit_lens_tensor.to(batch.seq_lens.dtype))
                batch.seq_lens_cpu.add_(
                    torch.tensor(
                        [int(c) for c in commit_lens],
                        dtype=batch.seq_lens_cpu.dtype,
                    )
                )
                batch.seq_lens_sum += sum(commit_lens)
            else:
                # Fallback path
                batch.seq_lens += commit_lens_tensor

        with gpu_ctx("hidden_select"):
            hidden = logits_output.hidden_states
            if hidden is not None:
                hidden = hidden.view(bs, self.draft_token_num, -1)
                segments = []
                for i, n in enumerate(commit_lens):
                    if n <= 0:
                        continue
                    if self.tree_is_spine:
                        segments.append(hidden[i, :n, :])
                    else:
                        keep = self.accepted_indices[i][:n]
                        if keep == list(range(n)):
                            segments.append(hidden[i, :n, :])
                        else:
                            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
                            segments.append(hidden[i].index_select(0, keep_t))
                if not segments:
                    next_target_hidden = hidden[:0]
                elif len(segments) == 1:
                    next_target_hidden = segments[0]
                else:
                    next_target_hidden = torch.cat(segments, dim=0)
            else:
                next_target_hidden = None

        num_correct_drafts_cpu = num_correct_drafts_per_req

        return (
            self.next_tokens,
            commit_lens_tensor,
            next_target_hidden,
            num_correct_drafts_cpu,
        )
