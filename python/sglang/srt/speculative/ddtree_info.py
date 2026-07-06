from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.common import (
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


# CODEX_DDTREE_ACCEPT_LENGTH_EXPORT disabled:
# _build_ddtree_accept_round_info used to build per-round response metadata.


@dataclass
class DDTreeVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    tree_budget: int
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    child_maps: List[Dict[int, Dict[int, int]]] = field(default_factory=list)
    actual_tree_sizes: Optional[torch.Tensor] = None

    custom_mask: Optional[torch.Tensor] = None
    topk: int = 1

    accepted_indices: List[List[int]] = field(default_factory=list)
    next_tokens: Optional[torch.Tensor] = None

    # When True, the tree is a pure linear chain (no branching siblings).
    # In this mode, cascade attention is unnecessary and a standard causal
    # mask suffices, matching DFLASH's verify pattern exactly.
    tree_is_spine: bool = False
    # CODEX_DDTREE_ACCEPT_LENGTH_EXPORT disabled:
    # block_size: Optional[int] = None

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
        use_sampling = (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        )

        with gpu_ctx("accept_argmax"):
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

        # --- 1) Acceptance ---
        if self.tree_is_spine:
            if use_sampling:
                # Chain-based sampling verification via sgl_kernel.
                # candidates must include ALL N tokens (bonus + drafts).
                from sglang.srt.speculative.dflash_utils import (
                    compute_dflash_sampling_correct_drafts_and_bonus,
                )

                candidates = self.draft_token.view(bs, self.draft_token_num)
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
                candidates = self.draft_token.view(bs, self.draft_token_num)
                correct, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = (correct + 1).tolist()
                num_correct_drafts_per_req = [max(0, cl - 1) for cl in commit_lens]
                self.accepted_indices = [list(range(cl)) for cl in commit_lens]
                self.next_tokens = bonus
        else:
            # Full tree path: greedy-only for now (tree sampling requires
            # sgl_kernel tree topology which is expensive to build per-step).
            with cpu_ctx("follow_tree_cpu"):
                (
                    self.accepted_indices,
                    self.next_tokens,
                    next_tokens_cpu,
                ) = follow_verified_tree(self.child_maps, target_predict)
                draft_tokens_cpu = (
                    self.draft_token.view(bs, self.draft_token_num).cpu().tolist()
                )
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
                    # CODEX_DDTREE_ACCEPT_LENGTH_EXPORT disabled:
                    # Do not append per-verify accepted-length records.
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
                # CODEX_DDTREE_ACCEPT_LENGTH_EXPORT disabled:
                # Do not append per-verify accepted-length records.
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
