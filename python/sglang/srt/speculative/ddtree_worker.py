import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.ddtree_info import DDTreeVerifyInput
from sglang.srt.speculative.ddtree_utils import (
    build_ddtree_tree,
    build_ddtree_tree_gpu,
    compile_ddtree_tree,
    resolve_ddtree_cuda_graph_buckets,
    resolve_ddtree_target_backend_capability,
    select_ddtree_cuda_graph_bucket,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput
from sglang.srt.speculative.dflash_worker import DFlashWorker

logger = logging.getLogger(__name__)


class DDTreeWorker(DFlashWorker):
    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker,
    ):
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.tree_budget = getattr(server_args, "speculative_ddtree_budget", None)
        if self.tree_budget is None:
            self.tree_budget = self.block_size - 1

        self.is_ddtree_prune = bool(getattr(server_args, "is_ddtree_prune", False))
        self.use_tree_attention = bool(
            getattr(server_args, "use_tree_attention", False)
        )
        self.force_ddtree_cpu_build = bool(
            getattr(server_args, "speculative_ddtree_cpu_build", False)
        )
        self.force_ddtree_cpu_follow = bool(
            getattr(server_args, "speculative_ddtree_cpu_follow", False)
        )
        self.max_tree_nodes = self.tree_budget + 1
        self.ddtree_cuda_graph_buckets = resolve_ddtree_cuda_graph_buckets(
            tree_budget=self.tree_budget,
            is_ddtree_prune=self.is_ddtree_prune,
            configured_buckets=getattr(
                server_args, "speculative_ddtree_cuda_graph_buckets", None
            ),
        )
        self.use_ddtree_cuda_graph_buckets = self.is_ddtree_prune and not bool(
            getattr(server_args, "disable_cuda_graph", False)
        )

        # Pre-allocated tree-build output buffers (reused across steps).
        _max_bs = getattr(server_args, "cuda_graph_max_bs", None) or 64
        _budget = self.tree_budget
        _mn = self.max_tree_nodes
        dev = self.device
        self._tree_node_token_ids_buf: torch.Tensor = torch.zeros(
            _max_bs, _budget, dtype=torch.long, device=dev
        )
        self._tree_node_depths_buf: torch.Tensor = torch.zeros(
            _max_bs, _budget, dtype=torch.long, device=dev
        )
        self._tree_parents_buf: torch.Tensor = torch.full(
            (_max_bs, _mn), -1, dtype=torch.long, device=dev
        )
        self._tree_visibility_buf: torch.Tensor = torch.zeros(
            _max_bs, _mn, _mn, dtype=torch.bool, device=dev
        )
        self._tree_actual_sizes_buf: torch.Tensor = torch.zeros(
            _max_bs, dtype=torch.long, device=dev
        )
        self._tree_pruned_node_token_ids_buf: Optional[torch.Tensor] = None
        self._verify_input_ids_buf: torch.Tensor = torch.empty(
            _max_bs * _mn, dtype=torch.long, device=dev
        )
        self._verify_position_ids_buf: torch.Tensor = torch.empty(
            _max_bs * _mn, dtype=torch.long, device=dev
        )
        self._tree_attention_mask_buf: Optional[torch.Tensor] = None
        self._tree_pruned_node_depths_buf: Optional[torch.Tensor] = None
        self._tree_pruned_parents_buf: Optional[torch.Tensor] = None
        self._tree_pruned_visibility_buf: Optional[torch.Tensor] = None
        if self.is_ddtree_prune and not self.force_ddtree_cpu_build:
            self._tree_pruned_node_token_ids_buf = torch.zeros(
                _max_bs, _budget, dtype=torch.long, device=dev
            )
            self._tree_pruned_node_depths_buf = torch.zeros(
                _max_bs, _budget, dtype=torch.long, device=dev
            )
            self._tree_pruned_parents_buf = torch.full(
                (_max_bs, _mn), -1, dtype=torch.long, device=dev
            )
            self._tree_pruned_visibility_buf = torch.zeros(
                _max_bs, _mn, _mn, dtype=torch.bool, device=dev
            )
        _heap_cap = 1 << (max(1, 2 * _budget + 3) - 1).bit_length()
        self._tree_heap_scores_buf: torch.Tensor = torch.empty(
            _max_bs, _heap_cap, dtype=torch.float64, device=dev
        )
        self._tree_heap_parents_buf: torch.Tensor = torch.empty(
            _max_bs, _heap_cap, dtype=torch.int32, device=dev
        )
        self._tree_heap_depths_buf: torch.Tensor = torch.empty(
            _max_bs, _heap_cap, dtype=torch.int32, device=dev
        )
        self._tree_heap_ranks_buf: torch.Tensor = torch.empty(
            _max_bs, _heap_cap, dtype=torch.int32, device=dev
        )
        self._follow_accepted_indices_buf: torch.Tensor = torch.empty(
            _max_bs, _mn, dtype=torch.long, device=dev
        )
        self._follow_accepted_token_ids_buf: torch.Tensor = torch.empty(
            _max_bs, _mn, dtype=torch.long, device=dev
        )
        self._follow_accepted_lens_buf: torch.Tensor = torch.empty(
            _max_bs, dtype=torch.long, device=dev
        )
        self._follow_next_tokens_buf: torch.Tensor = torch.empty(
            _max_bs, dtype=torch.long, device=dev
        )
        self._native_sampling_uniform_buf: torch.Tensor = torch.empty(
            _max_bs, _mn, dtype=torch.float32, device=dev
        )
        self._native_sampling_uniform_final_buf: torch.Tensor = torch.empty(
            _max_bs, dtype=torch.float32, device=dev
        )
        self._native_reject_indices_buf: torch.Tensor = torch.empty(
            _max_bs, dtype=torch.long, device=dev
        )
        self._native_reject_child_tokens_buf: torch.Tensor = torch.empty(
            _max_bs, _mn, dtype=torch.long, device=dev
        )
        self._native_reject_child_counts_buf: torch.Tensor = torch.empty(
            _max_bs, dtype=torch.long, device=dev
        )
        self._ddtree_vocab_id_cache_key = None
        self._ddtree_org_token_ids: Optional[torch.Tensor] = None
        self._ddtree_added_token_ids: Optional[torch.Tensor] = None

        if self.tp_rank == 0:
            logger.info(
                "Initialized DDTree worker. block_size=%s, tree_budget=%s, max_tree_nodes=%s, is_ddtree_prune=%s, use_tree_attention=%s, force_cpu_build=%s, force_cpu_follow=%s, cuda_graph_buckets=%s",
                self.block_size,
                self.tree_budget,
                self.max_tree_nodes,
                self.is_ddtree_prune,
                self.use_tree_attention,
                self.force_ddtree_cpu_build,
                self.force_ddtree_cpu_follow,
                self.ddtree_cuda_graph_buckets,
            )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError(
                "DDTREE batch has grammar constraints, but scheduler should have rejected this request."
            )

        bs = batch.batch_size()
        profiler = self.ddtree_profiler

        # --- 1) Append target hidden to draft KV cache.
        with profiler.gpu("draft_kv_append_before"):
            self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DDTREE requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block (reuse parent's shared implementation).
        with profiler.gpu("draft_forward"):
            draft_hidden, positions_2d, block_ids = self._run_draft_forward(
                batch, draft_input
            )

        # --- 3) Full tree path: compute TP-safe top-K
        # proposal ids/logprobs, beam search, compile mask.
        #
        # DDTree Algorithm 1 uses K = min(B, |V|), where |V| is the global,
        # unpadded target vocabulary size rather than a TP-local shard size.
        vocab_size = getattr(lm_head, "num_embeddings", None)
        if vocab_size is None:
            vocab_size = getattr(
                self.target_worker.model_runner.model_config, "vocab_size", None
            )
        if vocab_size is None or int(vocab_size) <= 0:
            raise RuntimeError(
                "DDTREE requires a positive target vocabulary size to compute "
                "K = min(tree_budget, vocab_size)."
            )
        proposal_topk = min(int(self.tree_budget), int(vocab_size))
        with profiler.gpu("draft_head_topk"):
            (
                draft_top_log_probs,
                draft_top_token_ids,
            ) = self._compute_draft_topk_log_probs_and_token_ids(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
                topk=proposal_topk,
            )
        draft_top_log_probs = draft_top_log_probs.view(
            bs, self.block_size - 1, proposal_topk
        )
        draft_top_token_ids = draft_top_token_ids.view(
            bs, self.block_size - 1, proposal_topk
        )

        can_use_gpu_build = (
            not self.force_ddtree_cpu_build and draft_top_log_probs.is_cuda
        )
        if can_use_gpu_build:
            with profiler.gpu("tree_gpu_build"):
                (
                    node_token_ids,
                    node_depths,
                    parents,
                    child_maps,
                    visibility,
                    actual_tree_sizes,
                    actual_tree_sizes_cpu,
                ) = build_ddtree_tree_gpu(
                    draft_top_log_probs=draft_top_log_probs,
                    draft_top_token_ids=draft_top_token_ids,
                    tree_budget=self.tree_budget,
                    device=batch.device,
                    _out_node_token_ids=self._tree_node_token_ids_buf,
                    _out_node_depths=self._tree_node_depths_buf,
                    _out_parents=self._tree_parents_buf,
                    _out_visibility=self._tree_visibility_buf,
                    _out_actual_tree_sizes=self._tree_actual_sizes_buf,
                    _heap_scores=self._tree_heap_scores_buf,
                    _heap_parents=self._tree_heap_parents_buf,
                    _heap_depths=self._tree_heap_depths_buf,
                    _heap_ranks=self._tree_heap_ranks_buf,
                    prune_to_deepest_chains=self.is_ddtree_prune,
                    _out_pruned_node_token_ids=self._tree_pruned_node_token_ids_buf,
                    _out_pruned_node_depths=self._tree_pruned_node_depths_buf,
                    _out_pruned_parents=self._tree_pruned_parents_buf,
                    _out_pruned_visibility=self._tree_pruned_visibility_buf,
                )
        else:
            (
                node_token_ids,
                node_depths,
                parents,
                child_maps,
                visibility,
                actual_tree_sizes,
                actual_tree_sizes_cpu,
            ) = build_ddtree_tree(
                draft_logits=None,
                draft_top_log_probs=draft_top_log_probs,
                draft_top_token_ids=draft_top_token_ids,
                tree_budget=self.tree_budget,
                device=batch.device,
                _out_node_token_ids=self._tree_node_token_ids_buf,
                _out_node_depths=self._tree_node_depths_buf,
                _out_parents=self._tree_parents_buf,
                _out_visibility=self._tree_visibility_buf,
                prune_to_deepest_chains=self.is_ddtree_prune,
                profiler=profiler,
            )

        raw_verify_token_num = max(1, max(int(x) for x in actual_tree_sizes_cpu))
        if self.use_ddtree_cuda_graph_buckets:
            verify_token_num = select_ddtree_cuda_graph_bucket(
                raw_verify_token_num, self.ddtree_cuda_graph_buckets
            )
        else:
            verify_token_num = raw_verify_token_num

        target_attn_backend = getattr(
            self.target_worker.model_runner, "attn_backend", None
        )
        target_backend_capability = resolve_ddtree_target_backend_capability(
            target_attn_backend,
            speculative_attention_mode=getattr(
                self.server_args, "speculative_attention_mode", "prefill"
            ),
        )
        if self.use_tree_attention and not target_backend_capability.use_visibility:
            raise RuntimeError(
                "--use-tree-attention currently requires an FA3/FA4 target "
                "attention backend (including a hybrid backend whose selected "
                "target-verify child is FA3/FA4). Got "
                f"{target_backend_capability.backend_name}."
            )
        if not target_backend_capability.supports_full_tree:
            reason = target_backend_capability.unsupported_reason or "unknown reason"
            raise RuntimeError(
                "DDTREE full-tree target verify supports target attention backends "
                "flashinfer, fa3/fa4, triton, or hybrid backends whose "
                "target-verify child backend resolves to one of those. Got "
                f"{target_backend_capability.backend_name}: {reason}."
            )

        with (
            profiler.cpu("mask_compile"),
            profiler.gpu("mask_compile_gpu"),
        ):
            past_lens_cpu = batch.seq_lens_cpu.tolist()
            if target_backend_capability.build_attention_mask:
                mask_numel = sum(
                    verify_token_num * (past_len + verify_token_num)
                    for past_len in past_lens_cpu
                )
                current_capacity = (
                    0
                    if self._tree_attention_mask_buf is None
                    else self._tree_attention_mask_buf.numel()
                )
                if current_capacity < mask_numel:
                    new_capacity = max(mask_numel, max(1, current_capacity * 2))
                    self._tree_attention_mask_buf = torch.empty(
                        new_capacity, dtype=torch.bool, device=batch.device
                    )
            (
                verify_input_ids,
                verify_position_ids,
                tree_attention_mask,
                actual_tree_sizes,
            ) = compile_ddtree_tree(
                root_token_ids=draft_input.bonus_tokens,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility=visibility,
                start_positions=batch.seq_lens,
                past_lengths=batch.seq_lens,
                tree_budget=self.tree_budget,
                actual_tree_sizes=actual_tree_sizes,
                device=batch.device,
                past_lens_cpu=past_lens_cpu,
                actual_sizes_cpu=actual_tree_sizes_cpu,
                verify_token_num=verify_token_num,
                build_attention_mask=target_backend_capability.build_attention_mask,
                _out_verify_input_ids=self._verify_input_ids_buf,
                _out_verify_position_ids=self._verify_position_ids_buf,
                _out_attention_mask=self._tree_attention_mask_buf,
            )

        tree_is_spine = (
            bool(child_maps)
            and all(
                all(len(children) <= 1 for children in cm.values()) for cm in child_maps
            )
            and all(int(size) == verify_token_num for size in actual_tree_sizes_cpu)
        )

        verify_input = DDTreeVerifyInput(
            draft_token=verify_input_ids.reshape(-1),
            positions=verify_position_ids.reshape(-1),
            draft_token_num=verify_token_num,
            tree_budget=self.tree_budget,
            child_maps=child_maps,
            actual_tree_sizes=actual_tree_sizes,
            parents=parents[:, :verify_token_num],
            visibility=visibility if target_backend_capability.use_visibility else None,
            custom_mask=tree_attention_mask,
            tree_is_spine=tree_is_spine,
            use_tree_attention=self.use_tree_attention,
            raw_tree_size=raw_verify_token_num,
            cuda_graph_bucket_size=verify_token_num,
            force_cpu_follow=self.force_ddtree_cpu_follow,
            follow_accepted_indices=self._follow_accepted_indices_buf[
                :bs, :verify_token_num
            ],
            follow_accepted_token_ids=self._follow_accepted_token_ids_buf[
                :bs, :verify_token_num
            ],
            follow_accepted_lens=self._follow_accepted_lens_buf[:bs],
            follow_next_tokens=self._follow_next_tokens_buf[:bs],
            native_sampling_uniform=self._native_sampling_uniform_buf[
                :bs, :verify_token_num
            ],
            native_sampling_uniform_final=self._native_sampling_uniform_final_buf[:bs],
            native_reject_indices=self._native_reject_indices_buf[:bs],
            native_reject_child_tokens=self._native_reject_child_tokens_buf[
                :bs, :verify_token_num
            ],
            native_reject_child_counts=self._native_reject_child_counts_buf[:bs],
        )
        with profiler.cpu("prepare_verify"):
            verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self, batch: ScheduleBatch, **kwargs
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "DDTREE batch requested return_logprob, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(batch, **kwargs)
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DDTREE requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if batch.extend_lens is None or batch.prefix_lens is None:
                raise RuntimeError(
                    "DDTREE expected extend_lens / prefix_lens to be populated in extend mode, but got None."
                )

            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(batch.extend_lens)
            draft_input = DFlashDraftInput(
                bonus_tokens=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(batch.prefix_lens)
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DDTREE decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        assert batch.forward_mode.is_target_verify()
        verify_input = batch.spec_info

        # Copy CUDA graph state from target worker BEFORE forward
        self.target_worker.capture_mode = getattr(
            self.target_worker.model_runner, "capture_mode", False
        )

        with self.ddtree_profiler.gpu("target_verify"):
            batch_result = self.target_worker.forward_batch_generation(
                batch, is_verify=True, **kwargs
            )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        # Full tree path: use DDTree verify.
        assert isinstance(verify_input, DDTreeVerifyInput)
        (
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            model_runner=self.target_worker.model_runner,
            profiler=self.ddtree_profiler,
        )

        draft_input.bonus_tokens = new_bonus_tokens
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        with self.ddtree_profiler.gpu("draft_kv_append_after"):
            self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
        bs = len(num_correct_drafts_per_req_cpu)
        self.ddtree_profiler.record_round(
            mode="ddtree_full",
            batch_size=bs,
            block_size=int(self.block_size),
            tree_budget=int(self.tree_budget),
            draft_token_num=int(verify_input.draft_token_num),
            raw_tree_size=int(
                getattr(verify_input, "raw_tree_size", verify_input.draft_token_num)
            ),
            cuda_graph_bucket_size=int(
                getattr(
                    verify_input,
                    "cuda_graph_bucket_size",
                    verify_input.draft_token_num,
                )
            ),
            mean_num_correct_drafts=(float(num_correct_drafts) / bs if bs > 0 else 0.0),
            mean_accept_len=(float(num_correct_drafts + bs) / bs if bs > 0 else 0.0),
            round_output_tokens=int(num_correct_drafts + bs),
            can_run_cuda_graph=bool(can_run_cuda_graph),
            use_tree_attention=self.use_tree_attention,
        )
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DDTREE verify completed. num_correct_drafts_per_req=%s",
                num_correct_drafts_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_bonus_tokens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _compute_draft_topk_log_probs_and_token_ids(
        self,
        hidden_states: torch.Tensor,  # [total_tokens, hidden]
        lm_head,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TP-safe draft top-k log-probs and true global token ids.

        This follows DFLASH's vocab-parallel greedy sampling pattern: each rank
        first converts local candidate indices into global token ids, then TP
        ranks exchange only candidate scores/ids.  DDTree must never use the
        concatenated TP-shard column number as a token id, because that column
        number includes per-rank padding layout rather than tokenizer ids.
        """
        if topk <= 0:
            raise ValueError(f"DDTree topk must be positive, got {topk}.")

        if hidden_states.numel() == 0:
            return (
                torch.empty(
                    (0, topk),
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                torch.empty(
                    (0, topk),
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
            )

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype

        if hidden_states.dtype != weight_dtype:
            hidden_states = hidden_states.to(weight_dtype)

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        device = hidden_states.device

        vocab_id_cache_key = (
            str(device),
            num_org,
            org_vocab_start,
            num_added,
            added_vocab_start,
        )
        if self._ddtree_vocab_id_cache_key != vocab_id_cache_key:
            self._ddtree_org_token_ids = (
                torch.arange(num_org, dtype=torch.long, device=device) + org_vocab_start
                if num_org > 0
                else None
            )
            self._ddtree_added_token_ids = (
                torch.arange(num_added, dtype=torch.long, device=device)
                + added_vocab_start
                if num_added > 0
                else None
            )
            self._ddtree_vocab_id_cache_key = vocab_id_cache_key

        local_logits_parts = []
        local_token_id_parts = []

        if num_org > 0:
            base_logits = torch.mm(hidden_states, weight[:num_org].t()).float()
            base_ids = self._ddtree_org_token_ids
            local_logits_parts.append(base_logits)
            local_token_id_parts.append(base_ids)

        if num_added > 0:
            added_slice_start = num_org_padded
            added_slice_end = num_org_padded + num_added
            added_logits = torch.mm(
                hidden_states, weight[added_slice_start:added_slice_end].t()
            ).float()
            added_ids = self._ddtree_added_token_ids
            local_logits_parts.append(added_logits)
            local_token_id_parts.append(added_ids)

        if len(local_logits_parts) == 1:
            local_logits = local_logits_parts[0]
            local_token_ids = local_token_id_parts[0]
            local_max = local_logits.amax(dim=-1)
        elif local_logits_parts:
            local_logits = torch.cat(local_logits_parts, dim=-1)
            local_token_ids = torch.cat(local_token_id_parts, dim=0)
            local_max = local_logits.amax(dim=-1)
        else:
            local_logits = torch.empty(
                (num_tokens, 0), dtype=torch.float32, device=device
            )
            local_token_ids = torch.empty((0,), dtype=torch.long, device=device)
            local_max = torch.full(
                (num_tokens,), -float("inf"), dtype=torch.float32, device=device
            )

        if tp_size > 1:
            gathered_max = torch.empty(
                (tp_size, num_tokens), dtype=torch.float32, device=device
            )
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            global_max = gathered_max.amax(dim=0)
        else:
            global_max = local_max

        if local_logits.shape[-1] > 0:
            local_exp_sum = torch.exp(local_logits - global_max[:, None]).sum(dim=-1)
        else:
            local_exp_sum = torch.zeros(
                (num_tokens,), dtype=torch.float32, device=device
            )

        if tp_size > 1:
            gathered_exp_sum = torch.empty(
                (tp_size, num_tokens), dtype=torch.float32, device=device
            )
            tp_group.all_gather_into_tensor(
                gathered_exp_sum, local_exp_sum.contiguous()
            )
            global_exp_sum = gathered_exp_sum.sum(dim=0)
        else:
            global_exp_sum = local_exp_sum

        log_z = global_max + torch.log(global_exp_sum.clamp_min(1e-38))

        if local_logits.shape[-1] > 0:
            local_k = min(topk, int(local_logits.shape[-1]))
            vals, idx = torch.topk(local_logits, k=local_k, dim=-1)
            if local_k == topk:
                local_top_vals = vals
                local_top_ids = local_token_ids[idx]
            else:
                local_top_vals = torch.full(
                    (num_tokens, topk),
                    -float("inf"),
                    dtype=torch.float32,
                    device=device,
                )
                local_top_ids = torch.zeros(
                    (num_tokens, topk), dtype=torch.long, device=device
                )
                local_top_vals[:, :local_k] = vals
                local_top_ids[:, :local_k] = local_token_ids[idx]
        else:
            local_top_vals = torch.full(
                (num_tokens, topk), -float("inf"), dtype=torch.float32, device=device
            )
            local_top_ids = torch.zeros(
                (num_tokens, topk), dtype=torch.long, device=device
            )

        if tp_size == 1:
            return local_top_vals - log_z[:, None], local_top_ids

        gathered_top_vals = torch.empty(
            (tp_size, num_tokens, topk), dtype=torch.float32, device=device
        )
        gathered_top_ids = torch.empty(
            (tp_size, num_tokens, topk), dtype=torch.long, device=device
        )
        tp_group.all_gather_into_tensor(
            gathered_top_vals, local_top_vals.contiguous()
        )
        tp_group.all_gather_into_tensor(
            gathered_top_ids, local_top_ids.contiguous()
        )

        flat_top_vals = gathered_top_vals.permute(1, 0, 2).reshape(num_tokens, -1)
        flat_top_ids = gathered_top_ids.permute(1, 0, 2).reshape(num_tokens, -1)
        global_top_vals, global_top_idx = torch.topk(flat_top_vals, k=topk, dim=-1)
        global_top_ids = flat_top_ids.gather(1, global_top_idx)
        global_top_log_probs = global_top_vals - log_z[:, None]

        return global_top_log_probs, global_top_ids
