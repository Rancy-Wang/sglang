import logging
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.sampler import (
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.ddtree_info import DDTreeVerifyInput
from sglang.srt.speculative.ddtree_profiler import DDTreeProfiler
from sglang.srt.speculative.ddtree_utils import (
    build_ddtree_tree,
    build_ddtree_tree_gpu,
    compile_ddtree_tree,
    follow_verified_tree,
    follow_verified_tree_gpu,
    resolve_ddtree_cuda_graph_buckets,
    resolve_ddtree_target_backend_capability,
    sample_ddtree_target_probs_gpu,
    select_ddtree_cuda_graph_bucket,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool_func,
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
)

logger = logging.getLogger(__name__)


def _can_use_native_tree_sampling(sampling_info) -> bool:
    return bool(
        sampling_info is not None
        and not sampling_info.is_all_greedy
        and getattr(sampling_info, "sampling_seed", None) is None
        and not bool(getattr(sampling_info, "need_min_p_sampling", False))
        and not bool(getattr(sampling_info, "need_top_p_sampling", False))
        and not bool(getattr(sampling_info, "need_top_k_sampling", False))
    )


def _sample_target_tokens(
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
    repeated_seed = (
        None
        if sampling_info.sampling_seed is None
        else torch.repeat_interleave(sampling_info.sampling_seed, q_len, dim=0)
    )
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
    tp_group = get_tp_group()
    if tp_group.world_size > 1:
        tp_group.broadcast(target_predict, src=0)
    return target_predict


class DDTreeWorkerV2(DFlashWorkerV2):
    """DDTree worker on top of the spec-v2 DFlash draft runtime."""

    def __init__(
        self,
        server_args,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=ps,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        self.tree_budget = int(
            getattr(server_args, "speculative_ddtree_budget", None)
            or (self.block_size - 1)
        )
        self.is_ddtree_prune = bool(
            getattr(server_args, "is_ddtree_prune", False)
        )
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

        graph_bs = list(getattr(server_args.cuda_graph_config.decode, "bs", []) or [])
        max_bs = max(
            max(graph_bs) if graph_bs else 0,
            int(getattr(server_args, "max_running_requests", None) or 64),
        )
        budget = self.tree_budget
        max_nodes = self.max_tree_nodes
        device = self.device

        self._tree_node_token_ids_buf = torch.zeros(
            max_bs, budget, dtype=torch.long, device=device
        )
        self._tree_node_depths_buf = torch.zeros(
            max_bs, budget, dtype=torch.long, device=device
        )
        self._tree_parents_buf = torch.full(
            (max_bs, max_nodes), -1, dtype=torch.long, device=device
        )
        self._tree_visibility_buf = torch.zeros(
            max_bs, max_nodes, max_nodes, dtype=torch.bool, device=device
        )
        self._tree_actual_sizes_buf = torch.zeros(
            max_bs, dtype=torch.long, device=device
        )
        self._verify_input_ids_buf = torch.empty(
            max_bs * max_nodes, dtype=torch.long, device=device
        )
        self._verify_position_ids_buf = torch.empty(
            max_bs * max_nodes, dtype=torch.long, device=device
        )
        self._tree_attention_mask_buf: Optional[torch.Tensor] = None

        self._tree_pruned_node_token_ids_buf: Optional[torch.Tensor] = None
        self._tree_pruned_node_depths_buf: Optional[torch.Tensor] = None
        self._tree_pruned_parents_buf: Optional[torch.Tensor] = None
        self._tree_pruned_visibility_buf: Optional[torch.Tensor] = None
        if self.is_ddtree_prune and not self.force_ddtree_cpu_build:
            self._tree_pruned_node_token_ids_buf = torch.zeros(
                max_bs, budget, dtype=torch.long, device=device
            )
            self._tree_pruned_node_depths_buf = torch.zeros(
                max_bs, budget, dtype=torch.long, device=device
            )
            self._tree_pruned_parents_buf = torch.full(
                (max_bs, max_nodes), -1, dtype=torch.long, device=device
            )
            self._tree_pruned_visibility_buf = torch.zeros(
                max_bs, max_nodes, max_nodes, dtype=torch.bool, device=device
            )

        heap_cap = 1 << (max(1, 2 * budget + 3) - 1).bit_length()
        self._tree_heap_scores_buf = torch.empty(
            max_bs, heap_cap, dtype=torch.float64, device=device
        )
        self._tree_heap_parents_buf = torch.empty(
            max_bs, heap_cap, dtype=torch.int32, device=device
        )
        self._tree_heap_depths_buf = torch.empty(
            max_bs, heap_cap, dtype=torch.int32, device=device
        )
        self._tree_heap_ranks_buf = torch.empty(
            max_bs, heap_cap, dtype=torch.int32, device=device
        )

        self._follow_accepted_indices_buf = torch.empty(
            max_bs, max_nodes, dtype=torch.long, device=device
        )
        self._follow_accepted_token_ids_buf = torch.empty(
            max_bs, max_nodes, dtype=torch.long, device=device
        )
        self._follow_accepted_lens_buf = torch.empty(
            max_bs, dtype=torch.long, device=device
        )
        self._follow_next_tokens_buf = torch.empty(
            max_bs, dtype=torch.long, device=device
        )
        self._native_sampling_uniform_buf = torch.empty(
            max_bs, max_nodes, dtype=torch.float32, device=device
        )
        self._native_sampling_uniform_final_buf = torch.empty(
            max_bs, dtype=torch.float32, device=device
        )
        self._native_reject_indices_buf = torch.empty(
            max_bs, dtype=torch.long, device=device
        )
        self._native_reject_child_tokens_buf = torch.empty(
            max_bs, max_nodes, dtype=torch.long, device=device
        )
        self._native_reject_child_counts_buf = torch.empty(
            max_bs, dtype=torch.long, device=device
        )
        self._out_tokens_buf = torch.zeros(
            max_bs, self.block_size, dtype=torch.long, device=device
        )

        self._ddtree_vocab_id_cache_key = None
        self._ddtree_org_token_ids: Optional[torch.Tensor] = None
        self._ddtree_added_token_ids: Optional[torch.Tensor] = None
        self.ddtree_profiler = DDTreeProfiler.from_server_args(
            server_args,
            name="ddtree_v2",
            rank=self.ps.tp_rank,
        )

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized DDTREE v2 worker. block_size=%s, tree_budget=%s, "
                "max_tree_nodes=%s, prune=%s, tree_attention=%s, cpu_build=%s, "
                "cpu_follow=%s, graph_buckets=%s",
                self.block_size,
                self.tree_budget,
                self.max_tree_nodes,
                self.is_ddtree_prune,
                self.use_tree_attention,
                self.force_ddtree_cpu_build,
                self.force_ddtree_cpu_follow,
                self.ddtree_cuda_graph_buckets,
            )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Keep the target allocator handles needed by tree compaction.

        DFlash v2 only forwards these handles to its draft worker because its
        linear verify path never moves sparse accepted nodes. DDTree does move
        accepted tree nodes, so it must retain the shared target handles too.
        """
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def _maybe_build_draft_sampler(self):
        # DDTree needs top-k scores and log-normalizers, not only greedy token ids.
        return None

    def _run_draft_backbone(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = len(batch.seq_lens)
        block_size = int(self.block_size)
        device = self.device
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DDTREE Triton prepare_block failed; using eager path: %s", e
                )

        if not self._use_triton_prepare_block:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=prefix_lens + block_size,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]

        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )
            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DDTREE,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        hidden = draft_out.logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DDTREE draft model returned no hidden states.")
        return hidden.view(bs, block_size, -1), positions_2d, block_ids

    def _build_verify_input(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        draft_hidden: torch.Tensor,
        lm_head,
    ) -> DDTreeVerifyInput:
        bs = len(batch.seq_lens)
        profiler = self.ddtree_profiler
        vocab_size = getattr(lm_head, "num_embeddings", None)
        if vocab_size is None:
            vocab_size = getattr(
                self.target_worker.model_runner.model_config, "vocab_size", None
            )
        if vocab_size is None or int(vocab_size) <= 0:
            raise RuntimeError("DDTREE requires a positive target vocabulary size.")
        proposal_topk = min(self.tree_budget, int(vocab_size))

        with profiler.gpu("draft_head_topk"):
            draft_top_log_probs, draft_top_token_ids = (
                self._compute_draft_topk_log_probs_and_token_ids(
                    hidden_states=draft_hidden[:, 1:, :].reshape(
                        -1, draft_hidden.shape[-1]
                    ),
                    lm_head=lm_head,
                    topk=proposal_topk,
                )
            )
        draft_top_log_probs = draft_top_log_probs.view(
            bs, self.block_size - 1, proposal_topk
        )
        draft_top_token_ids = draft_top_token_ids.view(
            bs, self.block_size - 1, proposal_topk
        )

        if not self.force_ddtree_cpu_build and draft_top_log_probs.is_cuda:
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

        raw_verify_token_num = max(1, max(actual_tree_sizes_cpu))
        verify_token_num = (
            select_ddtree_cuda_graph_bucket(
                raw_verify_token_num, self.ddtree_cuda_graph_buckets
            )
            if self.use_ddtree_cuda_graph_buckets
            else raw_verify_token_num
        )
        capability = resolve_ddtree_target_backend_capability(
            self.target_worker.model_runner.attn_backend,
            speculative_attention_mode=getattr(
                self.server_args, "speculative_attention_mode", "prefill"
            ),
        )
        if self.use_tree_attention and not capability.use_visibility:
            raise RuntimeError(
                "--use-tree-attention requires an FA3/FA4 target backend; got "
                f"{capability.backend_name}."
            )
        if not capability.supports_full_tree:
            raise RuntimeError(
                "DDTREE full-tree verify does not support target backend "
                f"{capability.backend_name}: {capability.unsupported_reason}."
            )

        if batch.seq_lens_cpu is None:
            past_lens_cpu = batch.seq_lens.to("cpu", dtype=torch.int32).tolist()
        else:
            past_lens_cpu = batch.seq_lens_cpu.tolist()
        if capability.build_attention_mask:
            mask_numel = sum(
                verify_token_num * (int(past_len) + verify_token_num)
                for past_len in past_lens_cpu
            )
            current_capacity = (
                0
                if self._tree_attention_mask_buf is None
                else self._tree_attention_mask_buf.numel()
            )
            if current_capacity < mask_numel:
                self._tree_attention_mask_buf = torch.empty(
                    max(mask_numel, max(1, current_capacity * 2)),
                    dtype=torch.bool,
                    device=batch.device,
                )

        with profiler.gpu("mask_compile"):
            (
                verify_input_ids,
                verify_position_ids,
                custom_mask,
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
                build_attention_mask=capability.build_attention_mask,
                _out_verify_input_ids=self._verify_input_ids_buf,
                _out_verify_position_ids=self._verify_position_ids_buf,
                _out_attention_mask=self._tree_attention_mask_buf,
            )

        tree_is_spine = bool(child_maps) and all(
            all(len(children) <= 1 for children in child_map.values())
            for child_map in child_maps
        )
        return DDTreeVerifyInput(
            draft_token=verify_input_ids.reshape(-1),
            positions=verify_position_ids.reshape(-1),
            draft_token_num=verify_token_num,
            tree_budget=self.tree_budget,
            actual_tree_sizes=actual_tree_sizes,
            parents=parents[:, :verify_token_num],
            visibility=visibility[:, :verify_token_num, :verify_token_num]
            if capability.use_visibility
            else None,
            custom_mask=custom_mask,
            tree_is_spine=tree_is_spine,
            use_tree_attention=self.use_tree_attention,
            raw_tree_size=raw_verify_token_num,
            cuda_graph_bucket_size=verify_token_num,
        )

    def _accept_tree(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: DDTreeVerifyInput,
        logits_output,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = len(batch.seq_lens)
        q_len = int(verify_input.draft_token_num)
        draft_tokens = verify_input.draft_token.view(bs, q_len)
        parents = verify_input.parents.view(bs, q_len)
        sampling_info = batch.sampling_info

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits_output.next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=q_len,
        )

        if (
            not self.force_ddtree_cpu_follow
            and _can_use_native_tree_sampling(sampling_info)
        ):
            target_probs = F.softmax(
                logits_output.next_token_logits.view(bs, q_len, -1)
                / sampling_info.temperatures.view(bs, 1, 1),
                dim=-1,
            )
            result = sample_ddtree_target_probs_gpu(
                target_probs=target_probs,
                draft_tokens=draft_tokens,
                parents=parents,
                actual_tree_sizes=verify_input.actual_tree_sizes,
                uniform_samples=self._native_sampling_uniform_buf[:bs, :q_len],
                uniform_final=self._native_sampling_uniform_final_buf[:bs],
                accepted_indices=self._follow_accepted_indices_buf[:bs, :q_len],
                accepted_token_ids=self._follow_accepted_token_ids_buf[:bs, :q_len],
                accepted_lens=self._follow_accepted_lens_buf[:bs],
                next_tokens=self._follow_next_tokens_buf[:bs],
                reject_indices=self._native_reject_indices_buf[:bs],
                reject_child_tokens=self._native_reject_child_tokens_buf[
                    :bs, :q_len
                ],
                reject_child_counts=self._native_reject_child_counts_buf[:bs],
            )
            tp_group = get_tp_group()
            if tp_group.world_size > 1:
                for tensor in result:
                    tp_group.broadcast(tensor, src=0)
            return result

        if sampling_info is not None and not sampling_info.is_all_greedy:
            target_predict = _sample_target_tokens(
                logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                positions=verify_input.positions,
                bs=bs,
                q_len=q_len,
            )
        else:
            target_predict = torch.argmax(
                logits_output.next_token_logits, dim=-1
            ).view(bs, q_len)

        if not self.force_ddtree_cpu_follow:
            return follow_verified_tree_gpu(
                draft_tokens=draft_tokens,
                target_predict=target_predict,
                parents=parents,
                actual_tree_sizes=verify_input.actual_tree_sizes,
                accepted_indices=self._follow_accepted_indices_buf[:bs, :q_len],
                accepted_token_ids=self._follow_accepted_token_ids_buf[:bs, :q_len],
                accepted_lens=self._follow_accepted_lens_buf[:bs],
                next_tokens=self._follow_next_tokens_buf[:bs],
            )

        draft_tokens_cpu = draft_tokens.cpu().tolist()
        parents_cpu = parents.cpu().tolist()
        sizes_cpu = verify_input.actual_tree_sizes.cpu().tolist()
        from sglang.srt.speculative.ddtree_utils import (
            build_child_maps_from_parent_metadata,
        )

        child_maps = build_child_maps_from_parent_metadata(
            draft_tokens_cpu, parents_cpu, sizes_cpu
        )
        paths, next_tokens, _ = follow_verified_tree(child_maps, target_predict.cpu())
        accepted_indices = self._follow_accepted_indices_buf[:bs, :q_len]
        accepted_token_ids = self._follow_accepted_token_ids_buf[:bs, :q_len]
        accepted_lens = self._follow_accepted_lens_buf[:bs]
        accepted_indices.fill_(-1)
        accepted_token_ids.zero_()
        for i, path in enumerate(paths):
            n = len(path)
            path_t = torch.tensor(path, dtype=torch.long, device=self.device)
            accepted_indices[i, :n] = path_t
            accepted_token_ids[i, :n] = draft_tokens[i].index_select(0, path_t)
            accepted_lens[i] = n
        self._follow_next_tokens_buf[:bs].copy_(next_tokens.to(self.device))
        return (
            accepted_indices,
            accepted_token_ids,
            accepted_lens,
            self._follow_next_tokens_buf[:bs],
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DDTREE speculative decoding does not support return_logprob yet."
            )
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch, on_publish=on_publish)

        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DDTREE spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )
        if batch.forward_mode.is_idle():
            return super().forward_batch_generation(batch, on_publish=on_publish)
        if batch.has_grammar:
            raise RuntimeError(
                "DDTREE does not support grammar-constrained requests."
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DDTREE requires a vocab-parallel target lm_head with weight "
                "and shard_indices."
            )

        with self.ddtree_profiler.gpu("draft_forward"):
            draft_hidden, _, _ = self._run_draft_backbone(batch, draft_input)
        verify_input = self._build_verify_input(
            batch=batch,
            draft_input=draft_input,
            draft_hidden=draft_hidden,
            lm_head=lm_head,
        )
        q_len = int(verify_input.draft_token_num)
        verify_end = prefix_lens + q_len
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=verify_end,
            batch_size=bs,
            draft_token_num=q_len,
            device=self.device,
        )
        verify_out_cache_loc_2d = verify_out_cache_loc.view(bs, q_len)
        batch.out_cache_loc = verify_out_cache_loc

        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_seq_lens = seq_lens_cpu_backup + q_len
            batch.seq_lens_cpu = verify_host_seq_lens
            # FlashInfer's host fast-plan needs total KV lengths (prefix + q),
            # while generate_attn_arg_prefill receives seq_lens_sum as the
            # committed-prefix sum and adds q exactly once.
            batch.seq_lens_sum = (
                int(seq_lens_sum_backup)
                if seq_lens_sum_backup is not None
                else int(seq_lens_cpu_backup.sum())
            )
        elif draft_input.reserved_seq_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            batch.seq_lens_sum = (
                int(seq_lens_sum_backup)
                if seq_lens_sum_backup is not None
                else int(prefix_lens.sum().item())
            )

        prepare_mamba_track_for_verify(batch)
        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup

        with self.ddtree_profiler.gpu("target_verify"):
            target_out = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        logits_output = target_out.logits_output
        with self.ddtree_profiler.gpu("accept_follow"):
            (
                accepted_indices,
                accepted_token_ids,
                commit_lens,
                bonus,
            ) = self._accept_tree(
                batch=batch,
                verify_input=verify_input,
                logits_output=logits_output,
            )
        commit_lens = commit_lens.to(torch.int32)

        path_width = int(self.block_size)
        local_accept = accepted_indices[:, :path_width]
        row_offsets = (
            torch.arange(bs, dtype=torch.long, device=self.device) * q_len
        ).unsqueeze(1)
        global_accept = torch.where(
            local_accept >= 0, local_accept + row_offsets, local_accept
        )

        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            commit_lens,
            global_accept,
            q_len,
        )
        move_accept_tokens_to_target_kvcache(
            batch,
            global_accept,
            commit_lens - 1,
            self.token_to_kv_pool_allocator,
        )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DDTREE verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, q_len, -1)
        safe_local_accept = local_accept.clamp(min=0)
        gathered_hidden = hidden.gather(
            1,
            safe_local_accept.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        )
        verify_positions = verify_input.positions.view(bs, q_len)
        gathered_positions = verify_positions.gather(1, safe_local_accept)
        committed_cache_locs = verify_out_cache_loc_2d[:, :path_width]
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=gathered_hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=committed_cache_locs.reshape(-1),
            cache_loc_2d=committed_cache_locs,
            positions=gathered_positions.reshape(-1),
            commit_lens=commit_lens,
        )

        out_tokens = self._out_tokens_buf[:bs]
        out_tokens.zero_()
        if path_width > 1:
            out_tokens[:, : path_width - 1].copy_(
                accepted_token_ids[:, 1:path_width]
            )
        out_tokens.scatter_(
            1, (commit_lens.to(torch.long) - 1).unsqueeze(1), bonus.unsqueeze(1)
        )

        logits_output.hidden_states = None
        next_draft_input = make_draft_input_v2(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        if self.ddtree_profiler.enabled:
            self.ddtree_profiler.record_round(
                batch_size=bs,
                block_size=int(self.block_size),
                tree_budget=int(self.tree_budget),
                verify_width=q_len,
                mean_accept_len=float(commit_lens.float().mean().item()),
                can_run_cuda_graph=bool(target_out.can_run_cuda_graph),
            )
        else:
            self.ddtree_profiler.record_round()
        if not self._logged_first_verify and self.ps.tp_rank == 0:
            logger.info(
                "DDTREE v2 verify completed. first commit_lens=%s",
                commit_lens.detach().cpu().tolist(),
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            next_draft_input=next_draft_input,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            speculative_num_draft_tokens=int(self.block_size),
            new_seq_lens=new_seq_lens,
        )

    def _compute_draft_topk_log_probs_and_token_ids(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TP-safe top-k log-probs and true global vocabulary ids."""
        if topk <= 0:
            raise ValueError(f"DDTree topk must be positive, got {topk}.")
        if hidden_states.numel() == 0:
            return (
                torch.empty(
                    (0, topk), dtype=torch.float32, device=hidden_states.device
                ),
                torch.empty(
                    (0, topk), dtype=torch.long, device=hidden_states.device
                ),
            )

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        shard = lm_head.shard_indices
        weight = lm_head.weight
        if hidden_states.dtype != weight.dtype:
            hidden_states = hidden_states.to(weight.dtype)

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
                torch.arange(num_org, dtype=torch.long, device=device)
                + org_vocab_start
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
            local_logits_parts.append(
                torch.mm(hidden_states, weight[:num_org].t()).float()
            )
            local_token_id_parts.append(self._ddtree_org_token_ids)
        if num_added > 0:
            local_logits_parts.append(
                torch.mm(
                    hidden_states,
                    weight[num_org_padded : num_org_padded + num_added].t(),
                ).float()
            )
            local_token_id_parts.append(self._ddtree_added_token_ids)

        if local_logits_parts:
            local_logits = (
                local_logits_parts[0]
                if len(local_logits_parts) == 1
                else torch.cat(local_logits_parts, dim=-1)
            )
            local_token_ids = (
                local_token_id_parts[0]
                if len(local_token_id_parts) == 1
                else torch.cat(local_token_id_parts, dim=0)
            )
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

        local_exp_sum = (
            torch.exp(local_logits - global_max[:, None]).sum(dim=-1)
            if local_logits.shape[-1] > 0
            else torch.zeros((num_tokens,), dtype=torch.float32, device=device)
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
                (num_tokens, topk),
                -float("inf"),
                dtype=torch.float32,
                device=device,
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
        global_top_vals, global_top_idx = torch.topk(
            flat_top_vals, k=topk, dim=-1
        )
        global_top_ids = flat_top_ids.gather(1, global_top_idx)
        return global_top_vals - log_z[:, None], global_top_ids
