from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class DDTreeVerifyInput(SpecInput):
    """Target-verify metadata for a uniform-width DDTree batch.

    ``custom_mask`` uses SGLang's canonical packed allow-mask convention:
    requests are concatenated, each request is row-major
    ``[q_len, committed_prefix_len + q_len]``, and True means visible.
    FlashAttention consumes ``visibility`` through its cascade suffix path;
    FlashInfer and Triton consume ``custom_mask``.
    """

    draft_token: Optional[torch.Tensor]
    positions: Optional[torch.Tensor]
    draft_token_num: int
    tree_budget: int
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    actual_tree_sizes: Optional[torch.Tensor] = None
    parents: Optional[torch.Tensor] = None
    visibility: Optional[torch.Tensor] = None
    custom_mask: Optional[torch.Tensor] = None
    tree_is_spine: bool = False
    use_tree_attention: bool = False
    raw_tree_size: Optional[int] = None
    cuda_graph_bucket_size: Optional[int] = None

    # Compatibility attribute read by tree-aware attention backends. DDTree's
    # proposal top-k is unrelated to EAGLE's layout top-k, so backend dispatch
    # must key on SpecInputType.DDTREE_VERIFY instead of this value.
    topk: int = 1
    num_tokens_per_req: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker,
    ) -> tuple[ForwardBatch, bool]:
        """Package a verify batch over scheduler-preallocated KV locations."""

        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            target_worker.model_runner,
            capture_hidden_mode=self.capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )

        graph_runner = target_worker.model_runner.decode_cuda_graph_runner
        can_run_cuda_graph = bool(
            graph_runner and graph_runner.can_run_graph(forward_batch)
        )
        if can_run_cuda_graph:
            graph_runner.load_batch(forward_batch)
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                forward_batch
            )
        return forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
    ):
        """Build FlashInfer paged indices without changing mask representation."""

        device = req_pool_indices.device
        bs = len(req_pool_indices)
        q_len = int(self.draft_token_num)
        qo_indptr = torch.arange(
            0,
            (bs + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=device,
        )

        kv_lens = paged_kernel_lens + q_len
        kv_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)
        kv_indices = torch.empty(
            paged_kernel_lens_sum + q_len * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            kv_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )

        mask = self.custom_mask
        if mask is not None:
            expected = paged_kernel_lens_sum * q_len + q_len * q_len * bs
            mask = mask.contiguous().view(-1).to(dtype=torch.bool)
            if mask.numel() < expected:
                mask = torch.cat(
                    (
                        mask,
                        torch.ones(
                            expected - mask.numel(),
                            dtype=torch.bool,
                            device=device,
                        ),
                    )
                )
            mask = mask[:expected]
        return kv_indices, kv_indptr, qo_indptr, mask
