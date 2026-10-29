"""Native one-loop execution shared by ordinary and speculative scheduling.

The scheduler owns admission, logical KV slots and state lifetimes. This runner
does not sample, commit tokens or run a private until-output loop. Its input is
the SGLang ForwardBatch plus carried hidden state and per-sequence depth, the
equivalent of vllm-rlt's StateSpan/ExecutionBatch execution metadata.
"""

import copy
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.timeline_pool import MixedTimelineRuntime
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.recurrent_trace import span


def use_recurrent_execution(model, server_args):
    """Select the validated native CDB storage/execution capabilities."""
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.environ import envs

    server_args = resolved_view(server_args)
    return (
        getattr(model, "supports_recurrent_execution", False)
        and server_args.speculative_algorithm is None
        and (server_args.tp_size, server_args.pp_size, server_args.dp_size,
             server_args.attn_cp_size) == (1, 1, 1, 1)
        and server_args.attention_backend == "triton"
        and server_args.device.startswith("cuda")
        and all(backend in (None, "triton") for backend in (
            server_args.prefill_attention_backend, server_args.decode_attention_backend))
        and not envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.get()
        and server_args.page_size == 1
        and server_args.disable_radix_cache
        and not any(getattr(server_args, name, False) for name in (
            "enable_hierarchical_cache", "enable_unified_memory", "enable_hisparse",
            "enable_unified_cache_external_linker", "enable_kv_cache_sharding",
            "enable_lora", "enable_two_batch_overlap", "enable_mixed_chunk",
            "enable_priority_preemption", "enable_torch_compile", "enable_memory_saver",
        ))
        and not server_args.cpu_offload_gb
        and server_args.disaggregation_mode == "null"
        and server_args.kv_cache_dtype in ("auto", "bf16", "bfloat16", "float16")
    )


@dataclass
class RecurrentState:
    hidden: torch.Tensor
    injected: torch.Tensor | None = None


class RecurrentRunner:
    def __init__(self, runner, *, max_rows, request_pool=None):
        self.runner, self.model = runner, runner.model
        self.timeline = MixedTimelineRuntime(
            runner, request_pool or runner.req_to_token_pool, max_rows
        )
        self.graphs = {}
        self.prefill_graphs = {}
        self.finish_graphs = {}
        self.graph_pool = None

    def init_graphs(self):
        from sglang.srt.model_executor.cuda_graph_config import check_cuda_graph_backend
        from sglang.srt.runtime_context import get_exec

        self.graph_pool = torch.cuda.graph_pool_handle()
        config = get_exec().graph.cuda_graph_config
        if not check_cuda_graph_backend("prefill", "disabled"):
            for tokens in sorted(config.prefill.bs):
                self.prefill_graphs[tokens] = RecurrentCoreGraph(
                    self, self.timeline.max_rows, tokens=tokens
                )
        if not check_cuda_graph_backend("decode", "disabled"):
            for rows in sorted(config.decode.bs):
                if rows <= self.timeline.max_rows:
                    self.graphs[rows] = RecurrentCoreGraph(self, rows)
            # Exact row counts retain the native readout GEMM shape. A decode
            # cohort finishing this loop can be smaller than the core batch.
            for rows in self.graphs:
                self.finish_graphs[rows] = RecurrentCoreGraph(self, rows, role="finish")

    @span("recurrent.execute_core")
    def execute_core(self, batch, state, depths, *, token_rows=None):
        if batch.forward_mode.is_decode():
            for rows, graph in self.graphs.items():
                if rows >= batch.batch_size:
                    return graph.replay(batch, state, depths), True
        elif batch.forward_mode.is_extend():
            for tokens, graph in self.prefill_graphs.items():
                if tokens >= len(state.hidden):
                    return graph.replay(batch, state, depths), True
        return self.advance(batch, state, depths, token_rows=token_rows), False

    @span("recurrent.prepare")
    def prepare(self, batch, *, token_rows=None):
        depths = torch.zeros_like(batch.req_pool_indices)
        mapped, backend = self.timeline.target_batch(
            batch, "pre", depths, token_rows
        )
        with forward_context(ForwardContext(backend)):
            hidden, injected = self.model.prepare_recurrent(
                batch.input_ids, batch.positions, mapped
            )
        return RecurrentState(hidden, injected)

    def advance(self, batch, state, depths, *, token_rows=None):
        mapped, backend = self.timeline.target_batch(
            batch, "core", depths, token_rows
        )
        # Timeline storage folds depth into token slots. Layer zero's core
        # plane therefore works for every row, even when depths differ.
        groups = [(0, 0, len(state.hidden), mapped, backend)]
        with forward_context(ForwardContext(backend)):
            hidden = self.model.advance_recurrent(
                state.hidden, batch.positions, groups, state.injected
            )
        return RecurrentState(hidden, state.injected)

    @span("recurrent.finish")
    def finish(self, batch, state, depths, *, token_rows=None):
        """Full target readout; keep native logprob/hidden-state semantics."""
        graph = self.finish_graphs.get(batch.batch_size)
        if (graph is not None and batch.forward_mode.is_decode()
                and batch.capture_hidden_mode == CaptureHiddenMode.NULL):
            return graph.replay(batch, state, depths)
        return self._finish(batch, state, depths, token_rows=token_rows)

    def _finish(self, batch, state, depths, *, token_rows=None):
        mapped, backend = self.timeline.target_batch(
            batch, "post", depths, token_rows
        )
        groups = [(0, 0, len(state.hidden), mapped, backend)]
        with forward_context(ForwardContext(backend)):
            hidden = self.model.finish_recurrent(
                state.hidden, batch.positions, groups
            )
            return self.model.logits_processor(
                batch.input_ids, hidden, self.model.lm_head, batch
            )

    def readout(self, batch, state, depths, *, token_rows=None):
        """Draft endpoints use the same core and depth-separated post KV."""
        mapped, backend = self.timeline.target_batch(
            batch, "post", depths, token_rows
        )
        groups = [(0, 0, len(state.hidden), mapped, backend)]
        with forward_context(ForwardContext(backend)):
            return self.model.readout_recurrent(
                state.hidden, batch.positions, groups
            )


class RecurrentCoreGraph:
    """One core graph per decode-row or packed-extend token bucket.

    Replay outputs are borrowed until the next replay. Owners must copy live
    carried states into their own storage before reusing this graph. Padding
    uses logical slot zero, which every timeline maps to the discard slot.
    """

    def __init__(self, recurrent, rows, *, tokens=None, role="core"):
        self.recurrent, self.rows = recurrent, rows
        self.role = role
        self.is_extend = tokens is not None
        self.tokens = tokens if self.is_extend else rows
        runner, model = recurrent.runner, recurrent.model
        device = runner.device
        self.depths = torch.zeros(rows, dtype=torch.int64, device=device)
        self.state = RecurrentState(
            torch.zeros(
                (self.tokens, model.config.hidden_size),
                dtype=model.get_input_embeddings().weight.dtype,
                device=device,
            )
        )
        if model.has_injection:
            self.state.injected = torch.zeros_like(self.state.hidden)
        self.batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND if self.is_extend else ForwardMode.DECODE,
            batch_size=rows,
            input_ids=torch.zeros(self.tokens, dtype=torch.int64, device=device),
            req_pool_indices=torch.zeros(rows, dtype=torch.int64, device=device),
            seq_lens=torch.ones(rows, dtype=torch.int32, device=device),
            seq_lens_sum=rows * recurrent.timeline.source.shape[1],
            out_cache_loc=torch.zeros(self.tokens, dtype=torch.int64, device=device),
            positions=torch.zeros(self.tokens, dtype=torch.int64, device=device),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        self.token_positions = None
        if self.is_extend:
            self.token_positions = torch.arange(self.tokens, device=device)
            self.batch.extend_seq_lens = torch.zeros(rows, dtype=torch.int32, device=device)
            self.batch.extend_seq_lens[0] = self.tokens
            self.batch.extend_prefix_lens = torch.zeros_like(self.batch.extend_seq_lens)
            self.batch.extend_start_loc = torch.zeros_like(self.batch.extend_seq_lens)
            self.batch.extend_num_tokens = self.tokens
            self.batch.extend_seq_lens_cpu = [self.tokens] + [0] * (rows - 1)
            # Prefix lengths remain dynamic; use the backend's allocation bound.
            self.batch.extend_prefix_lens_cpu = None
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), model_capture_mode():
            for _ in range(3):
                self.run()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(self.graph, stream=stream, pool=recurrent.graph_pool):
            self.run()
        torch.cuda.current_stream(device).wait_stream(stream)

    def run(self):
        token_rows = None
        if self.is_extend:
            token_rows = torch.bucketize(
                self.token_positions, self.batch.extend_seq_lens.cumsum(0), right=True
            ).clamp_max(self.rows - 1)
        execute = self.recurrent.advance if self.role == "core" else self.recurrent._finish
        self.output = execute(
            self.batch, self.state, self.depths, token_rows=token_rows
        )

    def replay(self, batch, state, depths):
        count = batch.batch_size
        tokens = len(state.hidden)
        if (batch.forward_mode.is_extend() != self.is_extend
                or not 0 < count <= self.rows or tokens > self.tokens):
            raise ValueError("Incompatible recurrent core graph batch")
        for name in ("input_ids", "req_pool_indices", "seq_lens", "out_cache_loc", "positions"):
            target, source = getattr(self.batch, name), getattr(batch, name)
            size = count if name in ("req_pool_indices", "seq_lens") else tokens
            target[:size].copy_(source)
            target[size:].fill_(1 if name == "seq_lens" else 0)
        if self.is_extend:
            for name in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
                target = getattr(self.batch, name)
                target[:count].copy_(getattr(batch, name))
                target[count:].zero_()
        self.depths[:count].copy_(depths)
        self.depths[count:].zero_()
        self.state.hidden[:tokens].copy_(state.hidden)
        self.state.hidden[tokens:].zero_()
        if self.state.injected is not None:
            if state.injected is None:
                raise ValueError("Recurrent model requires carried input injection")
            self.state.injected[:tokens].copy_(state.injected)
            self.state.injected[tokens:].zero_()
        self.graph.replay()
        if self.role == "finish":
            output = copy.copy(self.output)
            # Sampling can outlive the next core replay in the shared pool.
            output.next_token_logits = self.output.next_token_logits[:tokens].clone()
            return output
        return RecurrentState(self.output.hidden[:tokens], state.injected)
