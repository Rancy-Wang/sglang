"""LoopSpec V2 policy on the ordinary recurrent CDB scheduler and executor.

Request contexts own branches; temporary GPU row numbers never own requests.
Each call advances exactly ONE loop and may publish zero or several tokens.
"""

import logging
import time

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.model_executor.recurrent_runner import RecurrentRunner, RecurrentState
from sglang.srt.runtime_context import get_exec, get_observability, get_serving, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.recurrent_trace import span

from .config import pipeline_layout
from .info import LoopSpecInput
from .pipeline import Pipeline
from .sampling import decide_endpoints
from .utils import RequestContext, split_batch

logger = logging.getLogger(__name__)


def _step_metadata(self, batches, on_publish=None):
    rows = []
    for batch in batches:
        req = batch.reqs[0]
        context = self.contexts.get(req.rid)
        if context is not None and context.pipeline.lanes:
            p = context.pipeline
            root = next(l for l in p.lanes if l.index == p.committed)
            rows.append({"rid": req.rid, "output_index": p.committed,
                         "entry_depth": root.depth, "cycles": p.cycles,
                         "lanes": [{"uid": l.uid, "depth": l.depth, "position": len(l.prefix)} for l in p.lanes]})
    return {"requests": rows}


class LoopSpecWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        super().__init__()
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.server_args = server_args
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.adaptive_controller = None
        self.depths = tuple(get_spec().speculative_draft_depth)
        self.total = self.model_runner.model.config.total_ut_steps
        self.width = pipeline_layout(self.total, self.depths)[-1][2]
        self.contexts = {}
        self.role_replays = dict.fromkeys(("pre", "core", "post", "rejection"), 0)
        self.metrics = None
        if get_observability().enable_metrics:
            from .metrics import LoopSpecMetrics
            self.metrics = LoopSpecMetrics(get_serving().served_model_name)

    @property
    def recurrent_policy(self):
        return self

    @property
    def draft_worker(self):
        return None

    def alloc_memory_pool(self, **kwargs):
        # Worker construction precedes target pool sizing in the native lifecycle.
        self.max_rows = self.model_runner.max_running_requests * self.width
        self.req_to_token_pool, self.token_to_kv_pool_allocator = self.target_worker.get_memory_pool()
        self.private_pool = ReqToTokenPool(
            size=self.max_rows, max_context_len=self.req_to_token_pool.max_context_len,
            device=self.model_runner.device, enable_memory_saver=False,
        )
        self.private_pool.req_to_token.zero_()

    def init_attention_backends(self):
        if self.model_runner.recurrent_runner is None:
            raise RuntimeError("LoopSpec requires the ordinary recurrent CDB executor")
        self.executor = RecurrentRunner(self.model_runner, max_rows=self.max_rows, request_pool=self.private_pool)

    @torch.no_grad()
    def init_cuda_graphs(self):
        from sglang.srt.model_executor.cuda_graph_config import check_cuda_graph_backend
        if check_cuda_graph_backend("decode", "disabled"):
            return
        started, allocated = time.perf_counter(), torch.cuda.memory_allocated()
        sizes = get_exec().graph.cuda_graph_config.decode.bs
        sizes = set(sizes) | {self.width * size for size in sizes}
        self.executor.init_graphs(decode_sizes=sizes, prefill=False)
        self._additional_graph_memory_usage["draft_decode"] = (torch.cuda.memory_allocated() - allocated) / 2**30
        self._additional_graph_time_usage["draft_decode"] = time.perf_counter() - started
        logger.info("LoopSpec reuses CDB graphs, max requests=%d, branch width=%d",
                    self.model_runner.max_running_requests, self.width)

    def extra_owned_slots(self):
        # Debug census also catches accidental aliasing between request owners.
        slots = [slot for context in self.contexts.values() for slot in context.owned]
        if len(slots) != len(set(slots)):
            raise RuntimeError("LoopSpec private KV has multiple owners")
        return slots

    def clear_cache_pool(self):
        for context in self.contexts.values():
            context.retain(())
        self.contexts.clear()

    def note_request_retracted(self, *, rid):
        context = self.contexts.pop(rid, None)
        if context is not None:
            context.retain(())

    def note_request_finished(self, *, rid, natural_stop):
        context = self.contexts.pop(rid, None)
        if context is None:
            return
        p = context.pipeline
        logger.info("LoopSpec rid=%s cycles=%d reused=%d restarts=%d accepted=%s rejected=%d",
                    rid, p.cycles, p.reused, p.restarts, context.accepted, context.rejected)
        if self.metrics:
            # Each GPU replay is charged once globally, not once per owner.
            self.metrics.finish(context, p, self.role_replays, natural_stop)
            self.role_replays = dict.fromkeys(self.role_replays, 0)
        context.retain(())

    def split_recurrent_batch(self, batch):
        return split_batch(batch)

    @span("loopspec.prefill_complete")
    def finish_recurrent_prefill(self, batch, result):
        ids = result.next_token_ids.tolist()
        rows = batch.req_pool_indices.tolist()
        lengths = batch.seq_lens.tolist()
        for req, row, length, token in zip(batch.reqs, rows, lengths, ids, strict=True):
            if req is batch.chunked_req or req.finished() or req.sampling_params.max_new_tokens <= 1:
                continue
            if req.rid in self.contexts:
                raise RuntimeError("LoopSpec request identity reused before release")
            prefix = self.req_to_token_pool.req_to_token[row, :length].tolist()
            context = RequestContext(self, req, row, prefix)
            context.pipeline = Pipeline(context, token, prefix, req.sampling_params.max_new_tokens, self.total, self.depths)
            context.pipeline.committed = len(req.output_ids) + 1
            context.pipeline.lanes[0].index = context.pipeline.committed
            context.pipeline.lanes[0].history = tuple(req.output_ids) + (token,)
            self.contexts[req.rid] = context
        result.accept_lens = torch.ones_like(batch.seq_lens, dtype=torch.int32)
        result.new_seq_lens = batch.seq_lens
        result.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        result.speculative_output_stride = 1
        result.next_draft_input = LoopSpecInput(result.next_token_ids, batch.seq_lens)
        return result

    def _batch(self, rows):
        device = self.model_runner.device
        contexts, lanes = zip(*rows)
        lengths = [len(l.prefix) + 1 for l in lanes]
        return ForwardBatch(
            forward_mode=ForwardMode.DECODE, batch_size=len(rows),
            input_ids=torch.tensor([l.token for l in lanes], dtype=torch.int64, device=device),
            req_pool_indices=torch.arange(1, len(rows) + 1, device=device),
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
            seq_lens_sum=sum(lengths),
            out_cache_loc=torch.tensor([l.slot for l in lanes], dtype=torch.int64, device=device),
            positions=torch.tensor([n - 1 for n in lengths], dtype=torch.int64, device=device),
            capture_hidden_mode=CaptureHiddenMode.NULL, spec_algorithm=SpeculativeAlgorithm.NONE,
        )

    @span("loopspec.prepare_rows")
    def _prepare_rows(self, rows):
        if len(rows) > self.max_rows:
            raise RuntimeError("LoopSpec exceeded admitted branch capacity")
        new = [(context, lane) for context, lane in rows if lane.slot is None]
        if new:
            allocated = self.token_to_kv_pool_allocator.alloc(len(new))
            if allocated is None:
                raise RuntimeError("LoopSpec KV allocation exhausted")
            for (context, lane), slot in zip(new, allocated.tolist(), strict=True):
                lane.slot = slot
                context.owned.add(slot)
                context.materialized[slot] = 0
        for row, (context, lane) in enumerate(rows, 1):
            if any(context.materialized.get(slot, self.total) <= lane.depth for slot in lane.prefix):
                raise RuntimeError("LoopSpec read an unmaterialized prefix depth")
            chain = lane.prefix + (lane.slot,)
            self.private_pool.req_to_token[row, :len(chain)] = torch.tensor(
                chain, dtype=torch.int32, device=self.model_runner.device)
        batch = self._batch(rows)
        fresh = [i for i, (_, lane) in enumerate(rows) if lane.hidden is None]
        if fresh:
            import copy
            indices = torch.tensor(fresh, device=self.model_runner.device)
            initial = copy.copy(batch)
            initial.batch_size = len(fresh)
            for name in ("input_ids", "req_pool_indices", "seq_lens", "out_cache_loc", "positions"):
                setattr(initial, name, getattr(batch, name)[indices])
            state = self.executor.prepare(initial)
            for offset, index in enumerate(fresh):
                lane = rows[index][1]
                lane.hidden = state.hidden[offset:offset+1].clone()
                lane.injected = state.injected[offset:offset+1].clone() if state.injected is not None else None
        hidden = torch.cat([lane.hidden for _, lane in rows]) if len(rows) > 1 else rows[0][1].hidden
        injected = torch.cat([lane.injected for _, lane in rows]) if self.model_runner.model.has_injection else None
        state = RecurrentState(hidden, injected)
        depths = torch.tensor([lane.depth for _, lane in rows], dtype=torch.int64, device=self.model_runner.device)
        return batch, state, depths

    @span("loopspec.core")
    def _advance_rows(self, rows):
        batch, state, depths = self._prepare_rows(rows)
        state, used_graph = self.executor.execute_core(batch, state, depths)
        self.role_replays["core"] += int(used_graph)
        # Borrowed graph outputs must survive the finish graph's shared pool.
        state.hidden = state.hidden.clone()
        endpoints = []
        for i, (context, lane) in enumerate(rows):
            lane.depth += 1
            context.materialized[lane.slot] = lane.depth
            lane.hidden = state.hidden[i:i+1].clone()
            if lane.depth in self.depths or lane.depth == self.total:
                endpoints.append(i)
        if endpoints:
            import copy
            indices = torch.tensor(endpoints, device=self.model_runner.device)
            endpoint_batch = copy.copy(batch)
            endpoint_batch.batch_size = len(endpoints)
            for name in ("input_ids", "req_pool_indices", "seq_lens", "out_cache_loc", "positions"):
                setattr(endpoint_batch, name, getattr(batch, name)[indices])
            endpoint_state = RecurrentState(state.hidden[indices])
            logits = self.executor.finish(endpoint_batch, endpoint_state, depths[indices]).next_token_logits
            self.role_replays["post"] += int(len(endpoints) in self.executor.finish_graphs)
            # Preserve all endpoint logits with one copy before graph-pool reuse.
            logits = logits.clone()
            endpoint_rows = [rows[index] for index in endpoints]
            for j, (context, lane) in enumerate(endpoint_rows):
                context.logits[lane.uid] = logits[j]
            decisions, distributions = decide_endpoints(logits, endpoint_rows)
            decisions = self._read_decisions(decisions)
            for (context, lane), decision, distribution in zip(endpoint_rows, decisions, distributions, strict=True):
                context.decisions[lane.uid] = decision, distribution
        return used_graph

    @staticmethod
    @span("loopspec.decision_readback", gpu=False)
    def _read_decisions(decisions):
        # One dependent host read for the whole packet, never per endpoint.
        return decisions.tolist()

    @torch.no_grad()
    @span("loopspec.step", before=_step_metadata,
          after=lambda result, self, batches, on_publish=None: {"committed": [batch.reqs[0].rid for batch, _ in result]})
    def forward_recurrent(self, batches, on_publish=None):
        planned, rows = [], []
        for batch in batches:
            req = batch.reqs[0]
            context = self.contexts.get(req.rid)
            # The preceding overlapped result can still discover EOS/length.
            if context is None or not context.pipeline.lanes:
                continue
            if context.req is not req:
                raise RuntimeError("LoopSpec request identity mismatch")
            lanes = context.pipeline.plan()
            planned.append((batch, context, lanes))
            rows.extend((context, lane) for lane in lanes)
        if not rows:
            return []
        with device_timer_ctx(self.model_runner.device_timer, "decode"):
            used_graph = self._advance_rows(rows)
        outputs = []
        for batch, context, lanes in planned:
            verified = context.pipeline.collect(lanes)
            if verified is None:
                continue
            token, origin, logits = verified
            ids = torch.tensor([token], dtype=torch.int64, device=logits.device)
            output = LogitsProcessorOutput(next_token_logits=logits.unsqueeze(0))
            if batch.return_logprob:
                compute_spec_logprobs(batch, output, ids, chain_stride=1)
            lengths = torch.full_like(batch.seq_lens, context.committed_length)
            result = GenerationBatchResult(
                logits_output=output, next_token_ids=ids, can_run_cuda_graph=used_graph,
                accept_lens=torch.ones_like(lengths, dtype=torch.int32), new_seq_lens=lengths,
                next_draft_input=LoopSpecInput(ids, lengths),
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                speculative_output_stride=1, num_non_draft_tokens_per_req=int(origin is None),
            )
            if on_publish is not None:
                on_publish(batch, lengths)
            outputs.append((batch, result))
        return outputs

    def forward_batch_generation(self, batch, on_publish=None, pp_proxy_tensors=None):
        # V2 completion contract; the common scheduler calls forward_recurrent
        # for all independently completing requests in the selected loop.
        if batch.forward_mode.is_extend():
            raise RuntimeError("LoopSpec prefill must use the ordinary CDB executor")
        callback = (lambda _batch, lengths: on_publish(lengths)) if on_publish else None
        return GenerationBatchResult(recurrent_outputs=self.forward_recurrent(
            split_batch(batch), callback))
