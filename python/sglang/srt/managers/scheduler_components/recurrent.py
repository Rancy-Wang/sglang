"""Loop-boundary scheduling on native SGLang request and allocation ownership.

Corresponds to vllm-rlt's phase queues, take_phases and update_from_output.
There is no private serving loop: the ordinary/overlap event loops call one
step, receive zero or more completed native batches, and ingest again.
"""

import copy
from collections import deque
from dataclasses import dataclass

import torch

from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.recurrent_runner import RecurrentState
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils.recurrent_trace import span


@dataclass(eq=False)
class RecurrentWork:
    batch: ScheduleBatch
    depth: int = 0
    forward: ForwardBatch | None = None
    state: RecurrentState | None = None
    token_rows: torch.Tensor | None = None


class RecurrentScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.runner = scheduler.tp_worker.model_runner
        self.executor = self.runner.recurrent_runner
        self.total = self.runner.model.config.total_ut_steps
        self.pending = []
        self.completed = []
        self.chunks = deque()
        self.last_was_prefill = False

    def inflight_batches(self):
        return ([w.batch for w in self.pending] + self.completed
                + ([ScheduleBatch(reqs=list(self.chunks))] if self.chunks else []))

    def next_batch(self, running):
        s = self.scheduler
        # Publish only completed full-depth spans to normal decode admission.
        for batch in self.completed:
            chunk = batch.chunked_req
            if chunk is not None and not chunk.finished():
                s.stash_chunked_request(chunk)
                self.chunks.append(chunk)
            batch.filter_batch(reqs_to_exclude=[chunk] if chunk else None)
            batch.chunked_req = None
            if not batch.is_empty():
                if running.is_empty():
                    running = batch
                else:
                    running.merge_batch(batch)
        self.completed.clear()
        self.chunks = deque(req for req in self.chunks if not req.finished())

        # Pending rows count against the SAME native request/KV admission
        # budget. They cannot be prepared_for_decode a second time mid-token.
        view = copy.copy(running)
        view.reqs = list(dict.fromkeys(
            running.reqs + [r for w in self.pending for r in w.batch.reqs]
        ))
        view.batch_is_full = False
        if self.chunks:
            s.chunked_req = self.chunks.popleft()
        plan = s.get_new_batch_prefill(view)
        if plan.batch_to_run is not None:
            self.pending.append(RecurrentWork(plan.batch_to_run))
        elif s.chunked_req is not None:
            self.chunks.appendleft(s.chunked_req)
        # A partial chunk belongs to its work item until every depth is done.
        # This permits another request to enter at the next loop boundary.
        s.chunked_req = None

        if not running.is_empty() and not running.is_prefill_only:
            decode = s.update_running_batch(running)
            if decode is not None and not decode.is_empty():
                self.pending.append(RecurrentWork(decode))
                running = ScheduleBatch(reqs=[], batch_is_full=False)
        else:
            running.filter_batch()

        # Native overlap can have submitted the next token's first loop when
        # the previous result discovers EOS. Its KV was freed by that result
        # behind the WAR fence; do not advance those rows again.
        for work in self.pending:
            if work.forward is not None and work.batch.forward_mode.is_decode():
                self._filter_finished(work)
        self.pending = [w for w in self.pending if w.batch.reqs]
        prefills = [w for w in self.pending if w.batch.forward_mode.is_extend()]
        decodes = [w for w in self.pending if w.batch.forward_mode.is_decode()]
        if prefills and (not decodes or not self.last_was_prefill):
            selected = [prefills[0]]
            self.last_was_prefill = True
        else:
            selected = decodes
            self.last_was_prefill = False
        if not selected:
            return NextBatchPlan(batch_to_run=None, running_batch=running)
        # Rotate partial prefill work to avoid monopolizing all R loops.
        for work in selected:
            self.pending.remove(work)
            self.pending.append(work)
        carrier = copy.copy(selected[0].batch)
        carrier.reqs = [r for w in selected for r in w.batch.reqs]
        carrier.recurrent_plan = selected
        # Only initial spans count as new prefill input, not each loop.
        if selected[0].depth:
            carrier.extend_num_tokens = 0
        set_schedule_time_batch(carrier)
        return NextBatchPlan(batch_to_run=carrier, running_batch=running)

    def _filter_finished(self, work):
        keep = [i for i, req in enumerate(work.batch.reqs) if not req.finished()]
        if len(keep) == len(work.batch.reqs):
            return
        if not keep:
            work.batch.reqs = []
            return
        indices = torch.tensor(keep, device=self.runner.device)
        for name in ("input_ids", "req_pool_indices", "seq_lens", "positions", "out_cache_loc"):
            setattr(work.forward, name, getattr(work.forward, name)[indices])
        work.forward.batch_size = len(keep)
        work.forward.seq_lens_sum = None
        work.state = RecurrentState(work.state.hidden[indices],
            work.state.injected[indices] if work.state.injected is not None else None)
        out_cache_loc = work.batch.out_cache_loc[indices]
        work.batch.filter_batch(keep_indices=keep)
        work.batch.out_cache_loc = out_cache_loc
        # Rebuild only result/sampling metadata; the forward state/KV stays put.
        work.forward.sampling_info = work.batch.sampling_info.copy_for_forward()

    def _start(self, work):
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs

        s, batch = self.scheduler, work.batch
        resolve_forward_inputs(batch, s.future_map)
        with s._forward_isolation(batch, overlap=s.enable_overlap):
            work.forward = ForwardBatch.init_new(batch, self.runner, return_hidden_states_before_norm=False)
        # Reserve once per logical token, never once per loop. Timeline maps
        # are device resident; attention never uses a different depth's KV.
        slots = batch.out_cache_loc.tolist()
        timeline = self.runner.token_to_kv_pool.timeline
        timeline.ensure(slots, "pre", 0)
        for depth in range(self.total):
            timeline.ensure(slots, "core", depth)
        timeline.ensure(slots, "post", self.total - 1)
        if batch.forward_mode.is_extend():
            work.token_rows = torch.repeat_interleave(
                torch.arange(len(batch.reqs), device=self.runner.device),
                work.forward.extend_seq_lens, output_size=len(work.forward.input_ids),
            )
        work.state = self.executor.prepare(work.forward, token_rows=work.token_rows)

    @span("recurrent.core", before=lambda self, works: {
        "owners": [[req.rid for req in w.batch.reqs] for w in works],
        "depths": [w.depth for w in works],
        "modes": [w.batch.forward_mode.name for w in works],
    })
    def _advance(self, works):
        for work in works:
            if work.forward is None:
                self._start(work)
        if len(works) == 1:
            work = works[0]
            batch, state = work.forward, work.state
            depths = torch.full_like(batch.req_pool_indices, work.depth)
            token_rows = work.token_rows
        else:
            batch = copy.copy(works[0].forward)
            for name in ("input_ids", "req_pool_indices", "seq_lens", "positions", "out_cache_loc"):
                setattr(batch, name, torch.cat([getattr(w.forward, name) for w in works]))
            batch.batch_size = sum(w.forward.batch_size for w in works)
            batch.seq_lens_sum = None
            batch.seq_lens_cpu = None
            state = RecurrentState(torch.cat([w.state.hidden for w in works]))
            if works[0].state.injected is not None:
                state.injected = torch.cat([w.state.injected for w in works])
            depths = torch.cat([torch.full_like(w.forward.req_pool_indices, w.depth) for w in works])
            token_rows = None
        advanced, used_graph = self.executor.execute_core(batch, state, depths, token_rows=token_rows)
        offset = 0
        for work in works:
            length = len(work.state.hidden)
            work.state = RecurrentState(advanced.hidden[offset:offset+length].clone(), work.state.injected)
            work.depth += 1
            offset += length
        return used_graph

    def forward(self, carrier):
        s = self.scheduler
        outputs = []
        with s.forward_stream_ctx:
            s.forward_stream.wait_stream(s.schedule_stream)
            used_graph = self._advance(carrier.recurrent_plan)
            for work in carrier.recurrent_plan:
                if work.depth != self.total:
                    continue
                batch = work.batch
                depths = torch.full_like(work.forward.req_pool_indices, self.total - 1)
                logits = self.executor.finish(work.forward, work.state, depths, token_rows=work.token_rows)
                result = GenerationBatchResult(
                    logits_output=logits, can_run_cuda_graph=used_graph,
                    extend_input_len_per_req=batch.extend_lens if (batch.return_logprob or batch.return_hidden_states) else None,
                    extend_logprob_start_len_per_req=batch.extend_logprob_start_lens if batch.return_logprob else None,
                )
                result = s.tp_worker.sample_generation(result, work.forward)
                s.future_map.publish(batch.req_pool_indices, batch.seq_lens + 1)
                result.copy_done = s.device_module.Event()
                if result.delay_sample_func is None:
                    s._relay_forward_payload(batch, batch.req_pool_indices, result)
                    s.copy_stream.wait_stream(s.forward_stream)
                    with s.copy_stream_ctx:
                        result.copy_to_cpu(batch.return_logprob, batch.return_hidden_states)
                else:
                    result.future_indices = batch.req_pool_indices
                batch.forward_iter, batch.launch_ts = carrier.forward_iter, carrier.launch_ts
                batch.after_idle_gap = carrier.after_idle_gap
                outputs.append((batch.copy(), result))
                batch.input_ids = None
                self.completed.append(batch)
                self.pending.remove(work)
        # The event-loop result queue keeps completed snapshots/tensors alive;
        # partial work keeps its own state until a later loop or cancellation.
        return GenerationBatchResult(recurrent_outputs=outputs)
