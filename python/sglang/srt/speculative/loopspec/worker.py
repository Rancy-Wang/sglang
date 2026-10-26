"""SGLang V2 worker for an incremental recurrent speculative pipeline."""

import logging
import time

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_observability, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.speculative.spec_info import (
    SpecInput,
    SpecInputType,
    SpeculativeAlgorithm,
)

from .pipeline import Pipeline
from .sampling import probabilities, second_proposal, verify

logger = logging.getLogger(__name__)


class LoopSpecInput(SpecInput):
    cuda_graph_compatible = False
    future_indices = None

    def __init__(self, tokens, lengths):
        super().__init__(SpecInputType.LOOPSPEC_STATE)
        self.bonus_tokens = tokens
        self.new_seq_lens = lengths
        self.num_tokens_per_req = self.num_tokens_for_logprob_per_req = 1

    def prepare_for_decode(self, batch):
        # Private speculative slots are allocated by the worker, not published
        # in the scheduler's committed request row until all depths are done.
        batch.input_ids = self.bonus_tokens

    def filter_batch(self, indices, new_indices_cpu=None):
        self.bonus_tokens = self.bonus_tokens[indices]
        self.new_seq_lens = self.new_seq_lens[indices]

    def merge_batch(self, other):
        self.bonus_tokens = torch.cat((self.bonus_tokens, other.bonus_tokens))
        self.new_seq_lens = torch.cat((self.new_seq_lens, other.new_seq_lens))


class RequestExecutor:
    def __init__(self, worker, req, row, prefix):
        self.worker, self.req, self.row = worker, req, row
        self.model = worker.model_runner.model
        self.backend, self.translator = worker.backend, worker.translator
        self.device = worker.model_runner.device
        self.allocator = worker.token_to_kv_pool_allocator
        self.params = req.sampling_params
        self.depths = tuple(get_spec().speculative_draft_depth)
        self.total = self.model.config.total_ut_steps
        self.committed_length = len(prefix)
        self.owned = set()
        self.materialized = {}  # Only transient slots; committed slots are full depth.
        self.logits = {}
        self.accepted = [0, 0]
        self.proposed = [0, 0]
        self.peak_slots = worker.model_runner.token_to_kv_pool.timeline.live_slots
        self.rejected = 0
        self.stop_ids = set(self.params.stop_token_ids or ()) | set(
            req.eos_token_ids or ()
        )
        self.stop_ids |= set(req.tokenizer.additional_stop_token_ids or ())
        if req.tokenizer.eos_token_id is not None:
            self.stop_ids.add(req.tokenizer.eos_token_id)
        self.generator = None
        if self.params.sampling_seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.params.sampling_seed
            )

    def advance_many(self, lanes, count):
        from .graph import run_core

        for lane in lanes:
            if lane.slot is None:
                allocated = self.allocator.alloc(1)
                if allocated is None:
                    raise RuntimeError("LoopSpec KV allocation exhausted")
                lane.slot = int(allocated.item())
                self.owned.add(lane.slot)
                self.materialized[lane.slot] = 0
        rows = {}
        hidden = torch.zeros(
            (self.worker.max_rows, self.model.config.hidden_size),
            dtype=self.model.get_input_embeddings().weight.dtype,
            device=self.device,
        )
        positions = torch.zeros(
            self.worker.max_rows, dtype=torch.int64, device=self.device
        )
        injected = torch.zeros_like(hidden) if self.model.has_injection else None
        batches = []
        for base, start, end in self.worker.layout:
            active = [lane for lane in lanes if lane.depth == base]
            if len(active) > end - start:
                raise RuntimeError("LoopSpec exceeded its stage capacity")
            request_rows, lengths, slots, tokens = [], [], [], []
            for offset in range(end - start):
                row = start + offset
                if offset < len(active):
                    lane = active[offset]
                    rows[lane.uid] = row
                    positions[row] = len(lane.prefix)
                    chain = lane.prefix + (lane.slot,)
                    self.worker.private_pool.req_to_token[row + 1, : len(chain)] = (
                        torch.tensor(chain, dtype=torch.int32, device=self.device)
                    )
                    if lane.hidden is None:
                        self.prepare_lane(lane, row + 1)
                    hidden[row : row + 1] = lane.hidden
                    if injected is not None:
                        injected[row : row + 1] = lane.injected
                    request_rows.append(row + 1)
                    lengths.append(len(chain))
                    slots.append(lane.slot)
                    tokens.append(lane.token)
                else:
                    request_rows.append(0)
                    lengths.append(1)
                    slots.append(0)
                    tokens.append(0)
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=end - start,
                input_ids=torch.tensor(tokens, dtype=torch.int64, device=self.device),
                req_pool_indices=torch.tensor(
                    request_rows, dtype=torch.int64, device=self.device
                ),
                seq_lens=torch.tensor(lengths, dtype=torch.int32, device=self.device),
                seq_lens_sum=sum(lengths),
                out_cache_loc=torch.tensor(
                    slots, dtype=torch.int64, device=self.device
                ),
                positions=positions[start:end],
                capture_hidden_mode=CaptureHiddenMode.NULL,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            self.translator.bind_own_plan(batch)
            batches.append(batch)
        for micro in range(count):
            for lane in lanes:
                if any(
                    self.materialized.get(slot, self.total) <= lane.depth
                    for slot in lane.prefix
                ):
                    raise RuntimeError(
                        "LoopSpec attempted to read unmaterialized prefix depth"
                    )
                self.worker.timeline.pool.timeline.ensure(
                    [lane.slot], "core", lane.depth
                )
            if self.worker.graphs:
                hidden, logits = self.worker.graphs[micro].replay(
                    hidden, positions, batches, injected
                )
                self.worker.graph_replays += 1
                self.worker.role_replays["core"] += 1
            else:
                hidden, logits = run_core(
                    self.worker, hidden, positions, batches, micro, injected
                )
            for lane in lanes:
                lane.depth += 1
                self.materialized[lane.slot] = lane.depth
        for lane in lanes:
            if lane.depth in self.depths or lane.depth == self.total:
                self.worker.timeline.pool.timeline.ensure(
                    [lane.slot], "post", lane.depth - 1
                )
        if self.worker.graphs:
            _, logits = self.worker.post_graph.replay(hidden, positions, batches)
            self.worker.role_replays["post"] += 1
        else:
            _, logits = run_core(
                self.worker, hidden, positions, batches, count - 1, role="post"
            )
        self.peak_slots = max(
            self.peak_slots, self.worker.timeline.pool.timeline.live_slots
        )
        for lane in lanes:
            row = rows[lane.uid]
            lane.hidden = hidden[row : row + 1].clone()
            if lane.depth in self.depths or lane.depth == self.total:
                self.logits[lane.uid] = logits[row].clone()

    def prepare_lane(self, lane, row):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        device = self.device
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([lane.token], dtype=torch.int64, device=device),
            positions=torch.tensor(
                [len(lane.prefix)], dtype=torch.int64, device=device
            ),
            req_pool_indices=torch.tensor([row], dtype=torch.int64, device=device),
            seq_lens=torch.tensor(
                [len(lane.prefix) + 1], dtype=torch.int32, device=device
            ),
            seq_lens_sum=len(lane.prefix) + 1,
            out_cache_loc=torch.tensor([lane.slot], dtype=torch.int64, device=device),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        self.translator.bind_own_plan(batch)
        self.worker.timeline.pool.timeline.ensure([lane.slot], "pre", 0)
        if self.worker.graphs:
            graph = self.worker.pre_graph
            hidden, _ = graph.replay(
                torch.zeros_like(graph.hidden), batch.positions, [batch]
            )
            injected = graph.output_injected
            self.worker.role_replays["pre"] += 1
        else:
            batch, backend = self.worker.timeline.target_batch(batch, "pre", 0)
            with forward_context(ForwardContext(backend)):
                hidden, injected = self.model.prepare_recurrent(
                    batch.input_ids, batch.positions, batch
                )
        lane.hidden = hidden.clone()
        lane.injected = None if injected is None else injected.clone()

    def distribution(self, lane):
        return probabilities(
            self.logits[lane.uid], self.params, lane.history, self.stop_ids
        )

    def propose(self, lane):
        distribution = self.distribution(lane)
        if lane.depth != self.depths[0]:
            token, first = lane.proposals[self.depths[0]]
            distribution = second_proposal(first, distribution, token)
            if distribution is None:
                return None
        self.proposed[self.depths.index(lane.depth)] += 1
        token = int(torch.multinomial(distribution, 1, generator=self.generator))
        return token, distribution

    def verify(self, lane):
        first_token, first = lane.proposals[self.depths[0]]
        second_token, second = lane.proposals.get(
            self.depths[1] if len(self.depths) == 2 else -1, (None, None)
        )
        verifier = verify
        if self.worker.rejection_graph is not None:
            verifier = self.worker.rejection_graph.replay
            self.worker.role_replays["rejection"] += 1
        token, accepted = verifier(
            self.distribution(lane),
            first,
            first_token,
            second,
            second_token,
            self.generator,
        )
        if accepted is None:
            self.rejected += 1
        else:
            self.accepted[accepted] += 1
        return (
            token,
            None if accepted is None else self.depths[accepted],
            self.logits[lane.uid],
        )

    def commit(self, lane):
        if lane.depth != self.total or lane.slot not in self.owned:
            raise RuntimeError("LoopSpec committed a slot without full target KV")
        table = self.worker.req_to_token_pool.req_to_token
        table[self.row, self.committed_length] = lane.slot
        self.committed_length += 1
        self.req.kv.kv_allocated_len = self.committed_length
        self.owned.remove(lane.slot)
        self.materialized.pop(lane.slot)

    def retain(self, lanes):
        retained = set()
        for lane in lanes:
            retained.update(lane.prefix)
            if lane.slot is not None:
                retained.add(lane.slot)
        freed = self.owned - retained
        if freed:
            self.allocator.free(
                torch.tensor(sorted(freed), dtype=torch.int64, device=self.device)
            )
        self.owned -= freed
        for slot in freed:
            self.materialized.pop(slot)
        live = {lane.uid for lane in lanes}
        self.logits = {key: value for key, value in self.logits.items() if key in live}


class LoopSpecWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        super().__init__()
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.server_args = server_args
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.adaptive_controller = None
        self.executor = self.pipeline = None
        self.graphs = []
        self.graph_replays = 0
        self.role_replays = dict.fromkeys(("pre", "core", "post", "rejection"), 0)
        self.rejection_graph = None
        self.metrics = None
        if get_observability().enable_metrics:
            from .metrics import LoopSpecMetrics

            self.metrics = LoopSpecMetrics(server_args.served_model_name)
        depths = get_spec().speculative_draft_depth
        self.depths = depths
        self.stride = depths[0]
        from .config import pipeline_layout

        self.layout = pipeline_layout(
            self.model_runner.model.config.total_ut_steps, depths
        )
        self.max_rows = self.layout[-1][2]

    @property
    def draft_worker(self):
        return None

    def alloc_memory_pool(self, **kwargs):
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.target_worker.get_memory_pool()
        )
        self.private_pool = ReqToTokenPool(
            size=self.max_rows,
            max_context_len=self.req_to_token_pool.max_context_len,
            device=self.model_runner.device,
            enable_memory_saver=False,
        )

    def init_attention_backends(self):
        from sglang.srt.mem_cache.timeline_pool import TimelineRuntime

        self.translator = KVIndexTranslator(
            req_to_token=self.private_pool.req_to_token,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            page_size=1,
            device=self.model_runner.device,
        )
        self.timeline = TimelineRuntime(
            self.model_runner.token_to_kv_pool,
            self.token_to_kv_pool_allocator,
            self.private_pool,
            self.model_runner,
        )
        self.model_runner.model.timeline_runtime = TimelineRuntime(
            self.model_runner.token_to_kv_pool,
            self.token_to_kv_pool_allocator,
            self.req_to_token_pool,
            self.model_runner,
        )
        self.backends = [
            self.timeline.backends[("core", depth)]
            for depth in range(self.model_runner.model.config.total_ut_steps)
        ]
        self.backend = self.backends[0]

    @torch.no_grad()
    def init_cuda_graphs(self):
        from sglang.srt.model_executor.cuda_graph_config import (
            check_cuda_graph_backend,
        )
        from .graph import RecurrentGraph

        if check_cuda_graph_backend("decode", "disabled"):
            return
        started = time.perf_counter()
        allocated = torch.cuda.memory_allocated()
        self.graphs = [RecurrentGraph(self, micro) for micro in range(self.stride)]
        self.pre_graph = RecurrentGraph(self, role="pre")
        self.post_graph = RecurrentGraph(self, self.stride - 1, role="post")
        from .sampling import RejectionGraph

        self.rejection_graph = RejectionGraph(
            self.model_runner.model.config.vocab_size, self.model_runner.device
        )
        self._additional_graph_memory_usage["draft_decode"] = (
            torch.cuda.memory_allocated() - allocated
        ) / 2**30
        self._additional_graph_time_usage["draft_decode"] = (
            time.perf_counter() - started
        )
        logger.info(
            "LoopSpec captured pre/post/rejection and %d mixed-depth core graphs, rows=%d",
            len(self.graphs),
            self.max_rows,
        )

    def clear_cache_pool(self):
        if self.executor is not None:
            self.executor.retain(())
        self.executor = self.pipeline = None

    def note_request_finished(self, *, rid, natural_stop):
        if self.executor is not None and self.executor.req.rid == rid:
            logger.info(
                "LoopSpec rid=%s cycles=%d reused=%d restarts=%d accepted=%s rejected=%d graph_replays=%d",
                rid,
                self.pipeline.cycles,
                self.pipeline.reused,
                self.pipeline.restarts,
                self.executor.accepted,
                self.executor.rejected,
                self.graph_replays,
            )
            if self.metrics is not None:
                self.metrics.finish(
                    self.executor, self.pipeline, self.role_replays, natural_stop
                )
            self.clear_cache_pool()

    @torch.no_grad()
    def forward_batch_generation(self, batch, on_publish=None, pp_proxy_tensors=None):
        if len(batch.reqs) != 1 or on_publish is not None:
            raise RuntimeError("LoopSpec requires one synchronous request")
        req = batch.reqs[0]
        try:
            if batch.forward_mode.is_extend():
                self.clear_cache_pool()
                self.graph_replays = 0
                self.role_replays = dict.fromkeys(self.role_replays, 0)
                result = self.target_worker.forward_batch_generation(batch)
                result.accept_lens = torch.ones_like(batch.seq_lens, dtype=torch.int32)
                result.new_seq_lens = batch.seq_lens
                result.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                result.speculative_output_stride = 1
                result.next_draft_input = LoopSpecInput(
                    result.next_token_ids, batch.seq_lens
                )
                if req.sampling_params.max_new_tokens > 1:
                    row = int(batch.req_pool_indices[0])
                    length = int(batch.seq_lens[0])
                    prefix = self.req_to_token_pool.req_to_token[row, :length].tolist()
                    self.executor = RequestExecutor(self, req, row, prefix)
                    self.pipeline = Pipeline(
                        self.executor,
                        int(result.next_token_ids[0]),
                        prefix,
                        req.sampling_params.max_new_tokens,
                        self.executor.total,
                        self.executor.depths,
                    )
                return result
            if self.executor is None or self.executor.req.rid != req.rid:
                raise RuntimeError("LoopSpec decode has no matching prefill state")
            # Prefill is timed by the target runner. Our decode bypasses that
            # runner, so account for its stream interval through the same timer.
            with device_timer_ctx(self.model_runner.device_timer, "decode"):
                token, accepted_depth, logits = self.pipeline.advance_until_commit()
            ids = torch.tensor([token], dtype=torch.int64, device=logits.device)
            output = LogitsProcessorOutput(next_token_logits=logits.unsqueeze(0))
            if batch.return_logprob:
                compute_spec_logprobs(batch, output, ids, chain_stride=1)
            new_lengths = batch.seq_lens + 1
            return GenerationBatchResult(
                logits_output=output,
                next_token_ids=ids,
                can_run_cuda_graph=bool(self.graphs),
                accept_lens=torch.ones_like(batch.seq_lens, dtype=torch.int32),
                new_seq_lens=new_lengths,
                next_draft_input=LoopSpecInput(ids, new_lengths),
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                speculative_output_stride=1,
                num_non_draft_tokens_per_req=int(accepted_depth is None),
            )
        except Exception:
            self.clear_cache_pool()
            raise
