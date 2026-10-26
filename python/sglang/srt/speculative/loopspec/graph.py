"""Mixed-depth recurrent core: shared projections, depth-separated attention."""

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def run_core(worker, hidden, positions, batches, micro, injected=None, role="core"):
    groups = []
    for (base, start, end), batch in zip(worker.layout, batches):
        depth = base + micro
        backend = worker.backends[depth]
        backend.init_forward_metadata(batch)
        groups.append((depth, start, end, batch, backend))
    model = worker.model_runner.model
    with forward_context(ForwardContext(worker.backends[0])):
        logits = None
        if role == "core":
            hidden = model.advance_recurrent(hidden, positions, groups, injected)
        else:
            endpoints = set(worker.depths) | {model.config.total_ut_steps}
            selected = [g for g in groups if g[0] + 1 in endpoints]
            h = torch.cat([hidden[start:end] for _, start, end, _, _ in selected])
            pos = torch.cat([positions[start:end] for _, start, end, _, _ in selected])
            compact, offset = [], 0
            for depth, start, end, batch, backend in selected:
                compact.append((depth, offset, offset + end - start, batch, backend))
                offset += end - start
            readout = model.readout_recurrent(h, pos, compact)
            logits = torch.zeros(
                (worker.max_rows, model.config.vocab_size),
                dtype=torch.float32,
                device=hidden.device,
            )
            offset = 0
            for _, start, end, _, _ in selected:
                logits[start:end] = readout[offset : offset + end - start]
                offset += end - start
    return hidden, logits


class RecurrentGraph:
    def __init__(self, worker, micro=0, role="core"):
        self.worker, self.micro = worker, micro
        self.role = role
        model = worker.model_runner.model
        device = worker.model_runner.device
        rows = 1 if role == "pre" else worker.max_rows
        self.hidden = torch.zeros(
            (rows, model.config.hidden_size),
            dtype=model.get_input_embeddings().weight.dtype,
            device=device,
        )
        self.injected = torch.zeros_like(self.hidden) if model.has_injection else None
        self.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        self.batches = []
        for _, start, end in [(0, 0, 1)] if role == "pre" else worker.layout:
            size = end - start
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=size,
                input_ids=torch.zeros(size, dtype=torch.int64, device=device),
                req_pool_indices=torch.zeros(size, dtype=torch.int64, device=device),
                seq_lens=torch.ones(size, dtype=torch.int32, device=device),
                # Fixed allocation bound; actual lengths remain GPU inputs.
                seq_lens_sum=size * worker.private_pool.max_context_len,
                out_cache_loc=torch.zeros(size, dtype=torch.int64, device=device),
                positions=self.positions[start:end],
                capture_hidden_mode=CaptureHiddenMode.NULL,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            worker.translator.bind_own_plan(batch)
            self.batches.append(batch)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), model_capture_mode():
            for _ in range(3):
                self.run()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.run()
        torch.cuda.current_stream(device).wait_stream(stream)

    def run(self):
        if self.role == "pre":
            batch = self.batches[0]
            self.worker.pre_backend.init_forward_metadata(batch)
            with forward_context(ForwardContext(self.worker.pre_backend)):
                self.output_hidden, self.output_injected = (
                    self.worker.model_runner.model.prepare_recurrent(
                        batch.input_ids, self.positions, batch
                    )
                )
            self.output_logits = None
            return
        self.output_hidden, self.output_logits = run_core(
            self.worker,
            self.hidden,
            self.positions,
            self.batches,
            self.micro,
            self.injected,
            self.role,
        )

    def replay(self, hidden, positions, batches, injected=None):
        self.hidden.copy_(hidden)
        if self.injected is not None and injected is not None:
            self.injected.copy_(injected)
        self.positions.copy_(positions)
        for target, source in zip(self.batches, batches):
            target.input_ids.copy_(source.input_ids)
            target.req_pool_indices.copy_(source.req_pool_indices)
            target.seq_lens.copy_(source.seq_lens)
            target.out_cache_loc.copy_(source.out_cache_loc)
        self.graph.replay()
        # The executor clones live lanes before the next cycle overwrites this.
        return self.output_hidden, self.output_logits
