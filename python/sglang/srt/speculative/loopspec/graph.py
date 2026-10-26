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


def run_core(worker, hidden, positions, batches, micro):
    groups = []
    for (base, start, end), batch in zip(worker.layout, batches):
        depth = base + micro
        backend = worker.backends[depth]
        backend.init_forward_metadata(batch)
        groups.append((depth, start, end, batch, backend))
    model = worker.model_runner.model
    with forward_context(ForwardContext(worker.backends[0])):
        hidden = model.model.advance_batch(hidden, positions, groups)
        # Native Ouro is TP=1, unquantized, with no LM-head softcap/scaling.
        logits = (
            torch.nn.functional.linear(hidden, model.lm_head.weight).float()
            if micro == worker.stride - 1
            else None
        )
    return hidden, logits


class RecurrentGraph:
    def __init__(self, worker, micro):
        self.worker, self.micro = worker, micro
        model = worker.model_runner.model
        device = worker.model_runner.device
        self.hidden = torch.zeros(
            (worker.max_rows, model.config.hidden_size),
            dtype=model.model.embed_tokens.weight.dtype,
            device=device,
        )
        self.positions = torch.zeros(worker.max_rows, dtype=torch.int64, device=device)
        self.batches = []
        for _, start, end in worker.layout:
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
        self.output_hidden, self.output_logits = run_core(
            self.worker, self.hidden, self.positions, self.batches, self.micro
        )

    def replay(self, hidden, positions, batches):
        self.hidden.copy_(hidden)
        self.positions.copy_(positions)
        for target, source in zip(self.batches, batches):
            target.input_ids.copy_(source.input_ids)
            target.req_pool_indices.copy_(source.req_pool_indices)
            target.seq_lens.copy_(source.seq_lens)
            target.out_cache_loc.copy_(source.out_cache_loc)
        self.graph.replay()
        # The executor clones live lanes before the next cycle overwrites this.
        return self.output_hidden, self.output_logits
