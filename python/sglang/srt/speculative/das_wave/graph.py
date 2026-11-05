"""Wave graphs keep all core launches and readout within one replay.

The inherited staging contract uses discard slot zero for padding. Draft uses
the depth input buffer for dynamic mature-prefix lengths; verifier ignores it
and always computes R independent recurrences from the supplied embeddings.
"""

import torch

from sglang.srt.model_executor.recurrent_runner import RecurrentCoreGraph, RecurrentState


class WaveGraph(RecurrentCoreGraph):
    def __init__(self, worker, rows, *, draft):
        self.worker = worker
        self.draft = draft
        super().__init__(worker.executor, rows)

    def run(self):
        hidden = self.state.hidden
        if self.draft:
            for _ in range(self.worker.updates):
                hidden = self.worker._draft_core(self.batch, hidden, self.depths)
        else:
            for depth in range(self.worker.total):
                hidden = self.recurrent.advance(
                    self.batch, RecurrentState(hidden), torch.full_like(self.depths, depth)
                ).hidden
        self.output = RecurrentState(hidden)
        self.predictions = self.worker._readout(self.batch, hidden).argmax(-1)

    def replay_wave(self, batch, hidden, prefix):
        output = super().replay(batch, RecurrentState(hidden), prefix)
        # All buckets share a pool. Both live outputs must outlast the next
        # graph, including a graph of a different role or row count.
        return output.hidden.clone(), self.predictions[:batch.batch_size].clone()
