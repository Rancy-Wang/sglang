"""V2 relay: only verified anchor and materialized sequence length are public."""

import torch
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class DASWaveInput(SpecInput):
    cuda_graph_compatible = False
    future_indices = None
    topk_p = topk_index = hidden_states = None

    def __init__(self, tokens, lengths):
        super().__init__(SpecInputType.DAS_WAVE_STATE)
        self.bonus_tokens, self.new_seq_lens = tokens, lengths
        self.num_tokens_per_req = self.num_tokens_for_logprob_per_req = 1

    def prepare_for_decode(self, batch):
        batch.input_ids, batch.seq_lens = self.bonus_tokens, self.new_seq_lens
        batch.seq_lens_cpu = batch.seq_lens_sum = None

    def filter_batch(self, new_indices, new_indices_cpu=None):
        for name in ("bonus_tokens", "new_seq_lens", "future_indices"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[new_indices])

    def merge_batch(self, other):
        for name in ("bonus_tokens", "new_seq_lens", "future_indices"):
            a, b = getattr(self, name), getattr(other, name)
            setattr(self, name, torch.cat((a, b)) if a is not None and b is not None else None)
