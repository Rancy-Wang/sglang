"""Native V2 relay state. Branch tensors remain owned by request contexts."""

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class LoopSpecInput(SpecInput):
    cuda_graph_compatible = False
    future_indices = None
    topk_p = topk_index = hidden_states = None

    def __init__(self, tokens, lengths):
        super().__init__(SpecInputType.LOOPSPEC_STATE)
        self.bonus_tokens, self.new_seq_lens = tokens, lengths
        self.num_tokens_per_req = self.num_tokens_for_logprob_per_req = 1

    def prepare_for_decode(self, batch):
        # No speculative branch is published in the native row before verify.
        batch.input_ids = self.bonus_tokens
        batch.seq_lens = self.new_seq_lens
        batch.seq_lens_cpu = None
        batch.seq_lens_sum = None

    def filter_batch(self, new_indices, new_indices_cpu=None):
        for name in ("bonus_tokens", "new_seq_lens", "future_indices"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[new_indices])

    def merge_batch(self, other):
        for name in ("bonus_tokens", "new_seq_lens", "future_indices"):
            left, right = getattr(self, name), getattr(other, name)
            setattr(self, name, torch.cat((left, right)) if left is not None and right is not None else None)
