"""Per-request branch ownership and distribution decisions, independent of scheduling."""

import copy

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils.recurrent_trace import span

from .sampling import probabilities, verify_tensor


def split_batch(batch):
    """Metadata views of native requests; never create child Req objects.

    Finish frontiers differ per request. The common executor still packs all
    their branches into one GPU batch. Avoid shared mutable penalty state.
    """
    if len(batch.reqs) == 1:
        return [batch]
    views = []
    for i in range(len(batch.reqs)):
        view = copy.copy(batch)
        view.spec_info = copy.copy(batch.spec_info)
        view.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, batch.model_config.vocab_size
        )
        view.filter_batch(keep_indices=[i])
        views.append(view)
    return views


class RequestContext:
    def __init__(self, worker, req, row, prefix):
        self.worker, self.req, self.row = worker, req, row
        self.device = worker.model_runner.device
        self.allocator = worker.token_to_kv_pool_allocator
        self.params = req.sampling_params
        self.depths, self.total = worker.depths, worker.total
        self.committed_length = len(prefix)
        self.owned, self.materialized, self.logits = set(), {}, {}
        self.decisions = {}
        self.accepted, self.proposed = [0, 0], [0, 0]
        self.rejected = 0
        self.peak_slots = self.committed_length * self.total
        self.stop_ids = set(self.params.stop_token_ids or ()) | set(req.eos_token_ids or ())
        self.stop_ids |= set(getattr(req.tokenizer, "additional_stop_token_ids", ()) or ())
        if req.tokenizer.eos_token_id is not None:
            self.stop_ids.add(req.tokenizer.eos_token_id)
        self.generator = None
        if self.params.sampling_seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(self.params.sampling_seed)

    def distribution(self, lane):
        return probabilities(self.logits[lane.uid], self.params, lane.history, self.stop_ids)

    @span("loopspec.decision")
    def queue_decision(self, lane):
        """Enqueue only device work; worker collects all decisions in one D2H."""
        target = self.distribution(lane)
        if lane.depth == self.total:
            first_token, first = lane.proposals[self.depths[0]]
            second_token, second = lane.proposals.get(
                self.depths[1] if len(self.depths) == 2 else -1, (0, None)
            )
            uniforms = torch.rand(3, device=self.device, generator=self.generator)
            pair = verify_tensor(
                target, first, torch.tensor(first_token, device=self.device),
                torch.zeros_like(first) if second is None else second,
                torch.tensor(second_token, device=self.device),
                torch.tensor(second is not None, device=self.device), uniforms,
            )
            return pair, None
        enabled = torch.ones((), device=self.device, dtype=torch.bool)
        if lane.depth != self.depths[0]:
            token, first = lane.proposals[self.depths[0]]
            delta = (target - first).clamp_min(0)
            delta[token] = 0
            mass = delta.sum()
            enabled = (target[token] < first[token]) & (mass > 0)
            # Even a disabled proposal must have a valid sampling distribution;
            # its sampled value is discarded and cannot create a branch.
            target = torch.where(enabled, delta / mass.clamp_min(1e-30), target)
        token = torch.multinomial(target, 1, generator=self.generator)[0]
        return torch.stack((token, enabled.long())), target

    def propose(self, lane):
        (token, enabled), distribution = self.decisions.pop(lane.uid)
        if not enabled:
            return None
        self.proposed[self.depths.index(lane.depth)] += 1
        return token, distribution

    def verify(self, lane):
        (token, accepted), _ = self.decisions.pop(lane.uid)
        if accepted < 0:
            self.rejected += 1
        else:
            self.accepted[accepted] += 1
        return token, None if accepted < 0 else self.depths[accepted], self.logits[lane.uid]

    @span("loopspec.commit")
    def commit(self, lane):
        if lane.depth != self.total or lane.slot not in self.owned:
            raise RuntimeError("LoopSpec committed a slot without full target KV")
        self.worker.req_to_token_pool.req_to_token[self.row, self.committed_length] = lane.slot
        self.committed_length += 1
        self.req.kv.kv_allocated_len = self.committed_length
        self.owned.remove(lane.slot)
        self.materialized.pop(lane.slot)

    @span("loopspec.retain")
    def retain(self, lanes):
        retained = set()
        for lane in lanes:
            retained.update(lane.prefix)
            if lane.slot is not None:
                retained.add(lane.slot)
        freed = self.owned - retained
        if freed:
            self.allocator.free(torch.tensor(sorted(freed), dtype=torch.int64, device=self.device))
        self.owned -= freed
        for slot in freed:
            self.materialized.pop(slot)
        live = {lane.uid for lane in lanes}
        self.logits = {key: value for key, value in self.logits.items() if key in live}
        self.peak_slots = max(self.peak_slots,
            self.committed_length * self.total + sum(self.materialized.values()))
