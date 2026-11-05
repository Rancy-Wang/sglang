"""Saved proposals and lossless residual verification for LoopSpec."""

import torch

from sglang.srt.utils.recurrent_trace import span


def processed_logits(logits, contexts, lanes):
    """Apply per-owner policies without launching work once per endpoint.

    Sparse history/bias/stop metadata is assembled on the host; all vocabulary
    operations are batched. The common unpenalized path needs no history tensor.
    """
    biases, stops, histories = [], [], []
    params = [context.params for context in contexts]
    for row, (context, lane, param) in enumerate(zip(contexts, lanes, params, strict=True)):
        biases.extend((row, int(token), value) for token, value in (param.logit_bias or {}).items())
        if len(lane.history) < param.min_new_tokens:
            stops.extend((row, token) for token in context.stop_ids)
        if param.frequency_penalty or param.presence_penalty or param.repetition_penalty != 1:
            histories.extend((row, token) for token in lane.history)
    if not (biases or stops or histories):
        return logits
    values = logits.float().clone()
    if biases:
        rows, tokens, weights = zip(*biases)
        values[torch.tensor(rows, device=values.device), torch.tensor(tokens, device=values.device)] += torch.tensor(weights, device=values.device)
    if histories:
        rows, tokens = zip(*histories)
        flat = torch.tensor(rows, device=values.device) * values.shape[1] + torch.tensor(tokens, device=values.device)
        counts = torch.zeros_like(values).flatten()
        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=values.dtype))
        counts = counts.view_as(values)
        frequency, presence, repetition = torch.tensor(
            [(p.frequency_penalty, p.presence_penalty, p.repetition_penalty) for p in params],
            device=values.device, dtype=values.dtype,
        ).unbind(1)
        values -= counts * frequency[:, None]
        values -= (counts > 0) * presence[:, None]
        scale = torch.where(counts > 0, repetition[:, None], 1.0)
        values = torch.where(values < 0, values * scale, values / scale)
    if stops:
        rows, tokens = zip(*stops)
        values[torch.tensor(rows, device=values.device), torch.tensor(tokens, device=values.device)] = -torch.inf
    return values


def batch_probabilities(values, params):
    """Scalar probabilities() semantics, including its joint support cutoff."""
    temperature, top_k, top_p, min_p = torch.tensor(
        [(p.temperature, p.top_k, p.top_p, p.min_p) for p in params],
        device=values.device, dtype=torch.float32,
    ).unbind(1)
    probs = (values.float() / temperature[:, None]).softmax(-1)
    ordered = probs.sort(descending=True).values
    k_index = (top_k.long() - 1).clamp(0, probs.shape[1] - 1)
    cutoff = torch.where(top_k < probs.shape[1], ordered.gather(1, k_index[:, None])[:, 0], 0.0)
    p_index = torch.searchsorted(ordered.cumsum(-1), top_p[:, None].contiguous()).clamp(max=probs.shape[1] - 1)
    cutoff = torch.maximum(cutoff, torch.where(top_p < 1, ordered.gather(1, p_index)[:, 0], 0.0))
    cutoff = torch.maximum(cutoff, ordered[:, 0] * min_p)
    probs = torch.where(probs >= cutoff[:, None], probs, 0)
    return probs / probs.sum(-1, keepdim=True)


def sample_uniform(probs, uniforms):
    cdf = probs.cumsum(-1)
    total = cdf[:, -1]
    threshold = torch.minimum(uniforms * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cdf.contiguous(), threshold[:, None].contiguous(), right=True)[:, 0]


def greedy_decisions(values, metadata):
    """Metadata columns: role (first/second/verify), first, second, enabled."""
    role, first, second, enabled = metadata.unbind(1)
    token = values.argmax(-1)
    origin = torch.where(token == first, 0, torch.where((enabled != 0) & (token == second), 1, -1))
    proposed = torch.where(role == 0, 1, (token != first).long())
    return torch.stack((token, torch.where(role == 2, origin, proposed)), -1)


@span("loopspec.decision", before=lambda logits, endpoints: {"endpoints": len(endpoints)})
def decide_endpoints(logits, endpoints):
    """One device packet, no host read. Owner loops only assemble metadata.

    Proposal views share an owned probability tensor (never a graph output).
    Their lifetime is bounded by the owning lane, at most R loop advances.
    Greedy proposals need no probability tensor at all.
    """
    contexts, lanes = zip(*endpoints)
    metadata, greedy, stochastic = [], [], []
    for i, (context, lane) in enumerate(endpoints):
        role = 2 if lane.depth == context.total else int(lane.depth != context.depths[0])
        first = lane.proposals[context.depths[0]][0] if role else 0
        second_depth = context.depths[1] if len(context.depths) == 2 else -1
        second = lane.proposals.get(second_depth)
        metadata.append((role, first, second[0] if second else 0, second is not None))
        (greedy if context.params.top_k == 1 or context.params.temperature == 0 else stochastic).append(i)
    values = processed_logits(logits, contexts, lanes)
    meta = torch.tensor(metadata, device=logits.device, dtype=torch.int64)
    distributions = [None] * len(endpoints)
    if not stochastic:
        return greedy_decisions(values, meta), distributions
    result = torch.empty((len(endpoints), 2), device=logits.device, dtype=torch.int64)
    if greedy:
        idx = torch.tensor(greedy, device=logits.device)
        result[idx] = greedy_decisions(values[idx], meta[idx])
    # Roles are a fixed number of batches, independent of the number of owners.
    for role in range(3):
        selected = [i for i in stochastic if metadata[i][0] == role]
        if not selected:
            continue
        idx = torch.tensor(selected, device=logits.device)
        group = [endpoints[i] for i in selected]
        probs = batch_probabilities(values[idx], [c.params for c, _ in group])
        uniforms = torch.stack([torch.rand(3, device=logits.device, generator=c.generator) for c, _ in group])
        first_tokens = meta[idx, 1]
        if role:
            first = torch.stack([l.proposals[c.depths[0]][1] for c, l in group])
        if role == 2:
            second = torch.zeros_like(first)
            enabled_rows = [(j, l.proposals[c.depths[1]][1]) for j, (c, l) in enumerate(group)
                            if len(c.depths) == 2 and c.depths[1] in l.proposals]
            if enabled_rows:
                indices, saved = zip(*enabled_rows)
                second[torch.tensor(indices, device=logits.device)] = torch.stack(saved)
            result[idx] = verify_tensor(probs, first, first_tokens, second, meta[idx, 2], meta[idx, 3].bool(), uniforms)
        else:
            enabled = torch.ones(len(selected), device=logits.device, dtype=torch.bool)
            if role == 1:
                delta = (probs - first).clamp_min(0)
                delta.scatter_(1, first_tokens[:, None], 0)
                mass = delta.sum(-1, keepdim=True)
                enabled = (probs.gather(1, first_tokens[:, None])[:, 0] < first.gather(1, first_tokens[:, None])[:, 0]) & (mass[:, 0] > 0)
                probs = torch.where(enabled[:, None], delta / mass.clamp_min(1e-30), probs)
            result[idx] = torch.stack((sample_uniform(probs, uniforms[:, 0]), enabled.long()), -1)
            for i, distribution in zip(selected, probs.unbind(0), strict=True):
                distributions[i] = distribution
    return result, distributions


def residual(target, proposal, exclude=None):
    delta = (target - proposal).clamp_min(0)
    if exclude is not None:
        delta[exclude] = 0
    mass = delta.sum()
    if mass <= 0:
        # Zero residual after a rejection is unreachable in exact arithmetic.
        # Keep the floating-point fallback finite, as in the reference.
        return target / target.sum()
    return delta / mass


def second_proposal(first, deeper, first_token):
    if deeper[first_token] >= first[first_token]:
        return None
    delta = (deeper - first).clamp_min(0)
    delta[first_token] = 0
    if delta.sum() <= 0:
        return None
    return delta / delta.sum()


def verify(target, first, first_token, second, second_token, generator=None):
    """Return (committed token, accepted proposal index or None)."""

    def accepts(p, q, token):
        ratio = (p[token] / q[token]).clamp(max=1)
        return bool(torch.rand((), device=p.device, generator=generator) < ratio)

    if accepts(target, first, first_token):
        return first_token, 0
    fallback = residual(target, first)
    if second is not None:
        if accepts(fallback, second, second_token):
            return second_token, 1
        fallback = residual(fallback, second)
    return int(torch.multinomial(fallback, 1, generator=generator)), None


def probabilities(logits, params, output_ids, stop_ids):
    """Apply SGLang's output-only penalties and joint top-k/top-p support."""
    logits = logits.float().clone()
    if params.logit_bias:
        for token, bias in params.logit_bias.items():
            logits[int(token)] += bias
    if output_ids:
        ids = torch.tensor(output_ids, dtype=torch.long, device=logits.device)
        counts = torch.bincount(ids, minlength=logits.numel())
        logits -= counts * params.frequency_penalty
        logits -= (counts > 0) * params.presence_penalty
        if params.repetition_penalty != 1:
            scale = torch.where(counts > 0, params.repetition_penalty, 1.0)
            logits = torch.where(logits < 0, logits * scale, logits / scale)
    if len(output_ids) < params.min_new_tokens and stop_ids:
        logits[list(stop_ids)] = -torch.inf
    if params.top_k == 1 or params.temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(), logits.numel()).float()
    probs = (logits / params.temperature).softmax(-1)
    sorted_probs = probs.sort(descending=True).values
    cutoff = torch.zeros((), device=probs.device)
    if params.top_k < probs.numel():
        cutoff = sorted_probs[params.top_k - 1]
    if params.top_p < 1:
        index = torch.searchsorted(sorted_probs.cumsum(0), params.top_p).clamp(
            max=probs.numel() - 1
        )
        cutoff = torch.maximum(cutoff, sorted_probs[index])
    cutoff = torch.maximum(cutoff, sorted_probs[0] * params.min_p)
    probs = torch.where(probs >= cutoff, probs, 0)
    return probs / probs.sum()


def verify_tensor(target, first, first_token, second, second_token, enabled, uniforms):
    """Branch-free verification, including a batch dimension for statistical tests.

    ``second`` is the saved *actual* gated residual proposal, never the deeper
    head's raw distribution. Three independent U[0,1) variates are supplied by
    the caller; no host decisions or RNG state are captured in the graph.
    """

    def normalized_residual(p, q):
        delta = (p - q).clamp_min(0)
        mass = delta.sum(-1, keepdim=True)
        return torch.where(mass > 0, delta / torch.where(mass > 0, mass, 1), p)

    def accepts(p, q, token, uniform):
        pt = p.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        qt = q.gather(-1, token.unsqueeze(-1)).squeeze(-1)
        ratio = pt / torch.where(qt > 0, qt, 1)
        return (qt > 0) & (uniform < ratio.clamp(max=1))

    accept_first = accepts(target, first, first_token, uniforms[..., 0])
    fallback = normalized_residual(target, first)
    accept_second = enabled & accepts(fallback, second, second_token, uniforms[..., 1])
    fallback = torch.where(
        enabled.unsqueeze(-1), normalized_residual(fallback, second), fallback
    )
    cdf = fallback.cumsum(-1)
    total = cdf[..., -1]
    # Rounding u*total up to total must not select a trailing zero-support bin.
    threshold = torch.minimum(
        uniforms[..., 2] * total, torch.nextafter(total, torch.zeros_like(total))
    )
    sampled = torch.searchsorted(
        cdf.contiguous(), threshold.unsqueeze(-1).contiguous(), right=True
    ).squeeze(-1)
    token = torch.where(
        accept_first, first_token, torch.where(accept_second, second_token, sampled)
    )
    origin = torch.where(accept_first, 0, torch.where(accept_second, 1, -1))
    return torch.stack((token, origin), dim=-1)


class RejectionGraph:
    """Fixed-vocabulary verifier graph; returned scalars never alias graph buffers."""

    def __init__(self, vocab_size, device):
        self.target = torch.full((vocab_size,), 1 / vocab_size, device=device)
        self.first = self.target.clone()
        self.second = torch.zeros_like(self.target)
        self.tokens = torch.zeros(2, device=device, dtype=torch.int64)
        self.enabled = torch.zeros((), device=device, dtype=torch.bool)
        self.uniforms = torch.zeros(3, device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = self._run()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self._run()
        torch.cuda.current_stream(device).wait_stream(stream)

    def _run(self):
        return verify_tensor(
            self.target,
            self.first,
            self.tokens[0],
            self.second,
            self.tokens[1],
            self.enabled,
            self.uniforms,
        )

    def replay(
        self,
        target,
        first,
        first_token,
        second,
        second_token,
        generator=None,
        *,
        uniforms=None,
    ):
        self.target.copy_(target)
        self.first.copy_(first)
        self.second.copy_(second) if second is not None else self.second.zero_()
        self.tokens.copy_(
            torch.tensor(
                [first_token, second_token if second is not None else 0],
                device=self.tokens.device,
            )
        )
        self.enabled.fill_(second is not None)
        if uniforms is None:
            self.uniforms.uniform_(generator=generator)
        else:
            self.uniforms.copy_(uniforms)
        self.graph.replay()
        token, origin = self.output.tolist()
        return token, None if origin < 0 else origin
