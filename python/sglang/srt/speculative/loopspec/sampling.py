"""Saved proposals and lossless residual verification for LoopSpec."""

import torch


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
