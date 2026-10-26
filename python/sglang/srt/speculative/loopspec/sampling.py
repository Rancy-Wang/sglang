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
