"""Fixed-input decisions, owner mapping, and distribution oracles.

Run on CPU and, when available, CUDA. Statistical checks use 100,000 samples
per distribution, a fixed seed and max(5 sigma, 5/n) per-bin bounds.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.loopspec.sampling import (
    batch_probabilities, decide_endpoints, probabilities, processed_logits,
    sample_uniform, verify_tensor,
)

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def params(**changes):
    values = dict(temperature=0., top_k=32768, top_p=1., min_p=0.,
                  frequency_penalty=0., presence_penalty=0., repetition_penalty=1.,
                  logit_bias=None, min_new_tokens=0)
    values.update(changes)
    return NS(**values)


def endpoint(i, device, param=None, depth=None, double=True):
    context = NS(params=param or params(), stop_ids={0, 6}, total=4,
                 depths=(1, 2) if double else (1,),
                 generator=torch.Generator(device=device).manual_seed(1700+i))
    depth = depth or (1, 2, 4)[i % 3]
    proposals = {} if depth == 1 else {1: (i % 7, None)}
    if depth == 4 and double and i % 2:
        proposals[2] = ((i+1) % 7, None)
    lane = NS(uid=i, depth=depth, history=(1, 1, 2, 3), proposals=proposals)
    return context, lane


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('count', [1, 2, 4, 8, 16, 32, 64, 112])
def test_greedy_exact_roles_and_mixed_policies(device, count):
    torch.manual_seed(901)
    logits = torch.randn(count, 7, device=device, dtype=torch.bfloat16)
    logits[0].zero_()  # Tie-breaking must retain the first surviving token.
    ends = []
    for i in range(count):
        p = params() if i % 4 == 0 else params(
            frequency_penalty=.2*(i%3), presence_penalty=.3, repetition_penalty=1.1,
            logit_bias={4: .7}, min_new_tokens=9 if i%2 else 0,
            temperature=1. if i%2 else 0., top_k=1)
        ends.append(endpoint(i, device, p))
    expected = []
    for row, (c, l) in zip(logits, ends):
        token = int(probabilities(row, c.params, l.history, c.stop_ids).argmax())
        if l.depth == 1:
            tag = 1
        elif l.depth == 2:
            tag = int(token != l.proposals[1][0])
        else:
            tag = next((j for j, d in enumerate(c.depths)
                        if d in l.proposals and token == l.proposals[d][0]), -1)
        expected.append([token, tag])
    with patch('torch.multinomial', side_effect=AssertionError('greedy sampled')):
        packet, saved = decide_endpoints(logits, ends)
    assert packet.tolist() == expected
    assert saved == [None]*count
    permutation = list(reversed(range(count)))
    reordered, _ = decide_endpoints(logits[permutation], [ends[i] for i in permutation])
    assert reordered.tolist() == list(reversed(expected))


@pytest.mark.parametrize('device', DEVICES)
def test_probabilities_match_scalar_reference(device):
    torch.manual_seed(65)
    logits = torch.randn(12, 7, device=device)
    ends = [endpoint(i, device, params(temperature=.6+i*.13,
        top_k=3 if i%2 else 32768, top_p=.75 if i%3 else 1., min_p=.05,
        frequency_penalty=.2, presence_penalty=.3, repetition_penalty=1.2,
        logit_bias={4: .8}, min_new_tokens=9)) for i in range(12)]
    c, l = zip(*ends)
    actual = batch_probabilities(processed_logits(logits, c, l), [x.params for x in c])
    expected = torch.stack([probabilities(row, x.params, y.history, x.stop_ids)
                            for row, (x, y) in zip(logits, ends)])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('device', DEVICES)
def test_mixed_sampling_owner_proposals_and_rng(device):
    torch.manual_seed(74)
    logits = torch.randn(18, 7, device=device)
    ends = [endpoint(i, device, params(temperature=0. if i%4==0 else .8)) for i in range(18)]
    for c, l in ends:
        if c.params.temperature:
            l.proposals = {d: (token, torch.softmax(torch.arange(7, device=device).float() * (d+.2), 0))
                           for d, (token, _) in l.proposals.items()}
    states = [c.generator.get_state().clone() for c, _ in ends]
    packet, saved = decide_endpoints(logits, ends)
    snapshot = [p.clone() if p is not None else None for p in saved]
    for c, state in zip((c for c, _ in ends), states):
        c.generator.set_state(state)
    repeated, _ = decide_endpoints(logits, ends)
    assert torch.equal(packet, repeated)
    for i, ((c, l), distribution) in enumerate(zip(ends, saved)):
        if not c.params.temperature:
            assert torch.equal(states[i], c.generator.get_state())
        if distribution is not None:
            torch.testing.assert_close(distribution, snapshot[i], rtol=0, atol=0)
            assert abs(float(distribution.sum())-1) < 1e-6
            assert distribution[int(packet[i, 0])] > 0
        else:
            assert l.depth == 4 or not c.params.temperature


@pytest.mark.parametrize('device', DEVICES)
def test_residual_statistics_and_batched_verify_oracle(device):
    torch.manual_seed(11907)
    n = 100000
    for p, q in [([.05,.15,.8],[.8,.1,.1]), ([0,.4,.6],[.9,.1,0]), ([.2,.3,.5],[.2,.3,.5])]:
        target = torch.tensor(p, device=device).expand(n,-1)
        first = torch.tensor(q, device=device).expand(n,-1)
        token = sample_uniform(first, torch.rand(n, device=device))
        uniform = torch.rand(n,3,device=device)
        second = torch.zeros_like(first)
        result = verify_tensor(target,first,token,second,torch.zeros_like(token),
                               torch.zeros_like(token,dtype=torch.bool),uniform)
        # Independent scalar probability formula for the complete target law.
        counts = torch.bincount(result[:,0],minlength=3).float()/n
        expected = torch.tensor(p,device=device)
        tolerance = torch.maximum(5*(expected*(1-expected)/n).sqrt(),torch.tensor(5/n,device=device))
        assert torch.all((counts-expected).abs() <= tolerance), (counts, expected)
        for i in range(12):
            single = verify_tensor(target[i],first[i],token[i],second[i],torch.tensor(0,device=device),
                                   torch.tensor(False,device=device),uniform[i])
            assert torch.equal(single,result[i])


@pytest.mark.parametrize('device', DEVICES)
def test_draft_gating_and_target_rejection_in_greedy(device):
    # First=2, second=4: target hits first, hits second, or neither.
    ends = [endpoint(i, device, depth=4) for i in range(3)]
    for _, lane in ends:
        lane.proposals = {1:(2,None),2:(4,None)}
    logits = torch.zeros(3,7,device=device)
    logits[range(3),[2,4,6]] = 3
    packet, _ = decide_endpoints(logits,ends)
    assert packet.tolist() == [[2,0],[4,1],[6,-1]]
