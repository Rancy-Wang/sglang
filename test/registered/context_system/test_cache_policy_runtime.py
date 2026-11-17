"""Actual cache allocator and CUDA attention checks; opt-in runtime coverage."""

import importlib
import os

import pytest
import torch
from cache_policy_reference import simulate
from test_cache_policy_ir import POLICIES, compile_policy

pytest_plugins = (
    "test_cache_policy_ir",
    "test_native_attention",
    "test_context_attention_plan",
    "test_context_transfer",
    "test_drop_eviction",
)


@pytest.mark.parametrize("policy", POLICIES)
def test_real_tree_exact_or_rotate_and_hole_metadata(
    native_cache, policy_modules, policy, monkeypatch
):
    from sglang.srt.context_system import retry
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    p = compile_policy(policy_modules, policy, list(range(32)), {20: [(4, 8)]})
    key = RadixKey.from_context(p.layout)
    values = allocator.alloc(len(key))
    cache.insert(InsertParams(key=key, value=values))
    if policy == POLICIES[1]:

        def forbidden(*args, **kwargs):
            raise AssertionError("B must not visit the retrigger index")

        monkeypatch.setattr(retry, "longest_compatible_prefix", forbidden)
    hit = cache.match_prefix(
        MatchPrefixParams(
            key=key, context_retry=p.allow_retry, need_context_metadata=True
        )
    )
    assert len(hit.device_indices) == len(key)
    assert hit.context_resident.all()
    assert torch.equal(hit.context_source_positions, p.layout.positions)
    assert not hit.context_retry
    if policy == POLICIES[0]:
        rotated = compile_policy(
            policy_modules, policy, list(range(32)), {20: [(4, 8)]}, [32]
        )
        target = RadixKey.from_context(rotated.layout)
        reused = cache.match_prefix(
            MatchPrefixParams(
                key=target, context_retry=True, need_context_metadata=True
            )
        )
        assert len(reused.device_indices) == len(target)
        assert reused.context_retry
        # Read-only matching must not mutate the cached source's positions.
        assert list(key.context.positions) == hit.context_source_positions.tolist()
    cache.sanity_check()


@pytest.mark.parametrize("policy", POLICIES)
def test_pd_transfer_sends_only_active_identity(
    policy_modules, transfer_modules, policy
):
    transfer, _ = transfer_modules
    p = compile_policy(
        policy_modules,
        policy,
        list(range(16)),
        {12: [(2, 7)]},
        [16] if policy == POLICIES[0] else [],
    )
    view = transfer.ContextTransferPlan.build(p, "cpu")
    assert view.active_count == 11
    assert view.decode.positions.tolist() == list(range(11))
    assert view.decode.raw_indices.tolist() == (
        list(range(11)) if policy == POLICIES[0] else [0, 1, *range(7, 16)]
    )
    restored = policy_modules[2].ContextProgram.from_wire(p.to_json_wire(), p.input_ids)
    assert (
        transfer.ContextTransferPlan.build(restored, "cpu").signature == view.signature
    )


@pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1", reason="isolated CUDA device required"
)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("window_size", [-1, 0, 16])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_real_cuda_attention_against_independent_event_reads(
    policy_modules,
    attention_plan,
    native_attention,
    policy,
    window_size,
    kv_heads,
    monkeypatch,
):
    import sys

    occurrence = importlib.import_module(policy_modules[2].__package__ + ".occurrence")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.context_system.window",
        importlib.import_module(policy_modules[2].__package__ + ".window"),
    )
    tokens = list(range(96))
    drops = {32: [(3, 17)], 65: [(21, 38)], 90: [(40, 51)]}
    repos = [50, 91] if policy == POLICIES[0] else []
    reference = simulate(tokens, drops, repos, policy=policy)
    p = compile_policy(policy_modules, policy, tokens, drops, repos)
    projection = p.execution_to_full.tolist()
    inverse = {raw: i for i, raw in enumerate(projection)}
    rows = [
        [(inverse[j], position) for j, position in reference.readers[i]]
        for i in projection
    ]
    w = occurrence.compile_occurrence_window(
        p.layout,
        p.visible_until,
        p.layout.positions,
        query_start=0,
        query_end=len(p.input_ids),
    )
    plan = attention_plan.ContextAttentionPlan.merge(
        [attention_plan.ContextSequence.from_window(w)]
    )
    torch.manual_seed(1711)
    slots = torch.randperm(plan.occurrence_count + 32, device="cuda")[
        : plan.occurrence_count
    ]
    metadata = plan.bind(slots)
    pool_k = torch.randn(
        plan.occurrence_count + 32, kv_heads, 64, device="cuda", dtype=torch.bfloat16
    )
    pool_v = torch.randn_like(pool_k)
    q = torch.randn(len(projection), 4, 64, device="cuda", dtype=torch.bfloat16)
    pairs = list(zip(w.occurrence_raw_tokens.tolist(), w.occurrence_positions.tolist()))
    index = {pair: i for i, pair in enumerate(pairs)}
    births = [
        index[(i, reference.compute_positions[raw])] for i, raw in enumerate(projection)
    ]
    k, v = pool_k[slots[births]], pool_v[slots[births]]
    actual, frozen = native_attention
    output = metadata.forward(
        actual,
        q,
        k,
        v,
        torch.empty_like(q),
        pool_k,
        pool_v,
        birth_slots=slots[births],
        page_size=1,
        sliding_window_size=window_size,
    )
    prefix, offsets = [], [0]
    for row in rows:
        query_position = row[-1][1]
        chosen = [
            pair
            for pair in row
            if window_size < 0 or pair[1] >= query_position - window_size
        ]
        if window_size >= 0:
            chosen = chosen[-(window_size + 1) :]
        prefix.extend(index[pair] for pair in chosen if pair != row[-1])
        offsets.append(len(prefix))
    expected = torch.empty_like(q)
    frozen(
        q,
        k,
        v,
        expected,
        pool_k,
        pool_v,
        torch.arange(len(q) + 1, device="cuda", dtype=torch.int64),
        torch.tensor(offsets, device="cuda"),
        slots[torch.tensor(prefix, device="cuda", dtype=torch.long)],
        None,
        True,
        None,
        1,
        1.0,
        1.0,
        page_size=1,
        extend_seq_lens_cpu=[1] * len(q),
    )
    torch.testing.assert_close(output, expected, rtol=0.0078125, atol=0.0078125)
