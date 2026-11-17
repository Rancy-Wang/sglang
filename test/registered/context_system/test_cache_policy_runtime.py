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
    "test_model_forward",
)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("neox", [False, True])
def test_model_oracle_rotation_against_unrotated_complex_attention(policy, neox):
    """Check the independent oracle before trusting its future GPU comparisons."""
    from types import SimpleNamespace

    from cache_policy_reference import reference_attention_forward

    tokens = list(range(12))
    reference = simulate(
        tokens,
        {5: [(1, 3)], 9: [(4, 6)]},
        [10] if policy == POLICIES[0] else [],
        policy=policy,
    )
    projection = reference.active if policy == POLICIES[0] else tokens
    inverse = {raw: i for i, raw in enumerate(projection)}
    n = len(projection)
    torch.manual_seed(17)
    q, k, v = torch.randn(n, 4, 8), torch.randn(n, 2, 8), torch.randn(n, 2, 8)
    angles = torch.arange(32)[:, None] * torch.tensor([0.1, 0.03])[None, :]
    # Non-unit amplitude exercises scaled RoPE; preserve the non-rotary tail.
    cache = 1.2 * torch.cat((angles.cos(), angles.sin()), -1)
    rope = SimpleNamespace(cos_sin_cache=cache, rotary_dim=4, is_neox_style=neox)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=8,
        v_head_dim=8,
        sliding_window_size=3,
        scaling=8**-0.5,
        logit_cap=0,
    )
    model = SimpleNamespace(
        modules=lambda: [SimpleNamespace(attn=layer, rotary_emb=rope)]
    )
    births = [reference.compute_positions[i] for i in projection]

    def rotate(x, positions):
        z = x.double().clone()
        left, right = (
            (z[..., :2], z[..., 2:4]) if neox else (z[..., :4:2], z[..., 1:4:2])
        )
        phase = torch.polar(
            torch.full_like(angles[positions], 1.2).double(), angles[positions].double()
        )[:, None]
        value = torch.complex(left, right) * phase
        if neox:
            z[..., :4] = torch.cat((value.real, value.imag), -1)
        else:
            z[..., :4] = torch.stack((value.real, value.imag), -1).flatten(-2)
        return z

    born_q, born_k = rotate(q, births).float(), rotate(k, births).float()
    actual = reference_attention_forward(model, reference, projection)(
        layer, born_q, born_k, v, SimpleNamespace(positions=torch.tensor(births))
    )
    expected = []
    for i, raw in enumerate(projection):
        chosen = [
            (inverse[j], pos)
            for j, pos in reference.readers[raw]
            if pos >= births[i] - 3
        ][-4:]
        ids, positions = zip(*chosen)
        keys = rotate(k[list(ids)], list(positions)).repeat_interleave(2, 1)
        scores = (
            torch.einsum("hd,khd->hk", rotate(q[i : i + 1], [births[i]])[0], keys)
            * layer.scaling
        )
        expected.append(
            torch.einsum(
                "hk,khd->hd",
                scores.softmax(-1),
                v[list(ids)].double().repeat_interleave(2, 1),
            )
        )
    torch.testing.assert_close(
        actual.double(), torch.stack(expected).reshape(n, -1), rtol=1e-5, atol=1e-6
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


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_MODEL_PATH"), reason="isolated real model required"
)
@pytest.mark.parametrize("policy", POLICIES)
@torch.no_grad()
def test_real_model_logits_against_independent_fp32_attention(
    runtime, policy_modules, policy
):
    """No candidate compiler, cache or occurrence plan is used by the oracle."""
    from unittest.mock import patch

    from cache_policy_reference import reference_attention_forward
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from test_model_forward import forward, prepare_batch

    wrapper, tokenizer, _ = runtime
    runner = wrapper.torch_runner
    tokens = tokenizer.encode("The librarian kept blue notebooks and red maps. " * 20)[
        :96
    ]
    assert len(tokens) == 96

    def pair(drops, repos):
        ref = simulate(tokens, drops, repos, policy=policy)
        projection = ref.active if policy == POLICIES[0] else list(range(len(tokens)))
        wrapper.clear()
        batch = prepare_batch(runner, [[tokens[i] for i in projection]])
        fb = ForwardBatch.init_new(
            batch, runner, return_hidden_states_before_norm=False
        )
        fb.positions = torch.tensor(
            [ref.compute_positions[i] for i in projection],
            dtype=fb.positions.dtype,
            device=fb.positions.device,
        )
        with (
            patch.object(
                RadixAttention,
                "forward",
                reference_attention_forward(runner.model, ref, projection),
            ),
            patch.object(runner, "prefill_cuda_graph_runner", None),
        ):
            expected = runner.forward(fb).logits_output.next_token_logits.clone()
        wrapper.clear()
        program = compile_policy(policy_modules, policy, tokens, drops, repos)
        actual = forward(
            runner, prepare_batch(runner, [program.input_ids], programs=[program])
        )
        delta = (actual.float() - expected.float()).abs()
        return {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "p99_abs": float(torch.quantile(delta.flatten(), 0.99)),
        }

    baseline = pair({}, [])
    limits = {
        name: max(floor, 2 * baseline[name])
        for name, floor in (("max_abs", 0.125), ("mean_abs", 0.02), ("p99_abs", 0.0625))
    }
    measured = pair(
        {32: [(3, 17)], 65: [(21, 38)], 90: [(40, 51)]},
        [50, 91] if policy == POLICIES[0] else [],
    )
    print(
        "CACHE_POLICY_MODEL_ORACLE",
        {
            "policy": policy,
            "baseline": baseline,
            "limits": limits,
            "measured": measured,
        },
        flush=True,
    )
    assert all(measured[name] <= limits[name] for name in limits)
