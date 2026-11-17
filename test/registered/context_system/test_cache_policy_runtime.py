"""Actual cache allocator and CUDA attention checks; opt-in runtime coverage."""

import importlib
import os
import random
from itertools import pairwise

import pytest
import torch
from cache_policy_reference import simulate, simulate_compact
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
@pytest.mark.parametrize("window", [-1, 0, 3])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_model_oracle_rotation_against_unrotated_complex_attention(
    policy, neox, window, with_sinks
):
    """Check the independent oracle before trusting its future GPU comparisons."""
    from types import SimpleNamespace

    from cache_policy_reference import (
        chunked_reference_attention_forward,
        reference_attention_forward,
    )

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
        sliding_window_size=window,
        scaling=8**-0.5,
        logit_cap=1.7 if with_sinks else 0,
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
    extra = {"sinks": torch.randn(4)} if with_sinks else {}
    actual = reference_attention_forward(model, reference, projection)(
        layer,
        born_q,
        born_k,
        v,
        SimpleNamespace(positions=torch.tensor(births)),
        **extra,
    )
    compact = simulate_compact(
        len(tokens),
        {5: [(1, 3)], 9: [(4, 6)]},
        [10] if policy == POLICIES[0] else [],
        policy=policy,
    )
    blocked = chunked_reference_attention_forward(
        model, compact, query_block=3, key_block=2
    )(
        layer,
        born_q,
        born_k,
        v,
        SimpleNamespace(positions=torch.tensor(births)),
        **extra,
    )
    torch.testing.assert_close(blocked, actual, rtol=1e-5, atol=1e-6)
    streaming = chunked_reference_attention_forward(
        model, compact, query_block=3, key_block=2, streaming=True
    )
    pieces = []
    # Cross event boundaries, then finish with one-token decode-shaped calls.
    cuts = sorted({0, min(3, n), min(7, n), n - 1, n})
    for start, end in pairwise(cuts):
        pieces.append(
            streaming(
                layer,
                born_q[start:end],
                born_k[start:end],
                v[start:end],
                SimpleNamespace(positions=torch.tensor(births[start:end])),
                **extra,
            )
        )
    torch.testing.assert_close(torch.cat(pieces), actual, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="consecutive token chunks"):
        streaming(layer, born_q[:1], born_k[:1], v[:1], None)
    expected = []
    for i, raw in enumerate(projection):
        chosen = [
            (inverse[j], pos)
            for j, pos in reference.readers[raw]
            if window < 0 or pos >= births[i] - window
        ]
        if window >= 0:
            chosen = chosen[-(window + 1) :]
        ids, positions = zip(*chosen)
        keys = rotate(k[list(ids)], list(positions)).repeat_interleave(2, 1)
        scores = (
            torch.einsum("hd,khd->hk", rotate(q[i : i + 1], [births[i]])[0], keys)
            * layer.scaling
        )
        if layer.logit_cap:
            scores = layer.logit_cap * torch.tanh(scores / layer.logit_cap)
        if with_sinks:
            scores = torch.cat((scores, extra["sinks"].double()[:, None]), -1)
        weights = scores.softmax(-1)
        if with_sinks:
            weights = weights[:, :-1]
        expected.append(
            torch.einsum(
                "hk,khd->hd",
                weights,
                v[list(ids)].double().repeat_interleave(2, 1),
            )
        )
    torch.testing.assert_close(
        actual.double(), torch.stack(expected).reshape(n, -1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_compact_reference_matches_sequential_reader_graph(policy):
    rng = random.Random(17112026)
    for _ in range(100):
        n = rng.randrange(3, 50)
        drops = {
            b: [(i, i + 1) for i in range(b - 1) if rng.random() < 0.2]
            for b in range(2, n + 1)
            if rng.random() < 0.25
        }
        repos = (
            [b for b in range(2, n + 1) if rng.random() < 0.25]
            if policy == POLICIES[0]
            else []
        )
        slow = simulate(list(range(n)), drops, repos, policy=policy)
        compact = simulate_compact(n, drops, repos, policy=policy)
        raw = compact.raw.tolist()
        assert raw == (slow.active if policy == POLICIES[0] else list(range(n)))
        assert compact.positions.tolist() == [slow.positions[i] for i in raw]
        assert compact.compute_positions.tolist() == [
            slow.compute_positions[i] for i in raw
        ]
        assert compact.next_position == slow.next_position
        for query, i in enumerate(raw):
            for window in (-1, 0, 4):
                expected = [
                    (j, position)
                    for j, position in slow.readers[i]
                    if window < 0 or position >= slow.compute_positions[i] - window
                ]
                if window >= 0:
                    expected = expected[-(window + 1) :]
                assert [(raw[j], p) for j, p in compact.row(query, window)] == expected


@pytest.mark.parametrize("policy", POLICIES)
def test_long_reference_keeps_linear_state_and_exact_boundary_reads(policy):
    # Synthetic 128K geometry, explicitly not a real BCP/model acceptance run.
    n = 128 * 1024
    drops = {32768: [(100, 8192)], 98304: [(16384, 24576)]}
    repos = [98304] if policy == POLICIES[0] else []
    compact = simulate_compact(n, drops, repos, policy=policy)
    arrays = (compact.raw, compact.positions, compact.compute_positions, compact.expiry)
    assert sum(a.nbytes for a in arrays) <= 4 * n * 8
    assert compact.next_position == n - (8192 - 100) - 8192
    last = compact.row(len(compact.raw) - 1)
    assert len(last) == compact.next_position
    assert [p for _, p in last] == list(range(compact.next_position))
    # The final row excludes dead history even when it remains on the B path.
    assert not any(
        100 <= compact.raw[j] < 8192 or 16384 <= compact.raw[j] < 24576 for j, _ in last
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
# Context recovery requires positive SWA windows. Native Triton does not
# interpret zero as self-only attention (the bounded mathematical oracle can).
@pytest.mark.parametrize("window_size", [-1, 1, 16])
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


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_window_is_rejected_by_context_recovery(policy_modules, policy):
    recovery = importlib.import_module(policy_modules[2].__package__ + ".recovery")
    p = compile_policy(policy_modules, policy, list(range(8)), {5: [(1, 3)]})
    with pytest.raises(ValueError, match="window must be positive"):
        recovery.sliding_window_starts(p.layout, p.visible_until, 0)


@pytest.fixture(scope="module")
def cache_policy_model_runtime():
    """Isolated TP fixture; torchrun supplies a consistent rank/port on all GPUs."""
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    port = os.environ.get("CONTEXT_TEST_NCCL_PORT")
    if world > 1 and port is None:
        raise ValueError("TP validation requires CONTEXT_TEST_NCCL_PORT on every rank")
    args = ServerArgs(
        model_path=os.environ["CONTEXT_MODEL_PATH"],
        tp_size=world,
        nccl_port=int(port) if port else None,
        page_size=1,
        max_total_tokens=4096,
        max_running_requests=8,
        context_length=2048,
        mem_fraction_static=0.85,
        attention_backend=os.environ.get("CONTEXT_TEST_ATTENTION_BACKEND", "triton"),
        # Match the frozen BCP server backend. Auto can select triton_kernels,
        # whose projection/MoE rounding varies substantially with tiny chunks.
        moe_runner_backend="triton",
        enable_deterministic_inference=os.environ.get("CONTEXT_TEST_DETERMINISTIC")
        == "1",
        cuda_graph_config={
            "decode": {"backend": "disabled"},
            "prefill": {"backend": "disabled"},
        },
    )
    args.resolve_once()
    _set_envs_and_config(args)
    publish(args, role="scheduler")
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    wrapper, tokenizer = load_model(args, PortArgs.init_new(args), local_rank, rank)
    if os.environ.get("CONTEXT_TEST_DETERMINISTIC") == "1":
        from sglang.srt.batch_invariant_ops import (
            disable_batch_invariant_mode,
            enable_batch_invariant_mode,
        )

        # Keep deterministic model GEMMs, but use native FP32 BMM in the
        # independent oracle: the invariant BMM kernel defaults to TF32 dots.
        disable_batch_invariant_mode()
        enable_batch_invariant_mode(enable_bmm=False)
    runner = wrapper.torch_runner
    binding = ContextModelBinding(
        runner.model, runner.token_to_kv_pool, runner.kv_index_translator, page_size=1
    )
    runner.context_model_binding = binding
    print("CACHE_POLICY_MODEL", args.model_path, world, rank, flush=True)
    yield wrapper, tokenizer, binding


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_MODEL_PATH"), reason="isolated real model required"
)
@pytest.mark.parametrize("policy", POLICIES)
@torch.no_grad()
def test_real_model_logits_against_independent_fp32_attention(
    cache_policy_model_runtime, policy_modules, policy
):
    """No candidate compiler, cache or occurrence plan is used by the oracle."""
    from unittest.mock import patch

    from cache_policy_reference import reference_attention_forward
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from test_model_forward import forward, prepare_batch

    wrapper, tokenizer, _ = cache_policy_model_runtime
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


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_MODEL_PATH"), reason="isolated real model required"
)
@pytest.mark.skipif(
    os.environ.get("CONTEXT_TEST_DETERMINISTIC") != "1",
    reason="chunk invariance requires the model's batch-invariant GEMM mode",
)
@pytest.mark.parametrize("policy", POLICIES)
def test_real_model_streaming_reference_chunk_invariance(
    cache_policy_model_runtime, policy
):
    from cache_policy_reference import run_reference_model

    wrapper, tokenizer, _ = cache_policy_model_runtime
    tokens = tokenizer.encode("The librarian kept blue notebooks and red maps. " * 20)[
        :96
    ]
    # Freeze continuation independently of either model result.
    continuation = tokenizer.encode(" A blue notebook.")[:3]
    drops = {32: [(3, 17)], 65: [(21, 38)], 90: [(40, 51)]}
    repos = [50, 91] if policy == POLICIES[0] else []
    no_event = [
        run_reference_model(
            wrapper,
            tokens,
            {},
            [],
            policy=policy,
            continuation=continuation,
            chunk_tokens=size,
        )
        for size in (128, 17)
    ]
    no_event_delta = (no_event[0].float() - no_event[1].float()).abs()
    print(
        "REFERENCE_NO_EVENT_CHUNKS",
        policy,
        no_event_delta.max(-1).values.tolist(),
        no_event_delta.mean(-1).tolist(),
        flush=True,
    )
    audits = [{}, {}]
    complete = run_reference_model(
        wrapper,
        tokens,
        drops,
        repos,
        policy=policy,
        continuation=continuation,
        chunk_tokens=128,
        audit=audits[0],
    )
    chunked = run_reference_model(
        wrapper,
        tokens,
        drops,
        repos,
        policy=policy,
        continuation=continuation,
        chunk_tokens=17,
        audit=audits[1],
    )
    assert (
        complete.shape == chunked.shape == (len(continuation) + 1, complete.shape[-1])
    )
    assert torch.isfinite(chunked).all() and torch.isfinite(complete).all()
    delta = (complete.float() - chunked.float()).abs()
    if int(os.environ.get("RANK", "0")) == 0:
        for key in sorted(audits[0]):
            errors = {
                name: float((value.float() - audits[1][key][name].float()).abs().max())
                for name, value in audits[0][key].items()
            }
            print("REFERENCE_CHUNK_LAYER", policy, key, errors, flush=True)
    print(
        "REFERENCE_CHUNK_ROWS",
        policy,
        delta.max(-1).values.tolist(),
        delta.mean(-1).tolist(),
        flush=True,
    )
    print(
        "REFERENCE_CHUNK_INVARIANCE",
        policy,
        float(delta.max()),
        float(delta.mean()),
        flush=True,
    )
    assert float(delta.max()) <= 0.125
    assert float(delta.mean()) <= 0.02
