"""Native allocator ownership decisions, independent of device page numbering."""

import os

import numpy as np
import pytest
import torch
from test_ir import ROOT, args, cases, load_file

pytest_plugins = (
    "test_ir",
    "test_reposition_kernel",
    "test_context_attention_plan",
    "test_native_attention",
)


@pytest.fixture(scope="module")
def occurrence():
    return load_file(
        "context_ownership_occurrence",
        ROOT / "python/sglang/srt/context_system/occurrence.py",
    )


def expiry_for(n, drops):
    expiry = torch.full((n,), n + 1, dtype=torch.int32)
    for boundary, ranges in drops.items():
        for start, end in ranges:
            expiry[start:end].clamp_(max=boundary)
    return expiry


def test_materialization_ownership_and_positions(compiler, occurrence):
    rng = np.random.default_rng(1417)
    for tokens, drops, reposition in cases():
        n = len(tokens)
        start, end = n // 2, n
        layout = compiler(*args(tokens, drops, reposition))
        window = occurrence.compile_occurrence_window(
            layout,
            expiry_for(n, drops),
            layout.positions,
            query_start=start,
            query_end=end,
        )
        present = torch.ones(start, dtype=torch.bool)
        owned = torch.from_numpy(rng.random(start) < 0.3)
        terminal = torch.from_numpy(rng.random(start) < 0.3)
        exact = start // 2
        plan = occurrence.plan_occurrence_materialization(
            window,
            layout.positions[:start],
            present,
            owned,
            terminal,
            torch.ones(end, dtype=torch.bool),
            query_start=start,
            query_end=end,
            exact_prefix_len=exact,
        )
        canonical = torch.arange(1, n + 1, dtype=torch.int64)
        terminal_slots = torch.arange(n + 1, 2 * n + 1, dtype=torch.int64)
        allocated = torch.arange(2 * n + 1, 2 * n + 1 + plan.extra_page_count)
        slots, sources, destinations, pairs = plan.bind(
            canonical, terminal_slots, allocated
        )
        assert sources.dtype == destinations.dtype == pairs.dtype == torch.int32
        assert len(torch.unique(destinations)) == len(destinations)
        assert not set(sources.tolist()) & set(destinations.tolist())
        slot_positions = dict(zip(canonical.tolist(), layout.birth_positions.tolist()))
        slot_positions.update(
            zip(canonical[:start].tolist(), layout.positions[:start].tolist())
        )
        slot_positions.update(zip(terminal_slots.tolist(), layout.positions.tolist()))
        for src, dst, (old, new) in zip(
            sources.tolist(), destinations.tolist(), pairs.tolist()
        ):
            assert slot_positions[src] == old
            slot_positions[dst] = new
        required = set(window.segment_key_occurrences.tolist()) | set(
            window.terminal_occurrences.tolist()
        )
        for occ in required:
            assert slot_positions[int(slots[occ])] == int(
                window.occurrence_positions[occ]
            )
        for raw, occ in enumerate(plan.terminal_occurrences.tolist()):
            slot = int(slots[occ])
            if exact <= raw < start and not owned[raw] and not terminal[raw]:
                assert slot in allocated.tolist()
        # No unknown/unreferenced occurrence acquires a page.
        assert len(set(allocated.tolist())) == plan.extra_page_count


def test_skipped_hole_never_allocates_or_reads_padded_slot(compiler, occurrence):
    n, start = 12, 8
    drops = {start: [(0, 4)]}
    layout = compiler(*args(list(range(n)), drops, []))
    window = occurrence.compile_occurrence_window(
        layout,
        expiry_for(n, drops),
        layout.positions,
        query_start=start,
        query_end=n,
    )
    present = torch.ones(start, dtype=torch.bool)
    present[:4] = False
    keep = torch.ones(n, dtype=torch.bool)
    keep[:4] = False
    plan = occurrence.plan_occurrence_materialization(
        window,
        layout.positions[:start],
        present,
        torch.zeros_like(present),
        torch.zeros_like(present),
        keep,
        query_start=start,
        query_end=n,
        exact_prefix_len=start,
    )
    assert plan.extra_page_count == 0
    assert plan.read_cached[:4].tolist() == [False] * 4
    assert plan.terminal_occurrences[:4].tolist() == [-1] * 4
    present[6] = False
    with pytest.raises(ValueError, match="recovery required.*6"):
        occurrence.plan_occurrence_materialization(
            window,
            layout.positions[:start],
            present,
            torch.zeros_like(present),
            torch.zeros_like(present),
            keep,
            query_start=start,
            query_end=n,
            exact_prefix_len=start,
        )


def test_same_position_retry_copy_is_not_reposition_usage(compiler, occurrence):
    n, start = 12, 8
    layout = compiler(*args(list(range(n)), {}, []))
    window = occurrence.compile_occurrence_window(
        layout,
        torch.full((n,), n + 1, dtype=torch.int32),
        layout.positions,
        query_start=start,
        query_end=n,
    )
    yes = torch.ones(start, dtype=torch.bool)
    no = torch.zeros(start, dtype=torch.bool)
    plan = occurrence.plan_occurrence_materialization(
        window,
        layout.positions[:start],
        yes,
        no,
        no,
        torch.ones(n, dtype=torch.bool),
        query_start=start,
        query_end=n,
        exact_prefix_len=3,
    )
    assert plan.extra_page_count == 5
    assert torch.equal(plan.copy_position_pairs[:, 0], plan.copy_position_pairs[:, 1])
    assert not plan.repositioned_cached.any()
    assert plan.read_cached[:start].all()


@pytest.mark.skipif(os.environ.get("RUN_CONTEXT_GPU") != "1", reason="requires test GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_start", [0, 22, 70])
def test_materialization_copy_and_attention_chain(
    compiler, occurrence, kernels, attention_plan, native_attention, dtype, query_start
):
    """Exercise the planner's device bindings through actual copy and attention."""
    from test_planner import query_visibility

    mini_copy, actual_copy = kernels
    actual_attention, baseline_attention = native_attention
    n, heads, kv_heads, dim = 141, 4, 2, 64
    drops, repos = {60: [(0, 16)], 100: [(20, 40)]}, [59, 99]
    layout = compiler(*args(list(range(n)), drops, repos))
    visible, expiry = query_visibility(list(range(n)), drops, repos)
    window = occurrence.compile_occurrence_window(
        layout, expiry, layout.positions, query_start=query_start, query_end=n
    )
    ownership = occurrence.plan_occurrence_materialization(
        window,
        layout.positions[:query_start],
        torch.ones(query_start, dtype=torch.bool),
        torch.zeros(query_start, dtype=torch.bool),
        torch.zeros(query_start, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool),
        query_start=query_start,
        query_end=n,
        exact_prefix_len=0,
    )
    torch.manual_seed(847)
    pool_size = 2 * n + ownership.extra_page_count + 20
    pages = torch.randperm(pool_size - 1, device="cuda", dtype=torch.int64) + 1
    canonical = pages[:n]
    terminal = pages[n : 2 * n]
    allocated = pages[2 * n : 2 * n + ownership.extra_page_count]
    slots, source, destination, pairs = ownership.bind(canonical, terminal, allocated)
    k = torch.randn(1, pool_size, kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    oracle_k, oracle_v = k.clone(), v.clone()
    theta = torch.arange(256, device="cuda")[:, None] * torch.linspace(
        0.01, 1, dim // 2, device="cuda"
    )
    rope = torch.cat((theta.cos(), theta.sin()), dim=-1) * 1.37
    mini_copy(oracle_k, oracle_v, source, destination, pairs, rope)
    actual_copy(
        torch.tensor([k[0].data_ptr()], device="cuda", dtype=torch.uint64),
        torch.tensor([v[0].data_ptr()], device="cuda", dtype=torch.uint64),
        k[0], v[0], source, destination, pairs, rope,
    )
    assert torch.equal(k, oracle_k)
    assert torch.equal(v, oracle_v)
    metadata = attention_plan.ContextAttentionPlan.merge(
        [attention_plan.ContextSequence.from_window(window)]
    ).bind(slots)
    q = torch.randn(n - query_start, heads, dim, device="cuda", dtype=dtype)
    keys, values = k[0, canonical[query_start:]], v[0, canonical[query_start:]]
    result = metadata.forward(
        actual_attention, q, keys, values, torch.empty_like(q), k[0], v[0], page_size=1
    )
    # Independently enumerate each query's semantic visible set. The reference
    # kernel is the frozen native implementation, with no Context extensions.
    pair_ids = dict(enumerate(zip(
        window.occurrence_raw_tokens.tolist(), window.occurrence_positions.tolist()
    )))
    pair_to_occurrence = {pair: ident for ident, pair in pair_ids.items()}
    selected, offsets = [], [0]
    for demand in visible[query_start:]:
        selected.extend(pair_to_occurrence[pair] for pair in demand[:-1])
        offsets.append(len(selected))
    oracle = torch.empty_like(q)
    baseline_attention(
        q, keys, values, oracle, oracle_k[0], oracle_v[0],
        torch.arange(len(q) + 1, device="cuda", dtype=torch.int64),
        torch.tensor(offsets, device="cuda"),
        slots[torch.tensor(selected, device="cuda", dtype=torch.int64)],
        None, True, None, 1, 1.0, 1.0, page_size=1,
        extend_seq_lens_cpu=[1] * len(q),
    )
    torch.testing.assert_close(
        result, oracle, rtol=torch.finfo(dtype).eps, atol=torch.finfo(dtype).eps
    )
