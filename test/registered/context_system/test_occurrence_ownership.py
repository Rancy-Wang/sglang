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


@pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1", reason="requires test GPU"
)
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
        k[0],
        v[0],
        source,
        destination,
        pairs,
        rope,
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
    pair_ids = dict(
        enumerate(
            zip(
                window.occurrence_raw_tokens.tolist(),
                window.occurrence_positions.tolist(),
            )
        )
    )
    pair_to_occurrence = {pair: ident for ident, pair in pair_ids.items()}
    selected, offsets = [], [0]
    for demand in visible[query_start:]:
        selected.extend(pair_to_occurrence[pair] for pair in demand[:-1])
        offsets.append(len(selected))
    oracle = torch.empty_like(q)
    baseline_attention(
        q,
        keys,
        values,
        oracle,
        oracle_k[0],
        oracle_v[0],
        torch.arange(len(q) + 1, device="cuda", dtype=torch.int64),
        torch.tensor(offsets, device="cuda"),
        slots[torch.tensor(selected, device="cuda", dtype=torch.int64)],
        None,
        True,
        None,
        1,
        1.0,
        1.0,
        page_size=1,
        extend_seq_lens_cpu=[1] * len(q),
    )
    torch.testing.assert_close(
        result, oracle, rtol=torch.finfo(dtype).eps, atol=torch.finfo(dtype).eps
    )


def test_chunk_lifetime_publication_and_deferred_retirement(compiler, occurrence):
    """Recycle slots after each GPU boundary; alias bugs become stale reads."""
    rng = np.random.default_rng(1021)
    for tokens, drops, repositions in cases():
        n = len(tokens)
        layout = compiler(*args(tokens, drops, repositions))
        expiry = expiry_for(n, drops)
        start = int(rng.integers(0, n))
        # Prefix comes from a separate live source lease, possibly a Retry.
        sources = torch.arange(1, start + 1, dtype=torch.int64)
        state = occurrence.OccurrenceState.from_match(
            sources, layout.positions[:start], exact_prefix_len=start // 2
        )
        values = {
            int(slot): (raw, int(layout.positions[raw]))
            for raw, slot in enumerate(sources)
        }
        cache_slots = set(sources.tolist())
        private = set()
        available = list(range(start + 1, 16 * n + 1))
        pending = []

        def alloc(count, available=available, private=private):
            selected = available[:count]
            del available[:count]
            private.update(selected)
            return torch.tensor(selected, dtype=torch.int64)

        def free(
            slots,
            private=private,
            cache_slots=cache_slots,
            available=available,
            values=values,
        ):
            items = slots.tolist()
            assert len(set(items)) == len(items)
            assert set(items) <= private
            assert not set(items) & cache_slots
            private.difference_update(items)
            # Immediate reuse is intentional: stale references fail deterministically.
            available[:0] = items
            for item in items:
                values.pop(item, None)

        while start < n:
            end = min(n, start + int(rng.integers(1, 9)))
            window = occurrence.compile_occurrence_window(
                layout, expiry, layout.positions, query_start=start, query_end=end
            )
            plan = state.plan(window, torch.ones(end, dtype=torch.bool))
            births, extra = alloc(end - start), alloc(plan.extra_page_count)
            step = state.advance(window, plan, births, extra, expiry)
            # Layer writes birth KV, then all independent canonical-source copies.
            for raw, slot in zip(range(start, end), births.tolist()):
                values[slot] = (raw, int(layout.birth_positions[raw]))
            updates = {}
            for src, dst, (old, new) in zip(
                step.copy_source_slots.tolist(),
                step.copy_destination_slots.tolist(),
                step.copy_position_pairs.tolist(),
            ):
                raw, pos = values[src]
                assert pos == old
                assert dst in private and dst not in cache_slots
                updates[dst] = (raw, new)
            values.update(updates)
            for occ in window.segment_key_occurrences.tolist():
                slot = int(step.occurrence_slots[occ])
                assert values[slot] == (
                    int(window.occurrence_raw_tokens[occ]),
                    int(window.occurrence_positions[occ]),
                )
            state = step.state
            terminal = state.terminal_slots()
            for raw, slot in enumerate(terminal.tolist()):
                assert values[slot] == (raw, int(layout.positions[raw]))
            assert not set(step.retired_slots.tolist()) & set(state.slots.tolist())
            # Delay one batch's releases: scheduler can retain a preceding ticket
            # while preparing another forward without mutating its ownership map.
            pending.append(step.retired_slots)
            if len(pending) > 1:
                free(pending.pop(0))
            # Native insertion may deduplicate independently owned terminal pages.
            published = terminal.clone()
            for raw, slot in enumerate(terminal.tolist()):
                if slot not in private:
                    continue
                if raw % 2:
                    replacement = int(alloc(1)[0])
                    values[replacement] = values[slot]
                    free(torch.tensor([slot]))
                    slot = replacement
                    published[raw] = replacement
                private.remove(slot)
                cache_slots.add(slot)
            state = state.publish(published)
            assert not set(state.private_slots().tolist()) & cache_slots
            assert set(state.private_slots().tolist()) <= private
            start = end
        for retired in pending:
            free(retired)
        # Final publication owns all surviving terminal KV; no birth/private leak.
        assert state.private_slots().numel() == 0
        assert private == set()


def test_hole_lifetime_and_publication_does_not_claim_missing_kv(compiler, occurrence):
    n, start = 16, 8
    drops = {start: [(0, 4)]}
    layout = compiler(*args(list(range(n)), drops, [start - 1]))
    expiry = expiry_for(n, drops)
    # Four old pages are now holes. Only the remaining source rows are resident.
    state = occurrence.OccurrenceState(
        torch.arange(101, 105, dtype=torch.int64),
        torch.zeros(4, dtype=torch.bool),
        torch.tensor([-1] * 4 + list(range(4))),
        torch.full((start,), -1),
        layout.positions[:start],
        start,
    )
    window = occurrence.compile_occurrence_window(
        layout, expiry, layout.positions, query_start=start, query_end=n
    )
    keep = torch.ones(n, dtype=torch.bool)
    keep[:4] = False
    plan = state.plan(window, keep)
    step = state.advance(
        window,
        plan,
        torch.arange(201, 209),
        torch.arange(301, 301 + plan.extra_page_count),
        expiry,
    )
    terminal = step.state.terminal_slots()
    assert terminal[:4].tolist() == [-1] * 4
    assert not set(step.retired_slots.tolist()) & {101, 102, 103, 104}
    published = step.state.publish(terminal)
    assert published.terminal_slots().tolist() == terminal.tolist()
    assert published.private_slots().numel() == 0


def test_sparse_recovery_reuses_gaps_with_independent_position_versions(
    compiler, occurrence
):
    recovery = load_file(
        "context_gap_recovery", ROOT / "python/sglang/srt/context_system/recovery.py"
    )
    rng = np.random.default_rng(17391)
    for trial in range(80):
        n, matched = 32, 26
        drops = {8: [(1, 4)], 20: [(8, 12)]}
        layout = compiler(*args(list(range(n)), drops, [15, 23] if trial % 2 else []))
        expiry = expiry_for(n, drops)
        resident = torch.from_numpy(rng.random(matched) > 0.2)
        source_positions = layout.positions[:matched].clone()
        rewind = resident & (source_positions != layout.birth_positions[:matched])
        recovery_plan = recovery.plan_recovery(resident, expiry, n, rewind)
        first = recovery_plan.start
        source_slots = torch.arange(1, matched + 1, dtype=torch.int64)
        source_slots[~resident] = -1
        state = occurrence.OccurrenceState.from_match(
            source_slots[:first],
            source_positions[:first],
            exact_prefix_len=first,
            resident=resident[:first],
        )
        next_slot = matched + 1
        page_positions = dict(
            zip(source_slots[resident].tolist(), source_positions[resident].tolist())
        )
        queried = []
        owned_live = set()
        for start, end in recovery_plan.intervals:
            # Chunk a repair interval as well as jumping over retained gaps.
            for a in range(start, end, 3):
                b = min(a + 3, end)
                state = (
                    state.reuse_match_gap(
                        source_slots,
                        source_positions,
                        resident,
                        a,
                        exact_prefix_len=matched,
                    )
                    if a <= matched
                    else state
                )
                assert len(state.canonical_rows) == a
                window = occurrence.compile_occurrence_window(
                    layout,
                    expiry,
                    layout.positions,
                    query_start=a,
                    query_end=b,
                )
                keep = torch.ones(b, dtype=torch.bool)
                keep[:a] = (state.canonical_rows >= 0) | (state.terminal_rows >= 0)
                plan = state.plan(window, keep)
                queries = torch.arange(next_slot, next_slot + b - a)
                next_slot += len(queries)
                extra = torch.arange(next_slot, next_slot + plan.extra_page_count)
                next_slot += len(extra)
                owned_live.update(queries.tolist() + extra.tolist())
                page_positions.update(
                    zip(queries.tolist(), layout.birth_positions[a:b].tolist())
                )
                step = state.advance(window, plan, queries, extra, expiry)
                for src, dst, (old, new) in zip(
                    step.copy_source_slots.tolist(),
                    step.copy_destination_slots.tolist(),
                    step.copy_position_pairs.tolist(),
                ):
                    assert src != dst
                    assert page_positions[src] == old
                    page_positions[dst] = new
                read = window.segment_key_occurrences
                for slot, pos in zip(
                    step.occurrence_slots[read].tolist(),
                    window.occurrence_positions[read].tolist(),
                ):
                    assert slot > 0 and page_positions[slot] == pos
                retired = set(step.retired_slots.tolist())
                assert retired <= owned_live
                owned_live -= retired
                state = step.state
                assert set(state.slots[state.owned].tolist()) == owned_live
                queried.extend(range(a, b))
        assert queried == [q for a, b in recovery_plan.intervals for q in range(a, b)]
        assert not set(queried) & set(
            torch.nonzero(recovery_plan.reusable_prefix).flatten().tolist()
        )
        final = state.terminal_slots()
        for raw in torch.nonzero(layout.keep_mask).flatten().tolist():
            assert page_positions[int(final[raw])] == int(layout.positions[raw])
