"""Sequential event execution checked against independent token visibility."""

from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, args, cases, load_file
from test_planner import query_visibility

pytest_plugins = ("test_ir",)


@pytest.fixture(scope="module")
def staged():
    return load_file(
        "staged_plan_test", ROOT / "python/sglang/srt/context_system/staged.py"
    )


@pytest.fixture(scope="module")
def ownership():
    return load_file(
        "staged_ownership_test", ROOT / "python/sglang/srt/context_system/occurrence.py"
    )


@pytest.mark.parametrize("chunk", [1, 7, 100000])
def test_staged_sequential_versions_and_page_lifetime(
    compiler, staged, ownership, chunk
):
    for tokens, drops, repos in cases():
        n = len(tokens)
        layout = compiler(*args(tokens, drops, repos))
        expected, expiry = query_visibility(tokens, drops, repos)
        program = SimpleNamespace(layout=layout, visible_until=expiry)
        stages = staged.StagePlan.from_program(program)
        # Exercise cold, partial-message match, and all-but-final-query hits.
        for matched in sorted({0, n // 2, n - 1}):
            borrowed = torch.arange(1, matched + 1, dtype=torch.int64)
            state = ownership.OccurrenceState.from_match(
                borrowed,
                layout.positions[:matched],
                exact_prefix_len=matched // 2,
            )
            values = {i + 1: (i, int(layout.positions[i])) for i in range(matched)}
            private = set()
            spare = list(range(matched + 1, 16 * n + 32))

            def alloc(count):
                ids = spare[:count]
                del spare[:count]
                assert len(ids) == count
                private.update(ids)
                return torch.tensor(ids, dtype=torch.int64)

            def release(ids):
                ids = ids.tolist()
                assert len(ids) == len(set(ids))
                assert set(ids) <= private
                private.difference_update(ids)
                for slot in ids:
                    values.pop(slot, None)
                spare[:0] = ids

            start = matched
            queried = []
            while start < n:
                end = stages.next_end(start, min(n, start + chunk))
                window = ownership.compile_occurrence_window(
                    layout,
                    expiry,
                    layout.positions,
                    query_start=start,
                    query_end=end,
                    staged=True,
                )
                assert window.segment_count == 1
                keep = state.prefill_terminal_keep(layout, end, expiry)
                if end < n:
                    keep &= expiry[:end] > end
                plan = state.plan(window, keep, staged=True)
                birth, extra = alloc(end - start), alloc(plan.extra_page_count)
                step = state.advance(window, plan, birth, extra, expiry)
                for raw, slot in zip(range(start, end), birth.tolist()):
                    values[slot] = (raw, int(layout.birth_positions[raw]))
                updates = {}
                for src, dst, (old, new) in zip(
                    step.copy_source_slots.tolist(),
                    step.copy_destination_slots.tolist(),
                    step.copy_position_pairs.tolist(),
                ):
                    raw, position = values[src]
                    assert position == old
                    assert dst in private
                    updates[dst] = (raw, new)
                values.update(updates)
                selected = window.segment_key_occurrences.tolist()
                prefix = len(selected) - (end - start)
                for query in range(start, end):
                    actual = [
                        values[int(step.occurrence_slots[i])]
                        for i in selected[: prefix + query - start + 1]
                    ]
                    assert actual == expected[query]
                    queried.append(query)
                state = step.state
                assert set(step.retired_slots.tolist()).isdisjoint(state.slots.tolist())
                release(step.retired_slots)
                assert set(state.private_slots().tolist()) == private
                if end < n:
                    # Terminal here means current stage, never a future rotation.
                    current = dict(expected[end - 1])
                    for raw, slot in enumerate(state.terminal_slots().tolist()):
                        if slot >= 0:
                            assert values[slot] == (raw, current[raw])
                start = end
            assert queried == list(range(matched, n))
            for raw, slot in enumerate(state.terminal_slots().tolist()):
                if slot >= 0:
                    assert values[slot] == (raw, int(layout.positions[raw]))
            # Abort/final cleanup owns exactly the surviving private pages.
            release(state.private_slots())
            assert not private


def test_sparse_repair_preserves_cached_gaps(compiler, staged):
    recovery = load_file(
        "staged_recovery_test", ROOT / "python/sglang/srt/context_system/recovery.py"
    )
    n, matched = 20, 16
    drops = {10: [(0, 5)], 17: [(5, 8)]}
    layout = compiler(*args(list(range(n)), drops, [16]))
    _, expiry = query_visibility(list(range(n)), drops, [16])
    resident = torch.ones(matched, dtype=torch.bool)
    resident[2] = False  # dead: no missing query can read it
    resident[12] = False  # live: must be repaired
    plan = recovery.plan_recovery(resident, expiry, n)
    assert plan.intervals == ((12, 13), (16, 20))
    stages = staged.StagePlan.from_program(SimpleNamespace(layout=layout))
    actual = []
    for a, b in plan.intervals:
        while a < b:
            end = stages.next_end(a, b)
            actual.extend(range(a, end))
            a = end
    assert actual == [12, 16, 17, 18, 19]
    assert plan.reusable_prefix[13:16].all()


def test_terminal_transition_and_cache_publication_guard(compiler, staged, ownership):
    n = 12
    layout = compiler(*args(list(range(n)), {6: [(0, 4)]}, [11]))
    _, expiry = query_visibility(list(range(n)), {6: [(0, 4)]}, [11])
    window = ownership.compile_occurrence_window(
        layout,
        expiry,
        layout.positions,
        query_start=6,
        query_end=n,
        staged=True,
    )
    assert torch.equal(
        window.occurrence_positions[window.terminal_occurrences], layout.positions
    )
    req = SimpleNamespace(
        context_staged=True,
        context_recompute_program=None,
        context_program=SimpleNamespace(layout=layout),
        context_state=SimpleNamespace(canonical_rows=torch.empty(6)),
    )
    assert staged.has_unpublished_stage(req)
    req.context_state.canonical_rows = torch.empty(n)
    assert not staged.has_unpublished_stage(req)
    assert not staged.has_unpublished_stage(SimpleNamespace())
