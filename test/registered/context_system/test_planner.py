"""Independent CPU visibility and hole-repair checks (no model approximation)."""

import random

import pytest
import torch

from test_ir import ROOT, args, cases, compiler, load_file


@pytest.fixture(scope="module")
def occurrence():
    return load_file(
        "context_test_occurrence",
        ROOT / "python/sglang/srt/context_system/occurrence.py",
    ).compile_occurrence_window


@pytest.fixture(scope="module")
def recovery():
    return load_file(
        "context_test_recovery", ROOT / "python/sglang/srt/context_system/recovery.py"
    )


def query_visibility(tokens, drops, reposition):
    positions, active, next_pos, expected = [], [], 0, []
    expiry = torch.full((len(tokens),), 2**31 - 1, dtype=torch.int32)
    for insertion in range(len(tokens) + 1):
        for begin, end in drops.get(insertion, ()):
            active = [raw for raw in active if not begin <= raw < end]
            expiry[begin:end] = torch.minimum(
                expiry[begin:end], torch.tensor(insertion, dtype=torch.int32)
            )
        if insertion - 1 in reposition:
            changed = any(positions[raw] != rank for rank, raw in enumerate(active))
            if changed:
                for rank, raw in enumerate(active):
                    positions[raw] = rank
                next_pos = len(active)
        if insertion < len(tokens):
            positions.append(next_pos)
            next_pos += 1
            active.append(insertion)
            expected.append([(raw, positions[raw]) for raw in active])
    return expected, expiry


def test_all_queries_and_chunk_windows(compiler, occurrence):
    checked = 0
    for tokens, drops, reposition in cases():
        layout = compiler(*args(tokens, drops, reposition))
        if len(layout.transition_offsets) < 2:
            continue
        expected, expiry = query_visibility(tokens, drops, reposition)
        n = len(tokens)
        for start, end in {(0, n), (n // 2, n), (0, max(1, n // 2))}:
            window = occurrence(layout, expiry, layout.positions,
                                query_start=start, query_end=end)
            pairs = list(zip(window.occurrence_raw_tokens.tolist(),
                             window.occurrence_positions.tolist()))
            covered = []
            for a, b, x, y in zip(window.segment_query_starts.tolist(),
                                  window.segment_query_ends.tolist(),
                                  window.segment_key_offsets[:-1].tolist(),
                                  window.segment_key_offsets[1:].tolist()):
                selected = window.segment_key_occurrences[x:y].tolist()
                prefix_len = len(selected) - (b - a)
                for query in range(a, b):
                    actual = [pairs[i] for i in selected[:prefix_len + query - a + 1]]
                    assert actual == expected[query], (start, end, query)
                    covered.append(query)
            assert covered == list(range(start, end))
            assert [pairs[i] for i in window.terminal_occurrences.tolist()] == list(
                enumerate(layout.positions.tolist())
            )
            checked += 1
    assert checked > 100


def test_hole_dependency_graph(recovery):
    rng = random.Random(20260917)
    for _ in range(1000):
        n = rng.randint(1, 32)
        matched = rng.randrange(n)
        present = [rng.choice([True, False]) for _ in range(matched)]
        rewind = [rng.choice([True, False]) for _ in range(matched)]
        incompatible = [rng.choice([True, False]) for _ in range(matched)]
        expiry = [rng.randint(raw + 1, n + 1) for raw in range(n)]
        needed = set(range(matched, n))
        for query in reversed(range(n)):
            if query in needed:
                for raw in range(min(query, matched)):
                    if query < expiry[raw] and (
                        not present[raw] or incompatible[raw]
                        or (query < matched and rewind[raw])
                    ):
                        needed.add(raw)
        plan = recovery.plan_recovery(
            torch.tensor(present, dtype=torch.bool),
            torch.tensor(expiry, dtype=torch.int32), n,
            torch.tensor(rewind, dtype=torch.bool),
            torch.tensor(incompatible, dtype=torch.bool),
        )
        assert {raw for a, b in plan.intervals for raw in range(a, b)} == needed
        assert plan.required_prefix.tolist() == [
            raw in needed or any(raw < query < expiry[raw] for query in needed)
            for raw in range(matched)
        ]
        assert plan.reusable_prefix.tolist() == [
            present[raw] and raw not in needed for raw in range(matched)
        ]


def test_drop_skip_requires_matched_ancestor_event(recovery):
    records = torch.tensor([
        [0, 7, -1, 0], [0, 8, -1, 1], [1, -1, -3, -1], [0, 9, -1, 2]
    ], dtype=torch.int32)
    # Raw 1 was dropped but an earlier repair query still needs its KV.
    assert recovery.proven_skip_ranges(records, torch.tensor([False, True, True])) == [(0, 1)]
    assert recovery.proven_skip_ranges(records[:2], torch.tensor([False, False])) == []
    records[2, 2] = -4  # References future raw 2: must never authorize eviction.
    with pytest.raises(ValueError, match="non-ancestor"):
        recovery.proven_skip_ranges(records, torch.tensor([False, True, True]))
