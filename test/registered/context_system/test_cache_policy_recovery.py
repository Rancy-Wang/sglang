"""Recover arbitrary Full/SWA holes against an independent query graph."""

import importlib
import random

import pytest
import torch
from cache_policy_reference import simulate
from test_cache_policy_ir import POLICIES, compile_policy

pytest_plugins = ("test_cache_policy_ir", "test_context_attention_plan")


@pytest.mark.parametrize("window", [None, 1, 3, 128])
def test_history_fast_plan_matches_general_with_terminal_and_sparse_queries(
    policy_modules, monkeypatch, window
):
    recovery = importlib.import_module(policy_modules[2].__package__ + ".recovery")
    rng = random.Random(1118)
    for _ in range(100):
        n = rng.randrange(8, 100)
        drops = {b: [(i, i+1) for i in range(b-1) if rng.random() < .1]
                 for b in range(3, n+1, 3)}
        p = compile_policy(policy_modules, POLICIES[1], list(range(n)), drops)
        m = rng.randrange(n)
        resident = torch.tensor([rng.random() < .8 for _ in range(m)], dtype=torch.bool)
        swa = torch.tensor([rng.random() < .6 for _ in range(m)], dtype=torch.bool)
        terminal = torch.tensor([rng.random() < .2 for _ in range(m)], dtype=torch.bool)
        arguments = (resident, p.visible_until, p.layout, None, None,
                     swa if window else None, window, terminal if window else None,
                     p.layout.positions[:m])
        fast = recovery._plan_insert_recovery(*arguments)
        intervals = tuple((i, i+1) for i in range(n) if rng.random() < .2)
        masks = recovery._insert_read_masks(p.layout, p.visible_until, intervals, m, window)
        with monkeypatch.context() as patch:
            patch.setattr(recovery, "_history_layout", lambda layout: False)
            general = recovery._plan_insert_recovery(*arguments)
            expected = recovery._insert_read_masks(p.layout, p.visible_until, intervals, m, window)
        assert fast.intervals == general.intervals
        assert torch.equal(fast.required_prefix, general.required_prefix)
        assert torch.equal(fast.reusable_prefix, general.reusable_prefix)
        assert all(torch.equal(a, b) for a, b in zip(masks, expected))


def test_history_fast_path_rejects_version_mismatch(policy_modules, monkeypatch):
    recovery = importlib.import_module(policy_modules[2].__package__ + ".recovery")
    p = compile_policy(policy_modules, POLICIES[1], list(range(12)), {8: [(1, 4)]})
    positions = p.layout.positions[:9].clone()
    positions[0] += 1
    # The generic path may call the fast read-mask helper, but it must not use
    # the fixed-version dependency planner for this source.
    original = recovery._history_read_plan
    def observe(*args, **kwargs):
        assert not kwargs["recover"]
        return original(*args, **kwargs)
    monkeypatch.setattr(recovery, "_history_read_plan", observe)
    recovery._plan_insert_recovery(
        torch.zeros(9, dtype=torch.bool), p.visible_until, p.layout,
        torch.ones(9, dtype=torch.bool), None, None, None, None, positions,
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_holes_and_rotated_versions(policy_modules, policy):
    rng = random.Random(1117)
    recovery = importlib.import_module(policy_modules[2].__package__ + ".recovery")
    for _ in range(160):
        n = rng.randrange(8, 45)
        drops = {
            b: [(i, i + 1) for i in range(b - 1) if rng.random() < 0.18]
            for b in range(3, n + 1, 3)
        }
        repos = [b for b in range(4, n + 1, 4)] if policy == POLICIES[0] else []
        ref = simulate(list(range(n)), drops, repos, policy=policy)
        p = compile_policy(policy_modules, policy, list(range(n)), drops, repos)
        projection = p.execution_to_full.tolist()
        inverse = {raw: i for i, raw in enumerate(projection)}
        rows = [[(inverse[j], pos) for j, pos in ref.readers[i]] for i in projection]
        n = len(projection)
        matched = rng.randrange(n)
        resident = torch.tensor(
            [rng.random() < 0.7 for _ in range(matched)], dtype=torch.bool
        )
        swa = torch.tensor(
            [bool(resident[i]) and rng.random() < 0.6 for i in range(matched)],
            dtype=torch.bool,
        )
        stored = p.layout.positions[:matched]
        rewind = resident & (stored != p.layout.birth_positions[:matched])
        window = rng.randrange(1, 10)
        planned, full, old_reads, sliding = set(range(matched, n)), {}, {}, set()
        for q in range(n - 1, -1, -1):
            if q < matched:
                versions = old_reads.get(q, set())
                if (
                    q in full
                    and (
                        not resident[q]
                        or (rewind[q] and any(pos != stored[q] for pos in versions))
                    )
                ) or (q in sliding and not swa[q]):
                    planned.add(q)
            if q in planned:
                for raw, pos in rows[q][:-1]:
                    full.setdefault(raw, set()).add(pos)
                    # Uncached suffix queries can COW/rotate a cached source;
                    # historical repair queries require its original version.
                    if q < matched:
                        old_reads.setdefault(raw, set()).add(pos)
                qpos = ref.compute_positions[projection[q]]
                chosen = [pair for pair in rows[q] if pair[1] >= qpos - (window - 1)][
                    -window:
                ]
                sliding.update(raw for raw, _ in chosen if raw != q)
        result = recovery.plan_recovery(
            resident,
            p.visible_until,
            n,
            rewind,
            swa_resident=swa,
            swa_query_starts=recovery.sliding_window_starts(
                p.layout, p.visible_until, window
            ),
            layout=p.layout,
            swa_window=window,
            source_positions=stored,
        )
        assert {q for a, b in result.intervals for q in range(a, b)} == planned, (
            projection,
            drops,
            repos,
            rows,
            matched,
            resident.tolist(),
            swa.tolist(),
            stored.tolist(),
            p.layout.birth_positions.tolist(),
            window,
        )
        required = (set(full) | sliding | planned).intersection(range(matched))
        assert set(result.required_prefix.nonzero().flatten().tolist()) == required
        demand = recovery.sliding_window_read_mask(
            torch.zeros(n, dtype=torch.int32),
            p.visible_until,
            result.intervals,
            matched,
            layout=p.layout,
            window=window,
        )
        assert set(demand.nonzero().flatten().tolist()) == sliding.intersection(
            range(matched)
        )


@pytest.mark.parametrize("policy", POLICIES)
def test_native_segment_gathers_with_position_gaps(
    policy_modules, attention_plan, policy, monkeypatch
):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.context_system.window",
        importlib.import_module(policy_modules[2].__package__ + ".window"),
    )
    occurrence = importlib.import_module(policy_modules[2].__package__ + ".occurrence")
    p = compile_policy(
        policy_modules,
        policy,
        list(range(16)),
        {8: [(2, 5)], 14: [(6, 9)]},
        [10, 15] if policy == POLICIES[0] else [],
    )
    w = occurrence.compile_occurrence_window(
        p.layout,
        p.visible_until,
        p.layout.positions,
        query_start=0,
        query_end=len(p.input_ids),
    )
    seq = attention_plan.ContextSequence.from_window(w)
    plan = attention_plan.ContextAttentionPlan.merge([seq])
    bound = plan.bind(torch.arange(plan.occurrence_count))
    # Validate actual ragged-native gather rows, not just the occurrence graph.
    from test_cache_policy_ir import reads

    rows = reads(p, policy_modules)
    birth_slots = w.birth_occurrences.long()
    for distance in (0, 1, 3, 20):
        qo, kv, _, _, slots, _, _ = bound.native_plan(birth_slots, distance)
        raw = w.occurrence_raw_tokens[slots.long()].tolist()
        positions = w.occurrence_positions[slots.long()].tolist()
        for a, b, x, y in zip(qo[:-1], qo[1:], kv[:-1], kv[1:]):
            a, b, x, y = map(int, (a, b, x, y))
            for q in range(a, b):
                prefix = y - x - (b - a)
                actual = list(
                    zip(
                        raw[x : x + prefix + q - a + 1],
                        positions[x : x + prefix + q - a + 1],
                    )
                )[-(distance + 1) :]
                qpos = int(p.layout.birth_positions[q])
                expected = [pair for pair in rows[q] if pair[1] >= qpos - distance][
                    -(distance + 1) :
                ]
                assert actual == expected
