"""Recover arbitrary Full/SWA holes against an independent query graph."""

import importlib
import random

import pytest
import torch
from cache_policy_reference import simulate
from test_cache_policy_ir import POLICIES, compile_policy

pytest_plugins = ("test_cache_policy_ir", "test_context_attention_plan")


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
