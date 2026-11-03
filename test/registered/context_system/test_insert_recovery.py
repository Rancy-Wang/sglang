"""Actual historical read-dependency closure, checked against a simple graph."""
import importlib
import random

import pytest
import torch
from test_insert_ir import all_reads, program

pytest_plugins = ("test_rules", "test_context_attention_plan")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("repos", [False, True])
def test_random_holes_against_read_graph(context_modules, nested, repos):
    p = program(context_modules, nested=nested, repos=repos, width=3)
    module = importlib.import_module(context_modules[2].__package__ + ".recovery")
    rows = all_reads(context_modules, p)
    n = len(p.input_ids)
    rng = random.Random(20261103)
    for _ in range(100):
        matched = rng.randrange(n)
        resident = torch.tensor([rng.random() < .7 for _ in range(matched)], dtype=torch.bool)
        swa = torch.tensor([bool(resident[i]) and rng.random() < .6 for i in range(matched)], dtype=torch.bool)
        window = rng.randrange(1, 10)
        planned = set(range(matched, n))
        full, sliding = set(), set()
        for q in range(n-1, -1, -1):
            if q < matched and ((q in full and not resident[q]) or (q in sliding and not swa[q])):
                planned.add(q)
            if q in planned:
                for raw, pos in rows[q][:-1]:
                    full.add(raw)
                chosen = [pair for pair in rows[q] if pair[1] >= int(p.layout.birth_positions[q])-(window-1)][-window:]
                sliding.update(raw for raw, _ in chosen if raw != q)
        result = module.plan_recovery(
            resident, p.visible_until, n, swa_resident=swa,
            swa_query_starts=module.sliding_window_starts(p.layout, p.visible_until, window),
            layout=p.layout, swa_window=window)
        actual = {i for a,b in result.intervals for i in range(a,b)}
        assert actual == planned
        required = {i for i in (full | sliding | planned) if i < matched}
        assert set(result.required_prefix.nonzero().flatten().tolist()) == required
        demand = module.sliding_window_read_mask(
            torch.zeros(n, dtype=torch.int32), p.visible_until, result.intervals,
            matched, layout=p.layout, window=window)
        assert set(demand.nonzero().flatten().tolist()) == sliding.intersection(range(matched))


def test_warm_summary_never_recomputes_retained_suffix(context_modules):
    p = program(context_modules)
    module = importlib.import_module(context_modules[2].__package__ + ".recovery")
    resident = torch.tensor([True, False, False, False, False, False, True, True, True])
    result = module.plan_recovery(resident, p.visible_until, len(p.input_ids), layout=p.layout)
    assert result.intervals == ((9,11),)
    assert result.required_prefix.nonzero().flatten().tolist() == [0,6,7,8]


@pytest.mark.parametrize("duplicate_only", [False, True])
def test_decode_capacity_uses_logical_position_counts(context_modules, attention_plan, duplicate_only, monkeypatch):
    import sys
    from types import SimpleNamespace
    occ = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    monkeypatch.setitem(sys.modules, "sglang.srt.context_system.occurrence", occ)
    positions = ([0] + list(range(1, 513)) * 2 if not duplicate_only else [0] + [512] * 1024)
    layout = SimpleNamespace(keep_mask=torch.ones(1025, dtype=torch.bool),
                             positions=torch.tensor(positions, dtype=torch.int32),
                             logical_order=torch.arange(1025, dtype=torch.int32), next_position=513)
    view = occ.ContextDecodeLayout.from_layout(layout, "cpu")
    assert not view.positions_monotone
    table = torch.zeros((5, 2048), dtype=torch.int32)
    registry = attention_plan.ContextDecodeRegistry(SimpleNamespace(
        page_size=1, defer_read_translate=False, req_to_token=table), SimpleNamespace())
    req = SimpleNamespace(context_program=object(), context_decode_layout=view)
    registry.bind_batch([req]*4, [1026]*4, torch.arange(4))
    expected = 129
    assert registry.window_lengths_cpu(128, 4).tolist() == [expected]*4
    assert registry.fits_graph(4, 8192, window=128, window_capacity=516)
    assert registry.fits_graph(4, 8192, window=128, window_capacity=4*expected)
    assert not registry.fits_graph(4, 4096)
    assert registry.fits_graph(4, 2048, window=128, window_capacity=129, table=True)
    assert not registry.fits_graph(4, 2048, window=128, window_capacity=128, table=True)
    # A regular repositioned summary retains the constant-time graph path.
    layout.positions = torch.arange(1025, dtype=torch.int32)
    layout.next_position = 1025
    req.context_decode_layout = occ.ContextDecodeLayout.from_layout(layout, "cpu")
    registry.bind_batch([req], [1026], torch.tensor([0]))
    assert registry.fits_graph(4, 8192, window=128, window_capacity=516)
    assert registry._window_lengths == {}  # No per-query search in the usual case.
    assert registry.window_lengths_cpu(128, 4).tolist() == [129, 1, 1, 1]


def test_post_reposition_swa_hole_keeps_correct_full_versions(context_modules):
    p = program(context_modules, nested=True, repos=True, width=7)
    module = importlib.import_module(context_modules[2].__package__ + '.recovery')
    n = len(p.input_ids)
    matched = n - 1
    resident = p.layout.keep_mask[:matched].clone()
    swa = resident.clone()
    swa[-1] = False
    rewind = resident & (p.layout.positions[:matched] != p.layout.birth_positions[:matched])
    assert rewind.any()
    result = module.plan_recovery(
        resident, p.visible_until, n, rewind, swa_resident=swa,
        swa_query_starts=module.sliding_window_starts(p.layout, p.visible_until, 5),
        layout=p.layout, swa_window=5,
    )
    # Both the missing query and its successor are after the final R. Every
    # cached Full prefix already has the exact positions these queries need.
    assert result.intervals == ((matched-1, n),)


def test_rotated_source_versions_follow_selected_queries(context_modules):
    p = program(context_modules, nested=True, repos=True, width=4)
    module = importlib.import_module(context_modules[2].__package__ + '.recovery')
    rows = all_reads(context_modules, p)
    n, window = len(p.input_ids), 5
    matched = n - 1
    resident = p.layout.keep_mask[:matched].clone()
    stored = p.layout.positions[:matched]
    rewind = resident & (stored != p.layout.birth_positions[:matched])
    rng = random.Random(81103)
    for _ in range(50):
        swa = resident & torch.tensor([rng.random() > .15 for _ in range(matched)])
        planned, full, sliding = set(range(matched, n)), {}, set()
        for q in range(n-1, -1, -1):
            if q < matched:
                versions = full.get(q, set())
                if ((versions and (not resident[q] or (rewind[q] and any(pos != stored[q] for pos in versions))))
                        or (q in sliding and not swa[q])):
                    planned.add(q)
            if q in planned:
                for raw, pos in rows[q][:-1]:
                    full.setdefault(raw, set()).add(pos)
                chosen = [pair for pair in rows[q] if pair[1] >= int(p.layout.birth_positions[q])-(window-1)][-window:]
                sliding.update(raw for raw, _ in chosen if raw != q)
        result = module.plan_recovery(
            resident, p.visible_until, n, rewind, swa_resident=swa,
            swa_query_starts=module.sliding_window_starts(p.layout, p.visible_until, window),
            layout=p.layout, swa_window=window, source_positions=stored,
        )
        assert {q for a, b in result.intervals for q in range(a, b)} == planned


def test_retrigger_rotates_future_versions_without_rewinding_retained_suffix(context_modules):
    p = program(context_modules, repos=True, width=7)
    module = importlib.import_module(context_modules[2].__package__ + '.recovery')
    matched, n = 63, len(p.input_ids)
    stored = p.layout.birth_positions[:matched].clone()
    stored[21:28] = torch.arange(7, 14, dtype=torch.int32)  # source's first R
    resident = torch.ones(matched, dtype=torch.bool)
    resident[7:21] = False  # the source has completed D(1,2), R, and A(4..8)
    swa = resident.clone()
    swa[-1] = False  # only the final old-assistant token requires repair
    rewind = resident & (stored != p.layout.birth_positions[:matched])
    for incompatible in (None, stored != p.layout.positions[:matched]):
        result = module.plan_recovery(
            resident, p.visible_until, n, rewind, incompatible,
            swa_resident=swa,
            swa_query_starts=module.sliding_window_starts(p.layout, p.visible_until, 5),
            layout=p.layout, swa_window=5, source_positions=stored,
        )
        assert result.intervals == ((matched-1, n),)
        assert result.reusable_prefix[42:62].all()


def test_window_index_matches_independent_filter_and_tail(context_modules):
    module = importlib.import_module(context_modules[2].__package__ + ".window")
    rng = random.Random(61103)
    for _ in range(300):
        positions = [rng.randrange(40) for _ in range(rng.randrange(200))]
        index = module.LogicalSWAWindowIndex(positions)
        for capacity in (0, 1, 3, 17, 129):
            lower = rng.randrange(-5, 45)
            expected = [i for i, position in enumerate(positions) if position >= lower]
            expected = expected[-capacity:] if capacity else []
            assert index.tail(lower, capacity).tolist() == expected


def test_window_candidate_can_reenter_after_later_keys_expire(context_modules):
    module = importlib.import_module(context_modules[2].__package__ + ".window")
    index = module.LogicalSWAWindowIndex([3, 1, 1])
    # W=3, first query@3 selects B,C,Q; next query@4 selects A,Q,Q.
    rows = list(index.prefix_reads(3, 20, 3))
    assert [(i, row.tolist()) for i, row in rows] == [(0, [1, 2]), (1, [0])]
    from types import SimpleNamespace
    layout = SimpleNamespace(positions=torch.tensor([3, 1, 1]),
                             keep_mask=torch.ones(3, dtype=torch.bool), next_position=3)
    assert module.terminal_swa_mask(layout, 3).tolist() == [True, True, True]


def test_terminal_window_union_against_decode_simulation(context_modules):
    from types import SimpleNamespace

    module = importlib.import_module(context_modules[2].__package__ + ".window")
    rng = random.Random(318)
    for monotone in (False, True):
        for _ in range(60):
            n, capacity = rng.randrange(1, 50), rng.randrange(1, 12)
            positions = ([i*2 for i in range(n)] if monotone else
                         [rng.randrange(50) for _ in range(n)])
            keep = [rng.random() > .2 for _ in range(n)]
            order = list(range(n))
            if not monotone:
                rng.shuffle(order)
            next_position = rng.randrange(60)
            layout = SimpleNamespace(positions=torch.tensor(positions, dtype=torch.int32),
                                     keep_mask=torch.tensor(keep), logical_order=torch.tensor(order),
                                     next_position=next_position)
            active = [(i, positions[i]) for i in order if keep[i]]
            expected = set()
            for offset in range(capacity+2):
                query_position = next_position+offset
                active.append((n+offset, query_position))
                selected = [(raw, pos) for raw, pos in active if pos >= query_position-capacity+1][-capacity:]
                expected.update(raw for raw, _ in selected if raw < n)
            assert module.terminal_swa_mask(layout, capacity).tolist() == [i in expected for i in range(n)]
