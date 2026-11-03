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
                    if pos >= int(p.layout.birth_positions[q])-(window-1):
                        sliding.add(raw)
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
    expected = 257 if not duplicate_only else 1025
    assert registry.window_lengths_cpu(128, 4).tolist() == [expected]*4
    assert not registry.fits_graph(4, 8192, window=128, window_capacity=516)
    assert registry.fits_graph(4, 8192, window=128, window_capacity=4*expected)
    assert not registry.fits_graph(4, 4096)
    assert not registry.fits_graph(4, 2048, window=128, window_capacity=129, table=True)
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
                    if pos >= p.layout.birth_positions[q] - (window-1):
                        sliding.add(raw)
        result = module.plan_recovery(
            resident, p.visible_until, n, rewind, swa_resident=swa,
            swa_query_starts=module.sliding_window_starts(p.layout, p.visible_until, window),
            layout=p.layout, swa_window=window, source_positions=stored,
        )
        assert {q for a, b in result.intervals for q in range(a, b)} == planned
