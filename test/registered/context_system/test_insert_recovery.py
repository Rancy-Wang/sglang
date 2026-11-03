"""Actual historical read-dependency closure, checked against a simple graph."""
import importlib
import random

import pytest
import torch
from test_insert_ir import all_reads, program

pytest_plugins = ("test_rules",)


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
