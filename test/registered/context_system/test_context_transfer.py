"""PD identity, compact ownership and accounting without a model launch."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)


def test_pd_final_versions_holes_identity_and_usage(compiler, monkeypatch):
    occurrence = load_file(
        "pd_occurrence", ROOT / "python/sglang/srt/context_system/occurrence.py"
    )
    usage = load_file("pd_usage", ROOT / "python/sglang/srt/context_system/usage.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.context_system.occurrence", occurrence)
    monkeypatch.setitem(sys.modules, "sglang.srt.context_system.usage", usage)
    recovery = load_file(
        "pd_recovery", ROOT / "python/sglang/srt/context_system/recovery.py"
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.context_system.recovery", recovery)
    transfer = load_file(
        "pd_transfer", ROOT / "python/sglang/srt/disaggregation/context_transfer.py"
    )
    layout = compiler(*args(list(range(12)), {6: [(1, 4)]}, [7]))
    program = SimpleNamespace(
        layout=layout, visible_until=torch.full((12,), 99, dtype=torch.int32)
    )
    req = SimpleNamespace(
        context_program=program,
        context_recompute_program=None,
        kv=SimpleNamespace(req_pool_idx=0),
    )
    table = torch.full((1, 15), -99, dtype=torch.int64)
    pool = SimpleNamespace(
        req_to_token=table, write=lambda index, values: table.__setitem__(index, values)
    )
    allocated = []

    def alloc(count):
        allocated.append(count)
        return torch.arange(100, 100 + count, dtype=torch.int64)

    allocator = SimpleNamespace(device="cpu", alloc=alloc)
    slots = transfer.allocate_context_destination(req, allocator, pool)
    assert allocated == [9]
    assert table[0, :12].tolist() == [
        100,
        -1,
        -1,
        -1,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
    ]
    assert req.context_decode_layout.positions.tolist() == list(range(9))
    assert req.kv.kv_allocated_len == 12
    assert torch.equal(req.context_state.private_slots(), slots)
    assert req.context_state.nonterminal_private_slots().numel() == 0
    plan = transfer.transfer_plan(req, "cpu")
    # Raw chunks cross Drop holes; every final page is sent exactly once.
    cursor, sent = 0, []
    for raw_end in (3, 6, 9, 12, 12):
        final = len(sent) == 8
        begin, end, cursor = plan.full_chunk(cursor, raw_end, last_chunk=final)
        sent.extend(plan.decode.raw_indices[begin:end].tolist())
        if not final:
            assert len(sent) < plan.active_count
    assert sent == plan.decode.raw_indices.tolist()
    assert cursor == 12
    assert plan.slots(req, pool, window=3).tolist() == [106, 107, 108]
    table[0, 12:] = torch.tensor([109, 110, 111])
    assert transfer.request_active_slots(req, pool, 15).tolist() == list(
        range(100, 112)
    )
    assert transfer.request_active_slots(req, pool, 13, window=3).tolist() == [
        107,
        108,
        109,
    ]
    assert transfer.request_active_slots(req, pool, 15, window=3).tolist() == [
        109,
        110,
        111,
    ]
    # The final raw map has holes, and D allocated only its three live SWA peers.
    req.context_state = replace(
        req.context_state, swa_resident=torch.tensor([False] * 6 + [True] * 3)
    )
    assert req.context_state.live_swa_ranges(0, 11) == [(9, 11)]
    assert req.context_state.live_swa_ranges(11, 14) == [(11, 14)]
    req.context_usage = usage.ContextUsage.from_snapshot(
        usage.ContextUsageSnapshot(2, 3, 4, 5, 0)
    )
    row = torch.full((16,), -99, dtype=torch.int32)
    transfer.write_context_metadata(req, row)
    assert row[:7].tolist() == [-99] * 7
    transfer.commit_context_metadata(req, row, "cpu")
    req.context_usage.record_decode(2)
    assert req.context_usage.snapshot() == usage.ContextUsageSnapshot(2, 3, 4, 5, 2)
    row[10] ^= 1
    with pytest.raises(ValueError, match="identity"):
        transfer.commit_context_metadata(req, row, "cpu")
    req.context_program = None
    transfer.write_context_metadata(req, row)
    assert not row[7:].any()
    transfer.commit_context_metadata(req, row, "cpu")
