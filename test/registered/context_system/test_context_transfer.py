"""PD identity, compact ownership and accounting without a model launch."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)


@pytest.fixture
def transfer_modules(monkeypatch):
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
    storage = load_file("pd_storage", ROOT / "python/sglang/srt/context_system/request_storage.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.context_system.request_storage", storage)
    transfer = load_file(
        "pd_transfer", ROOT / "python/sglang/srt/disaggregation/context_transfer.py"
    )
    return transfer, usage


def test_pd_final_versions_holes_identity_and_usage(compiler, transfer_modules):
    transfer, usage = transfer_modules
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
    # A deferred final rotation leaves an active hole, then a private copy.
    # Neither can be streamed early; later final publication sends it once.
    rows = req.context_state.terminal_rows.clone()
    owners = torch.zeros(len(slots), dtype=torch.bool)
    rows[4] = -1
    begin, end, cursor = plan.full_chunk(
        0, 9, last_chunk=False, terminal_rows=rows, owned=owners,
    )
    assert (begin, end, cursor) == (0, 1, 4)
    with pytest.raises(ValueError, match="missing terminal KV"):
        plan.full_chunk(cursor, 12, last_chunk=True, terminal_rows=rows, owned=owners)
    rows[4] = 1
    owners[1] = True
    assert plan.full_chunk(
        cursor, 12, last_chunk=False, terminal_rows=rows, owned=owners,
    ) == (1, 1, 4)
    assert plan.full_chunk(
        cursor, 12, last_chunk=True, terminal_rows=rows, owned=owners,
    ) == (1, 9, 12)
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


def test_decode_retry_sparse_reuse_keeps_source_and_private_ownership(compiler, transfer_modules):
    transfer, _ = transfer_modules
    layout = compiler(*args(list(range(12)), {6: [(1, 4)]}, [7]))
    program = SimpleNamespace(layout=layout, visible_until=torch.full((12,), 99, dtype=torch.int32))
    source_slots = torch.arange(20, 30, dtype=torch.int64)
    resident = torch.ones(10, dtype=torch.bool)
    resident[5] = False  # A Retry hole before later resident pages.
    source_positions = torch.arange(10, dtype=torch.int32)
    req = SimpleNamespace(
        context_program=program, context_recompute_program=None,
        context_resident=resident, context_source_positions=source_positions,
        context_exact_prefix_len=5, prefix_indices=source_slots,
        kv=SimpleNamespace(req_pool_idx=0),
    )
    plan = transfer.transfer_plan(req, "cpu")
    reuse = transfer.ContextDecodeReuse.build(req, plan)
    req.context_decode_reuse = reuse
    assert plan.decode.raw_indices.tolist() == [0, 4, 5, 6, 7, 8, 9, 10, 11]
    assert reuse.reusable.tolist() == [True, True, False, True, True, True, True, False, False]
    assert reuse.borrowed.tolist() == [True] + [False] * 8
    assert reuse.copy_indices.tolist() == [1, 3, 4, 5, 6]
    assert reuse.position_pairs.tolist() == [[4, 1], [6, 3], [7, 4], [8, 5], [9, 6]]
    table = torch.full((1, 12), -99, dtype=torch.int64)
    allocated = []

    def alloc(n):
        allocated.append(n)
        return torch.arange(100, 100 + n, dtype=torch.int64)

    transfer.allocate_context_destination(
        req, SimpleNamespace(device="cpu", alloc=alloc),
        SimpleNamespace(write=lambda index, value: table.__setitem__(index, value)),
    )
    assert allocated == [8]
    assert req.context_state.slots.tolist() == [20] + list(range(100, 108))
    assert req.context_state.private_slots().tolist() == list(range(100, 108))
    assert table.tolist() == [[20, -1, -1, -1, 100, 101, 102, 103, 104, 105, 106, 107]]
    assert torch.equal(source_slots, torch.arange(20, 30))
    assert torch.equal(source_positions, torch.arange(10, dtype=torch.int32))
    assert not req.context_cache_published
    src = np.arange(9, dtype=np.int32)
    dst = np.arange(40, 49, dtype=np.int32)
    sent, target = transfer.select_missing_transfer(src, dst, reuse.reusable)
    assert sent.tolist() == [2, 7, 8] and target.tolist() == [42, 47, 48]
    assert src.tolist() == list(range(9)) and dst.tolist() == list(range(40, 49))


def test_decode_changed_branch_copies_even_at_equal_positions(compiler, transfer_modules):
    transfer, _ = transfer_modules
    layout = compiler(*args(list(range(6)), {}, []))
    program = SimpleNamespace(layout=layout, visible_until=torch.full((6,), 99, dtype=torch.int32))
    req = SimpleNamespace(context_resident=None, context_source_positions=None,
                          context_exact_prefix_len=3, prefix_indices=torch.arange(6))
    reuse = transfer.ContextDecodeReuse.build(req, transfer.ContextTransferPlan.build(program, "cpu"))
    assert reuse.reusable.all() and reuse.allocation_count == 3
    assert reuse.copy_indices.tolist() == [3, 4, 5]
    assert reuse.position_pairs.tolist() == [[3, 3], [4, 4], [5, 5]]
    source = np.arange(6, dtype=np.int32)
    assert not transfer.select_missing_transfer(source, source + 10, reuse.reusable)[0].size
    with pytest.raises(ValueError, match="coordinates"):
        transfer.select_missing_transfer(source, source[:3], reuse.reusable)
    with pytest.raises(ValueError, match="boolean"):
        transfer.select_missing_transfer(source, source, reuse.reusable.astype(np.uint8))
