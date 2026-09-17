"""Raw-history overflow: request ownership, model-position bounds and GPU IO."""

from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, load_file

storage = load_file(
    "context_raw_storage", ROOT / "python/sglang/srt/context_system/request_storage.py"
)


def test_retry_self_pin_falls_back_but_external_pressure_waits(compiler):
    from test_ir import args
    from test_occurrence_ownership import expiry_for

    occurrence = load_file(
        "context_pressure_occurrence",
        ROOT / "python/sglang/srt/context_system/occurrence.py",
    )
    n, start, capacity = 100, 99, 180
    drops = {40: [(0, 20)]}
    # Reposition after the matched prefix: source KV is at the same birth
    # positions, so recovery does not replace this valid Retry by recomputation.
    layout = compiler(*args(list(range(n)), drops, [98]))
    expiry = expiry_for(n, drops)

    def demand(begin, exact):
        window = occurrence.compile_occurrence_window(
            layout, expiry, layout.positions, query_start=begin, query_end=n
        )
        plan = occurrence.plan_occurrence_materialization(
            window, torch.arange(begin, dtype=torch.int32),
            torch.ones(begin, dtype=torch.bool), torch.zeros(begin, dtype=torch.bool),
            torch.zeros(begin, dtype=torch.bool), torch.ones(n, dtype=torch.bool),
            query_start=begin, query_end=n, exact_prefix_len=exact,
        )
        return n - begin + plan.extra_page_count + 4 + 1

    req = SimpleNamespace(
        context_prefill_started=False, context_recovery_source=None,
        prefix_indices=torch.arange(start), context_resident=None,
        context_force_miss=False, context_admission_error=None,
    )
    needed = demand(start, 20)
    assert needed == 85 and needed >= capacity - start
    # The same available pages with a larger physical pool can be explained by
    # other live requests; do not throw away a reusable match in that case.
    storage.handle_prefill_capacity_pressure(req, 256, needed)
    assert not req.context_force_miss and req.context_admission_error is None
    storage.handle_prefill_capacity_pressure(req, capacity, needed)
    assert req.context_force_miss and req.context_admission_error is None
    # Cold full prefill also exceeds this pool, but native chunking plus
    # Drop-aware release makes progress with the same occurrence semantics.
    state = occurrence.OccurrenceState.from_match(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32),
        exact_prefix_len=0,
    )
    peaks = []
    for begin, end in ((0, 64), (64, n)):
        window = occurrence.compile_occurrence_window(
            layout, expiry, layout.positions, query_start=begin, query_end=end
        )
        keep = torch.ones(end, dtype=torch.bool)
        keep[:begin] = (state.canonical_rows >= 0) | (state.terminal_rows >= 0)
        plan = state.plan(window, keep)
        peaks.append(len(state.slots) + end - begin + plan.extra_page_count)
        advance = state.advance(
            window, plan, torch.arange(1000 + begin, 1000 + end),
            torch.arange(2000 + begin, 2000 + begin + plan.extra_page_count), expiry,
        )
        state = advance.state
        state = state.publish(state.terminal_slots())
        state = state.drop_borrowed_raw(expiry[:end] <= end)
    assert peaks == [108, 159] and max(peaks) + 4 + 1 < capacity
    cold_needed = 1 + 4 + 1
    req.prefix_indices = req.prefix_indices[:0]
    storage.handle_prefill_capacity_pressure(req, capacity, cold_needed)
    assert req.context_admission_error is None
    storage.handle_prefill_capacity_pressure(req, cold_needed, cold_needed)
    assert "without a cached prefix" in storage.prefill_capacity_error(req, cold_needed)


pytest_plugins = ("test_ir",)


def test_continuation_capacity_includes_future_terminal_copies(compiler):
    from test_ir import args
    from test_occurrence_ownership import expiry_for

    occurrence = load_file(
        "context_continuation_occurrence",
        ROOT / "python/sglang/srt/context_system/occurrence.py",
    )
    n, capacity = 100, 110
    drops = {80: [(0, 40)]}
    layout = compiler(*args(list(range(n)), drops, [98]))
    expiry = expiry_for(n, drops)
    state = occurrence.OccurrenceState.from_match(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32),
        exact_prefix_len=0,
    )
    start, serial = 0, 1
    while start < n:
        length = min(32, n - start)
        while length:
            end = start + length
            window = occurrence.compile_occurrence_window(
                layout, expiry, layout.positions, query_start=start, query_end=end
            )
            keep = torch.ones(end, dtype=torch.bool)
            keep[:start] = (state.canonical_rows >= 0) | (state.terminal_rows >= 0)
            plan = state.plan(window, keep)
            needed = length + plan.extra_page_count + 1 + int(end == n)
            if len(state.slots) + needed < capacity:
                break
            length //= 2
        if not length:
            break
        query = torch.arange(serial, serial + length)
        serial += length
        extra = torch.arange(serial, serial + plan.extra_page_count)
        serial += len(extra)
        state = state.advance(window, plan, query, extra, expiry).state
        state = state.publish(state.terminal_slots())
        state = state.drop_borrowed_raw(expiry[:end] <= end)
        start = end

    assert (start, len(state.slots), needed) == (74, 108, 3)
    req = SimpleNamespace(
        context_prefill_started=True, context_state=state,
        context_admission_error=None,
    )
    storage.handle_prefill_capacity_pressure(req, 256, needed)
    assert req.context_admission_error is None  # other live requests can finish
    storage.handle_prefill_capacity_pressure(req, capacity, needed)
    assert "retains 108" in storage.prefill_capacity_error(req, capacity)


def test_prefill_capacity_uses_actual_queries_and_invalidates_on_retry():
    # Drop 0:7 before q8: final active=3, but cold q7 must read eight keys.
    program = SimpleNamespace(
        visible_until=torch.tensor([8] * 7 + [100] * 3, dtype=torch.int32)
    )
    req = SimpleNamespace(
        context_program=program,
        context_recompute_program=None,
        context_recovery_plan=SimpleNamespace(intervals=((0, 10),)),
    )
    assert "at least 8" in storage.prefill_capacity_error(req, 6)
    assert storage.prefill_capacity_error(req, 8) is None
    # Same request can get a better cache match while waiting. Do not reject
    # historical queries which will never run, including gaps between repairs.
    req.context_recovery_plan.intervals = ((1, 2), (8, 10))
    assert storage.prefill_capacity_error(req, 3) is None
    assert "at least 3" in storage.prefill_capacity_error(req, 2)
    # A new recompute program after retraction must invalidate the old peak.
    req.context_recompute_program = SimpleNamespace(
        visible_until=torch.full((10,), 100, dtype=torch.int32)
    )
    assert "at least 10" in storage.prefill_capacity_error(req, 6)


def pool_and_request(device="cpu"):
    pool = SimpleNamespace(
        req_to_token=torch.zeros((3, 16), dtype=torch.int32, device=device),
        device=device,
    )
    req = SimpleNamespace(
        context_program=object(),
        origin_input_ids=list(range(40)),
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=SimpleNamespace(req_pool_idx=1),
    )
    return pool, req


def test_overflow_ownership_decode_reserve_and_slot_reuse():
    pool, req = pool_and_request()
    original = pool.req_to_token.data_ptr()
    storage.prepare_request_row(pool, req)
    row = storage.request_row(pool, 1)
    assert len(row) == 52 and pool.req_to_token.data_ptr() == original
    storage.prepare_request_row(pool, req)
    assert storage.request_row(pool, 1) is row
    storage.write_request_slots(pool, (1, slice(35, 40)), torch.arange(5))
    storage.write_request_slots(
        pool, (torch.tensor([1, 2]), torch.tensor([40, 3])), torch.tensor([55, 66])
    )
    assert row[35:41].tolist() == [0, 1, 2, 3, 4, 55]
    assert pool.req_to_token[2, 3] == 66
    assert pool.req_to_token[1].count_nonzero() == 0
    storage.release_request_row(pool, 1)
    assert storage.row_pointers(pool) is None
    assert storage.request_row(pool, 1).data_ptr() == pool.req_to_token[1].data_ptr()
    req.context_program = None
    storage.prepare_request_row(pool, req)
    assert not pool._context_rows


def test_positions_validate_intermediate_rope_not_raw_history():
    # Raw length 40, every real materialization below model limit 24.
    layout = SimpleNamespace(
        birth_positions=torch.arange(40) % 20,
        transition_old_positions=torch.tensor([19]),
        transition_new_positions=torch.tensor([3]),
        positions=torch.arange(40) % 20,
        keep_mask=torch.arange(40) >= 20,
        next_position=20,
    )
    program = SimpleNamespace(layout=layout)
    assert storage.validate_positions(program, 24) == 20
    layout.transition_old_positions[0] = 24
    with pytest.raises(ValueError, match="execution position"):
        storage.validate_positions(program, 24)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_fused_prefix_decode_scatter_and_graph_gather():
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry
    from test_context_decode import layout

    pool, req = pool_and_request("cuda")
    storage.prepare_request_row(pool, req)
    prefix = torch.arange(100, 130, dtype=torch.int64, device="cuda")
    out = torch.arange(130, 140, dtype=torch.int64, device="cuda")
    rows = torch.tensor([1], device="cuda")
    pointers = torch.tensor([prefix.data_ptr()], dtype=torch.uint64, device="cuda")
    lens = [torch.tensor([v], device="cuda") for v in (30, 40, 10)]
    write_req_to_token_pool_triton[(1,)](
        pool.req_to_token, rows, pointers, *lens, out, 16, storage.row_pointers(pool)
    )
    storage.write_request_slots(
        pool,
        (rows, torch.tensor([40], device="cuda")),
        torch.tensor([140], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        storage.request_row(pool, 1)[:41],
        torch.arange(100, 141, device="cuda", dtype=torch.int32),
    )
    view, program = layout([0, 31, 39], [0, 1, 2], 40, 3, "cuda")
    req.context_program = SimpleNamespace(layout=program)
    req.context_decode_layout = view
    translator = SimpleNamespace(
        page_size=1,
        defer_read_translate=False,
        req_to_token=pool.req_to_token,
        is_translating=False,
    )
    registry = ContextDecodeRegistry(translator, SimpleNamespace(), pool)
    _, lengths, _, refs = registry.bind_batch([req], [41], rows)
    indptr = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    output = torch.empty(4, device="cuda", dtype=torch.int64)
    registry.fill(rows, lengths, indptr, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        registry.fill(rows, lengths, indptr, output)
    graph.replay()
    assert output.tolist() == [100, 131, 139, 140]
    assert pool.req_to_token[1].count_nonzero() == 0
    # Reuse the same slot for native input without stale pointer reads.
    storage.release_request_row(pool, 1)
    pool.req_to_token[1, :4] = torch.tensor([7, 8, 9, 10], device="cuda")
    req.context_program = None
    registry.bind_batch([req], [4], rows)
    graph.replay()
    assert output.tolist() == [7, 8, 9, 10]
