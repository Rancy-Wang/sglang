"""Native prefill admission must pay for copies without charging model queries."""

import sys
from array import array

import pytest
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")


@pytest.mark.parametrize(
    "prefill,transport_error", [(False, False), (True, False), (True, True)]
)
def test_capacity_rejection_preserves_program_and_releases_pd_metadata(
    prefill, transport_error
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler

    original = [11, 12, 13]
    program = object()
    sender = Mock()
    if transport_error:
        sender.abort.side_effect = RuntimeError("peer unavailable")
    req = SimpleNamespace(
        rid="capacity",
        origin_input_ids=original,
        context_program=program,
        disagg_kv_sender=sender,
        metadata_buffer_index=4,
        pending_bootstrap=True,
        time_stats=SimpleNamespace(trace_ctx=Mock()),
        return_logprob=False,
    )
    scheduler = SimpleNamespace(
        _release_aborted_request=Mock(),
        beam_coordinator=Mock(),
        disaggregation_mode=(
            DisaggregationMode.PREFILL if prefill else DisaggregationMode.NULL
        ),
        req_to_metadata_buffer_idx_allocator=Mock(),
        output_streamer=Mock(),
    )
    Scheduler._reject_context_prefill_capacity(scheduler, req, "capacity exhausted")
    assert req.origin_input_ids is original and req.context_program is program
    assert req.finished_reason.to_json()["status_code"] == 503
    scheduler._release_aborted_request.assert_called_once_with(req)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
    if prefill:
        sender.abort.assert_called_once()
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(4)
        assert req.metadata_buffer_index == -1 and not req.pending_bootstrap
    else:
        sender.abort.assert_not_called()
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()


@pytest.fixture
def factory():
    from functools import partial
    from unittest.mock import patch

    module = load_file(
        "context_native_prefill_fixture",
        ROOT / "test/registered/unit/managers/test_prefill_adder.py",
    )
    fixture = module.TestPrefillAdder()
    # These native admission tests allocate CPU metadata only. Do not probe or
    # initialize a GPU occupied by a concurrent model benchmark.
    with patch.object(module, "ServerArgs", partial(module.ServerArgs, device="cpu")):
        fixture.setUp()
    fixture.mock_tree_cache.supports_mamba.return_value = False
    fixture.mock_token_allocator.size_full = 1_000_000
    yield fixture
    fixture.doCleanups()


def make_req(compiler, *, ignore_eos=False):
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from test_occurrence_ownership import expiry_for

    tokens = list(range(128))
    drops = {24: [(4, 12)], 56: [(16, 36)]}
    program = ContextProgram(
        compiler(*args(tokens, drops, [23, 55])), expiry_for(len(tokens), drops)
    )
    req = Req(
        "context",
        "",
        array("q", tokens),
        SamplingParams(max_new_tokens=4, ignore_eos=ignore_eos),
        context_program=program.to_wire(),
    )
    req._refresh_fill_ids()
    return req


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_native_admission_charges_copies_once(factory, compiler, ignore_eos):
    from sglang.srt.managers.schedule_policy import AddReqResult

    factory.mock_token_allocator.available_size.return_value = 4096
    factory.mock_tree_cache.disable = ignore_eos
    adder = factory.create_adder(factory.create_running_batch(), rem_chunk_tokens=64)
    req = make_req(compiler, ignore_eos=ignore_eos)
    verdict = adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
    assert verdict in (AddReqResult.CONTINUE, AddReqResult.OTHER)
    assert adder.can_run_list == [req]
    assert req.extend_range.length == 64
    _, plan = req.context_window_plan
    assert plan.extra_page_count > 0
    assert adder.memory_budget.current_offset == 64 + 1 + plan.extra_page_count
    assert adder.rem_chunk_tokens == 0
    assert adder.rem_input_tokens == 10000 - 64


def test_continuation_shrinks_before_allocation(factory, compiler):
    factory.mock_token_allocator.available_size.return_value = 90
    adder = factory.create_adder(factory.create_running_batch(), rem_chunk_tokens=64)
    req = make_req(compiler)
    assert adder.add_chunked_req(req) is req
    assert adder.can_run_list == [req]
    assert 0 < req.extend_range.length < 64
    _, plan = req.context_window_plan
    assert adder.memory_budget.current_offset == (
        req.extend_range.length + plan.extra_page_count + 1
    )
    assert adder.memory_budget.remaining_current > 0
    factory.mock_token_allocator.alloc.assert_not_called()


def test_no_capacity_does_not_publish_a_plan(factory, compiler):
    factory.mock_token_allocator.available_size.return_value = 1
    adder = factory.create_adder(factory.create_running_batch(), rem_chunk_tokens=64)
    req = make_req(compiler)
    assert adder.add_chunked_req(req) is req
    assert adder.can_run_list == []
    assert req.context_window_plan is None
    assert adder.memory_budget.current_offset == 0


def test_retry_self_pin_releases_lease_and_rematches_cold(factory, compiler):
    import torch
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import AddReqResult
    from sglang.srt.mem_cache.base_prefix_cache import MatchResult
    from sglang.srt.sampling.sampling_params import SamplingParams
    from test_occurrence_ownership import expiry_for

    drops = {40: [(0, 20)]}
    program = ContextProgram(
        compiler(*args(list(range(100)), drops, [98])), expiry_for(100, drops)
    )
    req = Req(
        "self-pin", "", array("q", range(100)), SamplingParams(max_new_tokens=4),
        context_program=program.to_wire(),
    )
    req._refresh_fill_ids()
    req.prefix_indices = torch.arange(1, 100, dtype=torch.int64)
    req.context_source_positions = torch.arange(99, dtype=torch.int32)
    req.context_exact_prefix_len = 20
    req.prepare_context_recovery()
    assert req.context_recovery_plan.start == 99
    factory.mock_token_allocator.size_full = 180
    factory.mock_token_allocator.available_size.return_value = 81
    adder = factory.create_adder(factory.create_running_batch(), rem_chunk_tokens=64)
    assert adder.add_one_req(req, False, None) == AddReqResult.NO_TOKEN
    assert req.context_force_miss and req.context_window_plan is None
    assert not adder.can_run_list
    factory.mock_tree_cache.dec_lock_ref.assert_called_once()
    factory.mock_token_allocator.alloc.assert_not_called()

    cache = factory.mock_tree_cache
    cache.swa_reprefill_tail_tokens.return_value = 0
    cache.match_prefix.return_value = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=cache.root, last_host_node=cache.root,
        best_match_node=cache.root,
    )
    req.init_next_round_input(cache)
    assert len(cache.match_prefix.call_args.args[0].key) == 0
    assert req.context_recovery_plan.start == 0
    assert req.context_state is None and req.context_usage.resident.size == 0
    factory.mock_token_allocator.available_size.return_value = 180
    assert adder.add_one_req(req, False, None) in (
        AddReqResult.CONTINUE, AddReqResult.OTHER
    )
    assert adder.can_run_list == [req] and req.extend_range.start == 0


@pytest.mark.parametrize("chunk_limit", [64, None])
def test_repair_admission_excludes_reused_gap_from_total_and_tile_budget(
    factory, compiler, chunk_limit
):
    import torch
    from sglang.srt.managers.schedule_policy import AddReqResult

    req = make_req(compiler)
    req.prefix_indices = torch.arange(1, 101, dtype=torch.int64)
    req.context_source_positions = req.context_program.layout.positions[:100]
    req.context_resident = torch.ones(100, dtype=torch.bool)
    req.context_resident[36:40] = False
    req.context_exact_prefix_len = 100
    req.kv.cache_protected_len = 100
    req.prepare_context_recovery()
    recovery = req.context_recovery_plan
    assert recovery.start < 100
    assert recovery.remaining_queries(recovery.start) < 128 - recovery.start
    factory.mock_token_allocator.available_size.return_value = 110
    adder = factory.create_adder(
        factory.create_running_batch(), rem_chunk_tokens=chunk_limit
    )
    verdict = adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
    assert verdict in (AddReqResult.CONTINUE, AddReqResult.OTHER)
    assert adder.can_run_list == [req]
    first, end = recovery.intervals[0]
    assert req.extend_range.start == first
    assert req.extend_range.end == end
    assert adder.new_chunked_req is req
    assert adder.log_input_tokens == end - first
    assert adder.rem_input_tokens == 10000 - (end - first)
    factory.mock_token_allocator.alloc.assert_not_called()
