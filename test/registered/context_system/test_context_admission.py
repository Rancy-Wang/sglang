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
        drain_context_capacity_abort=Mock(side_effect=lambda *_: (sender.abort(), True)[1]),
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


@pytest.mark.parametrize("prefill", [False, True])
def test_continuation_capacity_abort_uses_native_cleanup(prefill):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler

    req = SimpleNamespace(
        rid="capacity-continuation", context_admission_error="retains 108 KV tokens",
        time_stats=SimpleNamespace(trace_ctx=Mock()), to_finish=None,
        disagg_kv_sender=Mock(), metadata_buffer_index=4, pending_bootstrap=True,
        return_logprob=False,
    )
    scheduler = SimpleNamespace(
        _pending_chunked_abort_req=req, chunked_req=req,
        disaggregation_mode=DisaggregationMode.PREFILL if prefill else DisaggregationMode.NULL,
        _release_aborted_request=Mock(), tree_cache=Mock(),
        drain_context_capacity_abort=Mock(return_value=True),
        clear_pending_chunk_send=Mock(), req_to_metadata_buffer_idx_allocator=Mock(),
        ipc_channels=SimpleNamespace(send_to_tokenizer=Mock()),
    )
    with (
        patch("sglang.srt.managers.scheduler.release_kv_cache") as release,
        patch("sglang.srt.managers.scheduler._make_abort_req") as notification,
    ):
        if prefill:
            scheduler.drain_context_capacity_abort.return_value = False
            Scheduler.process_pending_chunked_abort(scheduler)
            release.assert_not_called()
            scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
            assert scheduler.chunked_req is req
            scheduler.drain_context_capacity_abort.return_value = True
        Scheduler.process_pending_chunked_abort(scheduler)
        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        reason = notification.call_args.kwargs["finished_reason"]
        assert reason["status_code"] == 503
        assert "retains 108" in reason["message"]
        Scheduler.process_pending_chunked_abort(scheduler)
        release.assert_called_once()  # retrying the scheduling step cannot double-free
    assert scheduler.chunked_req is None and scheduler._pending_chunked_abort_req is None
    if prefill:
        scheduler.clear_pending_chunk_send.assert_called_once_with(req)
        req.disagg_kv_sender.abort.assert_not_called()  # drain owns transport abort
        assert scheduler.drain_context_capacity_abort.call_count == 2
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(4)
        assert not req.pending_bootstrap and req.metadata_buffer_index == -1
    else:
        req.disagg_kv_sender.abort.assert_not_called()



def test_capacity_abort_waits_for_all_rank_transfers_before_notifying_decode():
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    room = 57
    targets = [("127.0.0.1", 1234)]
    manager = SimpleNamespace(
        _staging_outstanding={room: 2},
        _room_notify_targets=Mock(return_value=targets),
        conclude_failure=Mock(), failure_lock=threading.Lock(),
        failure_records={room: "aborted"},
    )
    sender = Mock(kv_mgr=manager, bootstrap_room=room)
    req = SimpleNamespace(disagg_kv_sender=sender)
    scheduler = SimpleNamespace(attn_tp_cpu_group=object())
    drain = SchedulerDisaggregationPrefillMixin.drain_context_capacity_abort
    with patch("torch.distributed.all_reduce") as reduce:
        assert not drain(scheduler, req, "capacity")
        manager.conclude_failure.assert_not_called()
        sender.clear.assert_not_called()
        manager._staging_outstanding[room] = 0
        # Local writes ended, but another TP rank still owns a write.
        reduce.side_effect = lambda tensor, **_: tensor.fill_(1)
        assert not drain(scheduler, req, "capacity")
        manager.conclude_failure.assert_not_called()
        sender.clear.assert_not_called()
        reduce.side_effect = None
        assert drain(scheduler, req, "capacity")
        assert reduce.call_count == 3
    sender.abort.assert_called_once()
    manager._room_notify_targets.assert_called_once_with(room)
    manager.conclude_failure.assert_called_once_with(
        bootstrap_room=room, failure_reason="capacity", targets=targets
    )
    sender.clear.assert_called_once()
    assert room not in manager.failure_records


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


@pytest.mark.parametrize("existing_chunk", [False, True])
def test_short_repair_interval_preserves_single_chunk_slot(factory, compiler, existing_chunk):
    import torch
    from sglang.srt.managers.schedule_policy import AddReqResult

    factory.mock_token_allocator.available_size.return_value = 4096
    adder = factory.create_adder(factory.create_running_batch(), rem_chunk_tokens=128)
    first = make_req(compiler)
    first.prefix_indices = torch.arange(1, 101, dtype=torch.int64)
    first.context_source_positions = first.context_program.layout.positions[:100]
    first.context_resident = torch.ones(100, dtype=torch.bool)
    first.context_resident[36:40] = False
    first.context_exact_prefix_len = 100
    first.kv.cache_protected_len = 100
    first.prepare_context_recovery()
    if existing_chunk:
        assert adder.add_chunked_req(first) is first
    else:
        assert adder.add_one_req(first, False, None) == AddReqResult.CONTINUE
    assert 0 < adder.rem_chunk_tokens < 128
    second = make_req(compiler)
    assert adder.add_one_req(second, existing_chunk, None) == AddReqResult.OTHER
    assert adder.can_run_list == [first]
    assert adder.new_chunked_req is (None if existing_chunk else first)
    assert second.context_window_plan is None
    # Spare compute can still admit a request which finishes its entire prefill.
    third = make_req(compiler)
    third.prefix_indices = torch.arange(1, 128, dtype=torch.int64)
    third.context_source_positions = third.context_program.layout.positions[:127]
    third.context_exact_prefix_len = 127
    third.prepare_context_recovery()
    assert adder.add_one_req(third, existing_chunk, None) == AddReqResult.CONTINUE
    assert adder.can_run_list == [first, third]


@pytest.mark.parametrize("capacity_error", [None, "impossible"])
def test_parked_chunk_returns_control_to_decode(factory, capacity_error):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from sglang.srt.managers.scheduler import Scheduler

    req = SimpleNamespace(
        context_admission_error=capacity_error, init_next_round_input=Mock(),
        inflight_middle_chunks=0,
    )
    adder = Mock(can_run_list=[], add_chunked_req=Mock(return_value=req))
    running = factory.create_running_batch()
    running.batch_is_full = False
    scheduler = SimpleNamespace(
        grammar_manager=Mock(has_waiting_grammars=Mock(return_value=False)),
        enable_priority_preemption=False, is_hybrid_swa=False,
        waiting_queue=[object()], chunked_req=req, min_free_slots_delayer=None,
        get_num_allocatable_reqs=Mock(return_value=4), policy=Mock(),
        processed_tokens_counter=0, chunked_prefill_size=32, dynamic_chunk_sizer=None,
        tp_worker=Mock(), page_size=1, tree_cache=factory.mock_tree_cache,
        token_to_kv_pool_allocator=factory.mock_token_allocator,
        new_token_ratio_tracker=SimpleNamespace(current=1.0), max_prefill_tokens=32,
        is_mixed_chunk=False, priority_scheduling_preemption_threshold=0,
        max_prefill_bs=4, max_running_requests=4, dllm_config=None,
        _pending_chunked_abort_req=None,
    )
    with patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder):
        batch, remaining = Scheduler._get_new_batch_prefill_raw(scheduler, None, running)
    assert batch is None and remaining is running
    assert scheduler.chunked_req is req and req.inflight_middle_chunks == 0
    assert scheduler._pending_chunked_abort_req is (req if capacity_error else None)
    adder.add_one_req.assert_not_called()
