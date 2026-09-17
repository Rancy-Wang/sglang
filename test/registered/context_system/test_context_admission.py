"""Native prefill admission must pay for copies without charging model queries."""

import sys
from array import array

import pytest
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")


@pytest.fixture
def factory():
    module = load_file(
        "context_native_prefill_fixture",
        ROOT / "test/registered/unit/managers/test_prefill_adder.py",
    )
    fixture = module.TestPrefillAdder()
    fixture.setUp()
    fixture.mock_tree_cache.supports_mamba.return_value = False
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
