"""Native scheduler/cache integration and opt-in real BCP HTTP validation."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from test_ir import ROOT, args, load_file
from test_serving_runtime import server  # noqa: F401

pytest_plugins = ("test_ir",)


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
@pytest.mark.parametrize("mode", ["occurrence", "staged"])
def test_execution_mode_cli_and_request(mode):
    from sglang.srt.runtime_context import get_schedule, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = ServerArgs.from_cli_args(parser.parse_args(
        ["--model-path", "dummy", "--page-size", "1", "--context-execution-mode", mode]
    ))
    reset_context()
    try:
        publish(parsed, role="scheduler")
        assert get_schedule().context_execution_mode == mode
    finally:
        reset_context()


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
def test_intermediate_cache_keeps_private_state_and_source_lease():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.context_system.occurrence import OccurrenceState

    state = OccurrenceState.from_match(torch.arange(1, 5), torch.arange(4, dtype=torch.int32), exact_prefix_len=2)
    state.terminal_rows[:] = state.canonical_rows
    state.owned[2:] = True
    req = SimpleNamespace(context_staged=True, context_state=state,
                          context_program=SimpleNamespace(layout=SimpleNamespace(positions=torch.empty(10))),
                          context_recompute_program=None, kv=SimpleNamespace(req_pool_idx=0),
                          context_source_lease=object(), rid="stage", prefix_indices=None)
    cache = Mock(spec=UnifiedRadixCache)
    cache.req_to_token_pool = Mock()
    lease = req.context_source_lease
    method = UnifiedRadixCache.cache_unfinished_req
    getattr(method, "__wrapped__", method)(cache, req, chunked=True)
    cache._cache_unfinished_req_native.assert_not_called()
    assert req.context_state is state and req.context_source_lease is lease
    assert req.prefix_indices.tolist() == [1, 2, 3, 4]


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
@pytest.mark.parametrize("start", [0, 3, 6, 9, 19])
def test_native_admission_stops_at_next_event(compiler, start):
    from sglang.srt.managers.schedule_policy import PrefillAdder, _PrefillAdmission
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from test_occurrence_ownership import expiry_for
    from array import array

    reset_context()
    publish(ServerArgs(model_path="dummy", page_size=1, context_execution_mode="staged"), role="scheduler")
    try:
        drops = {6: [(0, 2)], 14: [(2, 4)]}
        program = ContextProgram(compiler(*args(list(range(20)), drops, [8, 13])), expiry_for(20, drops))
        req = Req("stage", "", array("q", range(20)), SamplingParams(max_new_tokens=4), context_program=program.to_wire(), context_staged=True)
        req._refresh_fill_ids()
        req.prefix_indices = torch.arange(1, start + 1, dtype=torch.int64)
        req.context_source_positions = program.layout.positions[:start]
        req.context_exact_prefix_len = start
        req.prepare_context_recovery()
        adder = SimpleNamespace(_context_decode_reserved=True, page_size=1, dllm_config=None,
                                rem_total_tokens=4096, cur_rem_tokens=4096, is_hybrid_swa=False)
        admission = PrefillAdder._fit_context_admission(adder, req, _PrefillAdmission(start, 20-start, 4, False))
        expected = next(x for x in [6, 9, 14, 20] if x > start)
        assert admission.extend_len == expected - start
        assert admission.is_chunked == (expected < 20)
        assert admission.max_new_tokens == (0 if expected < 20 else 4)
        assert req.context_window_plan[0].segment_count == 1
    finally:
        reset_context()


@pytest.mark.skipif(not os.environ.get("CONTEXT_BCP_ORACLE"), reason="explicit BCP staged oracle required")
def test_bcp_staged_reference(server):  # noqa: F811
    from test_bcp_numeric import test_bcp_default_reference

    reference = json.loads(Path(os.environ["CONTEXT_BCP_ORACLE"]).read_text())
    assert reference["reference_mode"] == "staged"
    extra = json.loads(os.environ.get("CONTEXT_SERVER_EXTRA_ARGS", "[]"))
    assert "--context-execution-mode" in extra
    assert extra[extra.index("--context-execution-mode") + 1] in ("staged", "occurrence")
    test_bcp_default_reference(server)
