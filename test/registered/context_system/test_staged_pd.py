"""Final-stage handoff guards and native sender integration."""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_context_transfer import transfer_modules  # noqa: F401
from test_ir import ROOT, args, load_file
from test_planner import query_visibility

pytest_plugins = ("test_ir",)


@pytest.mark.parametrize("chunk", [1, 5, 100])
@pytest.mark.parametrize("tail_reposition", [False, True])
def test_only_final_stage_is_valid_handoff(compiler, transfer_modules, chunk, tail_reposition):
    transfer, _ = transfer_modules
    occurrence = sys.modules["sglang.srt.context_system.occurrence"]
    stages = load_file("pd_stages", ROOT / "python/sglang/srt/context_system/staged.py")
    n, drops = 20, {6: [(0, 3)], 14: [(3, 5)]}
    repos = [8, 16] + ([19] if tail_reposition else [])
    layout = compiler(*args(list(range(n)), drops, repos))
    _, expiry = query_visibility(list(range(n)), drops, repos)
    program = SimpleNamespace(layout=layout, visible_until=expiry)
    req = SimpleNamespace(context_program=program, context_recompute_program=None)
    plan = transfer.transfer_plan(req, "cpu")
    req.context_state = occurrence.OccurrenceState.from_match(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32), exact_prefix_len=0
    )
    frontier = stages.StagePlan.from_program(program)
    start, slot = 0, 1
    while start < n:
        end = frontier.next_end(start, min(n, start + chunk))
        window = occurrence.compile_occurrence_window(
            layout, expiry, layout.positions, query_start=start, query_end=end, staged=True
        )
        state = req.context_state
        keep = state.prefill_terminal_keep(layout, end, expiry)
        if end < n:
            keep &= expiry[:end] > end
        materialization = state.plan(window, keep, staged=True)
        count = end - start
        step = state.advance(
            window, materialization,
            torch.arange(slot, slot + count),
            torch.arange(slot + count, slot + count + materialization.extra_page_count),
            expiry,
        )
        slot += count + materialization.extra_page_count
        req.context_state = step.state
        if end < n:
            with pytest.raises(ValueError, match="completed prefill"):
                transfer.validate_staged_handoff(req, plan)
        start = end
    transfer.validate_staged_handoff(req, plan)
    state = req.context_state
    rows = state.terminal_rows[torch.from_numpy(plan.decode.raw_indices.astype('int64'))]
    swa = torch.zeros(len(state.slots), dtype=torch.bool)
    swa[rows[plan.swa_start(4):]] = True
    req.context_state = replace(state, swa_resident=swa)
    transfer.validate_staged_handoff(req, plan, window=4)
    swa[rows[-1]] = False
    with pytest.raises(ValueError, match="missing final SWA"):
        transfer.validate_staged_handoff(req, plan, window=4)
    req.context_state = state
    raw = int(plan.decode.raw_indices[0])
    old = int(state.canonical_positions[raw])
    state.canonical_positions[raw] = old + 1
    with pytest.raises(ValueError, match="position transitions"):
        transfer.validate_staged_handoff(req, plan)
    state.canonical_positions[raw] = old
    state.terminal_rows[raw] = -1
    with pytest.raises(ValueError, match="missing final KV"):
        transfer.validate_staged_handoff(req, plan)


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
def test_native_sender_defers_every_intermediate_stage():
    from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

    # No allocator, sender or pool access is allowed, even if the scheduler
    # has already advanced the request state to its final overlapping forward.
    req = SimpleNamespace(context_staged=True, start_send_idx=7)
    for end in (8, 100):
        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            SimpleNamespace(), req, last_chunk=False, end_idx=end
        )
        assert req.start_send_idx == 7


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_staged_pd_capability(role):
    from sglang.srt.context_system.capabilities import validate_context_config
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.environ import envs

    config = ServerArgs(
        model_path="dummy", page_size=1, attention_backend="triton",
        context_execution_mode="staged", disaggregation_mode=role,
        disaggregation_transfer_backend="mooncake",
    )
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        is_multimodal=False, dtype=torch.bfloat16,
    )
    with envs.SGLANG_DISAGG_STAGING_BUFFER.override(False):
        validate_context_config(config, model)
        config.disaggregation_transfer_backend = "fake"
        with pytest.raises(ValueError, match="Mooncake"):
            validate_context_config(config, model)


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_PD_LIFECYCLE"),
    reason="explicit isolated PD lifecycle launch required",
)
def test_http_rebootstrap_abort_and_recovery(pd_servers):
    import concurrent.futures
    import json
    import os
    import time
    from pathlib import Path

    import requests
    from sglang.srt.context_system.ir import compile_context_layout, prewarm_context_layout
    from sglang.srt.context_system.planner import ContextProgram

    p_base, d_base, bootstrap = pd_servers
    prewarm_context_layout()
    n = 256
    drops = {80: [(10, 40)], 180: [(90, 120)]}
    repos = [100, 255]  # Includes a zero-query terminal transition.
    layout = compile_context_layout(*args([785] * n, drops, repos))
    _, expiry = query_visibility([785] * n, drops, repos)
    program = ContextProgram(layout, expiry).to_json_wire()
    evidence = {}

    def payload(name, count):
        return dict(
            rid=f"staged-pd-{name}-{time.time_ns()}", input_ids=[785] * n,
            context_program=program, background=True,
            sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True),
            bootstrap_host="127.0.0.1", bootstrap_port=bootstrap,
            bootstrap_room=time.time_ns() % (1 << 52),
        )

    def post(base, body):
        response = requests.post(base + "/generate", json=body, timeout=180)
        return response.status_code, response.json()

    def wait_running(future):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert not future.done(), "request finished before lifecycle intervention"
            response = requests.get(d_base + "/v1/loads?include=core", timeout=5)
            response.raise_for_status()
            if sum(x["num_running_reqs"] for x in response.json()["loads"]) > 0:
                return
            time.sleep(.05)
        pytest.fail("decode did not reach running_batch")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Abort after a real intermediate stage has allocated/written KV.
        # The isolated launch enables DEBUG so the native admission log is
        # evidence of entering prefill, rather than merely waiting to bootstrap.
        body = payload("prefill-abort", 8)
        long_n = 12000
        long_drops = {4000: [(100, 1000)], 8000: [(4500, 5500)]}
        long_repos = [6000, 11000]
        long_layout = compile_context_layout(*args([785] * long_n, long_drops, long_repos))
        long_expiry = torch.full((long_n,), torch.iinfo(torch.int32).max, dtype=torch.int32)
        for boundary, spans in long_drops.items():
            for left, right in spans:
                long_expiry[left:right] = boundary
        body["input_ids"] = [785] * long_n
        body["context_program"] = ContextProgram(long_layout, long_expiry).to_json_wire()
        p, d = [executor.submit(post, base, body) for base in (p_base, d_base)]
        log = Path(os.environ["CONTEXT_TRACE_DIR"]) / "prefill-server.log"
        deadline = time.monotonic() + 60
        marker = "Context KV versions rid=" + body["rid"]
        while marker not in log.read_text():
            assert not p.done(), "prefill finished before intermediate-stage abort"
            assert time.monotonic() < deadline, "missing native stage admission evidence"
            time.sleep(.02)
        for base in (p_base, d_base):
            requests.post(base + "/abort_request", json={"rid": body["rid"]}, timeout=10).raise_for_status()
        p_result, d_result = p.result(), d.result()
        evidence["prefill-abort"] = dict(prefill=p_result, decode=d_result)
        assert d_result[0] != 200 or d_result[1]["meta_info"]["finish_reason"]["type"] == "abort"
        for base in (p_base, d_base):
            requests.post(base + "/flush_cache", timeout=10).raise_for_status()

        body = payload("rebootstrap", 512)
        p, d = [executor.submit(post, base, body) for base in (p_base, d_base)]
        try:
            wait_running(d)
            response = requests.post(d_base + "/pause_generation", json={"mode": "retract"}, timeout=30)
            response.raise_for_status()
            assert not d.done()
            # Flush P to force actual staged recompute, rather than a fully
            # warm rebootstrap that could hide an invalid historical stage.
            requests.post(p_base + "/flush_cache", timeout=10).raise_for_status()
        finally:
            requests.post(d_base + "/continue_generation", json={}, timeout=30).raise_for_status()
        p_result, d_result = p.result(), d.result()
        evidence["rebootstrap"] = dict(prefill=p_result, decode=d_result)
        assert p_result[0] == d_result[0] == 200, evidence
        meta = d_result[1]["meta_info"]
        assert meta["num_retractions"] > 0, meta
        assert meta["completion_tokens"] == 512, meta
        assert len(d_result[1]["output_ids"]) == 512

        body = payload("abort", 512)
        p, d = [executor.submit(post, base, body) for base in (p_base, d_base)]
        wait_running(d)
        for base in (p_base, d_base):
            requests.post(base + "/abort_request", json={"rid": body["rid"]}, timeout=10).raise_for_status()
        p_result, d_result = p.result(), d.result()
        evidence["abort"] = dict(prefill=p_result, decode=d_result)
        assert d_result[0] != 200 or d_result[1]["meta_info"]["finish_reason"]["type"] == "abort"

        for base in (p_base, d_base):
            requests.post(base + "/flush_cache", timeout=10).raise_for_status()
        body = payload("recovery", 8)
        p, d = [executor.submit(post, base, body) for base in (p_base, d_base)]
        results = p.result(), d.result()
        evidence["recovery"] = results
        assert all(status == 200 for status, _ in results), results
        assert results[1][1]["meta_info"]["completion_tokens"] == 8
    (Path(os.environ["CONTEXT_TRACE_DIR"]) / "lifecycle.json").write_text(json.dumps(evidence, indent=2))


# Shared fixture keeps startup flags and cleanup identical to the BCP runs.
from test_bcp_pd_numeric import pd_servers  # noqa: E402,F401
