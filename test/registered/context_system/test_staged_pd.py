"""Final-stage handoff guards and native sender integration."""

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
        keep = state.prefill_terminal_keep(layout, end, staged=True)
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
        SchedulerDisaggregationPrefillMixin._send_kv_chunk(
            SimpleNamespace(), req, last_chunk=False, end_idx=end
        )
        assert req.start_send_idx == 7


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_staged_pd_capability(role, monkeypatch):
    from sglang.srt.context_system.capabilities import validate_context_request
    from sglang.srt.context_system.planner import ContextProgram
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
    req = SimpleNamespace(input_ids=[1], input_embeds=None, contains_mm_input=lambda:False,
                          session_id=None, session_params=None, lora_path=None,
                          sampling_params={}, context_program={})
    sentinel = object()
    monkeypatch.setattr(ContextProgram, 'from_wire', lambda *a: sentinel)
    with envs.SGLANG_DISAGG_STAGING_BUFFER.override(False):
        assert validate_context_request(config, model, req) is sentinel
        config.disaggregation_transfer_backend = "fake"
        with pytest.raises(ValueError, match="Mooncake"):
            validate_context_request(config, model, req)
