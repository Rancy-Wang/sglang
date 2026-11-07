"""Native scheduler/cache integration and opt-in real BCP HTTP validation."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_ir import ROOT, args, load_file
from test_drop_eviction import native_cache  # noqa: F401
from test_serving_runtime import server  # noqa: F401

pytest_plugins = ("test_ir",)


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")
@pytest.mark.parametrize("mode", ["occurrence", "staged"])
def test_execution_mode_cli_and_request(mode):
    from sglang.srt.runtime_context import get_schedule, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = ServerArgs.from_cli_args(
        parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--page-size",
                "1",
                "--context-execution-mode",
                mode,
            ]
        )
    )
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

    state = OccurrenceState.from_match(
        torch.arange(1, 5), torch.arange(4, dtype=torch.int32), exact_prefix_len=2
    )
    state.terminal_rows[:] = state.canonical_rows
    state.owned[2:] = True
    req = SimpleNamespace(
        context_staged=True,
        context_state=state,
        context_program=SimpleNamespace(
            layout=SimpleNamespace(positions=torch.empty(10))
        ),
        context_recompute_program=None,
        kv=SimpleNamespace(req_pool_idx=0),
        context_source_lease=object(),
        rid="stage",
        prefix_indices=None,
    )
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
    publish(
        ServerArgs(model_path="dummy", page_size=1, context_execution_mode="staged"),
        role="scheduler",
    )
    try:
        drops = {6: [(0, 2)], 14: [(2, 4)]}
        program = ContextProgram(
            compiler(*args(list(range(20)), drops, [8, 13])), expiry_for(20, drops)
        )
        req = Req(
            "stage",
            "",
            array("q", range(20)),
            SamplingParams(max_new_tokens=4),
            context_program=program.to_wire(),
            context_staged=True,
        )
        req._refresh_fill_ids()
        req.prefix_indices = torch.arange(1, start + 1, dtype=torch.int64)
        req.context_source_positions = program.layout.positions[:start]
        req.context_exact_prefix_len = start
        req.prepare_context_recovery()
        adder = SimpleNamespace(
            _context_decode_reserved=True,
            page_size=1,
            dllm_config=None,
            rem_total_tokens=4096,
            cur_rem_tokens=4096,
            is_hybrid_swa=False,
        )
        admission = PrefillAdder._fit_context_admission(
            adder, req, _PrefillAdmission(start, 20 - start, 4, False)
        )
        expected = next(x for x in [6, 9, 14, 20] if x > start)
        assert admission.extend_len == expected - start
        assert admission.is_chunked == (expected < 20)
        assert admission.max_new_tokens == (0 if expected < 20 else 4)
        assert req.context_window_plan[0].segment_count == 1
    finally:
        reset_context()


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("source_cache", [False, True])
def test_staged_native_ownership_cleanup(compiler, native_cache, abort, source_cache):
    from array import array

    from sglang.srt.context_system.occurrence import free_context_slots
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.context_system.staged import StagePlan
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs
    from test_drop_eviction import assert_allocator
    from test_occurrence_ownership import expiry_for

    cache, allocator = native_cache
    reset_context()
    publish(ServerArgs(model_path="dummy", page_size=1), role="scheduler")
    try:
        tokens, drops, repos = list(range(48)), {16: [(2, 6)], 32: [(8, 12)]}, [15, 31]
        program = ContextProgram(
            compiler(*args(tokens, drops, repos)), expiry_for(48, drops)
        )
        if source_cache:
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens[:20])), value=allocator.alloc(20)
                )
            )
        req = Req(
            "staged-owners",
            "",
            array("q", tokens),
            SamplingParams(max_new_tokens=1),
            context_program=program.to_wire(),
            context_staged=True,
        )
        pool = ReqToTokenPool(2, 64, "cpu", False)
        cache.req_to_token_pool = pool
        pool.alloc([req])
        req.init_next_round_input(cache)
        PrefillAdder._req_inc_lock_ref(SimpleNamespace(tree_cache=cache), req)
        stage_plan = StagePlan.from_program(program)
        end = 0
        while end < 48:
            req.advance_context_recovery_gap()
            req.commit_context_recovery_gap()
            start = len(req.prefix_indices)
            _, interval_end = req.context_recovery_plan.next_interval(start)
            end = stage_plan.next_end(start, min(start + 7, interval_end))
            extra = req.plan_context_prefill(end)
            window, plan = req.context_window_plan
            advance = req.context_state.advance(
                window,
                plan,
                allocator.alloc(end - start),
                allocator.alloc(extra),
                program.visible_until,
            )
            if hasattr(allocator, "free_swa"):
                allocator.free_swa(advance.unused_swa_slots)
            req.context_state = advance.state
            req.context_prefill_started = True
            req.set_extend_range(start, end)
            req.kv.kv_committed_len = end
            free_context_slots(
                allocator, advance.retired_slots, advance.retired_swa_resident
            )
            cache.cache_unfinished_req(req, chunked=end < 48)
            cache.evict(EvictParams(num_tokens=128, swa_num_tokens=128))
            cache.sanity_check()
            if end < 48:
                assert not req.context_cache_published
            if abort and end >= 24:
                break
        cache.cache_finished_req(req, kv_len_to_handle=end)
        assert req.context_state is None and req.context_source_lease is None
        pool.free(req)
        cache.evict(EvictParams(num_tokens=128, swa_num_tokens=128))
        cache.sanity_check()
        assert_allocator(cache, allocator, 0)
        if hasattr(allocator, "swa_attn_allocator"):
            assert allocator.swa_attn_allocator.available_size() == 128
    finally:
        reset_context()


@pytest.mark.parametrize("unsupported", ["insert", "hybrid", "pd"])
def test_staged_rejects_unsupported_execution(unsupported):
    capabilities = load_file(
        "staged_capabilities", ROOT / "python/sglang/srt/context_system/capabilities.py"
    )
    config = SimpleNamespace(
        page_size=1,
        context_execution_mode="staged",
        disaggregation_mode="prefill" if unsupported == "pd" else "null",
    )
    architecture = (
        "Qwen3_5ForCausalLM" if unsupported == "hybrid" else "Qwen3ForCausalLM"
    )
    model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture]))
    with pytest.raises(ValueError, match="staged execution does not yet support"):
        capabilities.validate_context_config(
            config, model, has_insert=unsupported == "insert"
        )


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_BCP_ORACLE"),
    reason="explicit BCP staged oracle required",
)
def test_bcp_staged_reference(server):  # noqa: F811
    import requests

    from bcp_numeric_fixture import request_for
    from serving_logits_probe import serialized_probe

    reference_path = Path(os.environ["CONTEXT_BCP_ORACLE"])
    reference = json.loads(reference_path.read_text())
    assert reference["reference_mode"] == "staged"
    extra = json.loads(os.environ.get("CONTEXT_SERVER_EXTRA_ARGS", "[]"))
    assert "--context-execution-mode" in extra
    assert extra[extra.index("--context-execution-mode") + 1] in (
        "staged",
        "occurrence",
    )
    mode = os.environ.get("CONTEXT_BCP_MODE", "all")
    assert mode in ("all", "fixed", "actual")
    expected_logits = torch.load(str(reference_path) + ".pt", weights_only=True)
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    expected_input = reference["runs"]["none"]["records"][0]["input"]["ids"]

    def flush():
        assert requests.post(server + "/flush_cache", timeout=5).status_code == 200

    def call(feature, name, reference_key, *, fixed=True, source=False):
        (record,) = reference["runs"][reference_key]["records"]
        tokens = record["tokens"]
        payload = {
            **request_for(reference["fixture"], feature),
            "model": os.environ["CONTEXT_SERVER_MODEL"],
            "temperature": 0,
            "max_tokens": len(tokens),
            "ignore_eos": True,
            "return_meta_info": True,
            "return_input_ids_in_sglext": True,
            "return_output_ids_in_sglext": True,
        }
        if source:
            payload["reposition"] = reference["fixture"]["reposition"][:1]
        trace = directory / (name + ".pt")
        if fixed:
            payload.update(
                custom_logit_processor=serialized_probe(),
                custom_params={
                    "context_trace_path": str(trace),
                    "context_trace_count": len(tokens),
                    "context_forced_tokens": tokens,
                },
            )
        response = requests.post(
            server + "/v1/chat/completions", json=payload, timeout=300
        )
        (directory / (name + ".json")).write_text(response.text)
        assert response.status_code == 200, response.text
        response = response.json()
        assert response["sglext"]["input_ids"] == expected_input, name
        (actual_tokens,) = response["sglext"]["output_ids"]
        assert len(actual_tokens) == len(tokens), (name, actual_tokens, tokens)
        value = {
            "reference_key": reference_key,
            "context_usage": response["choices"][0]
            .get("meta_info", {})
            .get("context_usage"),
            "input_tokens": len(expected_input),
            "tokens": len(tokens),
            "exact_token_match": actual_tokens == tokens,
        }
        if fixed:
            assert actual_tokens == tokens, name
            actual, expected = (
                torch.load(trace, weights_only=True),
                expected_logits[reference_key],
            )
            assert actual.shape == expected.shape and torch.isfinite(actual).all(), name
            delta = (actual - expected).abs()
            value.update(
                max_abs=delta.max().item(),
                mean_abs=delta.mean().item(),
                p99_abs=torch.quantile(delta.flatten(), 0.99).item(),
                per_token_max=delta.max(-1).values.tolist(),
                per_token_mean=delta.mean(-1).tolist(),
                raw_argmax_matching_tokens=int(
                    (actual.argmax(-1) == expected.argmax(-1)).sum()
                ),
            )
        else:
            value.update(
                comparison_kind="native_generation_observation",
                message=response["choices"][0]["message"],
                reference_message=reference["runs"][reference_key]["responses"][0][
                    "choices"
                ][0]["message"],
            )
        comparisons[name] = value
        (directory / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        print("BCP_STAGED_COMPARE", name, json.dumps(value), flush=True)

    for feature in ("none", "drop", "drop_repos"):
        if mode != "fixed":
            flush()
            call(feature, feature + "-actual", feature, fixed=False)
        if mode != "actual":
            flush()
            call(feature, feature + "-fixed", feature)
    if mode == "actual":
        return
    # Pair each reuse run with the same mini cache history, including hot reuse.
    call("drop_repos", "drop_repos-hot", "drop_repos-hot")
    flush()
    call("none", "none-retry-source", "none-retry-source")
    call("drop_repos", "drop_repos-retry", "drop_repos-retry")
    flush()
    call("drop_repos", "consecutive-source", "consecutive-source", source=True)
    call("drop_repos", "drop_repos-consecutive", "drop_repos-consecutive")
    usage = comparisons["drop_repos-consecutive"]["context_usage"]
    assert usage["cached_tokens"] + usage["repos_tokens"] > 0
    assert usage["actual_prefill_tokens"] < len(expected_input)

    baseline, failures = comparisons["none-fixed"], []
    for name, value in comparisons.items():
        if not name.startswith("drop") or "max_abs" not in value:
            continue
        for metric, floor in (
            ("max_abs", 0.125),
            ("mean_abs", 0.02),
            ("p99_abs", 0.0625),
        ):
            limit = max(floor, 2 * baseline[metric])
            if value[metric] > limit:
                failures.append((name, metric, value[metric], limit))
        for metric, floor in (("per_token_max", 0.125), ("per_token_mean", 0.02)):
            for index, (actual, control) in enumerate(
                zip(value[metric], baseline[metric], strict=True)
            ):
                limit = max(floor, 2 * control)
                if actual > limit:
                    failures.append((name, f"{metric}[{index}]", actual, limit))
    (directory / "gate-failures.json").write_text(json.dumps(failures, indent=2))
    assert not failures, failures
