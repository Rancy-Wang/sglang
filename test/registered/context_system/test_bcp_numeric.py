"""Native HTTP BCP comparison with mini default logits and actual tokens."""

import json
import os
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
torch = pytest.importorskip("torch")

from bcp_numeric_fixture import request_for
from serving_logits_probe import serialized_probe
from test_serving_runtime import server  # noqa: F401

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_BCP_ORACLE"), reason="explicit BCP reference required"
)


def test_bcp_default_reference(server):  # noqa: F811
    reference_path = Path(os.environ["CONTEXT_BCP_ORACLE"])
    reference = json.loads(reference_path.read_text())
    mode = os.environ.get("CONTEXT_BCP_MODE", "all")
    assert mode in ("all", "fixed", "actual")
    reference_logits = (
        torch.load(str(reference_path) + ".pt", weights_only=True)
        if mode != "actual"
        else None
    )
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = {}

    def call(feature, name, fixed, *, warm_source=False):
        (record,) = reference["runs"][feature]["records"]
        tokens = record["tokens"]
        if warm_source:
            tokens = tokens[:1]
        path = directory / (name + ".pt")
        params = {"context_trace_path": str(path), "context_trace_count": len(tokens)}
        if fixed:
            params["context_forced_tokens"] = tokens
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
        if warm_source:
            payload["reposition"] = reference["fixture"]["reposition"][:1]
        if fixed:
            payload.update(
                custom_logit_processor=serialized_probe(), custom_params=params
            )
        grammar_model = (
            os.environ.get("CONTEXT_BCP_GRAMMAR_MODEL") if not fixed else None
        )
        if grammar_model:
            import xgrammar

            # Exercise SGLang's public native grammar interface with the same
            # descriptor as mini's default tool grammar. No sampler replacement.
            payload["response_format"] = xgrammar.get_model_structural_tag(
                grammar_model,
                tools=reference["fixture"]["tools"],
                tool_choice="auto",
                reasoning=False,
            ).model_dump()
        response_path = directory / (name + ".json")
        response = requests.post(
            server + "/v1/chat/completions", json=payload, timeout=300
        )
        response_path.write_text(response.text)
        assert response.status_code == 200, response.text
        response = response.json()
        ids = response["sglext"]["input_ids"]
        expected_ids = reference["runs"]["none"]["records"][0]["input"]["ids"]
        # SGLang returns raw input; mini's legacy Drop record is compact active.
        assert ids == expected_ids, (name, len(ids), len(expected_ids))
        (output_ids,) = response["sglext"]["output_ids"]
        same = [a == b for a, b in zip(output_ids, tokens)]
        if warm_source:
            assert output_ids == tokens
            return response  # Cache setup has different events, so no logits oracle.
        if not fixed:
            choice = response["choices"][0]
            reference_choice = reference["runs"][feature]["responses"][0]["choices"][0]
            item = {
                "comparison_kind": "native_generation_observation",
                "same_input_tokens": ids == expected_ids,
                "reference_input_tokens": len(expected_ids),
                "matching_tokens": sum(same),
                "generated_tokens": len(output_ids),
                "reference_tokens": len(tokens),
                "exact_token_match": output_ids == tokens,
                "first_token_difference": next(
                    (i for i, equal in enumerate(same) if not equal),
                    len(same) if len(output_ids) != len(tokens) else None,
                ),
                "message": choice["message"],
                "reference_message": reference_choice["message"],
                "finish_reason": choice["finish_reason"],
                "reference_finish_reason": reference_choice["finish_reason"],
                "context_usage": choice.get("meta_info", {}).get("context_usage"),
                "grammar_model": grammar_model,
                "input_tokens": len(ids),
            }
            assert output_ids and choice["finish_reason"] in (
                "length",
                "stop",
                "tool_calls",
            ), response
            # Free-running contexts diverge after the first differing sample;
            # their later logits cannot serve as a fixed-token numerical gate.
            comparisons[name] = item
            (directory / "comparison.json").write_text(
                json.dumps(comparisons, indent=2)
            )
            print("BCP_ACTUAL_OBSERVATION", name, json.dumps(item), flush=True)
            return response
        assert output_ids == tokens, name
        assert path.exists(), name
        logits = torch.load(path, weights_only=True)
        reference_key = (
            name
            if name in ("drop_repos-retry", "drop_repos-consecutive")
            and name in reference_logits
            else feature
        )
        expected = reference_logits[reference_key]
        assert logits.shape == expected.shape, (name, logits.shape, expected.shape)
        assert torch.isfinite(logits).all(), name
        delta = (logits - expected).abs()
        item = {
            "reference_key": reference_key,
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "p99_abs": torch.quantile(delta.flatten(), 0.99).item(),
            "per_token_max": delta.max(-1).values.tolist(),
            "per_token_mean": delta.mean(-1).tolist(),
            "matching_tokens": sum(same),
            "tokens": len(tokens),
            "first_token_difference": next(
                (i for i, equal in enumerate(same) if not equal), None
            ),
            "context_usage": response["choices"][0]
            .get("meta_info", {})
            .get("context_usage"),
            "input_tokens": len(ids),
            "grammar_model": grammar_model,
            "raw_argmax_matching_tokens": int(
                (logits.argmax(-1) == expected.argmax(-1)).sum()
            ),
        }
        comparisons[name] = item
        (directory / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        print("BCP_COMPARE", name, json.dumps(item), flush=True)
        return response

    consecutive_only = os.environ.get("CONTEXT_BCP_CONSECUTIVE_ONLY") == "1"
    for feature in (() if consecutive_only else ("none", "drop", "drop_repos")):
        for fixed in (
            (False,)
            if mode == "actual"
            else (True,)
            if mode == "fixed"
            else (False, True)
        ):
            assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
            call(feature, f"{feature}-{'fixed' if fixed else 'actual'}", fixed)
    if mode == "actual":
        return
    # Same real task tests both direct final-version reuse and compatible Retry.
    if not consecutive_only:
        call("drop_repos", "drop_repos-hot", True)
        assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
        call("none", "none-retry-source", True)
        call("drop_repos", "drop_repos-retry", True)
    assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
    call("drop_repos", "consecutive-source", True, warm_source=True)
    call("drop_repos", "drop_repos-consecutive", True)
    usage = comparisons["drop_repos-consecutive"]["context_usage"]
    assert usage["cached_tokens"] + usage["repos_tokens"] > 0
    assert usage["actual_prefill_tokens"] < comparisons["drop_repos-consecutive"]["input_tokens"]
    baseline = (
        json.loads(Path(os.environ["CONTEXT_BCP_CALIBRATION"]).read_text())["none-fixed"]
        if consecutive_only else comparisons["none-fixed"]
    )
    names = ("drop_repos-consecutive",) if consecutive_only else (
        "drop-fixed",
        "drop_repos-fixed",
        "drop_repos-hot",
        "drop_repos-retry",
        "drop_repos-consecutive",
    )
    for name in names:
        item = comparisons[name]
        # Initial BF16 cross-backend gate, tied to this model's native control.
        for metric, floor in (
            ("max_abs", 0.125),
            ("mean_abs", 0.02),
            ("p99_abs", 0.0625),
        ):
            assert item[metric] <= max(floor, 2 * baseline[metric]), (
                name,
                metric,
                item,
                baseline,
            )
