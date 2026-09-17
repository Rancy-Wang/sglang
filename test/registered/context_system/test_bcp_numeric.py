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
    reference_logits = torch.load(str(reference_path) + ".pt", weights_only=True)
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = {}

    def call(feature, name, fixed):
        (record,) = reference["runs"][feature]["records"]
        tokens = record["tokens"]
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
            "custom_logit_processor": serialized_probe(),
            "custom_params": params,
        }
        response_path = directory / (name + ".json")
        if (
            not fixed
            and os.environ.get("CONTEXT_REUSE_BCP_ACTUAL") == "1"
            and path.exists()
            and response_path.exists()
        ):
            response = json.loads(response_path.read_text())
        else:
            response = requests.post(
                server + "/v1/chat/completions", json=payload, timeout=300
            )
            response_path.write_text(response.text)
            assert response.status_code == 200, response.text
            response = response.json()
        assert path.exists(), name
        logits = torch.load(path, weights_only=True)
        expected = reference_logits[feature]
        assert logits.shape == expected.shape, (name, logits.shape, expected.shape)
        assert torch.isfinite(logits).all(), name
        ids = response["sglext"]["input_ids"]
        expected_ids = reference["runs"]["none"]["records"][0]["input"]["ids"]
        # SGLang returns raw input; mini's legacy Drop record is compact active.
        assert ids == expected_ids, (name, len(ids), len(expected_ids))
        (output_ids,) = response["sglext"]["output_ids"]
        same = [a == b for a, b in zip(output_ids, tokens, strict=True)]
        if fixed:
            assert all(same), name
        delta = (logits - expected).abs()
        item = {
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
            "context_usage": response["choices"][0].get("meta_info", {}).get(
                "context_usage"
            ),
            "input_tokens": len(ids),
            "raw_argmax_matching_tokens": int(
                (logits.argmax(-1) == expected.argmax(-1)).sum()
            ),
        }
        comparisons[name] = item
        (directory / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        print("BCP_COMPARE", name, json.dumps(item), flush=True)
        return response

    for feature in ("none", "drop", "drop_repos"):
        for fixed in (False, True):
            assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
            call(feature, f"{feature}-{'fixed' if fixed else 'actual'}", fixed)
    # Same real task tests both direct final-version reuse and compatible Retry.
    call("drop_repos", "drop_repos-hot", True)
    assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
    call("none", "none-retry-source", True)
    call("drop_repos", "drop_repos-retry", True)
    baseline = comparisons["none-fixed"]
    for name in (
        "drop-fixed",
        "drop_repos-fixed",
        "drop_repos-hot",
        "drop_repos-retry",
    ):
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
