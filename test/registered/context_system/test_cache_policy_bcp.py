"""Acceptance of saved real runs; absence of evidence skips, never passes."""

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_CACHE_POLICY_REPORTS"),
    reason="four completed real BCP reports required",
)
def test_five_cases_two_rules_two_deployments():
    paths = json.loads(os.environ["CONTEXT_CACHE_POLICY_REPORTS"])
    assert len(paths) == 4
    reports = [json.loads(Path(path).read_text()) for path in paths]
    assert {(r["policy"], r["deployment"]) for r in reports} == {
        (rule, deployment)
        for rule in ("text_drop_with_pe", "message_drop_without_repos")
        for deployment in ("ordinary", "pd")
    }
    assert len({r["frozen_sha256"] for r in reports}) == 1
    assert len({r["head"] for r in reports}) == 1
    cohort = {c["case_id"] for c in reports[0]["cases"]}
    assert len(cohort) == 5
    for path, report in zip(paths, reports):
        assert report["completed"]
        assert {c["case_id"] for c in report["cases"]} == cohort
        for case in report["cases"]:
            assert case["drops"] and case["turns"]
            if report["policy"] == "text_drop_with_pe":
                assert case["repositions"]
                assert all(check["before"] >= 98304 for check in case["checks"])
            else:
                assert not case["repositions"]
            requests = list(
                (Path(path).parent / str(case["case_id"])).glob("*.request.json")
            )
            assert len(requests) == case["turns"]
            for request in requests:
                body = json.loads(request.read_text())
                response = json.loads(
                    Path(
                        str(request).replace(".request.json", ".response.json")
                    ).read_text()
                )
                assert body["max_tokens"] == response["usage"]["completion_tokens"]
                assert "custom_logit_processor" not in body  # formal free generation
                assert response["choices"][0]["finish_reason"] == "length"


def compare_logits(candidate, reference, limits):
    assert candidate.shape == reference.shape and candidate.ndim == 2
    assert torch.isfinite(candidate).all() and torch.isfinite(reference).all()
    delta = (candidate.float() - reference.float()).abs()
    per_token = {
        "max_abs": delta.max(-1).values,
        "mean_abs": delta.mean(-1),
        "p99_abs": torch.quantile(delta, 0.99, dim=-1),
    }
    assert all(
        bool(torch.all(value <= limits[name])) for name, value in per_token.items()
    )
    return {name: value.tolist() for name, value in per_token.items()}


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_CACHE_POLICY_NUMERIC_MANIFEST"),
    reason="independent model and cache diagnostics required",
)
def test_numerical_and_reuse_evidence():
    path = Path(os.environ["CONTEXT_CACHE_POLICY_NUMERIC_MANIFEST"])
    manifest = json.loads(path.read_text())
    assert manifest["reference"] == "independent_event_model_forward"
    assert manifest["tolerance_frozen_before_candidate"] is True
    floors = {"max_abs": 0.125, "mean_abs": 0.02, "p99_abs": 0.0625}
    expected_limits = {
        name: max(floor, 2 * manifest["noop_baseline"][name])
        for name, floor in floors.items()
    }
    assert manifest["limits"] == expected_limits
    seen = set()
    for item in manifest["comparisons"]:
        assert item["same_payload"] and item["same_fixed_continuation"]
        assert item["structure_errors"] == 0
        assert item["physical_reads_checked"] and item["per_layer_checked"]
        if item["policy"] == "message_drop_without_repos":
            assert item["retry_tokens"] == 0
        if item["mode"] == "retrigger":
            assert item["policy"] == "text_drop_with_pe" and item["retry_tokens"] > 0
        if item["mode"] == "partial_eviction":
            assert item["evicted_pages"] > 0 and item["recovered_queries"] > 0
        compare_logits(
            torch.load(path.parent / item["candidate"], weights_only=True),
            torch.load(path.parent / item["reference_logits"], weights_only=True),
            expected_limits,
        )
        seen.add(
            (
                str(item["case_id"]),
                item["policy"],
                item["deployment"],
                item["snapshot"],
                item["mode"],
            )
        )
    for case in manifest["cases"]:
        for policy in ("text_drop_with_pe", "message_drop_without_repos"):
            for deployment in ("ordinary", "pd"):
                for snapshot in (
                    "first_drop",
                    "multiple_drop",
                    "final",
                    *(("first_reposition",) if policy == "text_drop_with_pe" else ()),
                ):
                    for mode in (
                        "cold",
                        "hot",
                        "partial_eviction",
                        *(("retrigger",) if policy == "text_drop_with_pe" else ()),
                    ):
                        assert (str(case), policy, deployment, snapshot, mode) in seen
