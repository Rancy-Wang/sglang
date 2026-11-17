"""Runner controls only; synthetic inputs are never called real BCP results."""

import hashlib
import json

import pytest
from cache_policy_bcp import (
    POLICIES,
    RollingPolicy,
    independent_events,
    load_cases,
    require_coverage,
)


def test_independent_boundaries_keep_disjoint_ownership_and_native_suffix():
    payload = {
        "messages": [{}, {}, {}],
        "drop_rule": POLICIES[0],
        "drop_message": {"2": [0]},
        "reposition": [2],
    }
    # Unowned prefix/tail and disjoint message 0 ownership are intentional.
    owners = [-1, 0, 0, 1, 0, 1, 2, 2, -1]
    assert independent_events(payload, owners) == ({8: [[1, 3], [4, 5]]}, [8])
    payload["drop_message"] = {"0": [2]}
    with pytest.raises(ValueError, match="future text"):
        independent_events(payload, owners)


def test_independent_boundaries_preserve_harmony_generation_header():
    payload = {
        "messages": [{}, {}],
        "drop_rule": POLICIES[0],
        "drop_message": {"1": [0]},
        "reposition": [1],
    }
    assert independent_events(payload, [-1, 0, 0, 1, 1, 2, 2]) == (
        {5: [[1, 3]]}, [5]
    )
    with pytest.raises(ValueError, match="terminal suffix"):
        independent_events(payload, [0, 2, 1])
    with pytest.raises(ValueError, match="outside message"):
        independent_events(payload, [0, 1, 3])


def test_quick_jobs_preserve_real_history_and_isolate_policy_fields(tmp_path):
    from cache_policy_bcp import DEFAULT_CASES, quick_jobs, write_json

    names = []
    for case in DEFAULT_CASES:
        for snapshot in ("first_drop", "first_reposition", "final"):
            name = f"{case}-{snapshot}.json"
            names.append(name)
            write_json(tmp_path / name, {
                "case_id": case, "snapshot": snapshot, "policy": POLICIES[0],
                "payload": {"messages": [{"content": case}], "drop_rule": POLICIES[0],
                            "drop_message": {"2": [0]}, "reposition": [2]},
            })
    write_json(tmp_path / "jobs.json", {"model": "frozen", "jobs": names})
    _, selected = quick_jobs(tmp_path / "jobs.json")
    assert len(selected) == 12
    for job in selected:
        assert job["payload"]["messages"] == [{"content": job["case_id"]}]
        assert job["payload"]["drop_message"] == {"2": [0]}
        assert ("reposition" in job["payload"]) == (job["policy"] == POLICIES[0])
    assert "reposition" in json.loads((tmp_path / names[0]).read_text())["payload"]
    (tmp_path / names[-1]).unlink()
    with pytest.raises(FileNotFoundError):
        quick_jobs(tmp_path / "jobs.json")


@pytest.mark.parametrize("policy", POLICIES)
def test_rolling_keeps_assistants_full_history_and_real_threshold(policy):
    rolling = RollingPolicy(policy)
    messages, owners = [], []
    for i in range(40):
        messages.append(
            {"role": "assistant" if i % 2 == 0 else "tool", "content": str(i)}
        )
        owners.extend([i] * 5000)
        state = rolling.extend(messages, owners)
    removed = [j for targets in state["drop_message"].values() for j in targets]
    assert removed == list(range(1, 17, 2))
    assert len(messages) == 40
    assert all(messages[j]["role"] == "tool" for j in removed)
    if policy == POLICIES[0]:
        assert state["reposition"]
        assert all(check["before"] >= 98304 for check in rolling.checks)
    else:
        assert "reposition" not in state
        assert rolling.compacted == 0
    assert state == rolling.extend(messages, owners)
    messages[0]["content"] = "changed"
    with pytest.raises(ValueError, match="Historical messages changed"):
        rolling.extend(messages, owners)


def test_coverage_cannot_silently_pass_without_reposition():
    with pytest.raises(ValueError, match="five cases"):
        require_coverage([{"drops": 2, "repositions": 0}] * 5)
    require_coverage(
        [{"drops": 2, "repositions": 1, "drop_after_reposition": True}] * 5
    )


def test_hash_verified_exactly_five_case_loading(tmp_path):
    entries = []
    with (tmp_path / "trajectories.jsonl").open("wb") as target:
        for i in range(5):
            raw = (
                json.dumps(
                    {"case_id": str(i), "trial": 0, "turns": [{"max_new_tokens": 55}]}
                ).encode()
                + b"\n"
            )
            entries.append(
                {
                    "case_id": str(i),
                    "trial": 0,
                    "offset": target.tell(),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            target.write(raw)
    index = b"\n".join(json.dumps(e).encode() for e in entries)
    (tmp_path / "tasks_unique.jsonl").write_bytes(index)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "tools": [],
                "files": {"tasks_unique.jsonl": hashlib.sha256(index).hexdigest()},
            }
        )
    )
    cases, _, _ = load_cases(tmp_path, list(map(str, range(5))))
    assert [c["turns"][0]["max_new_tokens"] for c in cases] == [55] * 5
    with pytest.raises(ValueError, match="Exactly five"):
        load_cases(tmp_path, ["0"] * 5)
    (tmp_path / "trajectories.jsonl").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cases(tmp_path, list(map(str, range(5))))


def test_reference_jobs_freeze_calibration_without_changing_formal_payload(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import cache_policy_bcp as runner

    class Renderer:
        def __init__(self):
            self.renderer = SimpleNamespace(
                trace=SimpleNamespace(input_ids=list(range(9))),
                tokenizer_manager=SimpleNamespace(
                    tokenizer=SimpleNamespace(encode=lambda _: list(range(8)))
                ),
            )

        def render(self, messages, tools):
            return 9, [-1, 0, 0, 1, 1, 2, 2, 2, -1]

    monkeypatch.setattr(runner, "renderer_for", lambda _: Renderer())
    monkeypatch.setattr(runner, "model_fingerprint", lambda _: {})
    frozen = {
        "model": "fixture",
        "model_files": {},
        "template_kwargs": {},
        "cases": ["223"],
    }
    frozen_path = tmp_path / "frozen.json"
    runner.write_json(frozen_path, frozen)
    saved = tmp_path / "223"
    saved.mkdir()
    payload = {
        "messages": [{}, {}, {}],
        "tools": [],
        "drop_rule": POLICIES[0],
        "drop_message": {"2": [1]},
        "reposition": [2],
    }
    runner.write_json(saved / "13.request.json", payload)
    report = {
        "completed": True,
        "frozen_sha256": runner.digest(frozen),
        "head": "test",
        "policy": POLICIES[0],
        "deployment": "ordinary",
        "cases": [{"case_id": "223", "snapshots": {"first_drop": "13"}}],
    }
    runner.write_json(tmp_path / "report.json", report)
    root = tmp_path / "jobs"
    result = runner.reference_jobs(
        SimpleNamespace(
            report=tmp_path / "report.json",
            frozen=frozen_path,
            model="fixture",
            template_kwargs="{}",
            output=root,
            include_calibration=True,
        )
    )
    assert result["jobs"] == ["noop.json", "223-first_drop.json"]
    original = json.loads((root / "223-first_drop.json").read_text())
    baseline = json.loads((root / "noop.json").read_text())
    assert original["payload"] == payload
    assert original["drops"] == {"8": [[3, 5]]}
    assert baseline["drops"] == baseline["payload"]["drop_message"] == {}
    assert baseline["reposition"] == baseline["payload"]["reposition"] == []
    assert original["payload_sha256"] != baseline["payload_sha256"]


def test_numerical_join_preserves_failure_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cache_policy_bcp as runner
    import torch

    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"tree")
    candidate = tmp_path / "candidate.pt"
    expected = tmp_path / "223-final.logits.pt"
    torch.save(torch.zeros(2, 8), expected)
    torch.save(torch.ones(2, 8), candidate)
    runner.write_json(
        tmp_path / "223-final.json",
        {
            "reference": "independent_event_model_forward",
            "payload_sha256": "same",
            "continuation": [3],
            "head": "reference",
        },
    )
    runner.write_json(
        tmp_path / "diagnostics.json",
        {
            "completed": True,
            "head": "candidate",
            "calibration": {
                "frozen_before_policy_candidates": True,
                "noop_baseline": {"max_abs": 0, "mean_abs": 0, "p99_abs": 0},
                "limits": {"max_abs": 0.125, "mean_abs": 0.02, "p99_abs": 0.0625},
            },
            "comparisons": [
                {
                    "case_id": "223",
                    "snapshot": "final",
                    "payload_sha256": "same",
                    "continuation": [3],
                    "candidate": candidate.name,
                }
            ],
        },
    )
    args = SimpleNamespace(
        diagnostic_report=tmp_path / "diagnostics.json",
        reference_output=tmp_path,
        output=tmp_path / "comparison.json",
    )
    with pytest.raises(AssertionError, match="comparison failed"):
        runner.compare_artifacts(args)
    failed = json.loads(args.output.read_text())
    assert not failed["passed"] and failed["comparisons"][0]["metrics"]["max_abs"] == [
        1,
        1,
    ]
    torch.save(torch.zeros(2, 8), candidate)
    assert runner.compare_artifacts(args)["passed"]
