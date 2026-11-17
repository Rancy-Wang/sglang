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
