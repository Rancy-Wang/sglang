"""One real BCP history for functional diagnosis, not a throughput workload."""

import copy
import hashlib
import json
import os
from pathlib import Path


def load_fixture():
    root = Path(os.environ["CONTEXT_BCP_ROOT"])
    case_id = os.environ.get("CONTEXT_BCP_CASE", "778")
    entries = [
        json.loads(line)
        for line in (root / "tasks_unique.jsonl").read_text().splitlines()
    ]
    entry = next(item for item in entries if item["case_id"] == case_id)
    with (root / "trajectories.jsonl").open("rb") as source:
        source.seek(entry["offset"])
        raw = source.read(entry["bytes"])
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    case = json.loads(raw)
    turn = case["turns"][-1]
    messages = copy.deepcopy(case["trajectory"][: turn["source_assistant_message_id"]])
    names = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
        if message["role"] == "tool" and not message.get("name"):
            message["name"] = names[message["tool_call_id"]]
    tool_ids = [i for i, message in enumerate(messages) if message["role"] == "tool"]
    assert len(tool_ids) >= 4
    # Two historical transitions exercise changing positions and visibility.
    # Deliberately bounded diagnosis; this is not the benchmark's K=12/96K policy.
    drops = {str(tool_ids[-2]): [tool_ids[0]], str(tool_ids[-1]): [tool_ids[1]]}
    tools = json.loads((root / "manifest.json").read_text())["tools"]
    return {
        "case_id": case_id,
        "source_sha256": entry["sha256"],
        "source_turn": turn["turn"],
        "messages": messages,
        "tools": tools,
        "drop_message": drops,
        "reposition": [int(i) for i in drops],
    }


def request_for(fixture, feature):
    request = {"messages": fixture["messages"], "tools": fixture["tools"]}
    if feature != "none":
        request["drop_message"] = fixture["drop_message"]
    if feature == "drop_repos":
        request["reposition"] = fixture["reposition"]
    return request
