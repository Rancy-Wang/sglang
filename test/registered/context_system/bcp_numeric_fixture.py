"""One real BCP history for functional diagnosis, not a throughput workload."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path


def oracle_chat_template(reference_path, model, directory):
    """Keep the native Harmony template on the stored oracle's calendar date."""
    if not reference_path or "gpt-oss" not in model.lower():
        return None
    from transformers import AutoTokenizer

    reference = json.loads(Path(reference_path).read_text())
    ids = reference["runs"]["none"]["records"][0]["input"]["ids"]
    tokenizer = AutoTokenizer.from_pretrained(model)
    header = tokenizer.decode(ids[:256])
    dates = re.findall(r"(?m)^Current date: (\d{4}-\d{2}-\d{2})$", header)
    assert len(dates) == 1, "Oracle must contain one native Harmony system date"
    template = tokenizer.get_chat_template()
    clock = 'strftime_now("%Y-%m-%d")'
    assert template.count(clock) == 1, "Unrecognized native Harmony date template"
    path = Path(directory) / "oracle_date.jinja"
    path.write_text(template.replace(clock, json.dumps(dates[0])))
    return str(path)


def load_fixture():
    if path := os.environ.get("CONTEXT_BCP_FIXTURE"):
        return json.loads(Path(path).read_text())
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


if __name__ == "__main__":
    # Run with the SGLang environment before either model is loaded. Feeding
    # these same native tool dictionaries to mini aligns template tokens while
    # leaving both engines' renderers and tool implementations intact.
    import argparse

    from sglang.srt.entrypoints.openai.protocol import Tool

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-trace-model")
    args = parser.parse_args()
    fixture = load_fixture()
    fixture["tools"] = [Tool.model_validate(t).model_dump() for t in fixture["tools"]]
    fixture["tool_serialization"] = "SGLang native Tool.model_dump"
    if args.native_trace_model:
        import runpy

        benchmark = (
            Path(__file__).resolve().parents[3]
            / "benchmark/context_system/test_serving.py"
        )
        adapter = runpy.run_path(str(benchmark))["NativeTemplateAdapter"](
            args.native_trace_model, {}
        )
        adapter.render(fixture["messages"], fixture["tools"])
        trace = adapter.renderer.trace
        fixture["native_sglang_trace"] = {
            "model": args.native_trace_model,
            "input_ids": trace.input_ids,
            "owners": trace.owners,
            "generation_start": trace.owners.index(len(fixture["messages"])),
        }
    Path(args.output).write_text(json.dumps(fixture))
