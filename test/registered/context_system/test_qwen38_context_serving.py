"""Opt-in Qwen hybrid validation against an isolated, already running server."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN38_CONTEXT_URL"), reason="isolated Qwen hybrid GPU server required"
)


@pytest.fixture(scope="module")
def endpoint():
    base = os.environ["QWEN38_CONTEXT_URL"].rstrip("/")
    assert requests.get(base + "/health", timeout=10).status_code == 200
    return base


def request(endpoint, name, messages, **options):
    payload = dict(
        model=os.environ["QWEN38_TOKENIZER_PATH"],
        messages=messages,
        temperature=0,
        max_tokens=512,
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
        return_meta_info=True,
        return_input_ids_in_sglext=True,
        return_output_ids_in_sglext=True,
    )
    payload.update(options)
    directory = Path(os.environ["QWEN38_CONTEXT_RESULTS"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}-request.json").write_text(json.dumps(payload, ensure_ascii=False))
    response = requests.post(endpoint + "/v1/chat/completions", json=payload, timeout=1800)
    (directory / f"{name}-response.json").write_text(response.text)
    assert response.status_code == 200, response.text
    result = response.json()
    choice = result["choices"][0]
    text = choice["message"].get("content") or ""
    assert "\ufffd" not in text
    assert choice["finish_reason"] == "stop", choice
    assert requests.get(endpoint + "/health", timeout=10).status_code == 200
    print(name, json.dumps({"text": text, "meta": choice.get("meta_info")}, ensure_ascii=False))
    return result, text


def flush(endpoint):
    response = requests.post(endpoint + "/flush_cache", timeout=20)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("mode", ["native", "drop", "drop_reposition"])
def test_bcp_case771(endpoint, mode):
    """Replay one real BCP final-answer request; never include its stored answer."""
    source = Path(os.environ["QWEN38_BCP_TRAJECTORIES"])
    selected = None
    for line in source.read_bytes().splitlines():
        row = json.loads(line)
        if str(row["case_id"]) == "771" and row["trial"] == 1:
            selected = row
            digest = hashlib.sha256(line).hexdigest()
            break
    assert selected is not None
    messages = copy.deepcopy(selected["trajectory"][:-1])
    assert len(messages) == 5 and messages[-1]["role"] == "tool"
    # The discarded search response does not contain the answer. Keep the
    # complete supporting document (72111), including the 1982 brand evidence.
    assert "vakkorama" not in messages[2]["content"].lower()
    assert "vakkorama" in messages[4]["content"].lower()
    names = {}
    for message in messages:
        for call in message.get("tool_calls", []):
            names[call["id"]] = call["function"]["name"]
        if message["role"] == "tool":
            message["name"] = names[message["tool_call_id"]]
    tools = [
        {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]},
        }}
        for name, arg, description in [
            ("search", "query", "Search for documents; returns docids and snippets."),
            ("get_document", "docid", "Retrieve the full document by docid."),
        ]
    ]
    directory = Path(os.environ["QWEN38_CONTEXT_RESULTS"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bcp-source.json").write_text(json.dumps({
        "path": str(source), "case_id": "771", "trial": 1,
        "line_sha256": digest, "expected_answer": "Vakkorama", "evidence_docid": "72111",
    }, indent=2))
    # This is a final-answer replay, not a live search-engine rollout. The
    # evidence is complete; disable additional calls consistently in all modes.
    options = {"tools": tools, "tool_choice": "none"}
    if mode != "native":
        options["drop_message"] = {"3": [2]}
    if mode == "drop_reposition":
        options["reposition"] = [3]
    flush(endpoint)
    results = []
    for state in ("cold", "hot"):
        result, text = request(endpoint, f"bcp-{mode}-{state}", messages, **options)
        assert "vakkorama" in text.lower(), text
        assert "72111" in text and "exact answer" in text.lower(), text
        assert not result["choices"][0]["message"].get("tool_calls")
        results.append(result)
    if mode != "native":
        cold, hot = [r["choices"][0]["meta_info"]["context_usage"] for r in results]
        assert hot["actual_prefill_tokens"] < cold["actual_prefill_tokens"], (cold, hot)
        # Repeated requests must reuse native hybrid state and skip Full KV
        # that is no longer demanded, rather than replaying all old queries.
        assert hot["drop_skipped_tokens"] > 0, hot


def test_raw_history_above_model_limit(endpoint):
    """Exceed native 262144 tokens while keeping active Full positions small."""
    if not os.environ.get("QWEN38_CONTEXT_LONG"):
        pytest.skip("explicit long-history run required")
    messages = [{"role": "system", "content": "Follow the final user instruction."}]
    drops, reposition = {}, []
    for _ in range(96):
        messages += [
            {"role": "user", "content": "red blue green " * 1000},
            {"role": "assistant", "content": "Read."},
        ]
        event = len(messages) - 1
        drops[str(event)] = [event - 1]
        reposition.append(event)
    messages.append({"role": "user", "content": "Reply with exactly the word orange."})
    flush(endpoint)
    for state in ("cold", "hot"):
        result, text = request(endpoint, f"long-{state}", messages,
                               drop_message=drops, reposition=reposition, max_tokens=32)
        assert text.strip().lower().strip(".!") == "orange", text
        assert len(result["sglext"]["input_ids"]) > 262144
        usage = result["choices"][0]["meta_info"]["context_usage"]
        assert usage["actual_prefill_tokens"] <= len(result["sglext"]["input_ids"]), usage
        if state == "hot":
            assert usage["drop_skipped_tokens"] > 262144, usage
