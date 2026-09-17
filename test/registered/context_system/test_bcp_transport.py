"""Whole-method SSE accounting and paired P/D failure propagation, without GPU."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ir import ROOT, load_file


def test_gpt_replay_named_tool_after_generated_final(tmp_path):
    model = os.environ.get("CONTEXT_GPT_TOKENIZER")
    if not model:
        pytest.skip("real GPT-OSS tokenizer required")
    from transformers import AutoTokenizer

    launcher = load_file(
        "context_bcp_launcher", ROOT / "benchmark/context_system/run_minimal.py"
    )
    bench = load_file(
        "context_bcp_replay", ROOT / "benchmark/context_system/test_serving.py"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    template = launcher.replay_template(tokenizer.get_chat_template(), "2026-09-18")
    path = tmp_path / "replay.jinja"
    path.write_text(template)
    adapter = bench.NativeTemplateAdapter(
        model, {"preserve_thinking_history": True}, path
    )
    history = [
        {"role": "user", "content": "Find the answer."},
        {"role": "assistant", "content": "A generated final answer."},
        {"role": "tool", "name": "search", "content": "Recorded search result."},
    ]
    import copy

    original = copy.deepcopy(history)
    size, owners = adapter.render(history, [])
    assert history == original
    text = adapter.renderer.trace.rendered_text
    assert "<|start|>functions.search to=assistant" in text
    assert "Current date: 2026-09-18" in text
    assert "strftime_now" not in template
    assert "A generated final answer." in text
    assert size == len(owners)
    assert 2 in owners
    assert adapter.render(history, [])[0] == size
    # Preserve the native error when neither the replay nor the history has a name.
    history[-1].pop("name")
    with pytest.raises(ValueError, match="no previous assistant"):
        adapter.render(history, [])

    # Real fixed-length output can carry all three fields. No generated text
    # or additional call may disappear from the next turn's prompt.
    history[1] = {
        "role": "assistant",
        "content": "Generated commentary.",
        "reasoning_content": "Generated reasoning.",
        "tool_calls": [
            {
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": '{"query":"test"}'},
            }
            for name in ("search", "lookup")
        ],
    }
    history[-1]["name"] = "search"
    original = copy.deepcopy(history)
    size, owners = adapter.render(history, [])
    assert history == original
    text = adapter.renderer.trace.rendered_text
    for channel, value in (
        ("commentary", "Generated commentary."),
        ("analysis", "Generated reasoning."),
    ):
        assert f"<|channel|>{channel}<|message|>{value}<|end|>" in text
        assert text.count(value) == 1
    for name in ("search", "lookup"):
        assert f"assistant to=functions.{name}" in text
    assert size == len(owners)
    history += [
        {"role": "assistant", "content": "Later final."},
        {"role": "user", "content": "Continue."},
    ]
    adapter.render(history, [])
    from collections import Counter

    old_counts = Counter(owners)
    new_counts = Counter(adapter.renderer.trace.owners)
    assert all(new_counts[i] == old_counts[i] for i in range(3))
    assert "Generated reasoning." in adapter.renderer.trace.rendered_text


@pytest.mark.parametrize("blocked_at", ["headers", "stream"])
def test_prefill_failure_wakes_blocked_decode(blocked_at):
    bench = load_file(
        "context_bcp_bench_failure", ROOT / "benchmark/context_system/test_serving.py"
    )

    class Session:
        @asynccontextmanager
        async def post(self, url, json):
            if url == "prefill":

                async def fail():
                    return {"error": "transfer failed"}

                yield SimpleNamespace(status=500, json=fail)
            else:
                if blocked_at == "headers":
                    await asyncio.Event().wait()

                async def content():
                    await asyncio.Event().wait()
                    yield b"unreachable"

                yield SimpleNamespace(status=200, content=content(), text=None)

    async def events(content):
        async for event in content:
            yield event

    async def run():
        transport = bench.Transport(
            Session(), SimpleNamespace(sse_events=events), "prefill", 1
        )
        async with transport.post("decode", json={}) as response:
            async for _ in response.content:
                pass

    async def bounded():
        with pytest.raises(RuntimeError, match="P HTTP 500"):
            await asyncio.wait_for(run(), timeout=1)

    asyncio.run(bounded())


@pytest.mark.parametrize("pd_failure", [False, True])
def test_streaming_counters_through_pinned_method(pd_failure):
    mini = os.environ.get("MINI_SGLANG_REFERENCE")
    if not mini:
        pytest.skip("explicit pinned mini checkout required")
    bench = load_file(
        "context_bcp_bench", ROOT / "benchmark/context_system/test_serving.py"
    )
    method = bench.load_method(Path(mini))
    calls = []

    class Session:
        @asynccontextmanager
        async def post(self, url, json):
            calls.append((url, json))
            if url == "prefill":

                async def response_json():
                    return {"error": "transfer failed"} if pd_failure else {}

                yield SimpleNamespace(
                    status=500 if pd_failure else 200, json=response_json
                )
                return

            async def content():
                events = [
                    {
                        "choices": [
                            {"delta": {"content": "hello"}, "finish_reason": None}
                        ]
                    },
                    {
                        "sglext": {
                            "context_usage": {
                                "0": {
                                    "actual_prefill_tokens": 11,
                                    "actual_decode_tokens": 3,
                                    "cached_tokens": 70,
                                    "repos_tokens": 20,
                                    "drop_skipped_tokens": 50,
                                }
                            }
                        }
                    },
                    {
                        "choices": [{"delta": {}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 151, "completion_tokens": 2},
                    },
                ]
                for event in events:
                    yield ("data: " + __import__("json").dumps(event) + "\n").encode()
                    yield b"\n"
                yield b"data: [DONE]\n"
                yield b"\n"

            yield SimpleNamespace(status=200, content=content(), text=None)

    async def run():
        transport = bench.Transport(Session(), method, "prefill", 1234)
        row = await method.request(
            transport, "decode", {"max_tokens": 2, "stream": True}
        )
        row.update(filler=False)
        return row, method.summary([row], row["start_time"] - 1, row["end_time"] + 1)

    row, report = asyncio.run(run())
    assert row["success"] is (not pd_failure), row
    bodies = dict(calls)
    assert bodies["prefill"]["bootstrap_room"] == bodies["decode"]["bootstrap_room"]
    assert bodies["prefill"]["stream"] is False
    assert bodies["decode"]["stream"] is True
    if pd_failure:
        assert "P HTTP 500" in row["error"]
    else:
        assert report["actual"]["prefill_tokens"] == 11
        # Includes discarded/replayed overlap compute; not output_len - 1.
        assert report["actual"]["decode_tokens"] == 3
        assert bench.compute_metrics({"usage": {"prompt_tokens": 151}}) is None
