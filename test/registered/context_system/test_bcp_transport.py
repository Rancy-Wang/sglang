"""Whole-method SSE accounting and paired P/D failure propagation, without GPU."""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ir import ROOT, load_file


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
