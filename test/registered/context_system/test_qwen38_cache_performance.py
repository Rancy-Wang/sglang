"""Opt-in, paired cold/hot timings; the same probe runs before and after R3."""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import requests


pytestmark = pytest.mark.skipif(
    not os.environ.get("QWEN38_PERF_URL"), reason="isolated performance server required"
)


def test_cache_performance():
    base = os.environ["QWEN38_PERF_URL"].rstrip("/")
    out = Path(os.environ["QWEN38_PERF_RESULTS"])
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    response = session.get(base + "/get_server_info", timeout=30)
    response.raise_for_status()
    (out / "server-info.json").write_text(response.text)
    records = []
    for mode in ("native", "drop", "drop_reposition"):
        payload = dict(
            model=os.environ.get("QWEN38_TOKENIZER_PATH", "/share/public/public_models/Qwen3.8-27B"),
            messages=[
                {"role": "user", "content": "Read the background: " + "red blue green " * 2500},
                {"role": "assistant", "content": "Read."},
                {"role": "user", "content": "Neutral background. " * 256 + " List the integers from one to one hundred."},
            ],
            temperature=0, max_tokens=128, ignore_eos=True, stream=True,
            stream_options={"include_usage": True},
            chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
        )
        if mode != "native":
            payload["drop_message"] = {"2": [0]}
        if mode == "drop_reposition":
            payload["reposition"] = [2]
        (out / f"{mode}-request.json").write_text(json.dumps(payload))
        for repeat in range(6):  # first pair warms kernels and graph variants
            session.post(base + "/flush_cache", timeout=30).raise_for_status()
            for cache in ("cold", "hot"):
                started = time.perf_counter()
                first = None
                last = None
                events = []
                with session.post(base + "/v1/chat/completions", json=payload,
                                  stream=True, timeout=1800) as response:
                    assert response.status_code == 200, response.text
                    for line in response.iter_lines(chunk_size=1):
                        if not line.startswith(b"data: ") or line == b"data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        assert "error" not in event, event
                        events.append(event)
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                                last = time.perf_counter()
                                if first is None:
                                    first = last
                elapsed = time.perf_counter() - started
                assert first is not None and last is not None, events
                usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
                meta = next((c["meta_info"] for e in reversed(events)
                             for c in e.get("choices", []) if c.get("meta_info")), {})
                count = usage.get("completion_tokens", meta.get("completion_tokens", 0))
                assert count > 1, (usage, meta)
                record = dict(mode=mode, cache=cache, repeat=repeat, seconds=elapsed,
                              ttft_seconds=first-started, tpot_seconds=(last-first)/(count-1),
                              usage=usage, meta=meta)
                records.append(record)
                (out / f"{mode}-{repeat}-{cache}.json").write_text(json.dumps(events))
                with (out / "timings.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
    summary = {}
    for mode in ("native", "drop", "drop_reposition"):
        for cache in ("cold", "hot"):
            selected = [r for r in records if r["mode"] == mode and r["cache"] == cache and r["repeat"]]
            summary[f"{mode}-{cache}"] = {
                key: {"median": statistics.median(r[key] for r in selected),
                      "max": max(r[key] for r in selected)}
                for key in ("seconds", "ttft_seconds", "tpot_seconds")
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
