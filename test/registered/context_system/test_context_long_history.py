"""Compare identical execution positions with native and overflowing raw rows."""

import contextlib
import json
import os
from pathlib import Path

import pytest
import requests
import torch

from serving_logits_probe import serialized_probe
from test_serving_runtime import server

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_LONG_HISTORY"),
    reason="explicit GPU validation required",
)


def test_raw_overflow_matches_wide_table(tmp_path_factory, monkeypatch):
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": "Remember the final color."}]
    drops, reposition = {}, []
    for index in range(5):
        messages += [
            {"role": "user", "content": "red blue green " * 300},
            {"role": "assistant", "content": "I read those colors."},
        ]
        event = len(messages) - 1
        drops[str(event)] = [event - 1]
        reposition.append(event)
    messages.append(
        {"role": "user", "content": "The final color is orange. Repeat it."}
    )
    base = {
        "model": os.environ["CONTEXT_SERVER_MODEL"],
        "messages": messages,
        "drop_message": drops,
        "reposition": reposition,
        "temperature": 0,
        "max_tokens": 8,
        "ignore_eos": True,
        "return_input_ids_in_sglext": True,
        "return_output_ids_in_sglext": True,
        "return_meta_info": True,
    }
    outputs = {}
    forced = None
    monkeypatch.setenv("CONTEXT_CHUNK_SIZE", "256")
    monkeypatch.setenv("CONTEXT_KV_CAPACITY", "8192")
    for capacity in (8192, 2048):
        monkeypatch.setenv("CONTEXT_MAX_LENGTH", str(capacity))
        monkeypatch.setenv(
            "CONTEXT_SERVER_LOG", str(directory / f"server-{capacity}.log")
        )
        with contextlib.contextmanager(server.__wrapped__)(tmp_path_factory) as url:
            if forced is None:
                response = requests.post(
                    url + "/v1/chat/completions", json=base, timeout=300
                )
                assert response.status_code == 200, response.text
                forced = response.json()["sglext"]["output_ids"][0]
                assert len(forced) == 8
                assert requests.post(url + "/flush_cache", timeout=5).status_code == 200
            for state in ("cold", "hot"):
                name = f"{capacity}-{state}"
                path = directory / f"{name}.pt"
                payload = {
                    **base,
                    "custom_logit_processor": serialized_probe(),
                    "custom_params": {
                        "context_trace_path": str(path),
                        "context_trace_count": len(forced),
                        "context_forced_tokens": forced,
                    },
                }
                response = requests.post(
                    url + "/v1/chat/completions", json=payload, timeout=300
                )
                (directory / f"{name}.json").write_text(response.text)
                assert response.status_code == 200, response.text
                result = response.json()
                assert 4096 < len(result["sglext"]["input_ids"]) < 8192
                assert result["sglext"]["output_ids"] == [forced]
                logits = torch.load(path, weights_only=True)
                assert torch.isfinite(logits).all()
                outputs[capacity, state] = logits
                if state == "hot":
                    usage = result["choices"][0]["meta_info"]["context_usage"]
                    assert usage["drop_skipped_tokens"] > 4000, usage
                # The same materialization and chunk shapes must not change
                # numerics merely because page-ID storage moved out of line.
                if capacity == 2048:
                    torch.testing.assert_close(
                        logits, outputs[8192, state], rtol=0, atol=0
                    )
            assert requests.get(url + "/health", timeout=5).status_code == 200
    (directory / "comparison.json").write_text(
        json.dumps(
            {
                "raw_overflow_cold_and_hot_bit_exact": True,
                "tokens": forced,
                "capacities": [8192, 2048],
            },
            indent=2,
        )
    )
