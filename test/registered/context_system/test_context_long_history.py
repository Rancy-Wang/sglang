"""Compare identical execution positions with native and overflowing raw rows."""

import concurrent.futures
import contextlib
import json
import os
import time
from pathlib import Path

import pytest
import requests
import torch
from serving_logits_probe import serialized_probe
from test_bcp_pd_numeric import pd_servers  # noqa: F401
from test_serving_runtime import server

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_LONG_HISTORY"),
    reason="explicit GPU validation required",
)


def raw_overflow_request():
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
    return {
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


def test_raw_overflow_matches_wide_table(tmp_path_factory, monkeypatch):
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    base = raw_overflow_request()
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


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_PD_LONG_REFERENCE"),
    reason="reuse an existing normal-scheduler long-history reference",
)
def test_pd_raw_overflow(pd_servers):  # noqa: F811
    """Transfer overflowing raw history using mini-style shared SWA pages."""
    assert os.environ["CONTEXT_MAX_LENGTH"] == "2048"
    assert os.environ["CONTEXT_CHUNK_SIZE"] == "256"
    p_base, d_base, bootstrap = pd_servers
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    reference = Path(os.environ["CONTEXT_PD_LONG_REFERENCE"])
    tokens = json.loads((reference / "comparison.json").read_text())["tokens"]
    room = time.time_ns() % (1 << 53)
    comparisons = {}
    for state in ("cold", "hot"):
        expected_response = json.loads((reference / f"2048-{state}.json").read_text())
        payload = {
            **raw_overflow_request(),
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": bootstrap,
            "bootstrap_room": room,
            "custom_logit_processor": serialized_probe(),
        }
        room += 1

        def send(mode, base, count, offset, payload=payload, state=state):
            body = {
                **payload,
                "custom_params": {
                    "context_trace_path": str(directory / f"{state}-{mode}.pt"),
                    "context_trace_count": count,
                    "context_forced_tokens": tokens,
                    "context_forced_offset": offset,
                },
            }
            response = requests.post(
                base + "/v1/chat/completions", json=body, timeout=300
            )
            (directory / f"{state}-{mode}.json").write_text(response.text)
            assert response.status_code == 200, response.text
            return response.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            p = executor.submit(send, "prefill", p_base, 1, 0)
            d = executor.submit(send, "decode", d_base, len(tokens) - 1, 1)
            p.result()
            result = d.result()
        assert result["sglext"]["input_ids"] == expected_response["sglext"]["input_ids"]
        assert 4096 < len(result["sglext"]["input_ids"]) < 8192
        assert result["sglext"]["output_ids"] == [tokens]
        logits = torch.cat(
            [
                torch.load(directory / f"{state}-{mode}.pt", weights_only=True)
                for mode in ("prefill", "decode")
            ]
        )
        expected = torch.load(reference / f"2048-{state}.pt", weights_only=True)
        assert logits.shape == expected.shape and torch.isfinite(logits).all()
        delta = (logits - expected).abs()
        usage = result["choices"][0]["meta_info"]["context_usage"]
        item = {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "p99_abs": torch.quantile(delta.flatten(), 0.99).item(),
            "usage": usage,
        }
        comparisons[state] = item
        (directory / "comparison.json").write_text(json.dumps(comparisons, indent=2))
        assert (
            item["max_abs"] <= 0.125
            and item["mean_abs"] <= 0.02
            and item["p99_abs"] <= 0.0625
        ), item
        assert usage["actual_decode_tokens"] == len(tokens) - 1, usage
        if state == "hot":
            assert usage["drop_skipped_tokens"] > 4000, usage
            assert usage["actual_prefill_tokens"] == 1, usage
    for base in (p_base, d_base):
        assert requests.get(base + "/health", timeout=5).status_code == 200
