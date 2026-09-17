"""Opt-in HTTP scheduler checks; owns only the server process it starts."""

import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_SERVER_MODEL"), reason="isolated serving GPU required"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = int(os.environ.get("CONTEXT_SERVER_PORT", "28761"))
    base = f"http://127.0.0.1:{port}"
    log_path = Path(
        os.environ.get(
            "CONTEXT_SERVER_LOG",
            str(tmp_path_factory.mktemp("context-http") / "server.log"),
        )
    )
    graph = {
        "decode": {"bs": [1, 2, 4], "max_bs": 4},
        "prefill": {"bs": [16, 32, 64], "max_bs": 64},
    }
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        os.environ["CONTEXT_SERVER_MODEL"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--page-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-total-tokens",
        os.environ.get("CONTEXT_KV_CAPACITY", "4096"),
        "--context-length",
        os.environ.get("CONTEXT_MAX_LENGTH", "2048"),
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        os.environ.get("CONTEXT_CHUNK_SIZE", "64"),
        "--cuda-graph-config",
        json.dumps(graph),
        "--context-drop-aware-eviction",
        "--enable-mixed-chunk",
    ]
    cmd += ["--tp-size", os.environ.get("CONTEXT_TEST_TP", "1")]
    if os.environ.get("CONTEXT_BCP_ORACLE"):
        from bcp_numeric_fixture import oracle_chat_template

        template = oracle_chat_template(
            os.environ["CONTEXT_BCP_ORACLE"],
            os.environ["CONTEXT_SERVER_MODEL"],
            log_path.parent,
        )
        if template:
            cmd += ["--chat-template", template]
    if "gpt-oss" in os.environ["CONTEXT_SERVER_MODEL"].lower():
        cmd += [
            "--tool-call-parser",
            "gpt-oss",
            "--reasoning-parser",
            "gpt-oss",
            "--disable-hybrid-swa-memory",
        ]
    backend = os.environ.get("CONTEXT_TEST_ATTENTION_BACKEND")
    if backend:
        cmd += ["--attention-backend", backend]
    if os.environ.get("CONTEXT_TRACE_DIR"):
        cmd += ["--enable-custom-logit-processor"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent.resolve()), env.get("PYTHONPATH", "")]
    )
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    pytest.fail(log_path.read_text()[-12000:])
                try:
                    if requests.get(base + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail(
                    "Server startup timed out: " + log_path.read_text()[-12000:]
                )
            print(
                "CONTEXT_HTTP",
                "GPU",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                "LOG",
                log_path,
                flush=True,
            )
            yield base
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)


def call(base, feature, suffix="", *, trace_name=None, fixed=False):
    messages = [
        {"role": "user", "content": "Remember this text: " + "red blue green " * 32},
        {"role": "assistant", "content": "I have read the text."},
        {
            "role": "user",
            "content": "Reply with a short sentence about the ocean." + suffix,
        },
    ]
    payload = {
        "model": os.environ["CONTEXT_SERVER_MODEL"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "return_meta_info": True,
        "return_output_ids_in_sglext": True,
        "logprobs": True,
    }
    if feature == "identity":
        payload.update(reposition=[0])
    elif feature == "drop":
        payload.update(drop_message={"1": [0]})
    elif feature:
        payload.update(drop_message={"1": [0]}, reposition=[1])
    if trace_name is not None:
        from serving_logits_probe import serialized_probe

        params = {
            "context_trace_path": str(
                Path(os.environ["CONTEXT_TRACE_DIR"]) / (trace_name + ".pt")
            ),
            "context_trace_count": 8,
        }
        if fixed:
            params["context_forced_tokens"] = [
                785,
                17951,
                374,
                264,
                12767,
                323,
                25382,
                2487,
            ]
        payload.update(custom_logit_processor=serialized_probe(), custom_params=params)
    response = requests.post(base + "/v1/chat/completions", json=payload, timeout=120)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["usage"]["completion_tokens"] == 8, result
    assert result["choices"][0]["finish_reason"] == "length", result
    return result


def test_full_logits_http_diagnostic(server):
    if not os.environ.get("CONTEXT_TRACE_DIR"):
        pytest.skip(
            "full logits instrumentation explicitly enabled only for numerical runs"
        )
    outputs = []
    for fixed in (False, True):
        for feature, name in ((False, "none"), ("drop", "drop"), (True, "drop_repos")):
            assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
            path = f"{name}-{fixed}"
            result = call(server, feature, trace_name=path, fixed=fixed)
            outputs.append(result)
            assert (Path(os.environ["CONTEXT_TRACE_DIR"]) / (path + ".pt")).exists()
    (Path(os.environ["CONTEXT_TRACE_DIR"]) / "responses.json").write_text(
        json.dumps(outputs)
    )


@pytest.mark.skipif(
    not os.environ.get("CONTEXT_CAPACITY_PRESSURE"),
    reason="explicit 110-token KV pool required",
)
def test_context_capacity_rejection_recovers(server):
    """A self-pinned continuation must fail promptly and release its pages."""
    import torch

    from sglang.srt.context_system.ir import (
        compile_context_layout,
        prewarm_context_layout,
    )
    from sglang.srt.context_system.planner import ContextProgram

    assert os.environ["CONTEXT_KV_CAPACITY"] == "110"
    assert os.environ["CONTEXT_CHUNK_SIZE"] == "32"
    prewarm_context_layout()
    tokens = [785] * 100
    layout = compile_context_layout(
        *(
            torch.tensor(value, dtype=torch.int32)
            for value in (tokens, [80], [0, 1], [0, 40], [98], [99])
        )
    )
    expiry = torch.full((100,), torch.iinfo(torch.int32).max, dtype=torch.int32)
    expiry[:40] = 80
    body = {
        "input_ids": tokens,
        "context_program": ContextProgram(layout, expiry).to_json_wire(),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
    }
    started = time.monotonic()
    rejected = requests.post(server + "/generate", json=body, timeout=60)
    elapsed = time.monotonic() - started
    evidence = {
        "status": rejected.status_code,
        "seconds": elapsed,
        "body": rejected.json(),
    }
    output = Path(os.environ["CONTEXT_CAPACITY_RESULT"])
    output.write_text(json.dumps(evidence, indent=2))
    assert rejected.status_code == 503, evidence
    assert (
        "retains" in rejected.text and "KV pool holds 110" in rejected.text
    ), evidence
    assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
    # Almost the whole pool is needed again. Leaked private pages or a live
    # source receipt must not prevent a fresh native request from finishing.
    healthy = requests.post(
        server + "/generate",
        json={"input_ids": [785] * 90, "sampling_params": body["sampling_params"]},
        timeout=60,
    )
    evidence["subsequent_status"] = healthy.status_code
    evidence["subsequent_body"] = healthy.json()
    output.write_text(json.dumps(evidence, indent=2))
    assert healthy.status_code == 200, evidence
    assert healthy.json()["meta_info"]["completion_tokens"] == 8, evidence
    assert requests.get(server + "/health", timeout=5).status_code == 200


def test_chunk_retry_and_mixed_http_generation(server):
    baseline = call(server, False)
    identity = call(server, "identity")
    drop = call(server, "drop")
    print("HTTP_CONTROLS", json.dumps([baseline, identity, drop]), flush=True)
    retry = call(server, True)
    assert requests.post(server + "/flush_cache", timeout=5).status_code == 200
    cold = call(server, True)
    hot = call(server, True)
    print("HTTP_ISOLATED", json.dumps([retry, cold, hot]), flush=True)
    for item in (retry, cold, hot):
        for token in item["choices"][0]["logprobs"]["content"]:
            assert token["logprob"] is not None and math.isfinite(token["logprob"]), (
                item
            )
        # Native mini mask/occurrence produces these tokens for this extreme
        # first-user deletion too; decoded replacement bytes are not a failure.
        assert item["sglext"]["output_ids"] == [151645, 243] * 4, item
    assert cold["choices"][0]["message"] == hot["choices"][0]["message"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(
                lambda feature: call(server, feature, " Be concise."), [False, True]
            )
        )
    print("HTTP_RESULTS", json.dumps([baseline, cold, hot, *outputs]), flush=True)
    assert requests.get(server + "/health", timeout=5).status_code == 200
