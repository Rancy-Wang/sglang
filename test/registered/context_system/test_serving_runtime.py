"""Opt-in HTTP scheduler checks; owns only the server process it starts."""

import concurrent.futures
import json
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
        "4096",
        "--context-length",
        "2048",
        "--max-running-requests",
        "4",
        "--chunked-prefill-size",
        "64",
        "--cuda-graph-config",
        json.dumps(graph),
        "--context-drop-aware-eviction",
        "--enable-mixed-chunk",
    ]
    backend = os.environ.get("CONTEXT_TEST_ATTENTION_BACKEND")
    if backend:
        cmd += ["--attention-backend", backend]
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
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


def call(base, feature, suffix=""):
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
        "logprobs": True,
    }
    if feature == "identity":
        payload.update(reposition=[0])
    elif feature == "drop":
        payload.update(drop_message={"1": [0]})
    elif feature:
        payload.update(drop_message={"1": [0]}, reposition=[1])
    response = requests.post(base + "/v1/chat/completions", json=payload, timeout=120)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["usage"]["completion_tokens"] == 8, result
    assert result["choices"][0]["finish_reason"] == "length", result
    return result


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
            assert token["logprob"] is not None, item
        assert "\ufffd" not in (item["choices"][0]["message"].get("content") or ""), (
            item
        )
    assert cold["choices"][0]["message"] == hot["choices"][0]["message"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(
                lambda feature: call(server, feature, " Be concise."), [False, True]
            )
        )
    print("HTTP_RESULTS", json.dumps([baseline, cold, hot, *outputs]), flush=True)
    assert requests.get(server + "/health", timeout=5).status_code == 200
