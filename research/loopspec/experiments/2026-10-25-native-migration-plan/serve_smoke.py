"""Launch the standard server, exercise its standard API, then stop our group."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from verify_ouro import PROMPT, SAMPLING, http_request


def request(url, path, body=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=120,
    )


def lifecycle(url, expected):
    results = {}
    for count in (0, 1):
        with request(
            url,
            "/generate",
            {"text": PROMPT, "sampling_params": {**SAMPLING, "max_new_tokens": count}},
        ) as response:
            result = json.load(response)
        assert result["meta_info"]["completion_tokens"] == count, result
        results[f"max_{count}"] = result
    with request(
        url,
        "/generate",
        {"text": PROMPT, "sampling_params": {**SAMPLING, "stop": "Paris"}},
    ) as response:
        result = json.load(response)
    assert result["meta_info"]["finish_reason"] == {
        "type": "stop",
        "matched": "Paris",
    }, result
    results["stop_string"] = result
    last = None
    with request(
        url, "/generate", {"text": PROMPT, "sampling_params": SAMPLING, "stream": True}
    ) as response:
        for raw in response:
            if raw.startswith(b"data: ") and raw.strip() != b"data: [DONE]":
                last = json.loads(raw[6:])
    assert last is not None and last["text"] == expected["response"]["text"], last
    results["stream"] = last
    with request(
        url,
        "/generate",
        {
            "rid": "loopspec-lifecycle-cancel",
            "text": PROMPT,
            "sampling_params": {**SAMPLING, "ignore_eos": True, "max_new_tokens": 512},
            "stream": True,
        },
    ) as response:
        aborted = False
        for raw in response:
            if raw.startswith(b"data: ") and raw.strip() != b"data: [DONE]":
                last = json.loads(raw[6:])
                if not aborted:
                    with request(
                        url, "/abort_request", {"rid": "loopspec-lifecycle-cancel"}
                    ):
                        pass
                    aborted = True
    assert last["meta_info"]["finish_reason"]["type"] == "abort", last
    results["cancel"] = last
    result = http_request(url)
    assert result["output_ids"] == expected["output_ids"], result
    results["after_cancel"] = result
    with request(url, "/v1/models") as response:
        model_id = json.load(response)["data"][0]["id"]
    with request(
        url,
        "/v1/completions",
        {
            "model": model_id,
            "prompt": PROMPT,
            "temperature": 0,
            "max_tokens": 32,
        },
    ) as response:
        result = json.load(response)
    assert result["choices"][0]["text"] == expected["response"]["text"], result
    results["openai_completions"] = result
    with request(
        url,
        "/generate",
        {
            "text": PROMPT,
            "sampling_params": {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "max_new_tokens": 16,
                "sampling_seed": 42,
            },
        },
    ) as response:
        results["sampling"] = json.load(response)
    assert results["sampling"]["meta_info"]["completion_tokens"] > 0
    with request(url, "/flush_cache?timeout=10") as response:
        results["flush"] = response.read().decode()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30126)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument(
        "--bf16-gemm-backend", default="auto", choices=["auto", "torch", "tinygemm"]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/home/ruoxi/models/Ouro-2.6B",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--max-running-requests",
        "1",
        "--page-size",
        "1",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--disable-cuda-graph",
        "--context-length",
        "2048",
        "--mem-fraction-static",
        "0.7",
    ]
    if args.cuda_graph:
        command.remove("--disable-cuda-graph")
    command += ["--bf16-gemm-backend", args.bf16_gemm_backend]
    if args.depths:
        command += ["--speculative-algorithm", "loopspec", "--speculative-draft-depth"]
        command += list(map(str, args.depths))
    (args.output / "command.json").write_text(json.dumps(command))
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server failed with exit {process.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server did not become healthy")
                    time.sleep(1)
            for index in range(2):
                result = http_request(url)
                (args.output / f"response-{index}.json").write_text(
                    json.dumps(result, indent=2)
                )
                print(json.dumps(result), flush=True)
            if args.lifecycle:
                summary = lifecycle(url, result)
                (args.output / "lifecycle.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print("Lifecycle checks passed", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
