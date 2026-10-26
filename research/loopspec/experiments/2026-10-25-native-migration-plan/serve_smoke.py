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
    parser.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30126)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
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
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--enable-metrics",
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
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
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
            for _ in range(args.warmup):
                http_request(url)
            # The idle scheduler drains pending CUDA event intervals. Keep raw
            # metric snapshots so prefill/decode are not inferred from TTFT.
            time.sleep(0.2)
            with request(url, "/metrics") as response:
                before = response.read().decode()
            (args.output / "metrics-before.txt").write_text(before)
            samples = []
            for index in range(args.repeat):
                result = http_request(url)
                (args.output / f"response-{index}.json").write_text(
                    json.dumps(result, indent=2)
                )
                samples.append(result)
                print(json.dumps(result), flush=True)
            time.sleep(0.2)
            with request(url, "/metrics") as response:
                after = response.read().decode()
            (args.output / "metrics.txt").write_text(after)
            from prometheus_client.parser import text_string_to_metric_families

            def stages(text):
                out = {}
                for family in text_string_to_metric_families(text):
                    for sample in family.samples:
                        if sample.name == "sglang:forward_execution_seconds_total":
                            key = sample.labels["category"]
                            out[key] = out.get(key, 0) + sample.value
                return out

            first, last = stages(before), stages(after)
            stage_delta = {
                key: value - first.get(key, 0) for key, value in last.items()
            }
            assert any("extend" in key or "prefill" in key for key in stage_delta), (
                stage_delta
            )
            assert stage_delta.get("decode", 0) > 0, stage_delta
            (args.output / "benchmark.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "warmup": args.warmup,
                        "samples": samples,
                        "device_timer_enabled": True,
                        "gpu_stage_seconds": stage_delta,
                        "gpu_timing_note": "CUDA stream intervals; may include CPU submission gaps. Raw counters before/after retained; 0.2s idle allows scheduler event drain.",
                        "timing_note": "first_token_latency is service TTFT (includes scheduling); decode_throughput is native service decode throughput, not isolated GPU compute",
                    },
                    indent=2,
                )
            )
            metrics = after
            (args.output / "metrics.txt").write_text(metrics)
            if args.depths:
                assert "sglang:loopspec_events_total" in metrics, metrics
                if args.cuda_graph:
                    assert 'role="rejection"' in metrics, metrics
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
