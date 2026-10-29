"""Reproducible AIME cohorts against the unmodified SGLang HTTP contract.

Launches only the standard server. Output directories must be new, so failed
runs and their logs cannot be silently overwritten. Profiles are not benchmarks.
"""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def http(url, route, body=None, timeout=1200):
    return urllib.request.urlopen(
        urllib.request.Request(
            url + route, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ), timeout=timeout,
    )


def generate(url, problem, rid, scheduled_ns, max_tokens):
    remaining = (scheduled_ns - time.perf_counter_ns()) / 1e9
    if remaining > 0:
        time.sleep(remaining)
    start = time.perf_counter_ns()
    events, result = [], None
    with http(url, "/generate", {
        "rid": rid, **({"input_ids": problem} if isinstance(problem, list) else {"text": problem}), "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
    }) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            result = json.loads(payload)
            if "error" in result:
                raise RuntimeError(result)
            events.append({"received_ns": time.perf_counter_ns(), "response": {
                "meta_info": result["meta_info"], "output_ids": result.get("output_ids", [])}})
    end = time.perf_counter_ns()
    if result is None or result["meta_info"].get("finish_reason") is None:
        raise RuntimeError(f"Incomplete stream: {rid}: {result}")
    # The default API carries cumulative IDs. Validate instead of assuming deltas.
    ids = result.get("output_ids", [])
    if len(ids) != result["meta_info"]["completion_tokens"]:
        # Explicit incremental streaming is also a standard SGLang option.
        ids = [token for event in events for token in event["response"].get("output_ids", [])]
    if len(ids) != result["meta_info"]["completion_tokens"]:
        raise RuntimeError(f"Incomplete output IDs for {rid}")
    first = next(e["received_ns"] for e in events if e["response"]["meta_info"]["completion_tokens"])
    last = events[-1]["received_ns"]
    return {
        "rid": rid, "scheduled_ns": scheduled_ns, "sent_ns": start,
        "first_token_ns": first, "last_token_ns": last, "stream_end_ns": end,
        "e2e_ms": (last - start) / 1e6, "ttft_ms": (first - start) / 1e6,
        "send_lateness_ms": (start - scheduled_ns) / 1e6,
        "output_ids": ids, "finish_reason": result["meta_info"]["finish_reason"],
        "text": result.get("text", ""), "events": events,
    }


def cohort(args, rows, label):
    start = time.perf_counter_ns() + 100_000_000
    if args.serial:
        results = [generate(args.base_url, row["problem"], f"{label}-q{i+1}",
                            time.perf_counter_ns(), args.max_new_tokens)
                   for i, row in enumerate(rows)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
            futures = [pool.submit(generate, args.base_url, row["problem"],
                       f"{label}-q{i+1}", start + int(delay * 1e6), args.max_new_tokens)
                       for i, (row, delay) in enumerate(zip(rows, args.arrival_ms, strict=True))]
            results = [future.result() for future in futures]
    return {
        "label": label, "requests": results,
        "makespan_ms": (max(r["last_token_ns"] for r in results)
                        - min(r["sent_ns"] for r in results)) / 1e6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("/home/ruoxi/repo/test/datasets/aime.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first", type=int, default=8)
    p.add_argument("--arrival-ms", type=float, nargs="+", default=[0, 10, 25, 45, 75, 120, 200, 320])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--max-total-tokens", type=int)
    p.add_argument("--algorithm", choices=["ar", "loopspec"], default="ar")
    p.add_argument("--depths", type=int, nargs="+", default=[1])
    p.add_argument("--serial", action="store_true")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--no-overlap", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    p.add_argument("--base-url", default="http://127.0.0.1:30126")
    p.add_argument("--port", type=int, default=30126)
    p.add_argument("--external-server", action="store_true")
    p.add_argument("--chunked-prefill-size", type=int)
    p.add_argument("--lifecycle", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--graph-max-bs", type=int, default=8)
    p.add_argument("--mem-fraction", type=float, default=.7)
    p.add_argument("--report-mismatches", action="store_true",
                   help="Retain numerical divergences as unresolved in status; do not abort the entire benchmark.")
    args = p.parse_args()
    if args.first == 1 and args.arrival_ms == [0, 10, 25, 45, 75, 120, 200, 320]:
        args.arrival_ms = [0]
    if not args.serial and len(args.arrival_ms) != args.first:
        p.error("arrival-ms must match first")
    if args.profile and args.repeat != 1:
        p.error("profile runs require repeat=1; measure latency in separate unprofiled runs")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.dataset.read_bytes()
    all_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = all_rows[:args.first]
    assert len(rows) == args.first and all(isinstance(r["problem"], str) for r in rows)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_records = []
    for row in rows:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["problem"]}],
            tokenize=False, enable_thinking=True, add_generation_prompt=True) if args.thinking else row["problem"]
        ids = tokenizer.encode(rendered, add_special_tokens=not args.thinking)
        if args.thinking:
            assert "<think>" in rendered.split("<|im_start|>assistant")[-1]
        prompt_records.append({"id": row["id"], "rendered": rendered, "input_ids": ids})
        if args.thinking:
            row["problem"] = ids
    write_json(args.output / "prompts.json", {"chat_template": tokenizer.chat_template,
        "thinking": args.thinking, "eos_token_id": tokenizer.eos_token_id, "prompts": prompt_records})
    lengths = [len(r["input_ids"]) for r in prompt_records]
    assert max(lengths) + args.max_new_tokens <= args.context_length
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
               "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16",
               "--attention-backend", "triton", "--bf16-gemm-backend", "auto",
               "--max-running-requests", str(args.max_running_requests), "--page-size", "1",
               "--disable-radix-cache", "--context-length", str(args.context_length), "--mem-fraction-static", str(args.mem_fraction),
               "--cuda-graph-max-bs-decode", str(args.graph_max_bs), "--enable-metrics", "--stream-interval", "1"]
    if args.max_total_tokens is not None:
        command.extend(["--max-total-tokens", str(args.max_total_tokens)])
    if args.chunked_prefill_size is not None:
        command.extend(["--chunked-prefill-size", str(args.chunked_prefill_size)])
    if args.eager:
        command.append("--disable-cuda-graph")
    if args.no_overlap:
        command.append("--disable-overlap-schedule")
    if args.algorithm == "loopspec":
        command.extend(["--speculative-algorithm", "loopspec",
                        "--speculative-draft-depth", *map(str, args.depths)])
    environment = dict(os.environ, SGLANG_RUST_BUILD_MODE="auto")
    if args.lifecycle:
        environment["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY"] = "1"
    if args.profile:
        environment["SGLANG_RECURRENT_TRACE"] = "1"
        environment["SGLANG_PROFILE_V2"] = "0"
    else:
        environment.pop("SGLANG_RECURRENT_TRACE", None)
    import sglang

    manifest = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "command": command, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "question_ids": [r["id"] for r in rows], "prompt_tokens": lengths,
        "python": sys.executable, "sglang_file": sglang.__file__,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "triton"]},
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
    }
    write_json(args.output / "manifest.json", manifest)
    process = None
    profile_started = False
    try:
        if not args.external_server:
            with (args.output / "server.log").open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            write_json(args.output / "process.json", {"pid": process.pid, "pgid": process.pid})
        for _ in range(600):
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Server exited {process.returncode}; see server.log")
            try:
                with http(args.base_url, "/health", timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError("Server readiness timeout")
        with http(args.base_url, "/get_server_info") as response:
            write_json(args.output / "server_info.json", json.load(response))
        if args.lifecycle:
            from verify_lifecycle import verify
            write_json(args.output / "lifecycle.json", verify(args.base_url, rows))
            write_json(args.output / "status.json", {"success": True, "lifecycle": True})
            return
        for index in range(args.warmup):
            write_json(args.output / f"warmup-{index}.json", cohort(args, rows, f"warmup-{index}"))
        if args.profile:
            with http(args.base_url, "/start_profile", {
                "output_dir": str(args.output / "trace"), "activities": ["CPU", "GPU"],
                "profile_id": args.output.name, "with_stack": False,
                "record_shapes": False, "detailed_annotations": True,
            }) as response:
                write_json(args.output / "profile-start.json", {"status": response.status, "message": response.read().decode()})
            profile_started = True
        runs = []
        for index in range(args.repeat):
            result = cohort(args, rows, f"run-{index}")
            write_json(args.output / f"cohort-{index}.json", result)
            runs.append(result)
            print(json.dumps({"cohort": index, "makespan_ms": result["makespan_ms"],
                              "tokens": [len(r["output_ids"]) for r in result["requests"]]}), flush=True)
        if args.profile:
            with http(args.base_url, "/stop_profile", {}) as response:
                write_json(args.output / "profile-stop.json", {"status": response.status, "message": response.read().decode()})
            profile_started = False
        reference = json.loads((args.reference / "cohort-0.json").read_text()) if args.reference else runs[0]
        mismatches = []
        for n, run in enumerate(runs):
            for q, actual in enumerate(run["requests"]):
                expected = reference["requests"][q]
                if (actual["output_ids"], actual["finish_reason"]) != (expected["output_ids"], expected["finish_reason"]):
                    common = next((i for i, (a, b) in enumerate(zip(actual["output_ids"], expected["output_ids"])) if a != b),
                                  min(len(actual["output_ids"]), len(expected["output_ids"])))
                    mismatches.append({"cohort": n, "question": q + 1, "first_difference": common})
        write_json(args.output / "correctness.json", {"reference": str(args.reference), "mismatches": mismatches})
        if mismatches and not args.report_mismatches:
            raise RuntimeError(f"Output mismatches: {mismatches}")
        write_json(args.output / "status.json", {"success": not mismatches, "execution_complete": True, "numerical_equivalence": "unresolved" if mismatches else "repeat_exact", "profile": args.profile})
    except BaseException as error:
        write_json(args.output / "status.json", {"success": False, "error": repr(error)})
        raise
    finally:
        if profile_started:
            try:
                with http(args.base_url, "/stop_profile", {}, timeout=120):
                    pass
            except Exception:
                pass
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
