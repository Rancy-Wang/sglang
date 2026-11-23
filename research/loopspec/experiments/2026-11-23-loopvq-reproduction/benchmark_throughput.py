#!/usr/bin/env python3
"""Measure a real SGLang HTTP server; no model or server is launched here.

measure --base-url http://127.0.0.1:30000 --method bf16 --protocol common-shape
        --bs 16 --input-ids-json inputs.json --server-manifest server.json
        --server-log server.log --out-dir results/bf16-bs16
aggregate --results results/*/summary.json --out-dir results/paired [--wandb]

inputs.json: {"input_ids": [[...], ...]} (at least bs equal-length prompts).
server.json requires host, gpu_uuid, model_sha256, sglang_sha,
environment_sha256, server_pid, model_path, method, protocol, prefix_tokens,
recent_tokens, exclusive_server=true, and config (the actual launch settings).
config must include tp_size=pp_size=dp_size=1, total_ut_steps=4,
speculative_algorithm=null, disable_radix_cache=true, chunked_prefill_size=-1,
decode_log_interval=1, dtype, attention_backend and the canonical nested
cuda_graph_config returned by /server_info. Prefill backend must be disabled;
decode backend must be full or disabled. Actual capture buckets are read from
the server's startup capture log; convenience flags are not runtime evidence.
Use --incremental-streaming-output
on both servers to avoid cumulative text transfer at long output lengths.

The primary metric is output tokens / whole HTTP batch wall time. TTFT/TPOT
are CLIENT streaming observations. No server decode/prefill time is inferred
from TTFT. Scheduler logs provide actual active batch evidence, not GPU time.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOLS = {"common-shape": (2048, 512, 64, 256), "paper-long": (2048, 32768, 8, 64)}
IDENTITY_KEYS = ("host", "gpu_uuid", "model_sha256", "sglang_sha", "environment_sha256",
                 "model_path", "protocol", "prefix_tokens", "recent_tokens")
CONFIG_KEYS = ("tp_size", "pp_size", "dp_size", "dtype", "attention_backend",
               "speculative_algorithm", "disable_radix_cache", "chunked_prefill_size",
               "cuda_graph_config", "cuda_graph_bs_decode", "cuda_graph_backend_decode",
               "cuda_graph_backend_prefill", "disable_cuda_graph", "disable_decode_cuda_graph",
               "disable_prefill_cuda_graph", "disable_cuda_graph_padding",
               "decode_log_interval", "incremental_streaming_output", "context_length",
               "max_running_requests", "mem_fraction_static", "max_total_tokens",
               "page_size", "loopvq_codebook_dir", "loopvq_prefix_tokens", "loopvq_recent_tokens")
PAIR_CONFIG_KEYS = tuple(k for k in CONFIG_KEYS if k not in (
    "loopvq_codebook_dir", "loopvq_prefix_tokens", "loopvq_recent_tokens",
    "mem_fraction_static", "max_total_tokens", "page_size", "cuda_graph_bs_decode",
    "cuda_graph_backend_decode", "cuda_graph_backend_prefill", "disable_cuda_graph",
    "disable_decode_cuda_graph", "disable_prefill_cuda_graph"))
DECODE_PATTERN = re.compile(r"Decode batch.*?#running-req\s*[:=]\s*(\d+)")
GRAPH_PATTERN = re.compile(r"(?:cuda graph|cuda_graph)\s*[:=]\s*(True|False)", re.I)
CAPTURE_PATTERN = re.compile(r"Capture target decode CUDA graph end\.\s+bs=(\[[^\]]*\])", re.I)
OOM_PATTERN = re.compile(r"out of memory|OutOfMemoryError|CUDA OOM|not enough (?:memory|KV)|capacity.limit", re.I)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def get_server_config(base_url, timeout):
    with urllib.request.urlopen(base_url.rstrip("/") + "/server_info", timeout=timeout) as response:
        # Only retain a non-sensitive allowlist, never all resolved server args.
        raw = json.load(response)
    return {key: raw.get(key) for key in CONFIG_KEYS}


def validate_manifest(manifest, actual, method, protocol, bs, prefix, recent):
    required = set(IDENTITY_KEYS) | {"server_pid", "method", "exclusive_server", "config"}
    if required - manifest.keys():
        raise ValueError(f"Missing server identity fields: {sorted(required - manifest.keys())}")
    if manifest["method"] != method or manifest["protocol"] != protocol:
        raise ValueError("Method/protocol differs from the server manifest")
    if (manifest["prefix_tokens"], manifest["recent_tokens"]) != (prefix, recent):
        raise ValueError("Window differs from the selected protocol")
    if manifest["host"].split(".")[0] != platform.node().split(".")[0]:
        raise ValueError("Run beside the measured server so log/GPU observations are local")
    if manifest["exclusive_server"] is not True:
        raise ValueError("An exclusive benchmark server is required for batch-log attribution")
    cfg = manifest["config"]
    required_values = {"tp_size": 1, "pp_size": 1, "dp_size": 1,
                       "speculative_algorithm": None, "disable_radix_cache": True,
                       "chunked_prefill_size": -1, "decode_log_interval": 1}
    if cfg.get("total_ut_steps") != 4:
        raise ValueError("The manifest must attest the complete R4 model")
    for key, expected in required_values.items():
        if key not in cfg or cfg[key] != expected or actual.get(key) != expected:
            raise ValueError(f"Unsupported or unverified setting {key}: expected {expected!r}")
    for key in ("dtype", "attention_backend", "cuda_graph_config"):
        if key not in cfg or cfg[key] != actual.get(key):
            raise ValueError(f"Server config readback differs for {key}")
    graph = actual.get("cuda_graph_config")
    if not isinstance(graph, dict) or not isinstance(graph.get("decode"), dict) or not isinstance(graph.get("prefill"), dict):
        raise ValueError("Missing canonical per-phase CUDA graph configuration")
    if graph["prefill"].get("backend") != "disabled":
        raise ValueError("The prefill CUDA graph backend must be disabled")
    if graph["decode"].get("backend") not in ("full", "disabled"):
        raise ValueError("Only full or disabled decode CUDA graphs are supported by this protocol")
    if graph["decode"]["backend"] == "full":
        buckets = graph["decode"].get("bs")
        if not isinstance(buckets, list) or not buckets or any(type(size) is not int or size <= 0 for size in buckets):
            raise ValueError("Full decode graphs require valid canonical capture buckets")
    if actual.get("max_running_requests") is not None and actual["max_running_requests"] < bs:
        raise ValueError("Server max_running_requests is smaller than the requested batch")
    enabled = bool(actual.get("loopvq_codebook_dir"))
    if enabled != (method == "loopvq"):
        raise ValueError("The actual LoopVQ activation differs from the requested method")
    if enabled:
        if (actual.get("loopvq_prefix_tokens"), actual.get("loopvq_recent_tokens")) != (prefix, recent):
            raise ValueError("Actual LoopVQ prefix/recent windows differ")
        if not manifest.get("bundle_sha256"):
            raise ValueError("LoopVQ requires an immutable bundle hash in its manifest")


class StreamStats:
    def __init__(self, bs, output_len, input_len):
        self.bs, self.output_len, self.input_len = bs, output_len, input_len
        self.requests = [{"completion_tokens": 0, "first_chunk_seconds": None,
                          "first_chunk_tokens": 0, "last_token_seconds": None,
                          "finish_reason": None, "prompt_tokens": None} for _ in range(bs)]

    def feed(self, item, seconds):
        if "error" in item:
            raise RuntimeError(json.dumps(item["error"]))
        index = item.get("index", 0 if self.bs == 1 else None)
        if not isinstance(index, int) or not 0 <= index < self.bs:
            raise ValueError(f"Invalid stream index: {index!r}")
        meta = item["meta_info"]
        count = meta["completion_tokens"]
        row = self.requests[index]
        if not isinstance(count, int) or count < row["completion_tokens"] or count > self.output_len:
            raise ValueError("Completion count regressed or exceeded the fixed output budget")
        if count > row["completion_tokens"]:
            if row["first_chunk_seconds"] is None:
                row["first_chunk_seconds"] = seconds
                row["first_chunk_tokens"] = count
            row["last_token_seconds"] = seconds
        row["completion_tokens"] = count
        if "prompt_tokens" in meta:
            row["prompt_tokens"] = meta["prompt_tokens"]
        if meta.get("finish_reason") is not None:
            row["finish_reason"] = meta["finish_reason"]
        return {"index": index, "arrival_seconds": seconds, "completion_tokens": count,
                "prompt_tokens": row["prompt_tokens"], "finish_reason": meta.get("finish_reason")}

    def finalize(self, wall_seconds):
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("Invalid whole-batch wall time")
        for row in self.requests:
            if row["completion_tokens"] != self.output_len:
                raise ValueError("A request ended without the full fixed output length")
            if row["prompt_tokens"] != self.input_len:
                raise ValueError("Server prompt length differs from the fixed token IDs")
            if not isinstance(row["finish_reason"], dict) or row["finish_reason"].get("type") != "length":
                raise ValueError("Fixed-length performance requests must finish by length")
            after_first = row["completion_tokens"] - row["first_chunk_tokens"]
            row["client_ttft_seconds"] = row["first_chunk_seconds"]
            row["client_tpot_seconds"] = ((row["last_token_seconds"] - row["first_chunk_seconds"]) / after_first
                                          if after_first else None)
        tokens = sum(row["completion_tokens"] for row in self.requests)
        return {"output_tokens": tokens, "input_tokens": self.bs * self.input_len,
                "e2e_seconds": wall_seconds, "e2e_output_tokens_s": tokens / wall_seconds,
                "overall_input_output_tokens_s": (tokens + self.bs * self.input_len) / wall_seconds,
                "server_decode_seconds": None, "server_decode_tokens_s": None,
                "server_prefill_seconds": None,
                "timing_note": "TTFT/TPOT use client SSE arrival times; server decode/prefill timing is unavailable",
                "requests": self.requests}


def parse_capture_evidence(text):
    """Read actual runner buckets, which can differ from resolved launch buckets."""
    result = {"capture_bs": None, "source": "startup capture log unavailable"}
    for line in text.splitlines():
        match = CAPTURE_PATTERN.search(line)
        if match:
            buckets = json.loads(match.group(1))
            if not isinstance(buckets, list) or not buckets or any(type(size) is not int or size <= 0 for size in buckets):
                raise ValueError("Invalid actual capture bucket list in server log")
            result = {"capture_bs": buckets, "source": "target decode runner.capture_bs startup log",
                      "line": line}
    return result


def capture_from_log(path):
    result = {"capture_bs": None, "source": "startup capture log unavailable"}
    offset = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            if b"Capture target decode" in line and b"graph end." in line:
                parsed = parse_capture_evidence(line.decode(errors="replace"))
                if parsed["capture_bs"] is not None:
                    result = {**parsed, "byte_offset": offset}
            offset += len(line)
    return result


def decode_evidence(text, bs, output_len, config, capture=None):
    rows = []
    for line in text.splitlines():
        match = DECODE_PATTERN.search(line)
        if match:
            graph = GRAPH_PATTERN.search(line)
            rows.append({"active_requests": int(match.group(1)),
                         "cuda_graph": graph.group(1).lower() == "true" if graph else None})
    counts = collections.Counter(row["active_requests"] for row in rows)
    current = longest = 0
    for row in rows:
        current = current + 1 if row["active_requests"] == bs else 0
        longest = max(longest, current)
    fraction = counts.get(bs, 0) / len(rows) if rows else 0.0
    unexpected = any(row["active_requests"] > bs for row in rows)
    required_streak = min(8, max(1, output_len - 1))
    reasons = []
    if len(rows) < max(1, output_len - 2):
        reasons.append("incomplete_decode_log_coverage")
    if fraction < 0.9 or longest < required_streak:
        reasons.append("full_batch_not_sustained")
    if unexpected:
        reasons.append("unattributed_extra_requests")
    retractions = [line for line in text.splitlines() if re.search(r"retract|recomput", line, re.I)]
    if retractions:
        reasons.append("retraction_or_recomputation_log_requires_review")
    phase = config.get("cuda_graph_config", {}).get("decode", {})
    backend = phase.get("backend")
    captured = (capture or {}).get("capture_bs")
    target_rows = [row for row in rows if row["active_requests"] == bs]
    if backend == "full" and any(row["cuda_graph"] is not True for row in target_rows):
        reasons.append("full_decode_graph_not_verified")
    if backend == "disabled" and any(row["cuda_graph"] is not False for row in target_rows):
        reasons.append("eager_decode_mode_not_verified")
    if backend not in ("full", "disabled"):
        reasons.append("unsupported_or_missing_decode_graph_backend")
    padding = (0 if backend == "disabled" else
               0 if isinstance(captured, list) and bs in captured else "unknown")
    return {"verified": not reasons, "reasons": reasons,
            "source": "scheduler report_decode_stats: len(batch.reqs), ordinary AR, decode_log_interval=1",
            "decode_records": len(rows), "active_batch_counts": dict(sorted(counts.items())),
            "target_batch_fraction": fraction, "longest_target_batch_streak": longest,
            "cuda_graph_true_records": sum(row["cuda_graph"] is True for row in rows),
            "cuda_graph_unknown_records": sum(row["cuda_graph"] is None for row in rows),
            "graph_padding_at_target_bs": padding, "decode_graph_backend": backend,
            "canonical_capture_bs": phase.get("bs"), "actual_capture_evidence": capture,
            "retraction_log_lines": retractions,
            "criteria": {"min_target_fraction": 0.9, "min_target_streak": required_streak,
                         "min_decode_records": max(1, output_len - 2)}}


class GPUSampler:
    def __init__(self, gpu_uuid):
        self.gpu_uuid, self.samples, self.errors = gpu_uuid, [], []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        try:
            output = subprocess.check_output([
                "nvidia-smi", "-i", self.gpu_uuid,
                "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=5).strip()
            uuid, memory, utilization = [part.strip() for part in output.split(",")]
            if uuid != self.gpu_uuid:
                raise ValueError("GPU UUID readback mismatch")
            self.samples.append({"time_unix": time.time(), "memory_used_mib": float(memory),
                                 "utilization_percent": float(utilization)})
        except Exception as exc:
            self.errors.append({"time_unix": time.time(), "error": str(exc)})

    def start(self):
        self.sample()
        def work():
            while not self.stop.wait(1.0):
                self.sample()
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        self.sample()
        return {"samples": self.samples, "errors": self.errors,
                "sampled_peak_device_memory_mib": max((s["memory_used_mib"] for s in self.samples), default=None),
                "note": "Whole-device sampled usage, not allocator peak or exclusive process memory"}


def run_batch(args, inputs, phase, number, config, manifest, capture):
    stem = f"{phase}-{number:02d}"
    log = Path(args.server_log)
    log_stat = log.stat()
    offset = log_stat.st_size
    payload = {"input_ids": inputs, "sampling_params": {"temperature": 0.0,
               "max_new_tokens": args.output_len, "ignore_eos": True, "stream_interval": 1},
               "stream": True}
    atomic_json(Path(args.out_dir) / f"{stem}.request.json", payload)
    stats = StreamStats(args.bs, args.output_len, args.input_len)
    gpu = GPUSampler(manifest["gpu_uuid"])
    gpu.start()
    record = {"phase": phase, "number": number, "started_unix": time.time(),
              "status": "failed", "server_log_start_offset": offset}
    request = urllib.request.Request(args.base_url.rstrip("/") + "/generate",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with (Path(args.out_dir) / f"{stem}.events.jsonl").open("w", buffering=1024 * 1024) as events:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                done = False
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    item = json.loads(data)
                    observation = stats.feed(item, time.perf_counter() - started)
                    events.write(json.dumps(observation) + "\n")
                ended = time.perf_counter()
                if not done:
                    raise ValueError("SSE stream closed without [DONE]")
        record.update(stats.finalize(ended - started), status="ok")
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc),
                      e2e_seconds=time.perf_counter() - started, requests=stats.requests)
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
            record["http_error_body"] = exc.read().decode(errors="replace")
    finally:
        record["gpu"] = gpu.finish()
        # Outside timed interval; allow the final scheduler log line to flush.
        time.sleep(0.15)
        text = ""
        try:
            now_stat = log.stat()
            record["server_log_end_offset"] = now_stat.st_size
            if now_stat.st_ino != log_stat.st_ino or now_stat.st_size < offset:
                record["log_error"] = "Server log was rotated or truncated during the batch"
            else:
                with log.open("rb") as stream:
                    stream.seek(offset)
                    text = stream.read(now_stat.st_size - offset).decode(errors="replace")
        except OSError as exc:
            record["log_error"] = str(exc)
        (Path(args.out_dir) / f"{stem}.server.log").write_text(text)
        record["batch_evidence"] = decode_evidence(text, args.bs, args.output_len, config, capture)
        if OOM_PATTERN.search(text + str(record.get("error", "")) + record.get("http_error_body", "")):
            record["status"] = "oom_or_capacity_limited"
        elif record["status"] == "ok" and not record["batch_evidence"]["verified"]:
            record["status"] = "batch_not_verified"
        record["ended_unix"] = time.time()
        atomic_json(Path(args.out_dir) / f"{stem}.json", record)
    return record


def measure(args):
    if not os.environ.get("TMUX"):
        raise ValueError("Run long benchmark measurements inside the target machine's tmux")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.json")):
        raise ValueError("Use a fresh output directory; failed attempts must remain reviewable")
    shape = PROTOCOLS[args.protocol]
    args.input_len = shape[0] if args.input_len is None else args.input_len
    args.output_len = shape[1] if args.output_len is None else args.output_len
    if args.protocol == "paper-long" and (args.input_len, args.output_len) != shape[:2]:
        raise ValueError("paper-long must retain 2048 input / 32768 output")
    if args.output_len < 2 or args.input_len < 1:
        raise ValueError("Need positive input length and at least two generated tokens")
    raw_inputs = json.loads(Path(args.input_ids_json).read_text())
    all_inputs = raw_inputs["input_ids"] if isinstance(raw_inputs, dict) else raw_inputs
    if not isinstance(all_inputs, list) or len(all_inputs) < args.bs:
        raise ValueError("Input fixture must contain at least bs prompts")
    inputs = all_inputs[:args.bs]
    if any(not isinstance(row, list) or len(row) != args.input_len or
           any(type(token) is not int or token < 0 for token in row) for row in inputs):
        raise ValueError("Input token IDs must be nonnegative integers with the exact fixed length")
    manifest = json.loads(Path(args.server_manifest).read_text())
    (out / "input_ids.json").write_text(json.dumps({"input_ids": inputs}))
    summary = {"format_version": 1, "method": args.method, "protocol": args.protocol,
               "bs": args.bs, "input_len": args.input_len, "output_len": args.output_len,
               "input_ids_sha256": digest(inputs), "input_fixture_sha256": digest(all_inputs),
               "server_manifest": manifest, "server_config_readback": {},
               "base_url": args.base_url, "server_log": str(Path(args.server_log).resolve()),
               "client_host": platform.node(), "warmup_count": 2, "repeat_count": 3,
               "status": "running", "attempts": [], "started_unix": time.time(),
               "primary_metric": "e2e_output_tokens_s", "fixed_length_sampling": {"temperature": 0.0, "ignore_eos": True}}
    atomic_json(out / "summary.json", summary)
    try:
        Path(args.server_log).stat()
        config = get_server_config(args.base_url, min(args.timeout, 30))
        summary["server_config_readback"] = config
        validate_manifest(manifest, config, args.method, args.protocol, args.bs, shape[2], shape[3])
        capture = capture_from_log(args.server_log)
        summary["actual_capture_evidence"] = capture
    except Exception as exc:
        log_path = Path(args.server_log)
        startup_log = ""
        if log_path.exists():
            with log_path.open("rb") as stream:
                start_offset = max(0, log_path.stat().st_size - 2 * 1024 * 1024)
                stream.seek(start_offset)
                startup_log = stream.read().decode(errors="replace")
            summary["preflight_log_start_offset"] = start_offset
        (out / "preflight.server.log").write_text(startup_log)
        summary.update(status="oom_or_capacity_limited" if OOM_PATTERN.search(startup_log) else "preflight_failed",
                       error_type=type(exc).__name__, error=str(exc), ended_unix=time.time(),
                       unattempted_policy="No requests sent; preflight failed")
        atomic_json(out / "summary.json", summary)
        return 1
    atomic_json(out / "summary.json", summary)
    for phase, count in (("warmup", 2), ("measure", 3)):
        for number in range(1, count + 1):
            record = run_batch(args, inputs, phase, number, config, manifest, capture)
            summary["attempts"].append({"path": str((out / f"{phase}-{number:02d}.json").resolve()),
                                         "phase": phase, "number": number, "status": record["status"],
                                         "e2e_output_tokens_s": record.get("e2e_output_tokens_s")})
            atomic_json(out / "summary.json", summary)
            print(json.dumps(summary["attempts"][-1]), flush=True)
            if record["status"] != "ok":
                summary.update(status=record["status"], ended_unix=time.time(),
                               unattempted_policy="Stopped after failure; remaining repeats were not run")
                atomic_json(out / "summary.json", summary)
                return 1
    values = [a["e2e_output_tokens_s"] for a in summary["attempts"] if a["phase"] == "measure"]
    measured = [json.loads(Path(a["path"]).read_text()) for a in summary["attempts"] if a["phase"] == "measure"]
    ttfts = [r["client_ttft_seconds"] for m in measured for r in m["requests"]]
    tpots = [r["client_tpot_seconds"] for m in measured for r in m["requests"] if r["client_tpot_seconds"] is not None]
    memory = [m["gpu"]["sampled_peak_device_memory_mib"] for m in measured if m["gpu"]["sampled_peak_device_memory_mib"] is not None]
    summary.update(status="ok", ended_unix=time.time(), e2e_output_tokens_s_median=statistics.median(values),
                   e2e_output_tokens_s_min=min(values), e2e_output_tokens_s_max=max(values),
                   client_ttft_seconds_median=statistics.median(ttfts),
                   client_tpot_seconds_median=statistics.median(tpots) if tpots else None,
                   sampled_peak_device_memory_mib=max(memory) if memory else None,
                   min_target_batch_fraction=min(m["batch_evidence"]["target_batch_fraction"] for m in measured))
    atomic_json(out / "summary.json", summary)
    return 0


def paired_row(baseline, quantized):
    for key in ("protocol", "bs", "input_len", "output_len", "input_ids_sha256", "input_fixture_sha256",
                "warmup_count", "repeat_count", "fixed_length_sampling", "primary_metric"):
        if baseline.get(key) != quantized.get(key):
            raise ValueError(f"Cannot pair differing {key}")
    for key in IDENTITY_KEYS:
        if baseline["server_manifest"][key] != quantized["server_manifest"][key]:
            raise ValueError(f"Cannot pair differing server identity {key}")
    for key in PAIR_CONFIG_KEYS:
        left = baseline["server_config_readback"] if baseline["server_config_readback"] else baseline["server_manifest"]["config"]
        right = quantized["server_config_readback"] if quantized["server_config_readback"] else quantized["server_manifest"]["config"]
        if left.get(key) != right.get(key):
            raise ValueError(f"Cannot pair differing runtime setting {key}")
    if baseline["method"] != "bf16" or quantized["method"] != "loopvq":
        raise ValueError("Need one BF16 and one LoopVQ result")
    comparable = baseline["status"] == quantized["status"] == "ok"
    b = baseline.get("e2e_output_tokens_s_median") if baseline["status"] == "ok" else None
    q = quantized.get("e2e_output_tokens_s_median") if quantized["status"] == "ok" else None
    if comparable and any(value is None or not math.isfinite(value) or value <= 0 for value in (b, q)):
        raise ValueError("Successful measurements must have finite positive throughput")
    return {"protocol": baseline["protocol"], "bs": baseline["bs"],
            "input_len": baseline["input_len"], "output_len": baseline["output_len"],
            "host": baseline["server_manifest"]["host"], "gpu_uuid": baseline["server_manifest"]["gpu_uuid"],
            "prefix_tokens": baseline["server_manifest"]["prefix_tokens"],
            "recent_tokens": baseline["server_manifest"]["recent_tokens"],
            "bf16_status": baseline["status"], "loopvq_status": quantized["status"],
            "bf16_output_tokens_s": b, "loopvq_output_tokens_s": q,
            "bf16_ttft_seconds": baseline.get("client_ttft_seconds_median"),
            "loopvq_ttft_seconds": quantized.get("client_ttft_seconds_median"),
            "bf16_client_tpot_seconds": baseline.get("client_tpot_seconds_median"),
            "loopvq_client_tpot_seconds": quantized.get("client_tpot_seconds_median"),
            "bf16_sampled_peak_device_memory_mib": baseline.get("sampled_peak_device_memory_mib"),
            "loopvq_sampled_peak_device_memory_mib": quantized.get("sampled_peak_device_memory_mib"),
            "bf16_min_target_batch_fraction": baseline.get("min_target_batch_fraction"),
            "loopvq_min_target_batch_fraction": quantized.get("min_target_batch_fraction"),
            "speedup": q / b if comparable and b and q is not None else None,
            "speedup_status": "measured" if comparable else "N/A",
            "metric": "median full HTTP batch E2E output tokens/s"}


def aggregate(args):
    grouped = collections.defaultdict(dict)
    for path in args.results:
        result = json.loads(Path(path).read_text())
        key = (result["protocol"], result["bs"], result["input_len"], result["output_len"])
        if result["method"] in grouped[key]:
            raise ValueError(f"Duplicate method result for {key}; select attempts explicitly")
        grouped[key][result["method"]] = result
    rows = []
    for key, methods in sorted(grouped.items()):
        if set(methods) != {"bf16", "loopvq"}:
            raise ValueError(f"Missing pair for {key}")
        rows.append(paired_row(methods["bf16"], methods["loopvq"]))
    if not rows:
        raise ValueError("No result pairs")
    for protocol in {row["protocol"] for row in rows}:
        shapes = {(row["input_len"], row["output_len"]) for row in rows if row["protocol"] == protocol}
        if len(shapes) != 1:
            raise ValueError(f"All batches of {protocol} must use one frozen input/output shape")
        manifests = [methods["bf16"]["server_manifest"] for key, methods in grouped.items() if key[0] == protocol]
        for key in IDENTITY_KEYS:
            if len({digest(manifest[key]) for manifest in manifests}) != 1:
                raise ValueError(f"The {protocol} batch matrix mixes {key}")
    coverage = {protocol: {"present_bs": sorted(row["bs"] for row in rows if row["protocol"] == protocol),
                          "missing_bs": sorted({1, 2, 4, 8, 16} - {row["bs"] for row in rows if row["protocol"] == protocol})}
                for protocol in {row["protocol"] for row in rows}}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "throughput.json", {"rows": rows, "coverage": coverage,
                                          "sources": [str(Path(p).resolve()) for p in args.results]})
    with (out / "throughput.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: "N/A" if value is None else value for key, value in row.items()} for row in rows)
    if args.wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         group="2026-11-23-loopvq-reproduction", job_type="throughput-summary", mode="online",
                         config={"coverage": coverage})
        columns = list(rows[0])
        run.log({"throughput_results": wandb.Table(columns=columns, data=[[row[c] for c in columns] for row in rows])})
        for row in rows:
            if row["speedup"] is not None:
                run.summary[f"{row['protocol']}/bs{row['bs']}/speedup"] = row["speedup"]
        atomic_json(out / "wandb.json", {"url": run.url, "run_id": run.id})
        run.finish()
    print(json.dumps(rows, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measure_parser = sub.add_parser("measure")
    measure_parser.add_argument("--base-url", required=True)
    measure_parser.add_argument("--method", choices=("bf16", "loopvq"), required=True)
    measure_parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    measure_parser.add_argument("--bs", type=int, choices=(1, 2, 4, 8, 16), required=True)
    measure_parser.add_argument("--input-len", type=int, help="common-shape override frozen after capacity preflight")
    measure_parser.add_argument("--output-len", type=int, help="common-shape override frozen after capacity preflight")
    measure_parser.add_argument("--input-ids-json", required=True)
    measure_parser.add_argument("--server-manifest", required=True)
    measure_parser.add_argument("--server-log", required=True)
    measure_parser.add_argument("--out-dir", required=True)
    measure_parser.add_argument("--timeout", type=float, default=21600, help="HTTP read timeout; seconds")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--results", nargs="+", required=True)
    aggregate_parser.add_argument("--out-dir", required=True)
    aggregate_parser.add_argument("--wandb", action="store_true")
    aggregate_parser.add_argument("--wandb-entity", default="529513043-tsinghua-university")
    aggregate_parser.add_argument("--wandb-project", default="loopvq-ouro-2p6b-thinking")
    args = parser.parse_args()
    return measure(args) if args.command == "measure" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
