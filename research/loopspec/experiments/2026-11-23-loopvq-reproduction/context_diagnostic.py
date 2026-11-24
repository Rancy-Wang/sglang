#!/usr/bin/env python3
"""Frozen-inference context diagnostic; run on A10 in tmux with system Python.

fixtures uses the model environment; run uses stdlib only. Inference always
comes from the frozen checkout. No remote source mutation or process guessing.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request

import benchmark_throughput as bench
import a10_paperlong_pipeline as safe

ROOT = safe.TASK / "results/context-20261124"
GPU = "5"
UUID = "GPU-82009e07-07c1-3a6c-5613-df2e45f63980"
PORT = 30556
URL = f"http://127.0.0.1:{PORT}"
CONTEXTS = (1024, 2048, 4096, 8192)
FROZEN = safe.FROZEN_REPO / safe.SUBDIR


def request(path, payload=None, timeout=60):
    req = urllib.request.Request(URL + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fixtures():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(safe.MODEL, trust_remote_code=True)
    # Keep the model's exact native user/assistant and Thinking delimiters.
    marker = "LOOPVQ_CONTEXT_BODY"
    rendered = tok.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    assert rendered.count(marker) == 1
    prefix, suffix = rendered.split(marker)
    p = tok.encode(prefix, add_special_tokens=False)
    s = tok.encode(suffix, add_special_tokens=False)
    body = tok.encode("Read the following notes carefully. Each numbered observation describes a simple arithmetic example. " * 3000,
                      add_special_tokens=False)
    for length in CONTEXTS:
        count = length - 1 - len(p) - len(s)
        assert 0 < count <= len(body)
        ids = p + body[:count] + s
        safe.new_json(ROOT / f"input-{length}.json", {
            "input_ids": [ids for _ in range(16)], "context_start": length,
            "construction": "Native chat delimiters; body truncated in token space, supplied as input_ids",
            "chat_template": tok.chat_template, "rendered_template_with_marker": rendered,
            "prefix_ids": p, "suffix_ids": s, "enable_thinking": True,
            "model_path": str(safe.MODEL), "token_ids_sha256": bench.digest(ids)})


def gpu_pids():
    assert safe.capture(["nvidia-smi", "-i", GPU, "--query-gpu=uuid", "--format=csv,noheader"]) == UUID
    raw = safe.capture(["nvidia-smi", "-i", UUID, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    return {int(x) for x in raw.splitlines()}


def start(method, profile, *, launch_script=None, server_protocol="paper-long"):
    assert not gpu_pids(), "GPU5 must be empty"
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", PORT)) != 0
    label = f"context-{method}-{'nodes' if profile else 'timing'}-{time.time_ns()}"
    script = launch_script or FROZEN / "a10_throughput_server.sh"
    command = ["bash", str(script), method, server_protocol, GPU, str(PORT), label]
    trace = ROOT / f"{method}-nodes"
    if profile:
        assert not list(ROOT.glob(f"{method}-nodes*")), "Preserve previous traces"
        nsys_tmp = Path(f"/tmp/loopvq-nsys-{os.getuid()}")
        nsys_tmp.mkdir(mode=0o700, exist_ok=True)
        assert nsys_tmp.stat().st_uid == os.getuid(), "Nsight temp directory is not owned by this account"
        command = ["env", f"NSYS_TMPDIR={nsys_tmp}", f"TMPDIR={nsys_tmp}",
                   "nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                   "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                   "--capture-range-end=repeat:8:async", "--wait=primary", "--kill=none",
                   f"--output={trace}", *command]
    launcher = ROOT / f"{label}.launcher.log"
    shell = "exec " + shlex.join(command) + " > " + shlex.quote(str(launcher)) + " 2>&1"
    pane = safe.capture(["tmux", "new-window", "-d", "-t", "loopvq-20261123", "-n", label,
                         "-P", "-F", "#{pane_pid}", shell])
    deadline = time.monotonic() + 1800
    identity = None
    while time.monotonic() < deadline:
        assert Path(f"/proc/{pane}").exists(), f"Launcher exited before readiness; inspect {launcher}"
        assert not (safe.TASK / "logs" / f"{label}.exit").exists(), "Server exited during launch"
        owned = safe.process_tree(int(pane))
        assert gpu_pids() <= owned, "Unknown GPU5 task"
        candidates = []
        for pid in owned:
            try:
                snap = safe.proc_snapshot(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            a = snap["argv"]
            if any(a[i:i+2] == ["-m", "sglang.launch_server"] for i in range(len(a)-1)):
                safe.validate_parent(snap, safe.Paths().target_python, port=PORT)
                candidates.append(snap)
        assert len(candidates) <= 1
        if candidates:
            identity = candidates[0]
            try:
                request("/health", timeout=3)
                break
            except (OSError, TimeoutError):
                pass
        time.sleep(10)
    else:
        raise TimeoutError("Server readiness timeout")
    cfg = bench.get_server_config(URL, 30)
    base = safe.read_json(safe.Paths().bf_manifest)
    manifest = {k: copy.deepcopy(base[k]) for k in bench.IDENTITY_KEYS}
    manifest.update(method=method, protocol="context", prefix_tokens=8, recent_tokens=64,
                    server_pid=identity["pid"], gpu_uuid=UUID, gpu_index=5, exclusive_server=True,
                    config=cfg, process_identity=identity, inference_sha=safe.FROZEN_SHA)
    manifest["config"]["total_ut_steps"] = 4
    if method == "loopvq":
        bundle = safe.Paths().bundle
        manifest["bundle_sha256"] = {name: safe.sha256(bundle / name) for name in ("codebook.pt", "vq_v_codebook.pt")}
    safe.new_json(ROOT / f"{label}.manifest.json", manifest)
    log = safe.TASK / "logs" / f"{label}.log"
    return identity, manifest, log, launcher


def stop(identity):
    safe.same_process(identity, safe.proc_snapshot(identity["pid"]))
    safe.validate_parent(identity, safe.Paths().target_python, port=PORT)
    owned = safe.process_tree(identity["pid"])
    assert gpu_pids() and gpu_pids() <= owned and safe.listener_owned(owned, PORT)
    # Every submitted SSE request finished and the caller has no clients left.
    # Require token/queue gauges to be zero; running gauge can lag a finished batch.
    metrics = request("/metrics").decode()
    checked = []
    for line in metrics.splitlines():
        if not line.startswith("#") and any(line.startswith("sglang:" + k) for k in ("num_queue_reqs", "num_used_tokens")):
            assert float(line.rsplit(" ", 1)[1]) == 0, line
            checked.append(line)
    assert len(checked) >= 2, "Missing idle gauges"
    safe.new_json(ROOT / f"stop-{identity['pid']}.json", {"identity": identity, "gpu_pids": sorted(gpu_pids()), "idle": checked})
    fd = os.pidfd_open(identity["pid"])
    try:
        safe.same_process(identity, safe.proc_snapshot(identity["pid"]))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    deadline = time.monotonic() + 180
    while gpu_pids() and time.monotonic() < deadline:
        time.sleep(3)
    assert not gpu_pids(), "Service did not release GPU; no escalation attempted"


def cell(method, length, bs, manifest, log, profile):
    protocol = f"context-{length}"
    output = 9 if profile else 33
    bench.PROTOCOLS[protocol] = (length - 1, output, 8, 64)
    out = ROOT / ("profile" if profile else "timing") / method / str(length) / f"bs{bs}"
    assert not out.exists()
    out.mkdir(parents=True)
    m = copy.deepcopy(manifest)
    m["protocol"] = protocol
    mf = out / "manifest.input"
    safe.new_json(mf, m)
    args = argparse.Namespace(base_url=URL, method=method, protocol=protocol, bs=bs,
                              input_len=length-1, output_len=output,
                              input_ids_json=str(ROOT / f"input-{length}.json"),
                              server_manifest=str(mf), server_log=str(log), out_dir=str(out), timeout=1800)
    capacity = manifest["config"]["max_total_tokens"]
    required = bs * ((length - 1 + output + 7) // 8 * 8)
    if required > capacity:
        safe.new_json(out / "summary.json", {"status": "capacity_limited_not_run", "method": method,
                      "context": length, "bs": bs, "required_tokens": required, "capacity": capacity,
                      "requests_sent": 0, "speedup": None})
        return
    if not profile:
        assert bench.measure(args) == 0, f"Failed cell: {out}"
        return
    cfg = bench.get_server_config(URL, 30)
    bench.validate_manifest(m, cfg, method, protocol, bs, 8, 64)
    capture = bench.capture_from_log(log)
    inputs = safe.read_json(args.input_ids_json)["input_ids"][:bs]
    for number in (1, 2):
        assert bench.run_batch(args, inputs, "warmup", number, cfg, m, capture)["status"] == "ok"
    request("/start_profile", {"activities": ["CUDA_PROFILER"]})
    result = bench.run_batch(args, inputs, "profile", 1, cfg, m, capture)
    stopped = request("/stop_profile", {}, timeout=120)
    safe.new_json(out / "summary.json", {"status": result["status"], "context": length,
                  "bs": bs, "method": method, "decode_steps": 8,
                  "stop_profile_response": stopped.decode(errors="replace")})
    assert result["status"] == "ok"


def run(profile):
    assert os.environ.get("TMUX") and hasattr(os, "pidfd_open")
    assert safe.capture(["git", "-C", str(safe.FROZEN_REPO), "rev-parse", "HEAD"]) == safe.FROZEN_SHA
    safe.new_json(ROOT / ("profile-run.json" if profile else "timing-run.json"), {
        "started": time.time(), "helper_sha256": safe.sha256(__file__),
        "gpu": UUID, "profile": profile, "inference_sha": safe.FROZEN_SHA})
    for method in ("bf16", "loopvq"):
        identity, manifest, log, launcher = start(method, profile)
        for length in CONTEXTS:
            for bs in ((1, 4) if profile else (1, 2, 4, 8, 16)):
                print(json.dumps({"method": method, "context": length, "bs": bs, "profile": profile}), flush=True)
                cell(method, length, bs, manifest, log, profile)
        stop(identity)
        if profile:
            # Export only after the asynchronous profiler has finished writing.
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                reports = sorted(ROOT.glob(f"{method}-nodes.*.nsys-rep"))
                if len(reports) == 8 and all(p.stat().st_size > 0 for p in reports):
                    break
                time.sleep(10)
            assert len(reports) == 8, f"Incomplete Nsight export; inspect {launcher}"
            for index, (length, bs) in enumerate((l, b) for l in CONTEXTS for b in (1, 4)):
                report = ROOT / f"{method}-nodes.{index+1}.nsys-rep"
                target = ROOT / f"{method}-{length}-bs{bs}.sqlite"
                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={target}", str(report)], check=True)
    safe.new_json(ROOT / ("profile-completed.json" if profile else "timing-completed.json"), {"ended": time.time()})


def recover_long():
    p = safe.Paths()
    previous = safe.read_json(p.state)
    assert previous["status"] == "failed" and "filename-to-SHA256 map" in previous["error"]
    p.directory = p.matrix / "pipeline-recovery-20261124"
    p.state = p.directory / "status.json"
    reference = safe.read_json(p.reference)
    reference["prior_bundle_digest"] = reference["bundle_sha256"]
    reference["bundle_sha256"] = {name: safe.sha256(p.bundle / name) for name in ("codebook.pt", "vq_v_codebook.pt")}
    assert reference["bundle_sha256"] == {
        "codebook.pt": "01082911f0f352058aaffa8589f2a641e8bd6fd257ace7082beb7236653277f0",
        "vq_v_codebook.pt": "49641d553b891cc87ac571134111a3629891d8f8bdb7f54e0fb4a51d0cfdcfd4"}
    p.reference = p.matrix / "reference-filehash-manifest-20261124.json"
    safe.new_json(p.reference, reference)
    safe.run_locked(p, 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fixtures", "timing", "profile", "recover-long"))
    mode = parser.parse_args().mode
    ROOT.mkdir(parents=True, exist_ok=True)
    if mode == "fixtures":
        fixtures()
    elif mode == "recover-long":
        recover_long()
    else:
        run(mode == "profile")
