#!/usr/bin/env python3
"""Supervise the frozen A10 paper-long experiment; --inspect never writes.

Run from the validation checkout, inside the existing task's tmux session.
Only --run may stop/start a verified server. Interrupted or failed pipelines
require manual review: they never overwrite measurements or silently resume.
Inference, measure and aggregate always use the frozen performance checkout.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

FROZEN_SHA = "feba71a8927502fd334a94a86a6d312ff3f4e543"
FROZEN_REPO = Path("/homes/ruoxi07/repo/sglang-loopspec-tp2")
TASK = Path("/homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction")
MODEL = Path("/homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking")
BUILD_PYTHON = Path("/homes/ruoxi07/anaconda3/envs/loopvq-build/bin/python")
GPU = "6"
GPU_UUID = "GPU-cb6f61c7-9ca8-fe07-b00e-6f4b177a93e6"
PORT = 30555
URL = f"http://127.0.0.1:{PORT}"
SUBDIR = Path("research/loopspec/experiments/2026-11-23-loopvq-reproduction")
GAUGES = ("num_running_reqs", "num_queue_reqs", "num_used_tokens", "token_usage")
IDENTITY = ("pid", "ppid", "start_ticks", "uid", "exe", "argv", "cwd")


class Refuse(RuntimeError):
    """Stop without guessing ownership, overwriting data, or changing protocol."""


def require(condition, message):
    if not condition:
        raise Refuse(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def new_json(path, value):
    require(not Path(path).exists(), f"Refusing to overwrite {path}")
    atomic_json(path, value)


def capture(argv, *, env=None):
    return subprocess.check_output(argv, text=True, env=env, timeout=60).strip()


def parse_stat(text):
    # comm may contain spaces and ')'; fields after its final ')' start at 3.
    tail = text.rsplit(")", 1)[1].split()
    return {"state": tail[0], "ppid": int(tail[1]), "start_ticks": int(tail[19])}


def proc_snapshot(pid):
    directory = Path("/proc") / str(pid)
    before = parse_stat((directory / "stat").read_text())
    result = {
        "pid": int(pid), **before, "uid": directory.stat().st_uid,
        "argv": [a.decode(errors="replace") for a in (directory / "cmdline").read_bytes().split(b"\0") if a],
        "exe": os.readlink(directory / "exe"), "cwd": os.readlink(directory / "cwd"),
    }
    after = parse_stat((directory / "stat").read_text())
    require(all(before[k] == after[k] for k in ("ppid", "start_ticks")), "Process changed during inspection")
    return result


def option(argv, name):
    values = []
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(name + "="):
            values.append(token[len(name) + 1:])
    require(len(values) == 1, f"Expected one exact {name} in server argv")
    return values[0]


def validate_parent(snapshot, python, model=MODEL, repo=FROZEN_REPO, port=PORT):
    argv = snapshot["argv"]
    require(snapshot["uid"] == os.getuid(), "Server is not owned by this account")
    require(argv and Path(argv[0]).resolve() == Path(python).resolve(), "Wrong server Python argv")
    require(Path(snapshot["exe"]).resolve() == Path(python).resolve(), "Wrong server executable")
    require(Path(snapshot["cwd"]).resolve() == Path(repo).resolve(), "Wrong server working directory")
    require(any(argv[i:i + 2] == ["-m", "sglang.launch_server"] for i in range(len(argv) - 1)), "Not an SGLang launch_server parent")
    require(Path(option(argv, "--model-path")).resolve() == Path(model).resolve(), "Wrong server model")
    require(option(argv, "--port") == str(port), "Wrong server port")


def same_process(expected, actual):
    require(all(expected[k] == actual[k] for k in IDENTITY), "Server identity changed (including PID reuse)")


def process_tree(root_pid):
    parents = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid == os.getuid():
                parents[int(directory.name)] = parse_stat((directory / "stat").read_text())["ppid"]
        except (FileNotFoundError, ProcessLookupError):
            continue
    owned = {root_pid}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def gpu_pids():
    actual_uuid = capture(["nvidia-smi", "-i", GPU, "--query-gpu=uuid", "--format=csv,noheader"])
    require(actual_uuid == GPU_UUID, "Physical GPU index/UUID changed")
    raw = capture(["nvidia-smi", "-i", GPU_UUID, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    require(all(line.strip().isdigit() for line in raw.splitlines()), "GPU PID census is unavailable")
    return {int(line) for line in raw.splitlines()}


def check_ownership(parent, owned, gpu, schedulers):
    require(parent in owned, "Missing service parent")
    require(gpu and gpu <= owned, f"Unknown or missing GPU processes: {sorted(gpu - owned)}")
    require(schedulers and schedulers <= owned and bool(schedulers & gpu), "GPU scheduler is not a descendant of the service parent")


def listener_owned(owned, port):
    sockets = set()
    for name in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                sockets.add(f"socket:[{fields[9]}]")
    if not sockets:
        return False
    matched = set()
    for pid in owned:
        try:
            for fd in (Path("/proc") / str(pid) / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if target in sockets:
                        matched.add(target)
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue
    return sockets <= matched


def service_guard(identity, python, expected_gpu=None):
    same_process(identity, proc_snapshot(identity["pid"]))
    validate_parent(identity, python)
    owned = process_tree(identity["pid"])
    gpu = gpu_pids()
    require(gpu and gpu <= owned, f"Unknown or missing GPU processes: {sorted(gpu - owned)}")
    snapshots = {pid: proc_snapshot(pid) for pid in gpu}
    schedulers = {pid for pid, row in snapshots.items()
                  if row["argv"] and row["argv"][0].startswith("sglang::scheduler")}
    check_ownership(identity["pid"], owned, gpu, schedulers)
    require(listener_owned(owned, PORT), "Service family does not own the listening port")
    if expected_gpu is not None:
        require(set(snapshots) == set(expected_gpu), "GPU worker set changed")
        for pid, before in expected_gpu.items():
            same_process(before, snapshots[pid])
    return snapshots


def idle_values(text):
    values = {key: [] for key in GAUGES}
    for line in text.splitlines():
        match = re.match(r"^sglang:(\w+)(?:\{[^}]*\})?\s+(\S+)", line)
        if match and match[1] in values:
            values[match[1]].append(float(match[2]))
    require(all(rows and all(math.isfinite(x) for x in rows) for rows in values.values()), "Missing/nonfinite idle metrics")
    return values


def verified_sigterm(identity, guard, reader=proc_snapshot):
    # A pidfd continues referring to the inspected process even if its PID is reused.
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Linux pidfd signaling is required")
    fd = os.pidfd_open(identity["pid"])
    try:
        same_process(identity, reader(identity["pid"]))
        guard()
        same_process(identity, reader(identity["pid"]))
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def exit_ready(path):
    if not Path(path).exists():
        return False
    require(Path(path).read_text().strip() == "0", f"Nonzero or malformed completion marker: {path}")
    return True


def load_bench(repo):
    path = Path(repo) / SUBDIR / "benchmark_throughput.py"
    spec = importlib.util.spec_from_file_location("frozen_paperlong_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_complete(summary, bs, method, inputs, bench):
    require(summary.get("status") == "ok", "Measurement is not complete and successful")
    for key, expected in {"method": method, "protocol": "paper-long", "bs": bs,
                          "input_len": 2048, "output_len": 32768, "warmup_count": 2, "repeat_count": 3,
                          "input_ids_sha256": bench.digest(inputs[:bs]), "input_fixture_sha256": bench.digest(inputs),
                          "primary_metric": "e2e_output_tokens_s",
                          "fixed_length_sampling": {"temperature": 0.0, "ignore_eos": True}}.items():
        require(summary.get(key) == expected, f"Measurement differs for {key}")
    attempts = summary.get("attempts", [])
    require([(a.get("phase"), a.get("number")) for a in attempts] ==
            [("warmup", 1), ("warmup", 2), ("measure", 1), ("measure", 2), ("measure", 3)], "Incomplete repeat schedule")
    measured = []
    for attempt in attempts:
        row = read_json(attempt["path"])
        require(attempt.get("status") == row.get("status") == "ok", "Failed or incomplete raw attempt")
        require(row.get("output_tokens") == bs * 32768 and row.get("input_tokens") == bs * 2048, "Incorrect completed token totals")
        require(row.get("batch_evidence", {}).get("verified") is True, "Actual batch was not verified")
        requests = row.get("requests", [])
        require(len(requests) == bs and all(r.get("completion_tokens") == 32768 and
                r.get("prompt_tokens") == 2048 and r.get("finish_reason", {}).get("type") == "length"
                for r in requests), "A request did not finish the full fixed length")
        wall, speed = row.get("e2e_seconds", 0), row.get("e2e_output_tokens_s", 0)
        require(math.isfinite(wall) and wall > 0 and math.isfinite(speed) and speed > 0 and
                math.isclose(speed, bs * 32768 / wall, rel_tol=1e-9), "Invalid E2E throughput")
        if attempt["phase"] == "measure":
            measured.append(speed)
    require(math.isclose(summary.get("e2e_output_tokens_s_median", 0), statistics.median(measured), rel_tol=1e-9), "Summary median differs from raw measurements")


def capacity_summary(baseline, bs, inputs, evidence, bench):
    capacity = baseline["server_config_readback"].get("max_total_tokens")
    require(type(capacity) is int and capacity > 0 and bs * 34816 > capacity,
            "No hard pool-capacity proof for this skipped batch")
    keys = ("format_version", "protocol", "input_len", "output_len", "server_manifest", "server_config_readback",
            "base_url", "server_log", "client_host", "warmup_count", "repeat_count", "primary_metric", "fixed_length_sampling")
    result = {k: copy.deepcopy(baseline[k]) for k in keys}
    result.update(method="bf16", bs=bs, status="capacity_limited_not_run", requests_sent=0,
                  attempted=False, attempts=[], warmup_completed=0, repeats_completed=0,
                  input_ids_sha256=bench.digest(inputs[:bs]), input_fixture_sha256=bench.digest(inputs),
                  capacity_evidence=evidence,
                  capacity_proof={"required_tokens": bs * 34816, "configured_pool_upper_bound": capacity,
                                  "kv_bytes_per_token": 192 * 16 * 128 * 2 * 2,
                                  "note": "Hard simultaneous-batch capacity bound; no request sent and no CUDA OOM claimed"})
    return result


def completed_artifacts(state):
    require(state.get("status") == "completed" and state.get("gpu_released") is True, "Pipeline is not completed")
    hashes = state.get("artifact_sha256", {})
    require(bool(hashes), "Completed status has no artifact seals")
    for name, expected in hashes.items():
        require(Path(name).is_file() and sha256(name) == expected, f"Completed artifact changed: {name}")


class Paths:
    def __init__(self, task=TASK, repo=FROZEN_REPO):
        self.task, self.repo = Path(task), Path(repo)
        self.research = self.repo / SUBDIR
        self.matrix = self.task / "results/throughput-round7/paper-long"
        self.directory = self.matrix / "pipeline"
        self.state = self.directory / "status.json"
        self.bf_manifest = self.task / "results/throughput-r7-bf16-paperlong-server-manifest.json"
        self.bf_summary = self.matrix / "bf16/bs1/summary.json"
        self.bf_exit = self.task / "logs/throughput-r7-paper-long-bf16-bs1.exit"
        self.capacity = self.matrix / "bf16-capacity.json"
        self.inputs = self.task / "results/throughput-input2048.json"
        self.target_python = self.task / "envs/sglang-loopvq-test/bin/python"
        self.bundle = self.task / "artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e"
        self.reference = self.task / "results/throughput-r7-vq-common-b-server-manifest.json"


def inspect_plan(paths):
    if paths.state.exists():
        state = read_json(paths.state)
        if state.get("status") == "completed":
            completed_artifacts(state)
            return {"status": "completed", "meaning": "Recorded completion artifacts are unchanged; no experiment was run by inspect", "state": str(paths.state)}
        return {"status": "waiting", "recorded_status": state.get("status"), "phase": state.get("phase"),
                "action": "Existing incomplete pipeline requires manual review; --run will refuse it"}
    if not paths.bf_exit.exists():
        return {"status": "waiting", "reason": "BF16 completion marker is absent", "marker": str(paths.bf_exit)}
    code = paths.bf_exit.read_text().strip()
    return {"status": "planned", "bf16_exit_text": code,
            "action": "--run must still validate complete raw results, process ownership, idle state and GPU before any transition"}


class Pipeline:
    def __init__(self, paths, poll=20):
        self.p, self.poll = paths, poll
        self.state = {}
        self.vq_identity = self.vq_gpu = None
        self.artifacts = set()
        self.child_env = dict(os.environ, PYTHONPATH=str(paths.repo / "python"),
                              WANDB_BASE_URL="https://api.wandb.ai")

    def update(self, phase, **values):
        self.state.update(phase=phase, updated_unix=time.time(), **values)
        atomic_json(self.p.state, self.state)
        print(json.dumps({"phase": phase, "status": self.state.get("status"), "updated_unix": self.state["updated_unix"]}), flush=True)

    def save(self, name, value):
        path = self.p.directory / name
        new_json(path, value)
        self.artifacts.add(path)
        return path

    def frozen(self):
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        require(capture(["git", "-C", str(self.p.repo), "rev-parse", "HEAD"], env=env) == FROZEN_SHA, "Performance source SHA changed")
        require(not capture(["git", "-C", str(self.p.repo), "status", "--porcelain"], env=env), "Performance checkout is not clean")

    def environment(self):
        # Do not log pip output: installed direct URLs can contain private data.
        command = ["bash", "-c", 'set -e; source "$1"; shift; exec "$@"', "paperlong-env",
                   str(self.p.research / "a10_target_env.sh"), str(self.p.target_python)]
        raw = subprocess.check_output([*command, "-m", "pip", "freeze", "--all"], env=self.child_env, timeout=120)
        require(hashlib.sha256(raw).hexdigest() == self.bf_manifest["environment_sha256"], "Target environment differs from BF16 manifest")
        probe = capture([*command, "-c", "import json,sys,sglang; print(json.dumps([sys.executable,sglang.__file__]))"], env=self.child_env)
        executable, module = json.loads(probe)
        require(Path(executable).resolve() == self.p.target_python.resolve(), "Unexpected target Python")
        require(Path(module).resolve() == self.p.repo / "python/sglang/__init__.py", "Target import is not the frozen checkout")
        return {"environment_sha256": self.bf_manifest["environment_sha256"], "algorithm": "SHA256(target-python -m pip freeze --all stdout bytes)", "python": executable, "sglang_file": module}

    def http(self, endpoint):
        with urllib.request.urlopen(URL + endpoint, timeout=30) as response:
            return response.read().decode()

    def info(self):
        raw = json.loads(self.http("/server_info"))
        require(Path(raw.get("model_path", "")).resolve() == MODEL and raw.get("port") == PORT, "HTTP model/port differs from owned service")
        return {key: raw.get(key) for key in (*self.bench.CONFIG_KEYS, "model_path", "port", "max_total_num_tokens", "version", "startup_time")}

    def wait_idle(self, identity, gpu, timeout=180):
        deadline, streak, observations = time.monotonic() + timeout, 0, []
        while True:
            service_guard(identity, self.p.target_python, gpu)
            values = idle_values(self.http("/metrics"))
            observations.append({"unix": time.time(), "values": values})
            atomic_json(self.p.directory / "latest-idle.json", observations)
            streak = streak + 1 if all(x == 0 for rows in values.values() for x in rows) else 0
            if streak == 2:
                return observations
            require(time.monotonic() < deadline, "Owned server did not become idle; it was not stopped")
            time.sleep(self.poll)

    def stop_server(self, identity, gpu, name, launcher_exit=None):
        idle = self.wait_idle(identity, gpu)
        self.save(f"{name}-idle-before-stop.json", idle)
        def final_guard():
            service_guard(identity, self.p.target_python, gpu)
            values = idle_values(self.http("/metrics"))
            require(all(x == 0 for rows in values.values() for x in rows), "Server became busy before termination")
        verified_sigterm(identity, final_guard)
        deadline = time.monotonic() + 180
        while True:
            remaining = gpu_pids()
            require(remaining <= set(gpu), "Unknown GPU process appeared during server shutdown")
            for pid in remaining:
                try:
                    current = proc_snapshot(pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                # A child can be reparented during termination; identity must not be reused.
                require(all(current[k] == gpu[pid][k] for k in ("pid", "start_ticks", "uid", "exe", "argv")), "GPU PID reused during shutdown")
            import socket
            with socket.socket() as sock:
                port_closed = sock.connect_ex(("127.0.0.1", PORT)) != 0
            try:
                parent_stat = parse_stat((Path("/proc") / str(identity["pid"]) / "stat").read_text())
                require(parent_stat["start_ticks"] == identity["start_ticks"], "Parent PID reused during shutdown")
                parent_gone = parent_stat["state"] == "Z"
            except FileNotFoundError:
                parent_gone = True
            launcher_done = launcher_exit is None or launcher_exit.exists()
            if not remaining and port_closed and parent_gone and launcher_done:
                self.save(f"{name}-stopped.json", {"unix": time.time(), "gpu_compute_pids": [], "port_closed": True})
                return
            require(time.monotonic() < deadline, "Shutdown timed out; no escalation or unknown-process termination performed")
            time.sleep(self.poll)

    def verify_bf(self):
        self.frozen()
        self.bench = load_bench(self.p.repo)
        self.bf_manifest = read_json(self.p.bf_manifest)
        require(self.bf_manifest["sglang_sha"] == FROZEN_SHA and self.bf_manifest["gpu_uuid"] == GPU_UUID, "Wrong BF16 source/GPU")
        require(Path(self.bf_manifest["model_path"]).resolve() == MODEL, "Wrong BF16 model path")
        require(self.bf_manifest["host"].split(".")[0] == platform.node().split(".")[0] == "a10", "This pipeline is specific to A10")
        self.environment()
        self.inputs = read_json(self.p.inputs)
        self.inputs = self.inputs["input_ids"] if isinstance(self.inputs, dict) else self.inputs
        require(len(self.inputs) >= 16 and all(len(row) == 2048 and all(type(t) is int and t >= 0 for t in row) for row in self.inputs), "Invalid frozen input fixture")
        self.input_hash = sha256(self.p.inputs)
        identity = proc_snapshot(int(self.bf_manifest["server_pid"]))
        validate_parent(identity, self.p.target_python)
        if self.bf_manifest.get("server_start_ticks") is not None:
            require(identity["start_ticks"] == self.bf_manifest["server_start_ticks"], "BF16 PID differs from its recorded launch identity")
        gpu = service_guard(identity, self.p.target_python)
        info = self.info()
        self.bench.validate_manifest(self.bf_manifest, info, "bf16", "paper-long", 16, 8, 64)
        self.save("bf16-identity.json", {"parent": identity, "gpu_processes": gpu, "server_info": info,
                                       "captured_unix": time.time(), "launch_start_ticks_previously_recorded": self.bf_manifest.get("server_start_ticks") is not None})
        self.update("waiting_for_bf16", bf16_parent=identity, status="running")
        while not exit_ready(self.p.bf_exit):
            service_guard(identity, self.p.target_python, gpu)
            self.frozen()
            time.sleep(self.poll)
        baseline = read_json(self.p.bf_summary)
        validate_complete(baseline, 1, "bf16", self.inputs, self.bench)
        current = self.info()
        for key in self.bench.CONFIG_KEYS:
            require(baseline["server_config_readback"].get(key) == current.get(key), f"BF16 runtime setting changed: {key}")
        for key in self.bench.IDENTITY_KEYS:
            require(baseline["server_manifest"][key] == self.bf_manifest[key], f"BF16 summary identity differs: {key}")
        require(baseline["server_manifest"]["server_pid"] == identity["pid"], "BF16 measurement used another process")
        self.save("bf16-completion.json", {"summary_path": str(self.p.bf_summary), "summary_sha256": sha256(self.p.bf_summary), "exit_path": str(self.p.bf_exit), "exit_text": "0"})
        self.artifacts.update((self.p.bf_summary, self.p.bf_exit, self.p.bf_manifest, self.p.inputs))
        for row in baseline["attempts"]:
            self.artifacts.add(Path(row["path"]))
        self.frozen()
        self.save("environment-before-transition.json", self.environment())
        self.baseline = baseline
        return identity, gpu

    def capacity_rows(self):
        evidence = read_json(self.p.capacity)
        require(bool(evidence), "Missing BF16 capacity evidence")
        self.save("bf16-capacity-source.json", evidence)
        self.artifacts.add(self.p.capacity)
        paths = [self.p.bf_summary]
        for bs in (2, 4, 8, 16):
            directory = self.p.matrix / "bf16" / f"bs{bs}"
            require(not directory.exists(), f"Existing capacity result needs review: {directory}")
            row = capacity_summary(self.baseline, bs, self.inputs,
                                   {"path": str(self.p.capacity), "sha256": sha256(self.p.capacity)}, self.bench)
            new_json(directory / "input_ids.json", {"input_ids": self.inputs[:bs]})
            new_json(directory / "summary.json", row)
            paths.append(directory / "summary.json")
            self.artifacts.update((directory / "summary.json", directory / "input_ids.json"))
        return paths

    def validate_bundle(self):
        reference = read_json(self.p.reference)
        require(reference.get("method") == "loopvq" and reference.get("sglang_sha") == FROZEN_SHA, "VQ reference must use the frozen performance source")
        require(reference.get("environment_sha256") == self.bf_manifest["environment_sha256"], "Reference environment differs")
        require(reference.get("model_sha256") == self.bf_manifest["model_sha256"], "Reference model differs")
        require(Path(reference["config"]["loopvq_codebook_dir"]).resolve() == self.p.bundle, "Unexpected trained bundle")
        hashes = reference.get("bundle_sha256")
        require(isinstance(hashes, dict) and {"codebook.pt", "vq_v_codebook.pt"} <= hashes.keys(), "Reference bundle hashes must be a filename-to-SHA256 map")
        for name, expected in hashes.items():
            require(Path(name).name == name and isinstance(expected, str), "Unsafe bundle hash entry")
            require(sha256(self.p.bundle / name) == expected, f"Bundle changed: {name}")
        require(sha256(MODEL / "model.safetensors") == self.bf_manifest["model_sha256"], "Model weights changed")
        require(read_json(MODEL / "config.json")["total_ut_steps"] == 4, "Model is not full R4")
        self.save("vq-reference-manifest.json", reference)
        return hashes

    def start_vq(self, hashes):
        self.frozen()
        self.environment()
        require(not gpu_pids(), "GPU is no longer idle before launch")
        label = f"paperlong-vq-pipeline-{time.time_ns()}"
        launcher_log = self.p.task / "logs" / f"{label}.launcher.log"
        server_log = self.p.task / "logs" / f"{label}.log"
        launch_evidence = self.p.task / "logs" / f"{label}.launch.txt"
        launch_exit = self.p.task / "logs" / f"{label}.exit"
        self.vq_launch_exit = launch_exit
        session = capture(["tmux", "display-message", "-p", "-t", os.environ["TMUX_PANE"], "#{session_id}"])
        argv = ["bash", str(self.p.research / "a10_throughput_server.sh"), "loopvq", "paper-long", GPU, str(PORT), label]
        shell = "exec " + shlex.join(argv) + " > " + shlex.quote(str(launcher_log)) + " 2>&1"
        self.update("starting_vq", label=label, server_log=str(server_log), launcher_log=str(launcher_log))
        window = capture(["tmux", "new-window", "-d", "-t", session, "-n", label, "-P", "-F", "#{window_id}\t#{pane_id}\t#{pane_pid}", shell])
        window_id, pane_id, pane_pid = window.split("\t")
        self.update("starting_vq", window_id=window_id, pane_id=pane_id, pane_pid=int(pane_pid))
        deadline = time.monotonic() + 1800
        while True:
            require(not launch_exit.exists(), "VQ launcher exited before readiness; inspect preserved launcher/server logs")
            owned = process_tree(int(pane_pid))
            require(gpu_pids() <= owned, "Unknown GPU process appeared while starting VQ")
            candidates = []
            for pid in owned:
                try:
                    candidate = proc_snapshot(pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                args = candidate["argv"]
                if any(args[i:i + 2] == ["-m", "sglang.launch_server"] for i in range(len(args) - 1)):
                    validate_parent(candidate, self.p.target_python)
                    candidates.append(candidate)
            require(len(candidates) <= 1, "Multiple server parents beneath new tmux window")
            if candidates:
                if self.vq_identity is None:
                    self.vq_identity = candidates[0]
                    self.update("starting_vq", vq_parent=self.vq_identity)
                else:
                    same_process(self.vq_identity, candidates[0])
                try:
                    self.http("/health")
                except (OSError, TimeoutError):
                    pass
                else:
                    break
            require(time.monotonic() < deadline, "VQ readiness timeout")
            time.sleep(self.poll)
        info = self.info()
        self.vq_gpu = service_guard(self.vq_identity, self.p.target_python)
        text = launch_evidence.read_text()
        require(FROZEN_SHA in text.splitlines() and str(self.p.repo / "python/sglang/__init__.py") in text.splitlines(), "Launcher source/import evidence differs")
        self.save("environment-after-launch.json", self.environment())
        manifest = {key: copy.deepcopy(self.bf_manifest[key]) for key in self.bench.IDENTITY_KEYS}
        manifest.update(method="loopvq", exclusive_server=True, total_ut_steps=4,
                        server_pid=self.vq_identity["pid"], gpu_index=int(GPU), gpu_uuid=GPU_UUID,
                        server_command=shlex.join(self.vq_identity["argv"]), server_start_ticks=self.vq_identity["start_ticks"],
                        server_log=str(server_log), sglang_git_sha=FROZEN_SHA, sglang_source_dir=str(self.p.repo),
                        bundle_sha256=hashes, config={k: info.get(k) for k in self.bench.CONFIG_KEYS})
        manifest["config"].update(total_ut_steps=4, model_path=str(MODEL))
        self.bench.validate_manifest(manifest, info, "loopvq", "paper-long", 16, 8, 64)
        require(info["max_total_tokens"] == 557568 and info["context_length"] == 40960, "Wrong VQ long-protocol capacity")
        for key in self.bench.PAIR_CONFIG_KEYS:
            require(self.baseline["server_config_readback"].get(key) == info.get(key), f"Pairing config differs: {key}")
        manifest_path = self.save("vq-server-manifest.json", manifest)
        self.save("vq-server-identity.json", {"parent": self.vq_identity, "gpu_processes": self.vq_gpu, "server_info": info})
        self.artifacts.update((launch_evidence, launcher_log))
        return manifest_path, server_log

    def child(self, argv, name, guard):
        log = self.p.directory / f"{name}.log"
        new_json(self.p.directory / f"{name}.command.json", argv)
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=self.child_env)
            fd = os.pidfd_open(process.pid)
            try:
                self.update(self.state["phase"], active_client_pid=process.pid)
                while process.poll() is None:
                    guard()
                    time.sleep(self.poll)
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                new_json(self.p.directory / f"{name}.interrupted.json", {"pid": process.pid, "exit_code": process.poll(),
                                                                        "still_running": process.poll() is None})
                raise
            finally:
                os.close(fd)
        new_json(self.p.directory / f"{name}.exit", code)
        self.artifacts.update((log, self.p.directory / f"{name}.exit", self.p.directory / f"{name}.command.json"))
        require(code == 0, f"{name} failed with exit {code}; raw outputs preserved")

    def measurement_guard(self):
        self.frozen()
        require(sha256(self.p.inputs) == self.input_hash, "Frozen input IDs changed")
        service_guard(self.vq_identity, self.p.target_python, self.vq_gpu)

    def execute(self):
        require(os.environ.get("TMUX") and os.environ.get("TMUX_PANE"), "Execution requires a tmux pane")
        require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Execution requires Linux pidfd signaling")
        require(not Path(__file__).resolve().is_relative_to(self.p.repo.resolve()), "Run supervisor from the validation checkout, never the frozen performance checkout")
        self.update("preflight", status="running", supervisor_file=str(Path(__file__).resolve()),
                    supervisor_sha256=sha256(__file__), supervisor_pid=os.getpid(),
                    supervisor_git_sha=capture(["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"]),
                    inference_sha=FROZEN_SHA, started_unix=time.time())
        try:
            bf_identity, bf_gpu = self.verify_bf()
            hashes = self.validate_bundle()
            results = self.capacity_rows()
            self.update("stopping_bf16")
            self.stop_server(bf_identity, bf_gpu, "bf16")
            manifest, log = self.start_vq(hashes)
            for bs in (1, 2, 4, 8, 16):
                self.measurement_guard()
                self.environment()
                self.wait_idle(self.vq_identity, self.vq_gpu)
                out = self.p.matrix / "loopvq" / f"bs{bs}"
                require(not out.exists(), f"Existing VQ cell requires manual review: {out}")
                self.update(f"measuring_vq_bs{bs}")
                self.child([str(BUILD_PYTHON), str(self.p.research / "benchmark_throughput.py"), "measure",
                            "--base-url", URL, "--method", "loopvq", "--protocol", "paper-long", "--bs", str(bs),
                            "--input-ids-json", str(self.p.inputs), "--server-manifest", str(manifest),
                            "--server-log", str(log), "--out-dir", str(out)], f"vq-bs{bs}", self.measurement_guard)
                summary = read_json(out / "summary.json")
                validate_complete(summary, bs, "loopvq", self.inputs, self.bench)
                self.bench.paired_row(read_json(results[(1, 2, 4, 8, 16).index(bs)]), summary)
                self.artifacts.update(path for path in out.iterdir() if path.is_file())
                results.append(out / "summary.json")
            self.update("aggregating")
            paired = self.p.matrix / "paired"
            require(not paired.exists(), "Existing aggregate requires manual review")
            self.child([str(BUILD_PYTHON), str(self.p.research / "benchmark_throughput.py"), "aggregate",
                        "--results", *map(str, results), "--out-dir", str(paired), "--wandb"], "aggregate", self.measurement_guard)
            report = read_json(paired / "throughput.json")
            require(report["coverage"]["paper-long"]["missing_bs"] == [], "Aggregate matrix is incomplete")
            require(all(row["speedup"] is None and row["bf16_status"] == "capacity_limited_not_run"
                        for row in report["rows"] if row["bs"] > 1), "Capacity-limited rows acquired a false speedup")
            require(bool(read_json(paired / "wandb.json").get("url")), "No W&B completion evidence")
            self.artifacts.update(paired.iterdir())
            self.update("stopping_vq")
            self.stop_server(self.vq_identity, self.vq_gpu, "vq", self.vq_launch_exit)
            self.vq_identity = None
            self.update("completed", status="completed", gpu_released=True, ended_unix=time.time(),
                        artifact_sha256={str(path): sha256(path) for path in sorted(self.artifacts) if path.is_file()})
        except BaseException as exc:
            cleanup = "No owned VQ service was ready for cleanup"
            if self.vq_identity is not None and self.vq_gpu is not None:
                try:
                    self.stop_server(self.vq_identity, self.vq_gpu, "vq-failure-cleanup", self.vq_launch_exit)
                    cleanup = "Verified idle VQ service stopped and GPU released"
                except BaseException as cleanup_error:
                    cleanup = f"No further termination: {type(cleanup_error).__name__}: {cleanup_error}"
            self.update("failed", status="failed", error_type=type(exc).__name__, error=str(exc), cleanup=cleanup, ended_unix=time.time())
            raise


def run_locked(paths, poll):
    require(os.environ.get("TMUX") and os.environ.get("TMUX_PANE"), "Execution requires tmux")
    paths.directory.mkdir(parents=True, exist_ok=True)
    with (paths.directory / "pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refuse("Another pipeline holds the execution lock") from exc
        if paths.state.exists():
            state = read_json(paths.state)
            if state.get("status") == "completed":
                completed_artifacts(state)
                print(json.dumps({"status": "completed", "action": "Verified existing artifacts; nothing rerun"}))
                return
            refusal = paths.directory / f"refused-restart-{time.time_ns()}.json"
            new_json(refusal, {"status": "refused", "reason": "Incomplete prior pipeline requires manual review; no results overwritten", "prior_status": state.get("status")})
            raise Refuse("Incomplete prior pipeline; inspect its status and logs before manual recovery")
        existing = [p for p in paths.directory.iterdir() if p.name != "pipeline.lock"]
        require(not existing, "Pipeline directory has artifacts without a status; manual review required")
        Pipeline(paths, poll).execute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20, choices=range(10, 31), metavar="10..30")
    args = parser.parse_args()
    paths = Paths()
    try:
        if not args.run:
            print(json.dumps(inspect_plan(paths), indent=2))
        else:
            run_locked(paths, args.poll_seconds)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
