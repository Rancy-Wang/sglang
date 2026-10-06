"""Run the approved BF16 Top80 experiment sequence, with durable attempt state."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = "PLAN-CS-20260920-SWE-TOP80-BF16-PD-R1"
CASES = [(4, False), (6, True), (6, False), (8, True), (8, False),
         (10, True), (10, False), (12, True), (12, False), (14, True),
         (14, False), (16, True)]


def case_sequence(value):
    try:
        cases = []
        for item in value.split(","):
            concurrency, policy = item.split(":")
            concurrency = int(concurrency)
            if not 1 <= concurrency <= 32 or policy not in ("drop", "no_drop"):
                raise ValueError
            cases.append((concurrency, policy == "drop"))
        if len(set(cases)) != len(cases):
            raise ValueError
        return cases
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected distinct C:drop or C:no_drop entries, C in 1..32"
        ) from exc


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def command(args, head, root, concurrency, drop):
    pairs = {
        "output-dir": root, "server-repo": args.server_repo, "server-head": head,
        "source-launch": args.source_launch, "mini-root": args.mini_root,
        "requests-path": args.requests_path, "plan-id": PLAN, "seed": args.seed,
        "gpu-ids": "0,1,2,3,4,5,6,7", "port": args.port, "concurrency": concurrency,
        "transport": "nvlink", "mem-fraction-static": .85, "prefill-token-budget": 8192,
        "max-running-requests": 16 if concurrency <= 16 else 32,
        "rounds": 3, "model-context-limit": 131072,
        "warmup-wall-seconds": 60, "pd-timeout": 7200, "request-timeout": 21600,
        "workload-timeout": 172800, "http-keepalive-timeout": 60,
    }
    cmd = [sys.executable, str(HERE / "run_pd_matrix.py"), "--one",
           "--approved-overlap-matrix", "--unique-cohort", "--decode-radix"]
    for key, value in pairs.items():
        cmd += ["--" + key, str(value)]
    return cmd + (["--drop"] if drop else [])


def sample(root, port):
    row = dict(wall_time=time.time(), perf_time=time.perf_counter())
    row["host_loadavg"] = Path("/proc/loadavg").read_text().strip()
    row["host_cpu_counters"] = Path("/proc/stat").read_text().splitlines()[0]
    try:
        row["gpus"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except Exception as exc:
        row["gpu_error"] = repr(exc)
    for offset, mode in enumerate(("prefill", "decode")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port+offset}/metrics", timeout=2) as response:
                row[mode] = response.read().decode()
        except Exception as exc:
            row[mode + "_error"] = repr(exc)
    with (root / "telemetry.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=args.resume)
    head = subprocess.check_output(["git", "-C", args.server_repo, "rev-parse", "HEAD"], text=True).strip()
    selected = getattr(args, "cases", None)
    cases = selected or CASES
    stored_args = dict(vars(args))
    if selected is None:
        stored_args.pop("cases", None)
    elif "cases" in stored_args:
        stored_args["cases"] = [list(case) for case in selected]
    configuration = dict(args=stored_args, head=head, cases=cases, plan=PLAN)
    config_path = root / "configuration.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        # A repair can change HEAD, but never silently change the workload.
        previous = {k: v for k, v in old["args"].items() if k != "resume"}
        current = {k: v for k, v in stored_args.items() if k != "resume"}
        if previous != current:
            raise ValueError("Resume configuration differs from original workload")
    else:
        save(config_path, configuration)
    for number, (concurrency, drop) in enumerate(cases, 1):
        case = root / f"{number:02d}-c{concurrency}-{'drop' if drop else 'no_drop'}"
        case.mkdir(exist_ok=True)
        attempts = sorted(case.glob("attempt-*"))
        if any((attempt / "outcome.json").exists() and
               json.loads((attempt / "outcome.json").read_text()).get("valid") for attempt in attempts):
            continue
        attempt = case / f"attempt-{len(attempts)+1:02d}"
        cmd = command(args, head, attempt, concurrency, drop)
        save(case / f"command-{len(attempts)+1:02d}.json", cmd)
        state = dict(status="running", case=case.name, attempt=str(attempt), head=head,
                     seed=args.seed, start_wall=time.time(), pid=os.getpid())
        with (case / f"driver-{len(attempts)+1:02d}.log").open("x") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PD_MATRIX_FORWARD_TIMING="1"))
            state["driver_pid"] = proc.pid
            save(root / "state.json", state)
            print(json.dumps(state), flush=True)
            while proc.poll() is None:
                sample(case, args.port)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        state.update(exit_code=proc.returncode, end_wall=time.time(),
                     status="completed" if proc.returncode == 0 else "needs_investigation")
        state["allocated_gpu_hours"] = 8 * (state["end_wall"] - state["start_wall"]) / 3600
        save(case / f"allocation-{len(attempts)+1:02d}.json", state)
        save(root / "state.json", state)
        print(json.dumps(state), flush=True)
        if proc.returncode:
            raise RuntimeError("Attempt retained; investigate before explicit --resume")
    save(root / "state.json", dict(status="all_completed", time=time.time(), seed=args.seed))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "server-repo", "source-launch", "mini-root", "requests-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=41101)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cases", type=case_sequence,
                   help="Explicit sequence, e.g. 14:drop,16:drop,18:drop; use a new output directory")
    run(p.parse_args())
