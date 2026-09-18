"""Authorized OOM-only chunk backoff, then a fixed-budget PD comparison matrix."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_pd_matrix import HERE, NATIVE_HEAD, OVERLAP_PLAN, gpu_free, parser, write


def oom_evidence(case):
    for name in ("prefill.log", "decode.log", "client.log"):
        path = case / name
        if not path.exists():
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8 * 1024 * 1024))
            for line in stream.read().decode(errors="replace").splitlines():
                if "CUDA out of memory" in line or "torch.OutOfMemoryError:" in line:
                    return {"file": str(path), "line": line}
    return None


def stop_case(proc, case):
    # Only signal processes whose live argv still matches this case's manifest.
    targets = []
    launch = case / "launch.json"
    if launch.exists():
        for server in json.loads(launch.read_text())["servers"]:
            targets.append((server["pid"], server["argv"][:2], True))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if str(HERE / "test_serving.py") in argv and str(case / "workload") in argv:
            targets.append((int(entry.name), argv[:2], True))
    if proc.poll() is None:
        targets.append((proc.pid, [sys.executable, str(HERE / "run_pd_matrix.py")], False))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, prefix, group in targets:
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                if argv[:len(prefix)] == prefix:
                    (os.killpg if group else os.kill)(pid, sig)
            except (OSError, UnicodeError):
                pass
        if sig == signal.SIGTERM:
            time.sleep(10)
    proc.wait(timeout=30)


def await_gpu_release():
    deadline = time.monotonic() + 180
    while not gpu_free():
        if time.monotonic() > deadline:
            raise RuntimeError("GPU 0-3 not released; refusing to start another case")
        time.sleep(5)


def case_command(args, case, concurrency, drop, budget):
    return [sys.executable, str(HERE / "run_pd_matrix.py"), "--one", "--approved-overlap-matrix",
            "--output-dir", str(case), "--source-launch", args.source_launch,
            "--server-repo", args.drop_server_repo if drop else args.server_repo,
            "--server-head", args.drop_server_head if drop else args.server_head,
            "--mini-root", args.mini_root, "--requests-path", args.requests_path,
            "--concurrency", str(concurrency), "--rounds", "3", "--transport", "nvlink",
            "--decode-radix", "--port", str(args.port), "--prefill-token-budget", str(budget),
            "--max-running-requests", "32", "--pd-timeout", str(args.pd_timeout),
            "--warmup-timeout", str(args.warmup_timeout), "--request-timeout", str(args.request_timeout)
            ] + (["--drop"] if drop else [])


def run(args):
    if (args.server_head != NATIVE_HEAD or not args.drop_server_repo or not args.drop_server_head
            or args.rounds != 3 or args.profile_session or args.reserve_free_mib
            or not args.prefill_token_budget or not args.pd_timeout):
        raise ValueError("Explicit budget/timeouts, frozen servers, 3 rounds, no profiling/reservation required")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    state = dict(plan_id=OVERLAP_PLAN, amendment="OOM halving then fixed C32/1/2/4/8 pairs",
                 args=vars(args), state="running", cases=[], selected_budget=None)
    budget = args.prefill_token_budget
    write(root / "matrix.json", state)

    def run_case(concurrency, drop):
        await_gpu_release()
        case = root / f"chunk{budget}-c{concurrency}-{'drop' if drop else 'no_drop'}"
        command = case_command(args, case, concurrency, drop, budget)
        row = dict(name=case.name, path=str(case), concurrency=concurrency, drop=drop,
                   budget=budget, state="running", start=time.time(), command=command)
        state["cases"].append(row)
        with (root / (case.name + "-driver.log")).open("x") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = proc.pid
            write(root / "matrix.json", state)
            while proc.poll() is None:
                if oom_evidence(case):
                    stop_case(proc, case)
                    break
                time.sleep(10)
        evidence = oom_evidence(case)
        outcome = case / "outcome.json"
        valid = proc.returncode == 0 and outcome.exists() and json.loads(outcome.read_text()).get("valid")
        row.update(state="completed" if valid else "oom" if evidence else "failed",
                   end=time.time(), returncode=proc.returncode, oom=evidence)
        write(root / "matrix.json", state)
        await_gpu_release()
        if not valid and not evidence:
            raise RuntimeError(f"Non-OOM failure: {case}; no automatic retry")
        return bool(valid)

    try:
        while True:
            if run_case(32, False) and run_case(32, True):
                break
            # Both sides must complete under the same budget. An earlier successful
            # side is preserved, but is not paired with a smaller-budget run.
            if budget <= 1:
                raise RuntimeError("OOM even at the minimum token budget")
            budget //= 2
            state["next_budget"] = budget
            write(root / "matrix.json", state)
        state["selected_budget"] = budget
        state["c32_pair_completed"] = time.time()
        write(root / "matrix.json", state)
        for concurrency in (1, 2, 4, 8):
            for drop in (False, True):
                if not run_case(concurrency, drop):
                    raise RuntimeError("OOM after C32 budget was fixed; stopping to preserve comparability")
        state.update(state="completed", end=time.time())
    except Exception as exc:
        state.update(state="failed", error=repr(exc), end=time.time())
        raise
    finally:
        write(root / "matrix.json", state)


if __name__ == "__main__":
    run(parser().parse_args())
