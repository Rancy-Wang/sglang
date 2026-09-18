"""Eight frozen-native no-drop BCP cases, each with a separate full Nsight replay.

The baseline checkout is read-only. All instrumentation lives in this driver
checkout. No automatic deadline increases or inference repairs are applied.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.request

from run_minimal import warmup
from summarize_pd_matrix import summarize

NATIVE_HEAD = "03c2d9ec8a1ea0ef4e9151de2a65eb3c23c9dfab"
OVERLAP_PLAN = "PLAN-CS-20260918-PD-OVERLAP-C1-8-R1"
HERE = Path(__file__).resolve().parent


def write(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def overlap_command(command, max_running_requests=8, prefill_token_budget=None,
                    mem_fraction_static=0.9, kv_pages=None):
    """Apply the approved common settings to either frozen P/D launcher."""
    command = list(command)
    for flag in ("--max-total-tokens", "--mem-fraction-static", "--max-running-requests",
                 "--cuda-graph-config"):
        while flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    command = [arg for arg in command if arg != "--disable-overlap-schedule"]
    command += ["--mem-fraction-static", str(mem_fraction_static), "--max-running-requests", str(max_running_requests),
                "--cuda-graph-config", json.dumps({
                    "decode": {"bs": [n for n in (1, 2, 4, 8, 16, 32)
                                      if n <= max_running_requests],
                               "max_bs": max_running_requests},
                    "prefill": {"bs": [16, 32, 64], "max_bs": 64}})]
    if prefill_token_budget is not None:
        for flag in ("--chunked-prefill-size", "--max-prefill-tokens"):
            while flag in command:
                index = command.index(flag)
                del command[index:index + 2]
            command += [flag, str(prefill_token_budget)]
    if kv_pages is not None:
        if "--page-size" not in command or command[command.index("--page-size") + 1] != "1":
            raise ValueError("Exact KV page capacity requires page_size=1")
        command += ["--max-total-tokens", str(kv_pages)]
    return command


def verify_capacity(info, expected):
    capacities = [item.get("memory_usage", {}).get("token_capacity")
                  for item in info.get("internal_states", [])]
    if info.get("page_size") != 1 or not capacities or any(n != expected for n in capacities):
        raise ValueError(f"Expected {expected} KV pages, observed {capacities}")
    return capacities


def gpu_free():
    value = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1,2,3", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return all(int(v) < 256 for v in value.split())


class GPUIsolationGuard:
    """Pin TP clients before warmup; allow their NVLink peer contexts only."""

    def __init__(self, root):
        self.uuids = set(subprocess.check_output([
            "nvidia-smi", "-i", "0,1,2,3", "--query-gpu=uuid",
            "--format=csv,noheader"], text=True).split())
        self.root = root
        self.baseline = self.clients()
        self.last_check = 0.0
        self.worker_pids = None

    def clients(self):
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader"], text=True, timeout=15)
        clients = {uuid: set() for uuid in self.uuids}
        for line in output.splitlines():
            uuid, pid = (part.strip() for part in line.split(","))
            if uuid in clients:
                clients[uuid].add(int(pid))
        return clients

    def check(self, force=False, pin=False):
        now = time.monotonic()
        if not (force or pin) and now - self.last_check < 10:
            return
        self.last_check = now
        clients = self.clients()
        extra = {uuid: sorted(pids - self.baseline[uuid]) for uuid, pids in clients.items()}
        observed = set().union(*(set(pids) for pids in extra.values()))
        invalid = (any(len(pids) > 1 for pids in extra.values())
                   if self.worker_pids is None else bool(observed - self.worker_pids))
        if pin:
            # Before any PD workload, each of the four TP workers has one
            # primary context. IPC may subsequently expose the same worker
            # on a peer GPU; it must not admit a new process identity.
            invalid |= (self.worker_pids is not None or len(observed) != 4
                        or any(len(pids) != 1 for pids in extra.values()))
            if not invalid:
                self.worker_pids = observed
        record = dict(time=time.time(), monotonic=now, new_clients=extra,
                      pinned_worker_pids=sorted(self.worker_pids or []),
                      baseline={uuid: sorted(pids) for uuid, pids in self.baseline.items()})
        with (self.root / "gpu-isolation.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        if invalid:
            write(self.root / "comparison-invalid.json", dict(
                performance_comparable=False, reason="Concurrent GPU compute clients", **record))
            raise RuntimeError("GPU co-location detected; performance measurement invalid")


def run_one(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(args.server_repo)
    expected_head = args.server_head
    if args.drop and expected_head == NATIVE_HEAD:
        raise ValueError("The frozen native baseline does not implement Drop")
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != expected_head:
        raise ValueError("Pinned server revision changed")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
        raise ValueError("Native baseline is dirty")
    if not gpu_free():
        raise RuntimeError("GPU 0-3 unavailable; no co-located benchmark allowed")
    isolation = GPUIsolationGuard(root)
    source = json.loads(Path(args.source_launch).read_text())
    source_servers = source["servers"]
    template = root / "retained_history.jinja"
    src = source_servers[0]["argv"]
    template.write_text(Path(src[src.index("--chat-template") + 1]).read_text())
    procs, logs, temps, launch = [], [], [], []
    collecting = False
    cpu_sampler = None
    client_proc = None
    try:
        for i, mode in enumerate(("prefill", "decode")):
            cmd = list(source_servers[i]["argv"])
            if args.approved_overlap_matrix:
                cmd = overlap_command(cmd, args.max_running_requests or max(8, args.concurrency),
                                      args.prefill_token_budget, args.mem_fraction_static, args.kv_pages)
            cmd[0:2] = [sys.executable, str(HERE / "launch_pd_counted.py")]
            for flag, value in (("--port", args.port + i), ("--disaggregation-bootstrap-port", args.port + 10),
                                ("--nccl-port", args.port + 20 + i), ("--chat-template", template)):
                cmd[cmd.index(flag) + 1] = str(value)
            radix = "--disaggregation-decode-enable-radix-cache"
            if radix in cmd:
                cmd.remove(radix)
            if i == 1 and args.decode_radix:
                cmd.append(radix)
            eviction = "--context-drop-aware-eviction"
            if eviction in cmd:
                cmd.remove(eviction)
            if args.drop and (i == 0 or args.decode_radix):
                cmd.append(eviction)
            temp = tempfile.TemporaryDirectory(prefix="pd8-")
            temps.append(temp)
            env = dict(os.environ, PYTHONPATH=str(repo / "python"),
                       CUDA_VISIBLE_DEVICES="0,1" if i == 0 else "2,3", TMPDIR=temp.name,
                       PD_MATRIX_PROFILE="1" if args.profile_session else "0",
                       PD_MATRIX_RESERVE_FREE_MIB=str(args.reserve_free_mib),
                       TORCHELASTIC_USE_AGENT_STORE="False", SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1")
            if args.pd_timeout is not None:
                env.update(SGLANG_DISAGGREGATION_WAITING_TIMEOUT=str(args.pd_timeout),
                           SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=str(args.pd_timeout))
            if args.http_keepalive_timeout is not None:
                env["SGLANG_TIMEOUT_KEEP_ALIVE"] = str(args.http_keepalive_timeout)
            for key in ("MC_FORCE_TCP", "MC_INTRANODE_NVLINK", "MOONCAKE_PROTOCOL", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL"):
                env.pop(key, None)
            if args.transport == "tcp":
                env.update(MC_FORCE_TCP="1", MOONCAKE_PROTOCOL="tcp")
            else:
                env.update(MC_INTRANODE_NVLINK="1", MOONCAKE_PROTOCOL="nvlink_intra", SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK")
            for key in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
                path = root / mode / key.lower()
                path.mkdir(parents=True)
                env[key] = str(path)
            log = (root / f"{mode}.log").open("x")
            logs.append(log)
            proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            launch.append(dict(mode=mode, argv=cmd, pid=proc.pid, gpu=env["CUDA_VISIBLE_DEVICES"],
                               environment={k:v for k,v in env.items() if k.startswith(("MC_", "MOONCAKE_", "SGLANG_", "PD_MATRIX_"))}))
        write(root / "launch.json", dict(args=vars(args), head=expected_head, servers=launch))
        for i, proc in enumerate(procs):
            deadline = time.monotonic() + 900
            while True:
                isolation.check()
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Startup failed: {root}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(1)
        isolation.check(pin=True)
        capacities = {}
        for i, mode in enumerate(("prefill", "decode")):
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/server_info", timeout=30) as response:
                info = json.load(response)
                write(root / f"{mode}-server-info.json", info)
                if args.kv_pages is not None:
                    capacities[mode] = verify_capacity(info, args.kv_pages)
        if capacities:
            write(root / "capacity-check.json", dict(expected=args.kv_pages, actual=capacities))
        model = src[src.index("--model-path") + 1]
        warmup(SimpleNamespace(engine="pd", drop=args.drop, model=model, concurrency=args.concurrency,
                               port=args.port, warmup_timeout=args.warmup_timeout), root)
        isolation.check(force=True)
        if args.profile_session:
            subprocess.run(["nsys", "start", "--session=" + args.profile_session], check=True, timeout=60)
            collecting = True
            cpu_sampler = subprocess.Popen([
                sys.executable, str(HERE / "sample_pd_processes.py"),
                "--parent", str(os.getpid()), "--output", str(root / "cpu-sched.jsonl")])
        client = [sys.executable, str(HERE / "test_serving.py"), "--mini-root", args.mini_root,
                  "--model", model, "--tokenizer", model, "--requests-path", args.requests_path,
                  "--output-dir", str(root / "workload"), "--concurrency", str(args.concurrency),
                  "--num-tasks", str(args.concurrency * args.rounds), "--seed", "42", "--url",
                  f"http://127.0.0.1:{args.port+1}/v1/chat/completions", "--prefill-url",
                  f"http://127.0.0.1:{args.port}/v1/chat/completions", "--bootstrap-port", str(args.port+10),
                  "--chat-template", str(template), "--template-kwargs", '{"preserve_thinking_history":true}',
                  "--timeout", str(args.request_timeout)]
        if args.drop:
            client.append("--drop")
        write(root / "client.json", client)
        with (root / "client.log").open("x") as log:
            client_proc = subprocess.Popen(client, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # Scale the whole-workload watchdog for the larger task cohort only.
            # Per-request and KV-transfer deadlines are recorded separately.
            workload_timeout = args.workload_timeout or 14400 * max(1, args.concurrency // 8)
            deadline = time.monotonic() + workload_timeout
            while client_proc.poll() is None:
                isolation.check()
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(client, workload_timeout)
                time.sleep(5)
            isolation.check(force=True)
            if client_proc.returncode:
                raise subprocess.CalledProcessError(client_proc.returncode, client)
        # Completed request records may flush just after the HTTP final event.
        time.sleep(2)
        result = summarize(root, args.mini_root)
        result["measurement"]["profiler"] = bool(args.profile_session)
        write(root / "counted-result.json", result)
        write(root / "outcome.json", dict(valid=result["valid"], overall=result["overall"]))
        if not result["valid"]:
            raise RuntimeError("Incomplete first-pass tasks; do not continue matrix")
    except Exception as exc:
        write(root / "failure.json", {"error": repr(exc), "time": time.time()})
        raise
    finally:
        if client_proc is not None and client_proc.poll() is None:
            os.killpg(client_proc.pid, signal.SIGTERM)
            try:
                client_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(client_proc.pid, signal.SIGKILL)
                client_proc.wait()
        if collecting:
            # Periodic CUPTI flush plus an explicit stop while all CUDA workers
            # remain alive; never infer idleness from missing GPU tables.
            time.sleep(3)
            with (root / "nsys-stop.log").open("w") as log:
                try:
                    subprocess.run(["nsys", "stop", "--session=" + args.profile_session],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
                except Exception as exc:
                    write(root / "profile-stop-failure.json", {"error": repr(exc)})
            time.sleep(3)
        if cpu_sampler is not None:
            cpu_sampler.terminate()
            cpu_sampler.wait(timeout=10)
        for proc in procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()
        for temp in temps:
            temp.cleanup()


def case_process_running(pid, case, proc_root=Path("/proc")):
    """Identify the Linux case process without relying on signal-zero probes."""
    try:
        proc = proc_root / str(pid)
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        argv = (proc / "cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    if state in ("Z", "X"):
        return False
    # A reused PID belongs to a different process and cannot keep us waiting.
    return ("--one" in argv and "--output-dir" in argv
            and argv[argv.index("--output-dir") + 1] == str(case))


def predecessor_ready(case, pid):
    """Require a valid predecessor and its complete GPU cleanup before handoff."""
    case = Path(case)
    if (case / "failure.json").exists() or (case / "comparison-invalid.json").exists():
        raise RuntimeError(f"Predecessor failed: {case}")
    outcome = case / "outcome.json"
    valid = outcome.exists() and json.loads(outcome.read_text()).get("valid") is True
    if outcome.exists() and not valid:
        raise RuntimeError(f"Invalid predecessor outcome: {case}")
    running = case_process_running(pid, case)
    if not running and not valid:
        raise RuntimeError(f"Predecessor exited without a valid outcome: {case}")
    return valid and not running and gpu_free()


def overlap_matrix(args):
    """Run ordered pairs; failed cases remain invalid even when continuing."""
    if (args.rounds != 3 or args.reserve_free_mib or args.profile_session
            or not args.drop_server_repo or not args.drop_server_head
            or args.server_head != NATIVE_HEAD):
        raise ValueError("Approved matrix requires 3 rounds, frozen native/drop repos, no Nsight or VRAM reservation")
    if len(set(args.concurrencies)) != len(args.concurrencies):
        raise ValueError("Duplicate concurrency would repeat a case")
    if bool(args.wait_for_case) != bool(args.wait_for_case_pid):
        raise ValueError("Predecessor path and PID must be supplied together")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for c in args.concurrencies:
        for drop in (False, True):
            rows.append(dict(name=f"nvlink-d1-c{c}-{'drop' if drop else 'no_drop'}",
                             concurrency=c, drop=drop, num_tasks=3*c, state="pending"))
    state = dict(plan_id=OVERLAP_PLAN, args=vars(args), cases=rows, state="running")
    write(root / "matrix.json", state)
    if args.wait_for_case:
        state.update(state="waiting_predecessor", predecessor=args.wait_for_case,
                     predecessor_pid=args.wait_for_case_pid)
        write(root / "matrix.json", state)
        try:
            deadline = time.monotonic() + 43200
            while not predecessor_ready(args.wait_for_case, args.wait_for_case_pid):
                if time.monotonic() > deadline:
                    raise TimeoutError("Predecessor completion/cleanup wait exceeded 12 hours")
                time.sleep(30)
        except Exception as exc:
            state.update(state="blocked_predecessor", error=repr(exc))
            write(root / "matrix.json", state)
            raise
        state.update(state="running", predecessor_completed=time.time())
        write(root / "matrix.json", state)
    for index, row in enumerate(rows):
        command = [sys.executable, str(Path(__file__).resolve()), "--one", "--approved-overlap-matrix",
                   "--output-dir", str(root / row["name"]), "--source-launch", args.source_launch,
                   "--server-repo", args.drop_server_repo if row["drop"] else args.server_repo,
                   "--server-head", args.drop_server_head if row["drop"] else args.server_head,
                   "--mini-root", args.mini_root, "--requests-path", args.requests_path,
                   "--concurrency", str(row["concurrency"]), "--rounds", "3", "--transport", "nvlink",
                   "--decode-radix", "--port", str(args.port + 40 * index)]
        for option in ("prefill_token_budget", "max_running_requests", "pd_timeout",
                       "warmup_timeout", "request_timeout", "mem_fraction_static",
                       "kv_pages", "workload_timeout", "http_keepalive_timeout"):
            value = getattr(args, option)
            if value is not None:
                command += ["--" + option.replace("_", "-"), str(value)]
        if row["drop"]:
            command.append("--drop")
        row.update(state="running", command=command, start=time.time())
        write(root / "matrix.json", state)
        with (root / (row["name"] + "-driver.log")).open("x") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = proc.pid
            write(root / "matrix.json", state)
            code = proc.wait()
        row.update(state="completed" if code == 0 else "failed", returncode=code, end=time.time())
        if code and not args.continue_on_failure:
            state["state"] = "failed"
            write(root / "matrix.json", state)
            raise RuntimeError(f"Stopped on failed case: {row['name']}")
        write(root / "matrix.json", state)
        print(json.dumps(row), flush=True)
        # The child has joined its launchers; allow driver-level GPU teardown.
        deadline = time.monotonic() + 120
        while not gpu_free():
            if time.monotonic() > deadline:
                state.update(state="blocked_gpu_cleanup", blocked_after=row["name"])
                write(root / "matrix.json", state)
                raise RuntimeError("GPU 0-3 not released after completed case")
            time.sleep(5)
    state.update(state="completed_with_failures" if any(row["returncode"] for row in rows)
                 else "completed", end=time.time())
    write(root / "matrix.json", state)


def matrix(args):
    if args.drop or args.server_head != NATIVE_HEAD:
        raise ValueError("Modified-server validation requires --one; the old native matrix must not resume")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    write(root / "matrix.json", {"args": vars(args), "cases": rows})
    for profile in (False, True):
        for index, (transport, radix, c) in enumerate(itertools.product(("nvlink", "tcp"), (False, True), (1, 2))):
            name = f"{transport}-d{int(radix)}-c{c}" + ("-nsys" if profile else "-throughput")
            target = root / name
            row = dict(name=name, transport=transport, decode_radix=radix, concurrency=c, profile=profile, state="waiting_gpu")
            rows.append(row)
            write(root / "matrix.json", {"args": vars(args), "cases": rows})
            deadline = time.monotonic() + 43200
            while not gpu_free():
                if time.monotonic() > deadline:
                    raise RuntimeError("GPU wait exceeded 12 hours")
                time.sleep(30)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one", "--output-dir", str(target),
                   "--server-repo", args.server_repo, "--source-launch", args.source_launch,
                   "--mini-root", args.mini_root, "--requests-path", args.requests_path,
                   "--transport", transport, "--concurrency", str(c), "--port", str(args.port + index * 40)]
            if radix:
                cmd.append("--decode-radix")
            if profile:
                session = f"pd8_{os.getpid()}_{index}"
                cmd += ["--profile-session", session]
                cmd = ["nsys", "profile", "--session-new=" + session, "--start-later=true",
                       "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                       # Busy polling and short OSRT stacks can produce hundreds
                       # of GiB in a full BCP replay. Keep GPU activities, core
                       # CUDA APIs and the long host waits this audit examines.
                       "--cuda-trace-all-apis=false", "--osrt-threshold=1000000",
                       "--osrt-backtrace-threshold=50000000",
                       "--cuda-event-trace=false", "--cuda-graph-trace=graph",
                       "--cuda-flush-interval=1000", "--output=" + str(root / name), *cmd]
            row.update(state="running", command=cmd, start=time.time())
            write(root / "matrix.json", {"args": vars(args), "cases": rows})
            with (root / (name + "-driver.log")).open("x") as log:
                try:
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=18000).returncode
                except subprocess.TimeoutExpired:
                    # Do not launch another case over a possibly surviving GPU process.
                    row.update(state="driver_timeout", end=time.time())
                    write(root / "matrix.json", {"args": vars(args), "cases": rows})
                    raise
            row.update(state="finished" if code == 0 else "failed", returncode=code, end=time.time())
            if profile and target.exists():
                write(target / "profile-command.json", cmd)
                if (target / "counted-result.json").exists():
                    result = json.loads((target / "counted-result.json").read_text())
                    result["measurement"]["profiler"] = True
                    write(target / "counted-result.json", result)
            write(root / "matrix.json", {"args": vars(args), "cases": rows})
            print(json.dumps(row), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--server-repo", required=True)
    p.add_argument("--server-head", default=NATIVE_HEAD, help="Exact reviewed server commit")
    p.add_argument("--source-launch", required=True)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--requests-path", required=True)
    p.add_argument("--port", type=int, default=30041)
    p.add_argument("--one", action="store_true")
    p.add_argument("--transport", choices=("tcp", "nvlink"), default="nvlink")
    p.add_argument("--decode-radix", action="store_true")
    p.add_argument("--drop", action="store_true", help="Rolling K=12 / 96Ki-token Repos, with Drop-aware eviction")
    p.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8, 32), default=1)
    p.add_argument("--prefill-token-budget", type=int,
                   help="Explicit common chunked-prefill-size and max-prefill-tokens override")
    p.add_argument("--max-running-requests", type=int, choices=(8, 16, 32),
                   help="Keep the same server admission/graph configuration across concurrency cases")
    p.add_argument("--pd-timeout", type=int, help="Explicit PD bootstrap and waiting timeout in seconds")
    p.add_argument("--warmup-timeout", type=int, default=600)
    p.add_argument("--request-timeout", type=int, default=7200)
    p.add_argument("--workload-timeout", type=int, help="Whole measured case timeout in seconds")
    p.add_argument("--http-keepalive-timeout", type=int, help="Server HTTP idle connection timeout")
    p.add_argument("--mem-fraction-static", type=float, default=0.9)
    p.add_argument("--kv-pages", type=int, help="Exact page_size=1 capacity, verified before warmup")
    p.add_argument("--continue-on-failure", action="store_true",
                   help="Preserve failed cases and continue after GPU cleanup; never count them as valid")
    p.add_argument("--rounds", type=int, choices=range(1, 21), default=2)
    p.add_argument("--approved-overlap-matrix", action="store_true", help=OVERLAP_PLAN)
    p.add_argument("--concurrencies", type=int, nargs="+", choices=(1, 2, 4, 8, 32),
                   default=[1, 2, 4, 8], help="Ordered overlap-matrix concurrency pairs")
    p.add_argument("--wait-for-case", help="Finish this existing case before starting the new matrix")
    p.add_argument("--wait-for-case-pid", type=int)
    p.add_argument("--drop-server-repo")
    p.add_argument("--drop-server-head")
    p.add_argument("--profile-session")
    p.add_argument("--reserve-free-mib", type=int, default=0,
                   help="Reserve idle VRAM in each worker's reusable PyTorch cache, leaving this runtime headroom")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (run_one if args.one else overlap_matrix if args.approved_overlap_matrix else matrix)(args)
