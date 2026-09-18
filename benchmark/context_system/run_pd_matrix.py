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
HERE = Path(__file__).resolve().parent


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def gpu_free():
    value = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1,2,3", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return all(int(v) < 256 for v in value.split())


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
    source = json.loads(Path(args.source_launch).read_text())
    source_servers = source["servers"]
    template = root / "retained_history.jinja"
    src = source_servers[0]["argv"]
    template.write_text(Path(src[src.index("--chat-template") + 1]).read_text())
    procs, logs, temps, launch = [], [], [], []
    collecting = False
    cpu_sampler = None
    try:
        for i, mode in enumerate(("prefill", "decode")):
            cmd = list(source_servers[i]["argv"])
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
                       TORCHELASTIC_USE_AGENT_STORE="False", SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1")
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
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Startup failed: {root}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(1)
        model = src[src.index("--model-path") + 1]
        warmup(SimpleNamespace(engine="pd", drop=args.drop, model=model, concurrency=args.concurrency, port=args.port), root)
        if args.profile_session:
            subprocess.run(["nsys", "start", "--session=" + args.profile_session], check=True, timeout=60)
            collecting = True
            cpu_sampler = subprocess.Popen([
                sys.executable, str(HERE / "sample_pd_processes.py"),
                "--parent", str(os.getpid()), "--output", str(root / "cpu-sched.jsonl")])
        client = [sys.executable, str(HERE / "test_serving.py"), "--mini-root", args.mini_root,
                  "--model", model, "--tokenizer", model, "--requests-path", args.requests_path,
                  "--output-dir", str(root / "workload"), "--concurrency", str(args.concurrency),
                  "--num-tasks", str(args.concurrency * 2), "--seed", "42", "--url",
                  f"http://127.0.0.1:{args.port+1}/v1/chat/completions", "--prefill-url",
                  f"http://127.0.0.1:{args.port}/v1/chat/completions", "--bootstrap-port", str(args.port+10),
                  "--chat-template", str(template), "--template-kwargs", '{"preserve_thinking_history":true}']
        if args.drop:
            client.append("--drop")
        write(root / "client.json", client)
        with (root / "client.log").open("x") as log:
            subprocess.run(client, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=14400)
        # Completed request records may flush just after the HTTP final event.
        time.sleep(2)
        result = summarize(root, args.mini_root)
        write(root / "outcome.json", dict(valid=result["valid"], overall=result["overall"]))
    except Exception as exc:
        write(root / "failure.json", {"error": repr(exc), "time": time.time()})
        raise
    finally:
        if collecting:
            # Periodic CUPTI flush plus an explicit stop while all CUDA workers
            # remain alive; never infer idleness from missing GPU tables.
            time.sleep(3)
            with (root / "nsys-stop.log").open("w") as log:
                try:
                    subprocess.run(["nsys", "stop", "--session=" + args.profile_session],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
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
    p.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    p.add_argument("--profile-session")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (run_one if args.one else matrix)(args)
