"""Opt-in fixed-cohort PD profiling. No patches are installed by ordinary serving.

APPROVED_PLAN_ID: PLAN-CS-20260923-DECODE-BREAKDOWN-R1
All hooks live in this benchmark entrypoint; import frozen SGLang via PYTHONPATH.
"""

import argparse
from collections import deque
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

from build_decode_breakdown_fixture import digest, paired_payloads, validate_fixture


def cohort_ready(waiting, running, expected):
    """Hold before native prebuilt processing; do not advance a partial batch."""
    ids = [r.rid for r in waiting] + [r.rid for r in running]
    unexpected = set(ids) - set(expected)
    if unexpected:
        raise ValueError(f"Unexpected requests in isolated benchmark: {unexpected}")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request in fixed cohort")
    return set(ids) == set(expected)


def install():
    config = json.loads(Path(os.environ["DECODE_BREAKDOWN_CONFIG"]).read_text())
    fixture = validate_fixture(json.loads(Path(config["fixture"]).read_text()))
    requests = {r["rid"]: r for r in fixture["requests"]}
    role = config["role"]
    profile = config.get("profile", False)
    warm, steps = config.get("warm_steps", 128), config.get("measure_steps", 512)
    if warm + steps + 4 >= fixture["max_new_tokens"]:
        raise ValueError("Continuation too short for warmup, measurement and drain")

    import torch
    from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

    pending, records = deque(), []
    cursors, token_tables, native_requests = {}, {}, {}
    state = dict(rank=None, log=None, cohort_started=False, first_wait=None,
                 profiling_started=False, profiling_stopped=False, tokens_verified=False)

    def emit(row):
        row.update(role=role, rank=state["rank"], pid=os.getpid())
        if state["log"] is None:
            root = Path(config["output"])
            root.mkdir(parents=True, exist_ok=True)
            state["log"] = (root / f"{role}-rank{state['rank']}-{os.getpid()}.jsonl").open("x", buffering=1)
        state["log"].write(json.dumps(row) + "\n")

    def flush():
        while pending and pending[0][1].query():
            begin, end, row = pending.popleft()
            row["gpu_ms"] = begin.elapsed_time(end)
            records.append(row)
        # File IO and event queries happen outside ModelRunner.forward. Rows
        # stay buffered until the fixed cohort drains, avoiding per-step IO.
        if records and not pending:
            for row in records:
                emit(row)
            records.clear()
        if (role == "decode" and not state["tokens_verified"]
                and set(native_requests) == set(requests)
                and all(r.finished() for r in native_requests.values())):
            for rid, req in native_requests.items():
                actual = list(req.output_ids)
                expected = requests[rid]["replay_tokens"][:fixture["max_new_tokens"]]
                if actual != expected:
                    raise ValueError(f"Emitted tokens differ from fixed continuation: {rid}")
            emit(dict(kind="cohort_tokens_verified", hashes={
                rid: digest(list(req.output_ids)) for rid, req in native_requests.items()}))
            state["tokens_verified"] = True

    original_load = ModelRunner.load_model

    def load_model(self, *a, **kw):
        result = original_load(self, *a, **kw)
        state["rank"] = self.ps.tp_rank
        emit(dict(kind="runtime", wall=time.time(), config=config,
                  attention_backend=getattr(self.server_args, "attention_backend", None),
                  model_type=type(self.model).__name__,
                  cuda_graph_config=str(getattr(self.server_args, "cuda_graph_config", None))))
        if profile and role == "decode":
            # Include graph construction to retain graph-node attribution.
            # Python NVTX ranges do not execute again during graph replay.
            torch.cuda.profiler.start()
            state["profiling_started"] = True
            for name, module in self.model.named_modules():
                category = module_category(name, module)
                if category:
                    module.forward = nvtx_wrapper(module.forward, f"component:{category}:{name}", torch)
        return result

    ModelRunner.load_model = load_model
    original_init = ForwardBatch.init_new.__func__

    @classmethod
    def init_new(cls, batch, runner, *a, **kw):
        ret = original_init(cls, batch, runner, *a, **kw)
        if not batch.reqs:
            return ret
        identities = tuple(r.rid for r in batch.reqs)
        if any(rid not in requests for rid in identities):
            raise ValueError(f"Non-fixture request in benchmark: {identities}")
        decode = ret.forward_mode.is_decode()
        if decode:
            if role != "decode" or len(identities) != fixture["batch_size"]:
                raise ValueError("Decode batch differs from approved fixed cohort")
            counts = [cursors.get(rid, 0) for rid in identities]
            if len(set(counts)) != 1:
                raise ValueError("Requests have different Decode step indices")
            step = counts[0]
            for rid in identities:
                cursors[rid] = step + 1
        else:
            step = -1
        for req in batch.reqs:
            native_requests[req.rid] = req
            if not getattr(req, "_breakdown_checked", False):
                if digest(list(req.origin_input_ids)) != requests[req.rid]["input_sha256"]:
                    raise ValueError(f"Native server prefix differs from fixture: {req.rid}")
                req._breakdown_checked = True
        ret._breakdown = dict(identities=identities, step=step)
        return ret

    ForwardBatch.init_new = init_new
    original_sample = ModelRunner.sample

    def sample(self, logits_output, forward_batch, *a, **kw):
        result = original_sample(self, logits_output, forward_batch, *a, **kw)
        meta = getattr(forward_batch, "_breakdown", None)
        if meta is None:
            return result
        key = meta["identities"]
        if key not in token_tables:
            token_tables[key] = torch.tensor(
                [requests[rid]["replay_tokens"] for rid in key],
                dtype=result.dtype, device=result.device).t().contiguous()
        # P generates output[0]. D's first forward consumes it and generates
        # output[1]. Extra speculative-overlap tail steps are not measured.
        index = meta["step"] + 1 if meta["step"] >= 0 else 0
        if index >= token_tables[key].shape[0]:
            raise ValueError("Teacher-forced continuation exhausted")
        if profile:
            torch.cuda.nvtx.range_push("benchmark:teacher_force")
        result.copy_(token_tables[key][index])
        if profile:
            torch.cuda.nvtx.range_pop()
        return result

    ModelRunner.sample = sample
    original_forward = ModelRunner.forward

    def forward(self, batch, *a, **kw):
        meta = getattr(batch, "_breakdown", None)
        if meta is None or not batch.forward_mode.is_decode():
            return original_forward(self, batch, *a, **kw)
        step = meta["step"]
        measured = warm <= step < warm + steps
        row = dict(kind="decode_step", step=step, measured=measured,
                   batch_size=int(batch.batch_size), request_ids=meta["identities"],
                   start_perf=time.perf_counter(), start_wall=time.time(),
                   active_seq_lens=batch.seq_lens_cpu.tolist())
        if step == 0:
            expected = [(requests[rid]["drop_state"]["active_tokens"]
                         if config["strategy"] == "drop" else len(requests[rid]["input_ids"])) + 1
                        for rid in meta["identities"]]
            if row["active_seq_lens"] != expected:
                raise ValueError(f"Actual active KV lengths differ from fixture: {row['active_seq_lens']} vs {expected}")
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if profile:
            torch.cuda.nvtx.range_push(f"decode_step:{step}")
        begin.record()
        result = original_forward(self, batch, *a, **kw)
        end.record()
        if profile:
            torch.cuda.nvtx.range_pop()
        row.update(cpu_ms=1000 * (time.perf_counter() - row["start_perf"]),
                   cuda_graph=bool(result.can_run_graph))
        graph = getattr(self, "decode_cuda_graph_runner", None)
        if graph is None:
            graph = getattr(self, "cuda_graph_runner", None)
        row["graph_batch_size"] = getattr(graph, "bs", None) if result.can_run_graph else None
        pending.append((begin, end, row))
        if step == warm + steps + 2 and state["profiling_started"] and not state["profiling_stopped"]:
            # One synchronization outside the measurement window ensures the
            # last measured kernels finish before stopping collection.
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            state["profiling_stopped"] = True
        return result

    ModelRunner.forward = forward
    original_new = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch

    def get_new(self, running_batch):
        if role == "decode" and not state["cohort_started"]:
            if self.waiting_queue and state["first_wait"] is None:
                state["first_wait"] = time.monotonic()
            if not cohort_ready(self.waiting_queue, running_batch.reqs, requests):
                if state["first_wait"] and time.monotonic() - state["first_wait"] > config.get("gate_timeout", 1800):
                    raise TimeoutError("Fixed cohort did not become KV-ready")
                return None
            state["cohort_started"] = True
            emit(dict(kind="cohort_ready", wall=time.time(), request_ids=list(requests),
                      transfer_queue=len(self.disagg_decode_transfer_queue.queue),
                      prealloc_queue=len(self.disagg_decode_prealloc_queue.queue)))
        return original_new(self, running_batch)

    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch = get_new
    original_idle = Scheduler.on_idle

    def idle(self, *a, **kw):
        if pending:
            flush()
        return original_idle(self, *a, **kw)

    Scheduler.on_idle = idle


def module_category(name, module):
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "attn":
        window = getattr(module, "sliding_window_size", -1)
        return "swa_attention" if window and window > 0 else "full_attention"
    return {"qkv_proj": "qkv_projection", "o_proj": "output_projection",
            "rotary_emb": "rope", "router": "moe_router", "topk": "moe_topk",
            "experts": "moe_experts", "input_layernorm": "norm",
            "post_attention_layernorm": "norm", "norm": "norm",
            "lm_head": "lm_head", "logits_processor": "logits"}.get(leaf)


def nvtx_wrapper(original, label, torch):
    @functools.wraps(original)
    def wrapped(*a, **kw):
        torch.cuda.nvtx.range_push(label)
        try:
            return original(*a, **kw)
        finally:
            torch.cuda.nvtx.range_pop()
    return wrapped


async def client(args):
    import asyncio
    import aiohttp

    fixture = validate_fixture(json.loads(Path(args.fixture).read_text()))
    payloads = paired_payloads(fixture, args.strategy)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "request_manifest.json").write_text(json.dumps(dict(
        fixture_sha256=digest(fixture), strategy=args.strategy, payloads=payloads)))
    room_base = time.time_ns() % (1 << 50)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        async def send(index, payload):
            body = dict(payload, bootstrap_host="127.0.0.1", bootstrap_port=args.bootstrap_port,
                        bootstrap_room=room_base + index)
            started = time.perf_counter()
            async def prefill():
                async with session.post(args.prefill_url, json={**body, "stream": False}) as response:
                    value = await response.text()
                    (root / f"{index:02d}-prefill.json").write_text(value)
                    response.raise_for_status()
            async def decode():
                with (root / f"{index:02d}-sse.jsonl").open("x") as log:
                    async with session.post(args.decode_url, json=body) as response:
                        response.raise_for_status()
                        done = False
                        completion_tokens, finish = None, None
                        async for line in response.content:
                            text = line.decode().strip()
                            if not text:
                                continue
                            log.write(json.dumps(dict(time=time.perf_counter(), data=text)) + "\n")
                            if text == "data: [DONE]":
                                done = True
                            elif text.startswith("data: "):
                                value = json.loads(text[6:])
                                if value.get("usage"):
                                    completion_tokens = value["usage"].get("completion_tokens")
                                for choice in value.get("choices", []):
                                    finish = choice.get("finish_reason") or finish
                        if not done:
                            raise RuntimeError("Decode response ended without DONE")
                        if completion_tokens != fixture["max_new_tokens"] or finish != "length":
                            raise RuntimeError(f"Incomplete cohort output: {completion_tokens=}, {finish=}")
            await asyncio.gather(prefill(), decode())
            return dict(rid=payload["rid"], lifecycle_s=time.perf_counter() - started)
        result = await asyncio.gather(*(send(i, p) for i, p in enumerate(payloads)))
    (root / "completed.json").write_text(json.dumps(result, indent=2))


def run_one(args):
    """Start an isolated pair from the verified historical launch configuration."""
    from run_pd_matrix import GPUIsolationGuard, gpu_free
    from analyze_decode_breakdown import analyze
    from types import SimpleNamespace

    fixture = validate_fixture(json.loads(Path(args.fixture).read_text()))
    repo = Path(args.server_repo).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != "cfe9a570751218eab8ae8890777998f489ab5e21":
        raise ValueError("Runtime differs from the frozen R1 experimental version")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
        raise ValueError("Frozen runtime has local modifications")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if not gpu_free("0,1,2,3,4,5,6,7"):
        raise RuntimeError("GPUs occupied; no process was stopped")
    guard = GPUIsolationGuard(root, "0,1,2,3,4,5,6,7")
    source = json.loads(Path(args.source_launch).read_text())
    script = str(Path(__file__).resolve())
    procs, logs, launches = [], [], []
    client_proc = None
    success = False
    try:
        for i, role in enumerate(("prefill", "decode")):
            original = source["servers"][i]
            flags = original["argv"][2:]
            expected = {"--tp-size": "4", "--dtype": "bfloat16", "--page-size": "1",
                        "--chunked-prefill-size": "8192", "--max-prefill-tokens": "8192",
                        "--mem-fraction-static": "0.85", "--context-length": "131072"}
            for flag, value in expected.items():
                if flag not in flags or flags[flags.index(flag) + 1] != value:
                    raise ValueError(f"Unexpected source setting: {flag}")
            if "--attention-backend" in flags or "--disable-overlap-schedule" in flags:
                raise ValueError("Keep the native backend and overlap settings")
            if i == 1 and "--disaggregation-decode-enable-radix-cache" not in flags:
                raise ValueError("D Radix must be enabled")
            for flag, value in (("--port", args.port + i),
                                ("--disaggregation-bootstrap-port", args.port + 10),
                                ("--nccl-port", args.port + 20 + i)):
                flags[flags.index(flag) + 1] = str(value)
            eviction = "--context-drop-aware-eviction"
            flags = [f for f in flags if f != eviction]
            if args.strategy == "drop":
                flags.append(eviction)
            # Native startup probes have non-fixture RIDs. The controlled
            # cohort itself supplies 128 full-batch warmup decode steps.
            if "--skip-server-warmup" not in flags:
                flags.append("--skip-server-warmup")
            cfg = dict(fixture=str(Path(args.fixture).resolve()), role=role,
                       strategy=args.strategy, profile=args.profile, output=str(root / "timing"),
                       warm_steps=128, measure_steps=64 if args.profile else 512, gate_timeout=1800)
            config_path = root / f"{role}-config.json"
            config_path.write_text(json.dumps(cfg, indent=2))
            env = dict(os.environ, PYTHONPATH=str(repo / "python"),
                       CUDA_VISIBLE_DEVICES="0,1,2,3" if i == 0 else "4,5,6,7",
                       TORCHELASTIC_USE_AGENT_STORE="False",
                       SGLANG_DISAGGREGATION_WAITING_TIMEOUT="7200",
                       SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="7200",
                       SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
                       MC_INTRANODE_NVLINK="1", MOONCAKE_PROTOCOL="nvlink_intra",
                       SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK")
            env.pop("MC_FORCE_TCP", None)
            for key in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR",
                        "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
                folder = root / role / key.lower()
                folder.mkdir(parents=True)
                env[key] = str(folder)
            # AF_UNIX paths are limited to 107 bytes on this host. Keep IPC
            # temporary files isolated but outside the long archive path.
            env["TMPDIR"] = tempfile.mkdtemp(prefix="sgdb-", dir="/tmp")
            cmd = [sys.executable, script, "server", "--config", str(config_path), "--", *flags]
            if args.profile and role == "decode":
                cmd = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                       "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                       "--capture-range-end=stop", "--output=" + str(root / "decode-profile"), *cmd]
            log = (root / f"{role}.log").open("x")
            logs.append(log)
            proc = subprocess.Popen(cmd, env=env, cwd=repo, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            launches.append(dict(role=role, argv=cmd, pid=proc.pid,
                environment={k: v for k, v in env.items() if k.startswith(("CUDA_", "SGLANG_", "MC_", "MOONCAKE_", "PYTHONPATH", "LD_", "TMPDIR"))}))
        (root / "launch.json").write_text(json.dumps(dict(runtime_head=head,
            benchmark_head=subprocess.check_output(["git", "-C", str(Path(__file__).parents[2]), "rev-parse", "HEAD"], text=True).strip(),
            fixture_sha256=digest(fixture), launches=launches), indent=2))
        for i in range(2):
            deadline = time.monotonic() + 1200
            while True:
                guard.check()
                if any(p.poll() is not None for p in procs) or time.monotonic() > deadline:
                    raise RuntimeError("PD startup failed; inspect retained logs")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(1)
        guard.check(pin=True)
        for i, role in enumerate(("prefill", "decode")):
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/server_info", timeout=30) as response:
                info = json.load(response)
                (root / f"{role}-server-info.json").write_text(json.dumps(info, indent=2))
                capacity = min(s["memory_usage"]["token_capacity"] for s in info["internal_states"])
                required = sum(len(r["input_ids"]) + fixture["max_new_tokens"] for r in fixture["requests"])
                if required >= .85 * capacity:
                    raise ValueError(f"Insufficient No-Drop capacity margin: {required}/{capacity}")
        client_cmd = [sys.executable, script, "client", "--fixture", args.fixture,
                      "--strategy", args.strategy, "--output", str(root / "client"),
                      "--prefill-url", f"http://127.0.0.1:{args.port}/v1/chat/completions",
                      "--decode-url", f"http://127.0.0.1:{args.port+1}/v1/chat/completions",
                      "--bootstrap-port", str(args.port + 10)]
        with (root / "client.log").open("x") as log:
            client_proc = subprocess.Popen(client_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 3600
            while client_proc.poll() is None:
                guard.check()
                if any(p.poll() is not None for p in procs) or time.monotonic() > deadline:
                    raise RuntimeError("Cohort failed or exceeded timeout")
                time.sleep(1)
            if client_proc.returncode:
                raise RuntimeError("Client validation failed")
        # on_idle drains event buffers before timing files are analyzed.
        deadline = time.monotonic() + 30
        while True:
            try:
                analyze(SimpleNamespace(timings=str(root / "timing"), sqlite=None, output=str(root / "analysis")))
                break
            except ValueError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
        success = True
    finally:
        if client_proc and client_proc.poll() is None:
            os.killpg(client_proc.pid, signal.SIGTERM)
        for proc in reversed(procs):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        for log in logs:
            log.close()
        (root / "outcome.json").write_text(json.dumps(dict(
            success=success and not args.profile, timing_success=success,
            profile_pending=success and args.profile, time=time.time())))
    if args.profile:
        reports = list(root.glob("decode-profile*.nsys-rep"))
        if len(reports) != 1:
            raise ValueError("Expected one complete Nsight report for all four D ranks")
        database = root / "decode-profile.sqlite"
        subprocess.run([args.nsys, "export", "--type=sqlite", "--output=" + str(database), str(reports[0])], check=True)
        analyze(SimpleNamespace(timings=str(root / "timing"), sqlite=str(database),
                                output=str(root / "profile-analysis")))
        summary = json.loads((root / "profile-analysis" / "summary.json").read_text())
        if not summary["component_coverage_pass"]:
            raise ValueError("More than 5% unattributed GPU time; inspect trace before continuing")
        (root / "outcome.json").write_text(json.dumps(dict(success=True, time=time.time())))


def validate_completed_case(root, case):
    """Reuse only a successfully validated run with the exact frozen pairing."""
    root = Path(root).resolve()
    outcome = json.loads((root / "outcome.json").read_text())
    launch = json.loads((root / "launch.json").read_text())
    fixture = validate_fixture(json.loads(Path(case["fixture"]).read_text()))
    if not outcome.get("success"):
        raise ValueError(f"Cannot reuse unsuccessful run: {root}")
    if (launch["runtime_head"] != "cfe9a570751218eab8ae8890777998f489ab5e21"
            or launch["fixture_sha256"] != digest(fixture)):
        raise ValueError("Reused runtime or fixture differs")
    for role in ("prefill", "decode"):
        cfg = json.loads((root / f"{role}-config.json").read_text())
        if (cfg["strategy"] != case["strategy"] or cfg["profile"] != case["profile"]
                or cfg["warm_steps"] != 128
                or cfg["measure_steps"] != (64 if case["profile"] else 512)):
            raise ValueError("Reused measurement settings differ")
    folder = "profile-analysis" if case["profile"] else "analysis"
    folder = outcome.get("analysis_directory", folder)
    summary = json.loads((root / folder / "summary.json").read_text())
    if set(summary["ranks"]) != {"0", "1", "2", "3"}:
        raise ValueError("Reused result lacks four ranks")
    if case["profile"] and not summary.get("component_coverage_pass"):
        raise ValueError("Reused profile lacks validated component coverage")
    if any(r["steps"] != (64 if case["profile"] else 512)
           or r["batch_sizes"] != [fixture["batch_size"]]
           for r in summary["ranks"].values()):
        raise ValueError("Reused result has wrong batch or measurement length")
    return str(root)


def run_matrix(args):
    """Bounded R1 matrix; wait for resources without stopping other workloads."""
    from run_pd_matrix import gpu_free

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, str(Path(__file__).resolve()), "run",
              "--source-launch", args.source_launch, "--server-repo", args.server_repo,
              "--port", str(args.port), "--nsys", args.nsys]
    # First pair establishes that admission, timing and trace attribution work
    # before spending time on the remaining unprofiled repeats.
    cases = []
    for fixture_name in ("b8-48k.json", "b10-48k.json", "b4-112k-repos.json"):
        fixture_path = Path(args.fixtures) / fixture_name
        validate_fixture(json.loads(fixture_path.read_text()))
        for repeat in (0, 1, 2, 3):
            for strategy in (("no_drop", "drop") if repeat % 2 == 0 else ("drop", "no_drop")):
                cases.append(dict(fixture=str(fixture_path), strategy=strategy,
                                  profile=repeat == 1, repeat=repeat))
    (root / "matrix.json").write_text(json.dumps(cases, indent=2))
    reused = json.loads(Path(args.completed_cases).read_text()) if args.completed_cases else {}
    if any(not k.isdigit() or not 0 <= int(k) < len(cases) for k in reused):
        raise ValueError("Invalid completed-case index")
    if len(set(reused.values())) != len(reused):
        raise ValueError("One run cannot stand in for two independent repeats")
    reused = {k: validate_completed_case(p, cases[int(k)]) for k, p in reused.items()}
    (root / "reused-cases.json").write_text(json.dumps(reused, indent=2))
    state = root / "state.json"

    def report(**value):
        value.update(time=time.time())
        temp = state.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(state)
        print(json.dumps(value), flush=True)

    for index, case in enumerate(cases):
        if str(index) in reused:
            report(status="case_reused", index=index, case=case, source=reused[str(index)])
            continue
        label = f"{index:02d}-{Path(case['fixture']).stem}-{case['strategy']}-r{case['repeat']}"
        deadline = time.monotonic() + args.resource_timeout
        report(status="waiting_for_gpus", index=index, case=case)
        while not gpu_free("0,1,2,3,4,5,6,7"):
            if time.monotonic() > deadline:
                report(status="resource_timeout", index=index, case=case)
                raise TimeoutError("No isolated eight-GPU window; existing tasks preserved")
            time.sleep(30)
        cmd = common + ["--fixture", case["fixture"], "--strategy", case["strategy"],
                        "--output", str(root / label)]
        if case["profile"]:
            cmd.append("--profile")
        with (root / (label + ".log")).open("x") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            report(status="running", index=index, case=case, pid=proc.pid, command=cmd)
            status = proc.wait()
        if status:
            report(status="failed", index=index, case=case, exit_code=status)
            raise RuntimeError("Matrix stopped on first failure; inspect case log")
        report(status="case_completed", index=index, case=case)
    report(status="complete", cases=len(cases))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("server")
    s.add_argument("--config", required=True)
    c = sub.add_parser("client")
    for name in ("fixture", "output", "prefill-url", "decode-url"):
        c.add_argument("--" + name, required=True)
    c.add_argument("--strategy", choices=("drop", "no_drop"), required=True)
    c.add_argument("--bootstrap-port", type=int, required=True)
    c.add_argument("--timeout", type=int, default=3600)
    r = sub.add_parser("run")
    for name in ("fixture", "output", "source-launch", "server-repo"):
        r.add_argument("--" + name, required=True)
    r.add_argument("--strategy", choices=("drop", "no_drop"), required=True)
    r.add_argument("--port", type=int, default=45101)
    r.add_argument("--profile", action="store_true")
    r.add_argument("--nsys", default="/share/public/wangruoxi/cuda-12.1/bin/nsys")
    m = sub.add_parser("matrix")
    for name in ("fixtures", "output", "source-launch", "server-repo"):
        m.add_argument("--" + name, required=True)
    m.add_argument("--port", type=int, default=45101)
    m.add_argument("--nsys", default="/share/public/wangruoxi/cuda-12.1/bin/nsys")
    m.add_argument("--resource-timeout", type=int, default=14400)
    m.add_argument("--completed-cases", help="JSON mapping matrix indices to validated run directories")
    args, extra = p.parse_known_args()
    if args.command == "client":
        if extra:
            p.error(f"Unknown arguments: {extra}")
        import asyncio
        asyncio.run(client(args))
    elif args.command in ("run", "matrix"):
        if extra:
            p.error(f"Unknown arguments: {extra}")
        (run_one if args.command == "run" else run_matrix)(args)
    else:
        os.environ["DECODE_BREAKDOWN_CONFIG"] = str(Path(args.config).resolve())
        install()
        from sglang.launch_server import run_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.plugins import load_plugins
        from sglang.srt.utils import kill_process_tree
        load_plugins()
        try:
            run_server(prepare_server_args(extra[1:] if extra[:1] == ["--"] else extra))
        finally:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__mp_main__" and os.environ.get("DECODE_BREAKDOWN_CONFIG"):
    install()
if __name__ == "__main__":
    main()
