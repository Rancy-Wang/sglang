"""Opt-in E2E instrumentation; never imported by ordinary serving.

APPROVED_PLAN_ID: PLAN-CS-20260924-E2E-BREAKDOWN-R1
No admission gates, token forcing, cache changes, or synchronization in forwards.
CPU spans are elapsed call intervals, NOT CPU-active or GPU durations.
"""
import argparse
import asyncio
import contextvars
import concurrent.futures
from collections import deque
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

PLAN = "PLAN-CS-20260924-E2E-BREAKDOWN-R1"
FROZEN_HEAD = "cfe9a570751218eab8ae8890777998f489ab5e21"
HERE = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def seed_triton_cache(source, destination):
    """Copy compiled artifacts, relocating group paths into an isolated cache.

    No hard links, shared writable paths, or KV state. The source digest lets
    paired captures verify identical initial compilation state.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("Cache seed needs an existing source and empty destination")
    prepared, manifest = [], []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Cache seed symlink forbidden: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        data = path.read_bytes()
        manifest.append(dict(path=str(relative), bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest()))
        if path.name.startswith("__grp__") and path.suffix == ".json":
            group = json.loads(data)
            for key, child in group["child_paths"].items():
                child = Path(child)
                if not child.is_absolute():
                    child = path.parent / child
                local = child.resolve().relative_to(source)
                if not (source/local).is_file():
                    raise ValueError(f"Missing cache group artifact: {child}")
                group["child_paths"][key] = str(destination/local)
            data = json.dumps(group, sort_keys=True).encode()
        prepared.append((relative, data))
    if not prepared:
        raise ValueError("Empty compiled cache seed")
    # Validate all references before writing any artifacts.
    for relative, data in prepared:
        path = destination/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return dict(source=str(source), destination=str(destination), files=manifest,
                source_manifest_sha256=digest, isolation="independent copies; group paths relocated")


class Recorder:
    """Per-process asynchronous log; overflow is a fatal observability failure."""
    def __init__(self, root, role):
        self.root, self.role = Path(root), role
        self.pid = os.getpid()
        self.rank = None
        self.q = queue.Queue(65536)
        self.error = None
        self.worker = threading.Thread(target=self.write, daemon=True)
        self.worker.start()

    def emit(self, kind, **values):
        if self.error:
            raise RuntimeError("E2E recorder failed") from self.error
        self.q.put_nowait(dict(kind=kind, pid=self.pid, rank=self.rank,
                              tid=threading.get_native_id(), role=self.role, **values))

    def write(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / f"{self.role}-{self.pid}.jsonl").open("x") as f:
                while True:
                    try:
                        row = self.q.get(timeout=1)
                    except queue.Empty:
                        f.flush()
                        continue
                    if row is None:
                        f.flush()
                        return
                    f.write(json.dumps(row) + "\n")
                    if self.q.empty():
                        f.flush()
        except BaseException as exc:
            self.error = exc


def identity(value):
    """Only scalar identifiers; never copy tensors or full histories."""
    ret = {}
    for attr in ("rid", "bootstrap_room"):
        v = getattr(value, attr, None)
        if isinstance(v, (str, int)):
            ret[attr] = v
    req = getattr(value, "req", None)
    if req is not None:
        ret.update(identity(req))
    return ret


def bind_stats_identity(req):
    """Native tracing is optional; benchmark identity must not depend on it."""
    req.time_stats._e2e_identity = identity(req)
    return req.time_stats._e2e_identity


def batch_identity(value):
    """Read small host metadata only; no tensor materialization/synchronization."""
    reqs = getattr(value, "reqs", None)
    if reqs is None and isinstance(value, (list, tuple)):
        reqs = value
    if reqs is None:
        return {}
    identities = [identity(req) for req in reqs]
    return {"requests": [item for item in identities if item]}


def cache_result_metadata(value):
    ret = {}
    indices = getattr(value, "device_indices", None)
    if indices is not None:
        ret["matched_slots"] = len(indices)
    for name in ("context_exact_prefix_len", "context_retry", "full_kv_hit_length",
                 "num_tokens_evicted", "swa_num_tokens_evicted"):
        item = getattr(value, name, None)
        if isinstance(item, (bool, int, float)):
            ret[name] = item
    return ret


def mark_attention_backend(backend_class, nvtx, enabled):
    """Cover direct backend calls inside breakable Prefill graph custom ops.

    Those calls bypass RadixAttention.forward. Classify from the actual layer,
    never from a generic Triton kernel name; leave light runs unwrapped.
    """
    if not enabled:
        return
    original = backend_class.forward
    if getattr(original, "_e2e_attention_marker", False):
        return
    from profile_decode_breakdown import module_category

    @functools.wraps(original)
    def forward(*args, **kwargs):
        layer = args[4] if len(args) > 4 else kwargs["layer"]
        category = module_category("attn", layer)
        nvtx.range_push(f"component:{category}:attention_backend")
        try:
            return original(*args, **kwargs)
        finally:
            nvtx.range_pop()

    forward._e2e_attention_marker = True
    backend_class.forward = forward


def install():
    import torch
    # Preserve the historical benchmark's physical counters and P/D timestamps.
    os.environ["PD_MATRIX_PROFILE"] = "0"
    os.environ.pop("PD_MATRIX_FORWARD_LOG", None)
    from launch_pd_counted import install as install_counts
    install_counts()
    from profile_decode_breakdown import module_category, nvtx_wrapper
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.observability.req_time_stats import ReqTimeStatsBase, SchedulerReqTimeStats
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
    from sglang.srt.disaggregation.mooncake import conn
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine

    cfg = json.loads(Path(os.environ["PD_E2E_CONFIG"]).read_text())
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    mark_attention_backend(AttentionBackend, torch.cuda.nvtx, cfg["profile"])
    rec = Recorder(cfg["timing"], cfg["role"])
    current = contextvars.ContextVar("e2e_request", default={})
    pending = deque()
    state = dict(step=0, profile=False, stopped=False, checked=0.)
    transfer_context = threading.local()
    base_queue = conn.FastQueue
    class ObservedQueue(base_queue):
        def put(self, item):
            item._e2e_enqueued = time.perf_counter()
            return super().put(item)

        def get(self):
            item = super().get()
            transfer_context.chunk = dict(bootstrap_room=item.room,
                pages=len(item.prefill_kv_indices), last=item.is_last_chunk,
                slice_start=item.index_slice.start, slice_stop=item.index_slice.stop)
            rec.emit("transfer_dequeue", start=getattr(item, "_e2e_enqueued", None),
                     end=time.perf_counter(), **transfer_context.chunk)
            return item
    conn.FastQueue = ObservedQueue
    original_submit = concurrent.futures.ThreadPoolExecutor.submit
    def submit(self, fn, /, *args, **kw):
        chunk = getattr(transfer_context, "chunk", None)
        if chunk is None:
            return original_submit(self, fn, *args, **kw)
        def invoke():
            previous = getattr(transfer_context, "chunk", None)
            transfer_context.chunk = chunk
            try:
                return fn(*args, **kw)
            finally:
                transfer_context.chunk = previous
        return original_submit(self, invoke)
    concurrent.futures.ThreadPoolExecutor.submit = submit
    original_transfer = MooncakeTransferEngine.batch_transfer_sync
    def transfer(self, peer, src, dst, lengths):
        start = time.perf_counter()
        if cfg["profile"]:
            torch.cuda.nvtx.range_push("component:kv_transfer:batch_transfer_sync")
        try:
            result = original_transfer(self, peer, src, dst, lengths)
        finally:
            if cfg["profile"]:
                torch.cuda.nvtx.range_pop()
        rec.emit("transfer_sync", start=start, end=time.perf_counter(), bytes=sum(lengths),
                 blocks=len(lengths), result=result, peer=peer,
                 chunk=getattr(transfer_context, "chunk", None))
        return result
    MooncakeTransferEngine.batch_transfer_sync = transfer

    def clock():
        before = time.perf_counter_ns()
        if cfg["profile"] and state["profile"]:
            torch.cuda.nvtx.mark(f"e2e_clock:{before}")
        after = time.perf_counter_ns()
        rec.emit("clock", before_ns=before, after_ns=after, wall_ns=time.time_ns())

    def flush():
        while pending and pending[0][1].query():
            begin, end, row = pending.popleft()
            rec.emit("gpu_completed", gpu_ms=begin.elapsed_time(end), **row)
        now = time.monotonic()
        if now - state["checked"] < 1:
            return
        state["checked"] = now
        if state["profile"] and not state["stopped"]:
            clock()
            if Path(cfg["stop_file"]).exists():
                torch.cuda.synchronize()  # only AFTER workload/control stop
                while pending:
                    begin, end, row = pending.popleft()
                    rec.emit("gpu_completed", gpu_ms=begin.elapsed_time(end), **row)
                torch.cuda.profiler.stop()
                state["stopped"] = True
                rec.emit("capture_stopped", time=time.perf_counter())

    def wrap(cls, name, label=None):
        original = getattr(cls, name, None)
        if original is None:
            raise AttributeError(f"Required E2E hook missing: {cls.__name__}.{name}")
        label = label or f"{cls.__name__}.{name}"

        def enter(args):
            meta = dict(current.get())
            for obj in args[:3]:
                meta.update(identity(obj))
                meta.update(batch_identity(obj))
            token = current.set(meta)
            start = time.perf_counter()
            cpu_start = time.thread_time()
            if cfg["profile"]:
                torch.cuda.nvtx.range_push("e2e_cpu:" + label)
            return start, cpu_start, token, meta

        def leave(start, cpu_start, token, meta):
            end = time.perf_counter()
            cpu_s = time.thread_time()-cpu_start
            if cfg["profile"]:
                torch.cuda.nvtx.range_pop()
            current.reset(token)
            rec.emit("cpu_span", label=label, start=start, end=end,
                     thread_cpu_s=cpu_s, **meta)

        if inspect.iscoroutinefunction(original):
            # Async spans may interleave on one OS thread: no push/pop NVTX.
            @functools.wraps(original)
            async def wrapped(*args, **kw):
                meta = dict(current.get())
                for obj in args[:3]:
                    meta.update(identity(obj))
                token = current.set(meta)
                start = time.perf_counter()
                try:
                    return await original(*args, **kw)
                finally:
                    rec.emit("async_span", label=label, start=start,
                             end=time.perf_counter(), **meta)
                    current.reset(token)
        else:
            @functools.wraps(original)
            def wrapped(*args, **kw):
                if name == "process_input_requests" and len(args) > 1 and not args[1]:
                    return original(*args, **kw)
                start, cpu_start, token, meta = enter(args)
                try:
                    result = original(*args, **kw)
                    if name in ("match_prefix", "evict", "evict_for_alloc"):
                        rec.emit("cache_result", label=label, time=time.perf_counter(),
                                 **meta, **cache_result_metadata(result))
                    return result
                finally:
                    leave(start, cpu_start, token, meta)
        setattr(cls, name, wrapped)

    original_load = ModelRunner.load_model
    def load(self, *args, **kw):
        result = original_load(self, *args, **kw)
        rec.rank = self.ps.tp_rank
        if cfg["profile"]:
            torch.cuda.profiler.start()  # retain CUDA Graph node provenance
            state["profile"] = True
            for name, module in self.model.named_modules():
                category = module_category(name, module)
                if category:
                    module.forward = nvtx_wrapper(module.forward, f"component:{category}:{name}", torch)
        clock()
        rec.emit("runtime", config=cfg, time=time.perf_counter(),
                 backend=getattr(self.server_args, "attention_backend", None))
        return result
    ModelRunner.load_model = load

    original_init = ForwardBatch.init_new.__func__
    @classmethod
    def init_new(cls, batch, runner, *args, **kw):
        ret = original_init(cls, batch, runner, *args, **kw)
        ret._e2e_requests = [dict(identity(r), full_len=len(r.origin_input_ids),
                                 output_len=len(r.output_ids)) for r in batch.reqs]
        return ret
    ForwardBatch.init_new = init_new

    original_forward = ModelRunner.forward
    def forward(self, batch, *args, **kw):
        flush()
        step = state["step"]
        state["step"] += 1
        seq = getattr(batch, "seq_lens_cpu", None)
        row = dict(step=step, mode=str(batch.forward_mode), batch_size=int(batch.batch_size),
                   requests=getattr(batch, "_e2e_requests", []),
                   seq_lens=seq.tolist() if seq is not None else None,
                   extend_tokens=getattr(batch, "extend_num_tokens", None))
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        row["start"] = time.perf_counter()
        if cfg["profile"]:
            torch.cuda.nvtx.range_push(f"e2e_step:{step}")
        try:
            begin.record()
            result = original_forward(self, batch, *args, **kw)
            end.record()
        finally:
            if cfg["profile"]:
                torch.cuda.nvtx.range_pop()
        row["end"] = time.perf_counter()
        row["cuda_graph"] = bool(result.can_run_graph)
        graph = getattr(self, "decode_cuda_graph_runner", None) or getattr(self, "cuda_graph_runner", None)
        row["graph_batch_size"] = getattr(graph, "bs", None) if row["cuda_graph"] else None
        rec.emit("forward", **row)
        pending.append((begin, end, dict(step=step)))
        return result
    ModelRunner.forward = forward

    original_idle = Scheduler.on_idle
    def idle(self, *args, **kw):
        flush()
        return original_idle(self, *args, **kw)
    Scheduler.on_idle = idle

    original_trace = ReqTimeStatsBase.init_trace_ctx
    def trace(self, rid, bootstrap_room, *args, **kw):
        self._e2e_identity = dict(rid=rid, bootstrap_room=bootstrap_room)
        rec.emit("request_identity", time=time.perf_counter(), **self._e2e_identity)
        return original_trace(self, rid, bootstrap_room, *args, **kw)
    ReqTimeStatsBase.init_trace_ctx = trace
    original_req_init = Req.__init__
    @functools.wraps(original_req_init)
    def req_init(self, *args, **kw):
        original_req_init(self, *args, **kw)
        meta = bind_stats_identity(self)
        rec.emit("request_identity", time=time.perf_counter(), **meta)
        rec.emit("request_stage", stage="set_scheduler_recv_time", time=time.perf_counter(),
                 timestamp=self.time_stats.scheduler_recv_time, **meta)
    Req.__init__ = req_init
    for name in ("set_wait_queue_entry_time", "set_forward_entry_time", "set_prefill_finished_time",
                 "set_completion_time", "set_prefill_bootstrap_queue_entry_time",
                 "set_prefill_transfer_queue_entry_time", "set_decode_prealloc_queue_entry_time",
                 "set_decode_transfer_queue_entry_time"):
        original = getattr(SchedulerReqTimeStats, name)
        def stamp(self, *args, _original=original, _name=name, **kw):
            result = _original(self, *args, **kw)
            rec.emit("request_stage", stage=_name, time=time.perf_counter(),
                     timestamp=getattr(self, _name[4:], None),
                     **getattr(self, "_e2e_identity", {}))
            return result
        setattr(SchedulerReqTimeStats, name, stamp)

    hooks = [(TokenizerManager, "_tokenize_one_request"), (TokenizerManager, "_send_one_request"),
             (Scheduler, "process_input_requests"), (Scheduler, "run_batch"),
             (Scheduler, "process_batch_result"), (Scheduler, "send_kv_chunk"),
             (RadixCache, "match_prefix"), (RadixCache, "evict"),
             (UnifiedRadixCache, "match_prefix"), (UnifiedRadixCache, "evict"),
             (UnifiedRadixCache, "evict_for_alloc"),
             (Req, "init_next_round_input"),
             (Req, "plan_context_prefill"), (Req, "prepare_context_recovery"),
             (ContextModelBinding, "reposition_existing"),
             (ScheduleBatch, "_prepare_context_occurrences"),
             (ModelRunner, "sample"), (MooncakeKVSender, "send"),
             (MooncakeKVManager, "_transfer_data")]
    for cls, name in hooks:
        wrap(cls, name)
    rec.emit("hooks", names=[f"{cls.__name__}.{name}" for cls, name in hooks])


def launch_flags(source, role, port, drop):
    flags = list(source["servers"][role]["argv"][2:])
    expected = {"--tp-size": "4", "--dtype": "bfloat16", "--page-size": "1",
                "--chunked-prefill-size": "8192", "--max-prefill-tokens": "8192",
                "--mem-fraction-static": "0.85", "--context-length": "131072"}
    for key, value in expected.items():
        if key not in flags or flags[flags.index(key)+1] != value:
            raise ValueError(f"Unexpected launch setting {key}")
    if "--disable-overlap-schedule" in flags or "--attention-backend" in flags:
        raise ValueError("Preserve native backend and overlap")
    if role == 1 and "--disaggregation-decode-enable-radix-cache" not in flags:
        raise ValueError("D radix required")
    for key, value in (("--port", port+role), ("--nccl-port", port+20+role),
                       ("--disaggregation-bootstrap-port", port+10)):
        flags[flags.index(key)+1] = str(value)
    flags = [f for f in flags if f != "--context-drop-aware-eviction"]
    if drop:
        flags.append("--context-drop-aware-eviction")
    return flags


def run_client(args):
    import test_serving as serving
    original_load = serving.load_method
    def load(root):
        method = original_load(root)
        original_cases = method.load_cases
        def cases(*a, **kw):
            result, manifest = original_cases(*a, **kw)
            if args.pilot_turns:
                for case in result:
                    case["turns"] = case["turns"][:args.pilot_turns]
            return result, manifest
        method.load_cases = cases
        return method
    serving.load_method = load
    parsed = serving.parser().parse_args(json.loads(Path(args.client_config).read_text()))
    asyncio.run(serving.run(parsed))


def run_one(args):
    from run_pd_matrix import GPUIsolationGuard, gpu_free, bounded_warmup
    from types import SimpleNamespace
    repo = Path(args.server_repo).resolve()
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != FROZEN_HEAD:
        raise ValueError("Frozen runtime version mismatch")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
        raise ValueError("Frozen runtime dirty")
    if not gpu_free("0,1,2,3,4,5,6,7"):
        raise RuntimeError("GPUs occupied; no unrelated process stopped")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    trace_root = root
    if args.trace_root:
        trace_root = Path(args.trace_root).resolve()
        trace_root.mkdir(parents=True, exist_ok=False)
    if trace_root != root:
        # Logs/SSE are experiment data. JIT/model/IPC remain on local storage.
        for name in ("timing", "workload"):
            target = trace_root / name
            if name == "timing":
                target.mkdir()
            (root/name).symlink_to(target, target_is_directory=True)
    guard = GPUIsolationGuard(root, "0,1,2,3,4,5,6,7")
    source = json.loads(Path(args.source_launch).read_text())
    cache_seed = Path(args.triton_cache_seed).resolve() if args.triton_cache_seed else None
    if cache_seed:
        seed_launch = json.loads((cache_seed/"launch.json").read_text())
        if seed_launch["runtime_head"] != FROZEN_HEAD:
            raise ValueError("Compiled cache seed runtime mismatch")
    procs, logs = [], []
    client = None
    success = False
    try:
        launches = []
        for i, role in enumerate(("prefill", "decode")):
            cfg = dict(role=role, profile=args.profile, timing=str(root/"timing"),
                       stop_file=str(root/"stop_capture"), plan=PLAN,
                       ipc_tmp=tempfile.mkdtemp(prefix="e2e-", dir="/tmp"))
            save(root/f"{role}-config.json", cfg)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3" if i == 0 else "4,5,6,7",
                       PYTHONPATH=str(repo/"python"), TORCHELASTIC_USE_AGENT_STORE="False",
                       SGLANG_DISAGGREGATION_WAITING_TIMEOUT="7200", SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="7200",
                       SGLANG_TIMEOUT_KEEP_ALIVE="60", SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="0",
                       MC_INTRANODE_NVLINK="1", MOONCAKE_PROTOCOL="nvlink_intra",
                       SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK")
            env.pop("MC_FORCE_TCP", None)
            for key in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR",
                        "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
                path = root/role/key.lower()
                path.mkdir(parents=True)
                env[key] = str(path)
                if cache_seed and key == "TRITON_CACHE_DIR":
                    manifest = seed_triton_cache(cache_seed/role/key.lower(), path)
                    save(root/f"{role}-triton-cache-seed.json", manifest)
            env["TMPDIR"] = cfg["ipc_tmp"]
            cmd = [sys.executable, str(Path(__file__).resolve()), "server", "--config", str(root/f"{role}-config.json"),
                   "--", *launch_flags(source, i, args.port, args.drop)]
            if args.profile:
                # Nsight's bulk temporary trace data can be large. The server
                # restores a short local TMPDIR before spawning IPC workers.
                profile_tmp = trace_root / (role+"-nsys-tmp")
                profile_tmp.mkdir()
                env["TMPDIR"] = str(profile_tmp)
                cmd = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                       "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "--output="+str(trace_root/f"{role}-profile"), *cmd]
            log = (root/f"{role}.log").open("x")
            logs.append(log)
            proc = subprocess.Popen(cmd, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            launches.append(dict(role=role, argv=cmd, pid=proc.pid,
                                 environment={k:v for k,v in env.items() if k.startswith(("CUDA", "SGLANG", "MC_", "MOONCAKE", "PYTHONPATH", "TMPDIR"))}))
        save(root/"launch.json", dict(plan=PLAN, args=vars(args), runtime_head=FROZEN_HEAD, launches=launches))
        for i in (0, 1):
            deadline = time.monotonic()+1500
            while True:
                guard.check()
                if any(p.poll() is not None for p in procs) or time.monotonic()>deadline:
                    raise RuntimeError("Server startup failed; inspect retained logs")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                time.sleep(1)
        guard.check(pin=True)
        flags = launch_flags(source, 0, args.port, args.drop)
        model = flags[flags.index("--model-path")+1]
        bounded_warmup(SimpleNamespace(warmup_wall_seconds=60, drop=args.drop, concurrency=args.concurrency,
                                      port=args.port, warmup_timeout=600, unique_cohort=True), root, model)
        for i, role in enumerate(("prefill", "decode")):
            deadline = time.monotonic()+120
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/flush_cache", timeout=10) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic()>deadline:
                        raise
                    time.sleep(1)
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port+i}/server_info", timeout=30) as response:
                save(root/f"{role}-server-info.json", json.load(response))
        pairs = {"mini-root": args.mini_root, "requests-path": args.requests_path,
                 "output-dir": str((root/"workload").resolve()), "model": model, "tokenizer": model,
                 "url": f"http://127.0.0.1:{args.port+1}/v1/chat/completions",
                 "prefill-url": f"http://127.0.0.1:{args.port}/v1/chat/completions",
                 "bootstrap-port": args.port+10, "chat-template": flags[flags.index("--chat-template")+1],
                 "concurrency": args.concurrency, "num-tasks": args.tasks, "source-tasks": args.tasks,
                 "seed":42, "model-context-limit":131072, "timeout":21600}
        client_flags = [a for k,v in pairs.items() for a in ("--"+k, str(v))]
        client_flags += ["--unique-cohort", "--raw-sse", "--no-filler", "--e2e-timing"]
        if args.drop:
            client_flags.append("--drop")
        save(root/"client-config.json", client_flags)
        cmd = [sys.executable, str(Path(__file__).resolve()), "client", "--client-config", str(root/"client-config.json"),
               "--pilot-turns", str(args.pilot_turns)]
        with (root/"client.log").open("x") as log:
            client = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(root/"state.json", dict(status="workload_running", pid=client.pid, time=time.time()))
            while client.poll() is None:
                guard.check()
                if any(p.poll() is not None for p in procs):
                    raise RuntimeError("Server exited during workload")
                for storage in (root, trace_root):
                    stat = os.statvfs(storage)
                    if stat.f_bavail * stat.f_frsize < 5*1024**3:
                        raise RuntimeError(f"Capture filesystem below 5 GiB free: {storage}")
                time.sleep(5)
        if client.returncode:
            raise RuntimeError("Workload failed; retain all partial tokens/events")
        success = True
    finally:
        if client is not None and client.poll() is None:
            os.killpg(client.pid, signal.SIGTERM)
            client.wait(timeout=30)
        (root/"stop_capture").touch()
        # Let idle TP ranks stop capture and drain logs before terminating servers.
        time.sleep(5)
        for proc in procs:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in procs:
            try:
                # Long complete traces may take minutes to assemble after
                # cudaProfilerStop; do not truncate them with a short grace.
                proc.wait(timeout=1800 if args.profile else 120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()
        save(root/"outcome.json", dict(workload_complete=success, attribution_validated=False, time=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    server = sub.add_parser("server")
    server.add_argument("--config", required=True)
    client = sub.add_parser("client")
    client.add_argument("--client-config", required=True)
    client.add_argument("--pilot-turns", type=int, default=0)
    run = sub.add_parser("run")
    for name in ("output", "server-repo", "source-launch", "mini-root", "requests-path", "nsys"):
        run.add_argument("--"+name, required=True)
    run.add_argument("--port", type=int, default=46101)
    run.add_argument("--concurrency", type=int, choices=[8,10], required=True)
    run.add_argument("--tasks", type=int, default=30)
    run.add_argument("--pilot-turns", type=int, default=0)
    run.add_argument("--drop", action="store_true")
    run.add_argument("--profile", action="store_true")
    run.add_argument("--trace-root", help="Unique directory for bulk Nsight data; IPC/JIT stay local")
    run.add_argument("--triton-cache-seed", help="Completed run whose compiled Triton artifacts seed independent caches; never KV state")
    args, extra = p.parse_known_args()
    if args.command != "server" and extra:
        p.error(str(extra))
    if args.command == "run":
        if args.tasks < args.concurrency:
            p.error("Fixed task cohort must contain at least C tasks")
        run_one(args)
    elif args.command == "client":
        run_client(args)
    else:
        os.environ["PD_E2E_CONFIG"] = str(Path(args.config).resolve())
        cfg = json.loads(Path(args.config).read_text())
        if cfg.get("ipc_tmp"):
            os.environ["TMPDIR"] = cfg["ipc_tmp"]
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


if __name__ == "__mp_main__" and os.environ.get("PD_E2E_CONFIG"):
    install()
if __name__ == "__main__":
    main()
