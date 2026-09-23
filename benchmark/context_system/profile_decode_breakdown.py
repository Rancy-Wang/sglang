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
import sys
import time

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
    cursors, token_tables = {}, {}
    state = dict(rank=None, log=None, cohort_started=False, first_wait=None,
                 profiling_started=False, profiling_stopped=False)

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
                        async for line in response.content:
                            text = line.decode().strip()
                            if not text:
                                continue
                            log.write(json.dumps(dict(time=time.perf_counter(), data=text)) + "\n")
                            if text == "data: [DONE]":
                                done = True
                        if not done:
                            raise RuntimeError("Decode response ended without DONE")
            await asyncio.gather(prefill(), decode())
            return dict(rid=payload["rid"], lifecycle_s=time.perf_counter() - started)
        result = await asyncio.gather(*(send(i, p) for i, p in enumerate(payloads)))
    (root / "completed.json").write_text(json.dumps(result, indent=2))


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
    args, extra = p.parse_known_args()
    if args.command == "client":
        if extra:
            p.error(f"Unknown arguments: {extra}")
        import asyncio
        asyncio.run(client(args))
    else:
        os.environ["DECODE_BREAKDOWN_CONFIG"] = str(Path(args.config).resolve())
        install()
        from sglang.launch_server import run_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree
        try:
            run_server(prepare_server_args(extra[1:] if extra[:1] == ["--"] else extra))
        finally:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__mp_main__" and os.environ.get("DECODE_BREAKDOWN_CONFIG"):
    install()
if __name__ == "__main__":
    main()
