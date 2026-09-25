"""Whole BCP trajectories using the pinned mini scheduler; HTTP inference only.

No SLO or logits probes. Native SGLang templates and parsers remain in use.
The mini checkout supplies the exact dataset, scheduling and metric algorithms
from the approved test_serving method, not model or inference implementations.
"""

import argparse
import asyncio
import copy
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from serving_cohort import COMPLETED, Journal, UniqueCohort, context_stop

MINI_HEAD = "fb248835b908da925d14f607748536483e5fac54"


def load_method(root):
    root = Path(root).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != MINI_HEAD:
        raise ValueError(f"Benchmark reference changed: {head} != {MINI_HEAD}")
    paths = [
        "tests/benchmark/test_serving.py",
        "tests/benchmark/test_throughput.py",
        "tests/benchmark/throughput_compute.py",
    ]
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", *paths]):
        raise ValueError("Benchmark reference has uncommitted modifications")
    sys.path.insert(0, str(root / "tests/benchmark"))
    spec = importlib.util.spec_from_file_location("mini_bcp_method", root / paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_metrics = module.calculate_metrics
    module.calculate_metrics = lambda rows, duration: optional_ttft_metrics(
        original_metrics, rows, duration
    )
    return module


def optional_ttft_metrics(original, rows, duration):
    """Count completed invisible output without inventing a text arrival time."""
    missing = [r for r in rows if r["success"] and r["ttft"] is None]
    if not missing:
        return original(rows, duration)
    import numpy as np

    # The pinned metric implementation requires TTFT for every successful row.
    # Keep its visible-token timing statistics, then include all completed work.
    visible = [dict(r, success=False) if r["success"] and r["ttft"] is None else r for r in rows]
    metrics, _ = original(visible, duration)
    good = [r for r in rows if r["success"]]
    lengths = [r["output_len"] if r["success"] else 0 for r in rows]
    prompt = sum(r["prompt_len"] for r in good)
    output = sum(lengths)
    retokenized = sum(r["retokenized_len"] for r in good)
    metrics.update(completed=len(good), total_input=prompt, total_input_text=prompt,
        total_output=output, total_output_retokenized=retokenized,
        request_throughput=len(good) / duration, input_throughput=prompt / duration,
        output_throughput=output / duration, output_throughput_retokenized=retokenized / duration,
        total_throughput=(prompt + output) / duration,
        total_throughput_retokenized=(prompt + retokenized) / duration,
        concurrency=sum(r["latency"] for r in good) / duration,
        missing_ttft_requests=len(missing),
        max_output_tokens_per_s=None, max_concurrent_requests=None)
    values = [r["latency"] for r in good]
    funcs = {"mean": np.mean, "median": np.median, "std": np.std,
             **{f"p{p}": lambda v, p=p: np.percentile(v, p) for p in (90, 95, 99)}}
    for stat, function in funcs.items():
        metrics[f"{stat}_e2e_latency_ms"] = float(function(values) * 1000)
    if len(missing) == len(good):
        for key in metrics:
            if key.endswith(("_ttft_ms", "_tpot_ms")):
                metrics[key] = None
    return metrics, lengths


def accept_complete_invisible_output(row, payload):
    """Only relax the pinned client's visible-text requirement, never completion."""
    usage = row.get("usage") or {}
    assistant = row.get("assistant") or {}
    budget = payload.get("max_tokens")
    if (row.get("status") == "incomplete" and not row.get("success")
            and row.get("error") == "Missing DONE, usage, content or normal finish"
            and row.get("http_status") == 200 and row.get("done")
            and row.get("raw_done_time") is not None
            and row.get("finish_reason") == "length" and row.get("ttft") is None
            and type(budget) is int and budget > 0
            and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0
            and usage.get("completion_tokens") == row.get("output_len") == row.get("generated_tokens") == budget
            and row.get("last_received_usage") == usage
            and not any(assistant.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls"))):
        row.update(success=True, status="success", error=None,
                   completion_without_visible_delta=True)


class NativeTemplateAdapter:
    """Use the server's preprocessing method, retaining its canonical trace."""

    def __init__(self, path, template_kwargs, chat_template=None):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.parser.jinja_template_utils import (
            detect_jinja_template_content_format,
        )
        from sglang.srt.parser.template_manager import TemplateManager
        from transformers import AutoTokenizer

        class Renderer(OpenAIServingChat):
            def _render_and_encode_chat_template(self, messages, **kwargs):
                from sglang.srt.context_system.provenance import (
                    build_template_token_provenance,
                )

                self.trace = build_template_token_provenance(
                    self.tokenizer_manager.tokenizer,
                    messages,
                    tools=kwargs["tools"],
                    add_generation_prompt=True,
                    enable_thinking=None,
                    template_kwargs=kwargs["template_kwargs"],
                )
                return self.trace.rendered_text, self.trace.input_ids, None

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if chat_template is not None:
            tokenizer.chat_template = Path(chat_template).read_text()
        template = TemplateManager()
        template._jinja_template_content_format = detect_jinja_template_content_format(
            tokenizer.chat_template
        )
        template._run_template_detection(tokenizer.chat_template, tokenizer)
        self.renderer = Renderer.__new__(Renderer)
        self.renderer.tokenizer_manager = SimpleNamespace(
            tokenizer=tokenizer, served_model_name=path
        )
        self.renderer.template_manager = template
        self.renderer.chat_encoding_spec = None
        self.renderer._tokenizer_auto_adds_specials = bool(tokenizer.encode(""))
        self.template_kwargs = template_kwargs

    def render(self, messages, tools):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            messages=messages, tools=tools, chat_template_kwargs=self.template_kwargs
        )
        # Reordering would change semantic message IDs. Reject this benchmark
        # configuration rather than silently schedule Drop against another ID.
        if self.renderer.template_manager.jinja_template_may_reorder_tool_results:
            original = [message.model_dump() for message in request.messages]
            if self.renderer._canonicalize_tool_message_order(original) != original:
                raise ValueError("Benchmark history requires tool-result reordering")
        result = self.renderer._apply_jinja_template(
            request, [tool.model_dump() for tool in request.tools], False
        )
        return len(result.prompt_ids), self.renderer.trace.owners


def compute_metrics(event):
    """Translate only reported physical counters, never logical usage."""
    choices = (event.get("sglext") or {}).get("context_usage")
    if not choices:
        return None
    if set(choices) != {"0"}:
        raise ValueError("Benchmark requires exactly one Context choice")
    usage = choices["0"]
    return {
        "prefill_compute_tokens": usage["actual_prefill_tokens"],
        "decode_compute_tokens": usage["actual_decode_tokens"],
        "context_stage_count": 1,
        "cached_tokens": usage["cached_tokens"],
        "drop_skipped_tokens": usage["drop_skipped_tokens"],
        "repos_tokens": usage["repos_tokens"],
    }


class Transport:
    """Adapt final SSE counters; optional native paired P/D requests."""

    def __init__(self, session, method, prefill_url=None, bootstrap_port=None, emit=None):
        self.session, self.method = session, method
        self.emit = emit
        self.prefill_url, self.bootstrap_port = prefill_url, bootstrap_port
        self.room = time.time_ns() % (1 << 52)

    async def request(self, url, payload, identity=None):
        # Per-request state: concurrent turns must never share a DONE timestamp.
        timing = {"identity": identity or {}}
        adapter = SimpleNamespace(
            post=lambda url, **kwargs: self.post(url, _timing=timing, **kwargs)
        )
        row = await self.method.request(adapter, url, payload)
        row["pd_bootstrap_room"] = timing.get("room")
        row["server_metrics"] = dict(row.get("server_metrics") or {})
        if timing.get("room") is not None:
            row["server_metrics"]["pd_bootstrap_room"] = timing["room"]
        row["last_received_usage"] = timing.get("usage")
        row["usage_received_perf"] = timing.get("usage_perf")
        row["lifecycle_latency_s"] = row["latency"]
        row["raw_done_time"] = timing.get("raw_done_time")
        accept_complete_invisible_output(row, payload)
        row["latency"] = (
            row["raw_done_time"] - row["start_time"]
            if row["raw_done_time"] is not None else None
        )
        row["cleanup_time_s"] = (
            row["end_time"] - row["raw_done_time"]
            if row["raw_done_time"] is not None else None
        )
        row["tpot_s"] = (
            (row["latency"] - row["ttft"]) / (row["output_len"] - 1)
            if row["success"] and row["latency"] is not None
            and row["ttft"] is not None and row["output_len"] > 1 else None
        )
        # end_time remains the lifecycle boundary for real throughput windows.
        row["latency_boundary"] = "client_received_decode_done"
        return row

    @asynccontextmanager
    async def post(self, url, _timing=None, **kwargs):
        body = copy.deepcopy(kwargs["json"])
        def record(kind, **values):
            if self.emit:
                self.emit(dict(kind=kind, **((_timing or {}).get("identity") or {}),
                               received_perf=time.perf_counter(), received_wall=time.time(), **values))
        prefill = None
        owner = asyncio.current_task()
        interrupted_by_prefill = False
        active = True

        def prefill_finished(task):
            nonlocal interrupted_by_prefill
            if active and not task.cancelled() and task.exception() is not None:
                interrupted_by_prefill = True
                owner.cancel()

        if self.prefill_url:
            self.room += 1
            body.update(
                bootstrap_host="127.0.0.1",
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=self.room,
            )

            if _timing is not None:
                _timing["room"] = body["bootstrap_room"]

            async def send_prefill():
                record("prefill_request_start", room=body["bootstrap_room"])
                async with self.session.post(
                    self.prefill_url, json={**body, "stream": False}
                ) as response:
                    value = await response.json()
                    record("prefill_response", room=body["bootstrap_room"], http_status=response.status, response=value)
                    if response.status != 200:
                        raise RuntimeError(f"P HTTP {response.status}: {value}")
                    return value

            prefill = asyncio.create_task(send_prefill())
            # P may fail before D sends headers or a single SSE event. Wake
            # that blocked request once; add no task/race per decoded token.
            prefill.add_done_callback(prefill_finished)
        try:
            record("request_payload", url=url, payload=body)
            async with self.session.post(url, json=body) as response:

                async def content():
                    reported_metrics = None
                    sequence = 0
                    async for data in self.method.sse_events(response.content):
                        sequence += 1
                        record("raw_sse", sequence=sequence, room=body.get("bootstrap_room"), data=data)
                        if data == "[DONE]":
                            if _timing is not None:
                                _timing.setdefault("raw_done_time", time.perf_counter())
                            if prefill:
                                await prefill
                        else:
                            event = json.loads(data)
                            if _timing is not None and event.get("usage") is not None:
                                _timing["usage"] = copy.deepcopy(event["usage"])
                                _timing["usage_perf"] = time.perf_counter()
                            metrics = compute_metrics(event)
                            if metrics:
                                reported_metrics = metrics
                            if prefill is not None and event.get("usage"):
                                # Use this request's room, not the shared
                                # counter, which advances for concurrent turns.
                                metrics = dict(reported_metrics or {})
                                metrics["pd_bootstrap_room"] = body["bootstrap_room"]
                            if metrics:
                                # Core request() defaults generated_tokens from
                                # final usage when this key is absent.
                                event["server_metrics"] = metrics
                                data = json.dumps(event)
                        yield ("data: " + data + "\n").encode()
                        yield b"\n"

                yield SimpleNamespace(
                    status=response.status, content=content(), text=response.text
                )
        except asyncio.CancelledError:
            if interrupted_by_prefill:
                owner.uncancel()
                raise prefill.exception()
            raise
        finally:
            active = False
            if prefill is not None:
                prefill.remove_done_callback(prefill_finished)
                if not prefill.done():
                    prefill.cancel()
                await asyncio.gather(prefill, return_exceptions=True)


def check_context_budget(prompt_tokens, output_tokens, limit):
    if limit is not None and prompt_tokens + output_tokens > limit:
        raise ValueError(
            f"model_context_limit: prompt={prompt_tokens} output={output_tokens} limit={limit}"
        )


async def run(args):
    import aiohttp

    method = load_method(args.mini_root)
    if args.summary_policy_dir:
        from summary_replay import load_cases as load_summary
        if args.drop or not args.unique_cohort:
            raise ValueError("Summary replay requires unique cohort and no Drop")
        cases, manifest = load_summary(args.requests_path, args.source_tasks or args.num_tasks,
                                       args.seed, args.case_id, args.summary_policy_dir)
    else:
        cases, manifest = method.load_cases(
            args.requests_path, args.source_tasks or args.num_tasks, args.seed, args.case_id
        )
    renderer = NativeTemplateAdapter(
        args.tokenizer, json.loads(args.template_kwargs), args.chat_template
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if args.unique_cohort:
        method.write_json(root / "selection.json", dict(seed=args.seed, primary_count=args.num_tasks,
                          source_order=[case["case_id"] for case in cases]))
    rows = []
    user_turns = []
    writes = []
    raw_journal = Journal(root) if args.raw_sse else None
    loop = asyncio.get_running_loop()
    with (
        (raw_journal if raw_journal else nullcontext()),
        (root / ("journal-unused.log" if raw_journal else "events.jsonl")).open("w") as journal,
        ThreadPoolExecutor(max_workers=1) as rendering,
        ThreadPoolExecutor(max_workers=1) as writer,
    ):

        def emit(event):
            # Match mini's FIFO writer: snapshot in the event loop, serialize
            # and flush outside timed scheduling/network work.
            if raw_journal is not None:
                raw_journal.emit(event)
                return
            frozen = copy.deepcopy(event)

            def save():
                journal.write(json.dumps(frozen, ensure_ascii=False) + "\n")
                journal.flush()

            writes.append(writer.submit(save))

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.timeout),
            connector=aiohttp.TCPConnector(
                limit=args.concurrency * (2 if args.prefill_url else 1)
            ),
        ) as session:
            transport = Transport(
                session, method, args.prefill_url, args.bootstrap_port, emit=emit if args.raw_sse else None
            )

            async def execute(case, instance):
                if args.summary_policy_dir:
                    from summary_replay import execute_case
                    return await execute_case(case, instance, args=args, renderer=renderer,
                        rendering=rendering, transport=transport, emit=emit, rows=rows, user_turns=user_turns)
                history, rolling = [], method.RollingState(keep=12, threshold=96 * 1024)
                for turn in case["turns"]:
                    prepare_start = time.perf_counter()
                    history.extend(copy.deepcopy(turn["new_messages"]))
                    full, owners = await loop.run_in_executor(
                        rendering, renderer.render, history, manifest["tools"]
                    )
                    state = rolling.extend(history, owners, full) if args.drop else {}
                    if args.unique_cohort:
                        reason = context_stop(full, turn["max_new_tokens"], state, args.model_context_limit)
                        if reason:
                            emit(dict(kind="context_limit", case_id=case["case_id"], instance=instance["instance"],
                                      turn=turn["turn"], reason=reason, full_tokens=full, **state))
                            return reason
                    else:
                        check_context_budget(full, turn["max_new_tokens"], args.model_context_limit)
                    payload = {
                        "model": args.model,
                        "messages": history,
                        "tools": manifest["tools"],
                        "max_tokens": turn["max_new_tokens"],
                        "ignore_eos": True,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "chat_template_kwargs": json.loads(args.template_kwargs),
                    }
                    if args.drop:
                        payload.update(
                            drop_message=state["drop_message"],
                            reposition=state["reposition"],
                        )
                    identity = dict(case_id=case["case_id"], instance=instance["instance"], turn=turn["turn"], filler=instance["filler"])
                    if getattr(args, "e2e_timing", False):
                        emit(dict(kind="client_prepare", **identity, start=prepare_start,
                                  end=time.perf_counter(), wall=time.time()))
                    emit(dict(kind="turn_start", **identity, time=time.perf_counter(), full_tokens=full,
                              active_tokens=state.get("active_tokens", full), position_tokens=state.get("position_tokens", full),
                              max_new_tokens=turn["max_new_tokens"]))
                    row = await transport.request(args.url, payload, identity)
                    if row["success"] and row["prompt_len"] != full:
                        row.update(
                            success=False,
                            status="template_mismatch",
                            error="Client/server native template mismatch",
                        )
                    row.update(
                        case_id=case["case_id"],
                        trial=case["trial"],
                        turn=turn["turn"],
                        instance=instance["instance"],
                        filler=instance["filler"],
                        requested_max_tokens=turn["max_new_tokens"],
                        expected_prompt_tokens=full,
                        history_sha256=method.digest(history),
                        drop_state=state,
                    )
                    rows.append(row)
                    emit(dict(kind="turn_end", **row))
                    if not row["success"]:
                        return row["status"]
                    history.append(copy.deepcopy(row["assistant"]))
                return "all_turns_completed"

            scheduler = (UniqueCohort(cases if args.filler else cases[:args.num_tasks], args.concurrency, args.num_tasks, execute, emit,
                max_seconds=args.measurement_seconds, stop_file=root / "STOP.json",
                source_terminal_ok=bool(args.summary_policy_dir))
                         if args.unique_cohort else method.Scheduler(cases, args.concurrency, execute, emit, filler=args.filler))
            await scheduler.run()
    for future in writes:
        future.result()
    count = sum(
        item["instance"]["status"] in (COMPLETED | {"source_terminal_replayed"})
        for item in scheduler.completed
    )
    result = {
        "valid": (count == args.num_tasks or getattr(scheduler, "stop_reason", None) in
                  ("measurement_time_limit", "cache_pressure_after_1h")) and not getattr(scheduler, "failure", None),
        "stop_reason": getattr(scheduler, "stop_reason", None),
        "logical_user_turns": user_turns,
        "source_terminal_tasks": sum(t["status"] == "source_terminal_replayed" for t in scheduler.instances),
        "cohort_failure": getattr(scheduler, "failure", None),
        "successful_tasks": count,
        "args": vars(args),
        "mini_method_head": MINI_HEAD,
        "benchmark_head": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], text=True).strip(),
        "overall": method.summary(rows, scheduler.start, scheduler.cutoff),
        "user_e2e_latency_s": method.stats([r["latency"] for r in rows if r["success"]]),
        "cleanup_time_s": method.stats([r["cleanup_time_s"] for r in rows if r["success"]]),
        "client_tpot_s": method.stats([r["tpot_s"] for r in rows if r["tpot_s"] is not None]),
        "turns": rows,
        "tasks": scheduler.instances,
        "round_ends": scheduler.round_ends,
        "dataset": method.digest(manifest),
        "start": scheduler.start,
        "cutoff": scheduler.cutoff,
        "notes": [
            "No SLO. No logits instrumentation.",
            "Actual compute is null when any completed request lacks physical counters.",
            "Native generated assistant history; original tool responses are replayed.",
            "Legacy overall uses completed turns; counted-result accounting also reports known partial usage and forward work.",
            "User E2E/TPOT end at raw D DONE receipt; cleanup is reported separately.",
            "Throughput and task lifetimes retain real wall time, including P confirmation.",
        ],
    }
    method.write_json(root / "result.json", result)
    print(
        json.dumps({"valid": result["valid"], "overall": result["overall"]}), flush=True
    )
    if not result["valid"]:
        raise RuntimeError(f"Incomplete trajectories: {root / 'result.json'}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--requests-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--url", default="http://127.0.0.1:28761/v1/chat/completions")
    p.add_argument("--prefill-url", help="Native P URL; --url is the native D URL")
    p.add_argument("--bootstrap-port", type=int, default=28971)
    p.add_argument("--template-kwargs", default="{}")
    p.add_argument("--chat-template")
    p.add_argument("--concurrency", type=int, choices=[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 32], default=1)
    p.add_argument("--num-tasks", type=int, choices=range(1, 161), default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-tasks", type=int)
    p.add_argument("--unique-cohort", action="store_true")
    p.add_argument("--raw-sse", action="store_true")
    p.add_argument("--e2e-timing", action="store_true", help="Record client preparation outside user latency")
    p.add_argument("--case-id", action="append")
    p.add_argument("--summary-policy-dir", help="Hashed original pi compaction helpers for full summary replay")
    p.add_argument("--summary-smoke", action="store_true", help="Only first summary event from source prefix; never formal")
    p.add_argument("--measurement-seconds", type=float, help="Graceful cutoff preserving partial SSE and usage")
    p.add_argument("--drop", action="store_true")
    p.add_argument("--filler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model-context-limit", type=int)
    p.add_argument("--timeout", type=float, default=7200)
    return p


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
