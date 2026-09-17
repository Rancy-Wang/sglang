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
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

MINI_HEAD = "2966eb49a522041f9c42bce7dca07119ef6929de"


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
    return module


class NativeTemplateAdapter:
    """Use the server's preprocessing method, retaining its canonical trace."""

    def __init__(self, path, template_kwargs):
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

    def __init__(self, session, method, prefill_url=None, bootstrap_port=None):
        self.session, self.method = session, method
        self.prefill_url, self.bootstrap_port = prefill_url, bootstrap_port
        self.room = time.time_ns() % (1 << 52)

    @asynccontextmanager
    async def post(self, url, **kwargs):
        body = copy.deepcopy(kwargs["json"])
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

            async def send_prefill():
                async with self.session.post(
                    self.prefill_url, json={**body, "stream": False}
                ) as response:
                    value = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"P HTTP {response.status}: {value}")
                    return value

            prefill = asyncio.create_task(send_prefill())
            # P may fail before D sends headers or a single SSE event. Wake
            # that blocked request once; add no task/race per decoded token.
            prefill.add_done_callback(prefill_finished)
        try:
            async with self.session.post(url, json=body) as response:

                async def content():
                    async for data in self.method.sse_events(response.content):
                        if data == "[DONE]":
                            if prefill:
                                await prefill
                        else:
                            event = json.loads(data)
                            metrics = compute_metrics(event)
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


async def run(args):
    import aiohttp

    method = load_method(args.mini_root)
    cases, manifest = method.load_cases(
        args.requests_path, args.num_tasks, args.seed, args.case_id
    )
    renderer = NativeTemplateAdapter(args.tokenizer, json.loads(args.template_kwargs))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    writes = []
    loop = asyncio.get_running_loop()
    with (
        (root / "events.jsonl").open("w") as journal,
        ThreadPoolExecutor(max_workers=1) as rendering,
        ThreadPoolExecutor(max_workers=1) as writer,
    ):

        def emit(event):
            # Match mini's FIFO writer: snapshot in the event loop, serialize
            # and flush outside timed scheduling/network work.
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
                session, method, args.prefill_url, args.bootstrap_port
            )

            async def execute(case, instance):
                history, rolling = [], method.RollingState(keep=12, threshold=96 * 1024)
                for turn in case["turns"]:
                    history.extend(copy.deepcopy(turn["new_messages"]))
                    full, owners = await loop.run_in_executor(
                        rendering, renderer.render, history, manifest["tools"]
                    )
                    state = rolling.extend(history, owners, full) if args.drop else {}
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
                    row = await method.request(transport, args.url, payload)
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

            scheduler = method.Scheduler(cases, args.concurrency, execute, emit)
            await scheduler.run()
    for future in writes:
        future.result()
    count = sum(
        item["instance"]["status"] == "all_turns_completed"
        for item in scheduler.completed
    )
    result = {
        "valid": count == len(cases),
        "successful_tasks": count,
        "args": vars(args),
        "mini_method_head": MINI_HEAD,
        "overall": method.summary(rows, scheduler.start, scheduler.cutoff),
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
            "Completed-request accounting; cancelled filler compute is not counted.",
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
    p.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    p.add_argument("--num-tasks", type=int, choices=[2, 4], default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--case-id", action="append")
    p.add_argument("--drop", action="store_true")
    p.add_argument("--timeout", type=float, default=7200)
    return p


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
