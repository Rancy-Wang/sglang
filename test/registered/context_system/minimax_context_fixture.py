"""MiniMax HTTP acceptance helpers, independent of single-GPU model fixtures.

Servers are supplied explicitly and may use TP or separate hosts. No helper
starts/stops unrelated servers or flushes a shared cache. Responses and Context
usage are kept verbatim for review. This is not a substitute for kernel oracles.
"""

import concurrent.futures
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Read a key",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]


def output_findings(message):
    """Flag structural errors and degenerate repeats; preserve text for review."""
    findings = []
    for field in ("content", "reasoning_content"):
        value = message.get(field) or ""
        if not isinstance(value, str):
            findings.append(field + ":non_string")
            continue
        if "\ufffd" in value or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            findings.append(field + ":invalid_unicode")
        if any(c in value for c in ("\x00", "]~b]", "[e~[", "<minimax:tool_call>")):
            findings.append(field + ":leaked_protocol")
        # Long repeated units (not indentation, punctuation or repeated digits).
        if re.search(r"(.{32,512}?)\1{5,}", value, re.DOTALL):
            findings.append(field + ":repeated_block")
    for call in message.get("tool_calls") or []:
        try:
            assert call["type"] == "function"
            assert isinstance(call["id"], str) and call["id"]
            assert call["function"]["name"]
            assert isinstance(json.loads(call["function"]["arguments"]), dict)
        except (KeyError, TypeError, ValueError, AssertionError):
            findings.append("invalid_tool_call")
    return findings


class Endpoint:
    def __init__(
        self,
        *,
        decode,
        output,
        prefill=None,
        bootstrap_host="127.0.0.1",
        bootstrap_port=None,
    ):
        self.decode, self.prefill = decode.rstrip("/"), (
            prefill.rstrip("/") if prefill else None
        )
        self.bootstrap_host, self.bootstrap_port = bootstrap_host, bootstrap_port
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.timeout = float(os.environ.get("MINIMAX_REQUEST_TIMEOUT", "3600"))

    def request(self, payload, label, *, diagnostic=False):
        body = copy.deepcopy(payload)
        body["stream"] = False
        body["return_meta_info"] = True
        body.setdefault("model", os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"))
        body.setdefault("temperature", 0)
        identity = f"{label}-{uuid.uuid4().hex}"
        if self.prefill:
            if self.bootstrap_port is None:
                raise ValueError("PD requires bootstrap port")
            body.update(
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=time.time_ns() % (1 << 52),
            )
        record = dict(
            request=body, started=time.time(), decode=self.decode, prefill=self.prefill
        )
        path = self.output / (identity + ".json")
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

        def post(base):
            import requests

            response = requests.post(
                base + "/v1/chat/completions", json=body, timeout=self.timeout
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:4000]}"
                )
            return response.json()

        try:
            if self.prefill:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    p = executor.submit(post, self.prefill)
                    d = executor.submit(post, self.decode)
                    result = d.result()
                    record["prefill_response"] = p.result()
            else:
                result = post(self.decode)
            record["response"] = result
            choice = result["choices"][0]
            record["findings"] = output_findings(choice["message"])
            if diagnostic:
                record["diagnostic_only"] = True
                return result
            if choice["finish_reason"] not in ("stop", "tool_calls"):
                raise AssertionError(
                    f"Not a normal model stop: {choice['finish_reason']}"
                )
            if record["findings"]:
                raise AssertionError(record["findings"])
            assert choice["message"].get("content") or choice["message"].get(
                "tool_calls"
            )
            return result
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["ended"] = time.time()
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2))


def endpoint_from_env(mode):
    output = os.environ.get("MINIMAX_ACCEPTANCE_OUTPUT")
    if not output:
        raise ValueError("MINIMAX_ACCEPTANCE_OUTPUT is required")
    if mode == "ordinary":
        return Endpoint(
            decode=os.environ["MINIMAX_SERVER_URL"], output=Path(output) / mode
        )
    return Endpoint(
        decode=os.environ["MINIMAX_DECODE_URL"],
        prefill=os.environ["MINIMAX_PREFILL_URL"],
        output=Path(output) / mode,
        bootstrap_host=os.environ.get("MINIMAX_BOOTSTRAP_HOST", "127.0.0.1"),
        bootstrap_port=int(os.environ["MINIMAX_BOOTSTRAP_PORT"]),
    )


def tool_history():
    messages = [
        {"role": "system", "content": "Answer briefly using the latest lookup result."},
        {"role": "user", "content": "Look up the current marker."},
    ]
    for index in range(16):
        messages += [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": f"Read version {index}.",
                "tool_calls": [
                    {
                        "id": f"call{index}",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"key": "marker"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call{index}",
                "name": "lookup",
                "content": f"Version {index}: marker is M{index:02d}.",
            },
        ]
    messages.append(
        {
            "role": "user",
            "content": "What is the latest marker? Reply with the marker only.",
        }
    )
    return messages


def short_request(policy):
    payload = dict(
        messages=tool_history(),
        tools=TOOLS,
        max_tokens=2048,
        chat_template_kwargs={"preserve_thinking_history": True},
    )
    if policy in ("drop", "combined"):
        payload["drop_message"] = {
            str(3 + 2 * i): [3 + 2 * (i - 12)] for i in range(12, 16)
        }
    if policy in ("repos", "combined"):
        payload["reposition"] = [33]
    return payload


def context_usage(result):
    # Native non-stream responses expose these counters on the choice.
    choices = result.get("choices") or []
    if choices:
        usage = (choices[0].get("meta_info") or {}).get("context_usage")
        if usage is not None:
            return usage
    usage = (result.get("sglext") or {}).get("context_usage")
    if isinstance(usage, dict) and "0" in usage:
        return usage["0"]
    return usage


class RollingDrop96K:
    """Keep 12 tool responses; compact only after position reaches 96 * 1024.

    Ownership comes from one complete native template. Events occur AFTER the
    trigger message; previous request history and token boundaries must remain
    stable. Generated assistant text is always retained.
    """

    def __init__(self, tokenizer, *, provenance_builder=None):
        self.tokenizer = tokenizer
        if provenance_builder is None:
            from sglang.srt.context_system.provenance import (
                build_template_token_provenance,
            )

            provenance_builder = build_template_token_provenance
        self.provenance_builder = provenance_builder
        self.processed = 0
        self.previous = []
        self.bounds = {}
        self.tools, self.drops, self.repositions, self.checks = [], {}, [], []
        self.removed = self.compacted = 0

    def extend(self, messages, tools):
        # Match OpenAI serving's conversion of function argument JSON strings.
        rendered_messages = copy.deepcopy(messages)
        for message in rendered_messages:
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        trace = self.provenance_builder(
            self.tokenizer,
            rendered_messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=None,
            template_kwargs={"preserve_thinking_history": True},
        )
        assert (
            messages[: self.processed] == self.previous
        ), "Historical messages changed"
        counts, ends = Counter(trace.owners), {}
        for position, owner in enumerate(trace.owners):
            if owner < len(messages):
                ends[owner] = position + 1
        for owner, old in self.bounds.items():
            assert old == (
                counts[owner],
                ends.get(owner),
            ), "Historical token boundary changed"
        for index in range(self.processed, len(messages)):
            assert index in ends, f"Message {index} has no owned tokens"
            if messages[index]["role"] == "tool":
                self.tools.append(index)
                if len(self.tools) > 12:
                    old = self.tools[-13]
                    self.drops[str(index)] = [old]
                    self.removed += counts[old]
            before = ends[index] - self.compacted
            if before >= 96 * 1024 and self.removed > self.compacted:
                self.repositions.append(index)
                self.checks.append(
                    dict(message=index, before=before, after=ends[index] - self.removed)
                )
                self.compacted = self.removed
        self.processed, self.previous = len(messages), copy.deepcopy(messages)
        self.bounds = {i: (counts[i], ends[i]) for i in range(len(messages))}
        self.last_counts = dict(
            full=len(trace.input_ids),
            active=len(trace.input_ids) - self.removed,
            position=len(trace.input_ids) - self.compacted,
        )
        return dict(
            drop_message=copy.deepcopy(self.drops), reposition=list(self.repositions)
        )


def execute_container_command(container, command, output, *, capture="stdio"):
    """Keep commands inside Docker, including daemons with early exec returns."""
    if capture not in ("stdio", "files"):
        raise ValueError("MINIMAX_DOCKER_CAPTURE must be stdio or files")
    if capture == "stdio":
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/testbed",
                container,
                "timeout",
                "180",
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            timeout=195,
        )
        return result.returncode, result.stdout + result.stderr

    destination = Path(output) / ("tool-" + uuid.uuid4().hex)
    destination.mkdir(parents=True)
    stem = "/tmp/minimax-" + uuid.uuid4().hex
    # Positional arguments keep arbitrary tool text out of the wrapper's syntax.
    wrapper = (
        'timeout 180 bash -lc "$1" > "$2.out" 2>&1; status=$?; '
        'printf "%s" "$status" > "$2.tmp"; mv "$2.tmp" "$2.rc"'
    )
    deadline = time.monotonic() + 210
    subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/testbed",
            container,
            "/bin/sh",
            "-c",
            wrapper,
            "minimax-command",
            command,
            stem,
        ],
        capture_output=True,
        timeout=195,
        check=True,
    )
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "cp", container + ":" + stem + ".rc", str(destination / "rc")],
            capture_output=True,
            timeout=20,
        )
        if result.returncode == 0:
            code = int((destination / "rc").read_text())
            if not 0 <= code <= 255:
                raise ValueError("Invalid container command exit status")
            subprocess.run(
                [
                    "docker",
                    "cp",
                    container + ":" + stem + ".out",
                    str(destination / "output"),
                ],
                capture_output=True,
                timeout=20,
                check=True,
            )
            return code, (destination / "output").read_bytes()
        time.sleep(1)
    raise TimeoutError("Docker exec returned but no command completion marker arrived")


def run_swe_task(endpoint, task_path, image, *, max_turns=250):
    """Execute fresh model tool calls only inside a new isolated SWE container.

    No captured answers, forced EOS bypass, or host shell execution. A normal
    final answer is distinct from coverage: the report records whether actual
    Drop and 96K Reposition events occurred during this particular task.
    """
    from transformers import AutoTokenizer

    task = json.loads(Path(task_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["MINIMAX_TOKENIZER_PATH"], local_files_only=True
    )
    state = RollingDrop96K(tokenizer)
    container = "minimax-r2-" + uuid.uuid4().hex[:16]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command in /testbed. Each call has a fresh shell.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "Solve the repository issue in /testbed. Use the bash tool to inspect, edit source files and run relevant tests. Do not change tests or configuration. When finished, provide a final answer describing the changes and test results. Do not commit.",
        },
        {"role": "user", "content": task["problem_statement"]},
    ]
    report = dict(
        task=task,
        image=image,
        mode="pd" if endpoint.prefill else "ordinary",
        turns=[],
        completed=False,
    )
    report_path = endpoint.output / (container + "-swe.json")
    capture = os.environ.get("MINIMAX_DOCKER_CAPTURE", "stdio")
    limits = os.environ.get("MINIMAX_DOCKER_CGROUP_LIMITS", "1")
    assert capture in ("stdio", "files") and limits in ("0", "1")
    report["docker_capture"] = capture
    report["docker_cgroup_limits"] = limits == "1"

    def execute(command):
        # The command is an argument to bash inside the container, never the host.
        code, data = execute_container_command(
            container, command, endpoint.output / "tools", capture=capture
        )
        text = data.decode("utf-8", errors="replace")
        if len(text) > 100_000:
            text = text[:50_000] + "\n[output truncated]\n" + text[-50_000:]
        return code, text

    start = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--network",
            "none",
            *(
                ["--cpus", "4", "--memory", "8g", "--pids-limit", "256"]
                if limits == "1"
                else []
            ),
            image,
            "sleep",
            "infinity",
        ],
        capture_output=True,
    )
    if start.returncode:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        raise RuntimeError(start.stderr.decode("utf-8", errors="replace"))
    try:
        code, head = execute("git rev-parse HEAD")
        assert code == 0 and re.fullmatch(r"[0-9a-f]{40}", head.strip()), head
        assert re.fullmatch(r"[0-9a-f]{40}", task["base_commit"])
        # SWE images may add a metadata-only commit with an identical file tree.
        code, trees = execute(
            "git rev-parse HEAD^{tree} " + task["base_commit"] + "^{tree}"
        )
        tree_ids = trees.splitlines()
        assert code == 0 and len(tree_ids) == 2 and tree_ids[0] == tree_ids[1], trees
        code, dirty = execute("git status --porcelain --untracked-files=all")
        assert code == 0 and not dirty.strip(), dirty
        report["repository_head"] = head.strip()
        report["repository_tree"] = tree_ids[0]
        report["image_id"] = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.Image}}", container], text=True
        ).strip()
        for turn in range(max_turns):
            events = state.extend(messages, tools)
            result = endpoint.request(
                dict(
                    messages=messages,
                    tools=tools,
                    max_tokens=16384,
                    chat_template_kwargs={"preserve_thinking_history": True},
                    **events,
                ),
                "swe-" + str(turn),
            )
            message = result["choices"][0]["message"]
            messages.append(
                {
                    k: message[k]
                    for k in ("role", "content", "reasoning_content", "tool_calls")
                    if message.get(k) is not None
                }
            )
            report["turns"].append(
                dict(
                    turn=turn,
                    counts=state.last_counts,
                    events=events,
                    usage=context_usage(result),
                    finish_reason=result["choices"][0]["finish_reason"],
                )
            )
            calls = message.get("tool_calls") or []
            if not calls:
                assert turn > 0, "Task ended without executing any tools"
                report["completed"] = True
                break
            for call in calls:
                assert call["function"]["name"] == "bash"
                arguments = json.loads(call["function"]["arguments"])
                assert set(arguments) == {"command"} and isinstance(
                    arguments["command"], str
                )
                code, output = execute(arguments["command"])
                messages.append(
                    dict(
                        role="tool",
                        tool_call_id=call["id"],
                        content=f"exit_code={code}\n{output}",
                    )
                )
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        code, patch = execute("git diff --no-ext-diff")
        assert code == 0
        report["patch"] = patch
        report["drop_events"] = len(state.drops)
        report["reposition_checks"] = state.checks
        report["coverage_complete"] = bool(state.drops and state.repositions)
        assert report["completed"], "Turn limit reached, not a normal completion"
        return report
    finally:
        report["messages"] = messages
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        subprocess.run(
            ["docker", "rm", "-f", container], check=True, capture_output=True
        )


def numeric_timeline(length, drops, repos):
    """Independent raw-token/position timeline; events occur before token i."""
    active, positions, next_position, records = [], [], 0, []
    for raw in range(length):
        keep = [
            i
            for i, token in enumerate(active)
            if not any(a <= token < b for a, b in drops.get(raw, []))
        ]
        active = [active[i] for i in keep]
        positions = [positions[i] for i in keep]
        if raw - 1 in repos:
            positions = list(range(len(active)))
            next_position = len(active)
        records.append((list(active), list(positions), next_position))
        active.append(raw)
        positions.append(next_position)
        next_position += 1
    return records, active, positions


def numeric_program(tokens, drops, repos):
    """Candidate input only. Never used to construct the native reference."""
    import torch
    from sglang.srt.context_system.ir import compile_context_layout
    from sglang.srt.context_system.planner import ContextProgram

    offsets, spans = [0], []
    for ranges in drops.values():
        spans.extend(x for pair in ranges for x in pair)
        offsets.append(len(spans) // 2)
    layout = compile_context_layout(
        *(
            torch.tensor(x, dtype=torch.int32)
            for x in (
                tokens,
                list(drops),
                offsets,
                spans,
                repos,
                [x + 1 for x in repos],
            )
        )
    )
    expiry = torch.full((len(tokens),), len(tokens) + 1, dtype=torch.int32)
    for boundary, ranges in drops.items():
        for start, end in ranges:
            expiry[start:end].clamp_(max=boundary)
    return ContextProgram(layout, expiry)


def _native_prefix_batch(runner, tokens, prefix):
    """Ordinary attention over previously produced KV, with no Context IR."""
    from array import array
    from sglang.benchmark.one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    req = Req(
        "native-oracle",
        "",
        array("q", tokens),
        SamplingParams(temperature=0, max_new_tokens=8),
    )
    req.full_untruncated_fill_ids = req.origin_input_ids
    req.logprob_start_len = -1
    req.prefix_indices = prefix
    req.set_extend_range(len(prefix), len(tokens))
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=TreeCacheNamespace(
            page_size=1,
            device=runner.device,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        ),
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device)
    batch.prefill_input_ids_cpu = None
    return batch


def _native_rotate(layer, pages, old_positions, new_positions):
    """FP32 inverse old RoPE, then forward new RoPE; no production kernel."""
    import torch

    old = layer.k_buffer[pages].clone()
    value = layer.v_buffer[pages].clone()
    width = layer.rotary_dim
    part = old[..., :width].float()
    if layer.is_neox_style:
        x, y = part.chunk(2, dim=-1)
    else:
        x, y = part[..., ::2], part[..., 1::2]
    c0, s0 = layer.cos_sin_cache[old_positions].float().chunk(2, -1)
    c1, s1 = layer.cos_sin_cache[new_positions].float().chunk(2, -1)
    c0, s0, c1, s1 = (v[:, None, :] for v in (c0, s0, c1, s1))
    scale = c0.square() + s0.square()
    unrotated_x, unrotated_y = (x * c0 + y * s0) / scale, (y * c0 - x * s0) / scale
    x1, y1 = unrotated_x * c1 - unrotated_y * s1, unrotated_x * s1 + unrotated_y * c1
    rotated = (
        torch.cat((x1, y1), -1)
        if layer.is_neox_style
        else torch.stack((x1, y1), -1).flatten(-2)
    )
    replacement = old.clone()
    replacement[..., :width] = rotated.to(old.dtype)
    same = old_positions == new_positions
    replacement[same] = old[same]
    layer.k_buffer[pages] = replacement
    assert torch.equal(layer.k_buffer[pages][..., width:], old[..., width:])
    assert torch.equal(layer.v_buffer[pages], value)


def _numeric_model_run(wrapper, tokens, drops, repos, *, candidate):
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from test_model_forward import prepare_batch, forward

    runner = wrapper.torch_runner
    wrapper.clear()
    allocator = runner.token_to_kv_pool_allocator
    before = allocator.available_size()
    timeline, active, final_positions = numeric_timeline(len(tokens), drops, repos)
    binding = runner.context_model_binding
    cuts = sorted(set([24, 56, 96] + list(range(97, len(tokens) + 1))))
    state = usage = None
    program = numeric_program(tokens, drops, repos) if candidate else None
    slots = torch.full((len(tokens),), -1, dtype=torch.int64, device=runner.device)
    allocated, logits, start = [], [], 0
    for end in cuts:
        if candidate:
            batch = prepare_batch(
                runner,
                [tokens],
                programs=[program],
                ends=[end],
                states=[state] if state is not None else None,
                usages=[usage] if usage is not None else None,
            )
            row = forward(runner, batch)
            torch.cuda.synchronize()
            for receipt in batch.context_completions:
                receipt.complete(allocator)
                receipt.complete(allocator)  # idempotence is part of ownership.
            state, usage = batch.reqs[0].context_state, batch.reqs[0].context_usage
        else:
            live, positions, next_pos = timeline[start]
            if start and start - 1 in repos:
                # Tokens produced before an earlier reposition have since moved.
                previous_live, previous_pos, _ = timeline[start - 1]
                before_event = dict(
                    zip(
                        previous_live + [start - 1],
                        previous_pos + [timeline[start - 1][2]],
                    )
                )
                old_positions = [before_event[raw] for raw in live]
                ids = runner.kv_index_translator.translate_full_attn_ids(slots[live])
                for layer in binding.layers:
                    _native_rotate(
                        layer,
                        ids,
                        torch.tensor(old_positions, device=runner.device),
                        torch.tensor(positions, device=runner.device),
                    )
            batch = _native_prefix_batch(
                runner, [tokens[raw] for raw in live] + tokens[start:end], slots[live]
            )
            fb = ForwardBatch.init_new(
                batch, runner, return_hidden_states_before_norm=False
            )
            assert fb.context_attention is None
            fb.positions = torch.arange(
                next_pos,
                next_pos + end - start,
                dtype=fb.positions.dtype,
                device=runner.device,
            )
            row = runner.forward(fb).logits_output.next_token_logits.clone()
            torch.cuda.synchronize()
            slots[start:end] = batch.out_cache_loc
            allocated.append(batch.out_cache_loc.clone())
        if end >= 96:
            logits.append(row.cpu())
        runner.req_to_token_pool.free(batch.reqs[0])
        start = end
    if candidate:
        terminal = state.terminal_slots()
        assert (terminal[active] >= 0).all()
        assert int((terminal >= 0).sum()) == len(active)
        assert program.layout.positions[active].tolist() == final_positions
        assert usage.snapshot().actual_prefill_tokens == len(tokens)
        slots = terminal
    physical = runner.kv_index_translator.translate_full_attn_ids(slots[active])
    kv = [
        (layer.k_buffer[physical].cpu(), layer.v_buffer[physical].cpu())
        for layer in binding.layers
    ]
    allocator.free(state.private_slots() if candidate else torch.cat(allocated))
    assert allocator.available_size() == before, "Private KV pages leaked"
    return torch.cat(logits), kv


def _numeric_worker(rank, argv, ports, output):
    import torch
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(argv)
    args.resolve_once()
    _set_envs_and_config(args)
    publish(args, role="scheduler")
    wrapper, tokenizer = load_model(args, ports, rank, rank)
    runner = wrapper.torch_runner
    runner.context_model_binding = ContextModelBinding(
        runner.model, runner.token_to_kv_pool, runner.kv_index_translator, page_size=1
    )
    tokens = tokenizer.encode("The library has red books and blue notebooks. " * 32)[
        :103
    ]
    assert len(tokens) == 103
    reports, reference_rows = {}, {}
    metadata = {
        "humming_batch_invariant": os.environ.get("SGLANG_HUMMING_USE_BATCH_INVARIANT", "0"),
        "cuda_graph_disabled": args.disable_cuda_graph,
        "native_repeatability": {},
    }
    with torch.no_grad():
        # Fixed dtype thresholds, recorded before any candidate comparison.
        thresholds = {
            "logits_max": 0.04,
            "logits_mean": 0.002,
            "kv_atol": 0.04,
            "kv_rtol": 0.02,
        }
        for policy in ("native", "drop", "repos", "combined"):
            drops = (
                {24: [(4, 12)], 56: [(16, 36)]}
                if policy in ("drop", "combined")
                else {}
            )
            repos = [23, 55] if policy in ("repos", "combined") else []
            expected, expected_kv = _numeric_model_run(
                wrapper, tokens, drops, repos, candidate=False
            )
            repeated, repeated_kv = _numeric_model_run(
                wrapper, tokens, drops, repos, candidate=False
            )
            repeat_error = (repeated.float() - expected.float()).abs()
            metadata["native_repeatability"][policy] = {
                "max": repeat_error.max().item(),
                "mean": repeat_error.mean().item(),
                "kv_equal": all(
                    torch.equal(a, b)
                    for pair_a, pair_b in zip(expected_kv, repeated_kv)
                    for a, b in zip(pair_a, pair_b)
                ),
            }
            Path(output, f"repeatability-rank-{rank}.json").write_text(
                json.dumps(metadata, indent=2)
            )
            assert torch.equal(expected, repeated), metadata
            assert metadata["native_repeatability"][policy]["kv_equal"], metadata
            del repeated, repeated_kv
            actual, actual_kv = _numeric_model_run(
                wrapper, tokens, drops, repos, candidate=True
            )
            error = (actual.float() - expected.float()).abs()
            report = {
                "max": error.max().item(),
                "mean": error.mean().item(),
                "layers": [],
            }
            for index, (wanted, got) in enumerate(zip(expected_kv, actual_kv)):
                report["layers"].append(
                    {
                        "layer": index,
                        "k_max": (wanted[0].float() - got[0].float())
                        .abs()
                        .max()
                        .item(),
                        "v_max": (wanted[1].float() - got[1].float())
                        .abs()
                        .max()
                        .item(),
                    }
                )
            reports[policy] = report
            Path(output, f"numeric-rank-{rank}.json").write_text(
                json.dumps(
                    dict(thresholds=thresholds, reports=reports, tokens=tokens, metadata=metadata),
                    indent=2,
                )
            )
            torch.save(
                {"expected": expected, "actual": actual},
                Path(output, f"{policy}-rank-{rank}.pt"),
            )
            assert report["max"] <= thresholds["logits_max"], report
            assert report["mean"] <= thresholds["logits_mean"], report
            for wanted, got in zip(expected_kv, actual_kv):
                for a, b in zip(wanted, got):
                    torch.testing.assert_close(
                        a, b, atol=thresholds["kv_atol"], rtol=thresholds["kv_rtol"]
                    )
            reference_rows[policy] = expected
        if rank == 0:
            torch.save(reference_rows, Path(output, "native-reference.pt"))
            Path(output, "native-reference.json").write_text(
                json.dumps(
                    dict(
                        tokens=tokens,
                        prompt_length=96,
                        forced_tokens=tokens[96:] + [tokens[96]],
                        thresholds=thresholds,
                        policies=list(reports),
                        metadata=metadata,
                    ),
                    indent=2,
                )
            )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def run_native_numeric_oracle(argv_path, output):
    """Explicit isolated TP processes; never shares a live server's allocator."""
    import torch.multiprocessing as mp
    from sglang.srt.server_args import PortArgs, prepare_server_args
    from sglang.srt.arg_groups.model_override_base import resolved_view

    argv = json.loads(Path(argv_path).read_text())
    args = prepare_server_args(argv)
    args.resolve_once()
    cfg = resolved_view(args)
    Path(output).mkdir(parents=True, exist_ok=True)
    mp.spawn(
        _numeric_worker,
        args=(argv, PortArgs.init_new(args), output),
        nprocs=cfg.tp_size,
        join=True,
    )


def run_fixed_token_http(endpoint, reference_dir):
    """Compare native staged logits with cold/hot and mixed concurrent serving.

    This diagnostic requires custom-logit processing and a shared trace directory
    visible to these explicitly supplied endpoints; it is excluded from timing.
    """
    import requests
    import torch
    from serving_logits_probe import serialized_probe

    root = Path(reference_dir)
    manifest = json.loads((root / "native-reference.json").read_text())
    reference = torch.load(root / "native-reference.pt", weights_only=True)
    tokens = manifest["tokens"][: manifest["prompt_length"]]
    forced = manifest["forced_tokens"]
    processor = serialized_probe()
    thresholds = manifest["thresholds"]

    def call(policy, label):
        drops = (
            {24: [(4, 12)], 56: [(16, 36)]} if policy in ("drop", "combined") else {}
        )
        repos = [23, 55] if policy in ("repos", "combined") else []
        identity = label + "-" + policy + "-" + uuid.uuid4().hex
        body = dict(
            input_ids=tokens,
            stream=False,
            custom_logit_processor=processor,
            sampling_params=dict(
                temperature=0, max_new_tokens=len(forced), ignore_eos=True
            ),
        )
        if policy != "native":
            body["context_program"] = numeric_program(
                tokens, drops, repos
            ).to_json_wire()
        if endpoint.prefill:
            body.update(
                bootstrap_host=endpoint.bootstrap_host,
                bootstrap_port=endpoint.bootstrap_port,
                bootstrap_room=time.time_ns() % (1 << 52),
            )

        def post(base, mode, count, offset):
            payload = copy.deepcopy(body)
            path = endpoint.output / (identity + "-" + mode + ".pt")
            payload["sampling_params"]["custom_params"] = dict(
                context_trace_path=str(path.resolve()),
                context_trace_count=count,
                context_forced_tokens=forced,
                context_forced_offset=offset,
            )
            response = requests.post(
                base + "/generate", json=payload, timeout=endpoint.timeout
            )
            (endpoint.output / (identity + "-" + mode + ".json")).write_text(
                response.text
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["meta_info"]["finish_reason"]["type"] in (
                "length",
                "stop",
            ), result
            assert path.exists(), "Missing full numerical trace"
            rows = torch.load(path, weights_only=True)
            assert rows.shape[0] == count
            return rows

        if endpoint.prefill:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                p = pool.submit(post, endpoint.prefill, "prefill", 1, 0)
                d = pool.submit(post, endpoint.decode, "decode", len(forced) - 1, 1)
                actual = torch.cat((p.result(), d.result()))
        else:
            actual = post(endpoint.decode, "ordinary", len(forced), 0)
        error = (actual - reference[policy]).abs()
        report = dict(
            max=error.max().item(), mean=error.mean().item(), thresholds=thresholds
        )
        (endpoint.output / (identity + "-comparison.json")).write_text(
            json.dumps(report, indent=2)
        )
        assert report["max"] <= thresholds["logits_max"], report
        assert report["mean"] <= thresholds["logits_mean"], report

    for round_name in ("first", "repeat"):
        for policy in manifest["policies"]:
            call(policy, round_name)
    for concurrency in (2, 4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            tasks = [
                pool.submit(call, policy, f"concurrent-{concurrency}")
                for policy in manifest["policies"]
            ]
            for task in tasks:
                task.result()
