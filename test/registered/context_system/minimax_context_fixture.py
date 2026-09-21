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
