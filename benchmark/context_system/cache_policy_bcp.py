"""Frozen five-case BCP ablation validation, using real HTTP model generation.

Preflight uses recorded source histories solely to select event coverage.
Run feeds each newly generated assistant back into the next request, alongside
recorded tool feedback. This is trajectory replay, not live-search accuracy.
Every payload/response is saved; no reduced token budget is accepted.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

POLICIES = ("text_drop_with_pe", "message_drop_without_repos")
DEFAULT_CASES = ("771", "775", "778", "792", "844")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_cases(root, case_ids):
    root = Path(root)
    if len(case_ids) != 5 or len(set(case_ids)) != 5:
        raise ValueError("Exactly five distinct BCP cases are required")
    manifest = json.loads((root / "manifest.json").read_text())
    raw_index = (root / "tasks_unique.jsonl").read_bytes()
    if hashlib.sha256(raw_index).hexdigest() != manifest["files"]["tasks_unique.jsonl"]:
        raise ValueError("BCP task index hash mismatch")
    index = {
        str(row["case_id"]): row for row in map(json.loads, raw_index.splitlines())
    }
    cases, evidence = [], []
    with (root / "trajectories.jsonl").open("rb") as source:
        for case_id in case_ids:
            entry = index[case_id]
            source.seek(entry["offset"])
            raw = source.read(entry["bytes"])
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("BCP trajectory hash mismatch")
            case = json.loads(raw)
            if str(case["case_id"]) != case_id or case["trial"] != entry["trial"]:
                raise ValueError("BCP trajectory index identity mismatch")
            if not case["turns"] or any(
                type(t["max_new_tokens"]) is not int or t["max_new_tokens"] <= 0
                for t in case["turns"]
            ):
                raise ValueError(
                    "BCP requires original positive per-turn generation budgets"
                )
            cases.append(case)
            evidence.append(entry)
    return cases, manifest["tools"], evidence


class RollingPolicy:
    def __init__(self, policy, *, keep=12, threshold=96 * 1024):
        if policy not in POLICIES:
            raise ValueError("Unknown cache policy")
        self.policy, self.keep, self.threshold = policy, keep, threshold
        self.processed = self.removed = self.compacted = 0
        self.tools, self.reposition, self.checks = [], [], []
        self.drop_message, self.bounds = {}, {}
        self.prefix = None

    def extend(self, messages, owners):
        if (
            self.prefix is not None
            and digest(messages[: self.processed]) != self.prefix
        ):
            raise ValueError("Historical messages changed")
        counts, ends = Counter(owners), {}
        for i, owner in enumerate(owners):
            if owner >= 0:
                ends[owner] = i + 1
        if any((counts[i], ends.get(i)) != bound for i, bound in self.bounds.items()):
            raise ValueError("Native template changed a historical ownership range")
        for i in range(self.processed, len(messages)):
            if i not in ends:
                raise ValueError("Message has no native token ownership")
            if messages[i]["role"] in ("tool", "function"):
                self.tools.append(i)
                if len(self.tools) > self.keep:
                    old = self.tools[-self.keep - 1]
                    self.drop_message[str(i)] = [old]
                    self.removed += counts[old]
            before = ends[i] - self.compacted
            if (
                self.policy == POLICIES[0]
                and before >= self.threshold
                and self.removed > self.compacted
            ):
                self.reposition.append(i)
                self.checks.append(
                    {"message_id": i, "before": before, "after": ends[i] - self.removed}
                )
                self.compacted = self.removed
        self.processed, self.prefix = len(messages), digest(messages)
        self.bounds = {i: (counts[i], ends[i]) for i in range(len(messages))}
        result = {
            "drop_rule": self.policy,
            "drop_message": copy.deepcopy(self.drop_message),
        }
        if self.policy == POLICIES[0]:
            result["reposition"] = list(self.reposition)
        return result

    def coverage(self):
        return {
            "drops": len(self.drop_message),
            "repositions": len(self.reposition),
            "drop_after_reposition": any(
                int(d) > r for d in self.drop_message for r in self.reposition
            ),
            "checks": self.checks,
        }


def require_coverage(rows):
    if len(rows) != 5 or any(
        row["drops"] < 1 or row["repositions"] < 1 for row in rows
    ):
        raise ValueError(
            "Each of the five cases must exercise real Drop and 96K Reposition"
        )
    if not any(row["drops"] > 1 and row["drop_after_reposition"] for row in rows):
        raise ValueError("Cohort must include multiple Drops and Drop after Reposition")


def renderer_for(args):
    # Reuse native Harmony/Jinja provenance, not a second benchmark tokenizer.
    from test_serving import NativeTemplateAdapter

    return NativeTemplateAdapter(args.model, json.loads(args.template_kwargs))


def preflight(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    cases, tools, evidence = load_cases(args.dataset, args.cases)
    renderer = renderer_for(args)
    coverage = []
    for case in cases:
        rolling = RollingPolicy(POLICIES[0])
        for turn in case["turns"]:
            # Source assistants are allowed only in the input-only preflight.
            history = copy.deepcopy(
                case["trajectory"][: turn["source_assistant_message_id"]]
            )
            # Some data stores omit name on tool results; turns contain the
            # canonical tool name populated when the dataset was prepared.
            names = {
                m.get("tool_call_id"): m.get("name")
                for t in case["turns"]
                for m in t["new_messages"]
                if m["role"] == "tool"
            }
            for m in history:
                if m["role"] == "tool" and not m.get("name"):
                    m["name"] = names[m["tool_call_id"]]
            _, owners = renderer.render(history, tools)
            rolling.extend(history, owners)
        coverage.append(dict(case_id=case["case_id"], **rolling.coverage()))
    require_coverage(coverage)
    frozen = {
        "schema": 1,
        "cases": args.cases,
        "entries": evidence,
        "tools_sha256": digest(tools),
        "model": args.model,
        "template_kwargs": json.loads(args.template_kwargs),
        "source_coverage": coverage,
        "source": "recorded BCP trajectory; no inference yet",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write_json(root / "frozen.json", frozen)
    return frozen


async def infer(session, url, payload, prefill_url=None, bootstrap_port=None):
    async def post(endpoint, body):
        async with session.post(endpoint, json=body) as response:
            raw = await response.text()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {raw[:2000]}")
            return json.loads(raw)

    if not prefill_url:
        return await post(url, payload)
    if not bootstrap_port:
        raise ValueError("PD requires an explicit bootstrap port")
    body = {
        **payload,
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": time.time_ns() % (1 << 52),
    }
    # Both responses must complete; do not report D success after a P failure.
    tasks = [
        asyncio.create_task(post(endpoint, body)) for endpoint in (prefill_url, url)
    ]
    try:
        _, response = await asyncio.gather(*tasks)
        return response
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run(args):
    import aiohttp

    frozen = json.loads(Path(args.frozen).read_text())
    cases, tools, evidence = load_cases(args.dataset, frozen["cases"])
    if (
        evidence != frozen["entries"]
        or digest(tools) != frozen["tools_sha256"]
        or args.model != frozen["model"]
        or json.loads(args.template_kwargs) != frozen["template_kwargs"]
    ):
        raise ValueError("Run differs from frozen inputs/model/template")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    renderer = renderer_for(args)
    head = await asyncio.to_thread(
        subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
    )
    report = {
        "schema": 1,
        "frozen_sha256": digest(frozen),
        "deployment": "pd" if args.prefill_url else "ordinary",
        "policy": args.policy,
        "completed": False,
        "cases": [],
        "head": head.strip(),
        "model": args.model,
        "gpu_mapping": args.gpu_mapping,
        "meaning": "generated assistant replay with recorded tool feedback; not live-search accuracy",
    }
    write_json(root / "report.json", report)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        for case in cases:
            history, rolling = [], RollingPolicy(args.policy)
            saved = root / str(case["case_id"])
            saved.mkdir()
            snapshots, completed_turns = {}, 0
            for turn in case["turns"]:
                history.extend(copy.deepcopy(turn["new_messages"]))
                full, owners = renderer.render(history, tools)
                state = rolling.extend(history, owners)
                payload = dict(
                    model=args.model,
                    messages=history,
                    tools=tools,
                    **state,
                    temperature=0,
                    max_tokens=turn["max_new_tokens"],
                    ignore_eos=True,
                    stream=False,
                    chat_template_kwargs=json.loads(args.template_kwargs),
                )
                number = str(turn["turn"])
                write_json(saved / f"{number}.request.json", payload)
                started = time.perf_counter()
                response = await infer(
                    session, args.url, payload, args.prefill_url, args.bootstrap_port
                )
                write_json(saved / f"{number}.response.json", response)
                choice = response["choices"][0]
                expected_prompt = (
                    full - rolling.removed if args.policy == POLICIES[0] else full
                )
                if response["usage"]["prompt_tokens"] != expected_prompt:
                    raise ValueError("Native template/execution projection mismatch")
                if (
                    choice["finish_reason"] != "length"
                    or response["usage"]["completion_tokens"] != turn["max_new_tokens"]
                ):
                    raise ValueError(
                        "Formal replay did not complete its original generation budget"
                    )
                if rolling.drop_message:
                    snapshots.setdefault("first_drop", number)
                if rolling.reposition:
                    snapshots.setdefault("first_reposition", number)
                if len(rolling.drop_message) > 1:
                    snapshots.setdefault("multiple_drop", number)
                snapshots["final"] = number
                write_json(
                    saved / f"{number}.metrics.json",
                    {
                        "latency_s": time.perf_counter() - started,
                        "full_tokens": full,
                        "execution_tokens": expected_prompt,
                        "coverage": rolling.coverage(),
                        "payload_sha256": digest(payload),
                        "usage": response["usage"],
                    },
                )
                history.append(copy.deepcopy(choice["message"]))
                completed_turns += 1
            if not rolling.drop_message or (
                args.policy == POLICIES[0] and not rolling.reposition
            ):
                raise ValueError(
                    "Generated trajectory did not meet actual event coverage"
                )
            report["cases"].append(
                dict(
                    case_id=case["case_id"],
                    turns=completed_turns,
                    snapshots=snapshots,
                    **rolling.coverage(),
                )
            )
            write_json(root / "report.json", report)
    if args.policy == POLICIES[0]:
        require_coverage(report["cases"])
    report["completed"] = True
    write_json(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--template-kwargs", default='{"preserve_thinking_history": true}'
    )
    parser.add_argument("--cases", nargs=5, default=DEFAULT_CASES)
    parser.add_argument("--frozen")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--url")
    parser.add_argument("--prefill-url")
    parser.add_argument("--bootstrap-port", type=int)
    parser.add_argument("--gpu-mapping")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.action == "run" and not all(
        (args.frozen, args.policy, args.url, args.gpu_mapping)
    ):
        parser.error("run requires --frozen, --policy, --url and --gpu-mapping")
    result = preflight(args) if args.action == "preflight" else asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
