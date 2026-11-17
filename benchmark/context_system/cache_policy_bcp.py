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
DEFAULT_CASES = ("223", "231", "238", "775", "844")


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


def model_fingerprint(path):
    root = Path(path)
    names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
        if (root / name).is_file()
    }


def renderer_for(args):
    # Reuse native Harmony/Jinja provenance, not a second benchmark tokenizer.
    from test_serving import NativeTemplateAdapter

    return NativeTemplateAdapter(args.model, json.loads(args.template_kwargs))


def independent_events(payload, owners):
    """Translate native ownership into event boundaries without Context IR.

    Each message may own several disjoint runs. The trigger is after its last
    owned token, including its native suffix; unowned template tokens remain.
    This deliberately does not call the candidate Drop planner/compiler.
    """
    runs, ends = {}, {}
    for offset, owner in enumerate(owners):
        if owner < 0:
            continue
        if owner >= len(payload["messages"]):
            raise ValueError("Template owner outside message history")
        parts = runs.setdefault(owner, [])
        if parts and parts[-1][1] == offset:
            parts[-1][1] = offset + 1
        else:
            parts.append([offset, offset + 1])
        ends[owner] = offset + 1
    if len(ends) != len(payload["messages"]):
        raise ValueError("Message lacks native ownership")
    drops = {}
    for trigger, targets in payload.get("drop_message", {}).items():
        trigger = int(trigger)
        boundary = ends[trigger]
        spans = [part[:] for target in targets for part in runs[int(target)]]
        if any(end > boundary for _, end in spans):
            raise ValueError("Drop references future text")
        drops.setdefault(boundary, []).extend(spans)
    repos = [ends[int(trigger)] for trigger in payload.get("reposition", [])]
    if payload["drop_rule"] == POLICIES[1] and repos:
        raise ValueError("KV policy cannot reposition")
    return drops, repos


def reference_jobs(args):
    """Freeze diagnostic inputs from completed real trajectories, no inference."""
    report_path = Path(args.report)
    report = json.loads(report_path.read_text())
    frozen = json.loads(Path(args.frozen).read_text())
    if (
        not report["completed"]
        or report["frozen_sha256"] != digest(frozen)
        or args.model != frozen["model"]
        or model_fingerprint(args.model) != frozen["model_files"]
    ):
        raise ValueError("Reference jobs require the completed frozen cohort")
    renderer = renderer_for(args)
    if json.loads(args.template_kwargs) != frozen["template_kwargs"]:
        raise ValueError("Reference template differs from frozen template")
    continuation = renderer.renderer.tokenizer_manager.tokenizer.encode(
        " A blue notebook contains the recorded answer."
    )[:8]
    if len(continuation) < 4:
        raise ValueError("Fixed diagnostic continuation is too short")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for case in report["cases"]:
        for snapshot in ("first_drop", "multiple_drop", "first_reposition", "final"):
            if snapshot not in case["snapshots"]:
                continue
            turn = case["snapshots"][snapshot]
            request = report_path.parent / str(case["case_id"]) / f"{turn}.request.json"
            payload = json.loads(request.read_text())
            count, owners = renderer.render(payload["messages"], payload["tools"])
            tokens = list(renderer.renderer.trace.input_ids)
            assert len(tokens) == count == len(owners)
            drops, repos = independent_events(payload, owners)
            job = {
                "case_id": str(case["case_id"]),
                "snapshot": snapshot,
                "deployment": report["deployment"],
                "policy": report["policy"],
                "tokens": tokens,
                "drops": drops,
                "reposition": repos,
                "continuation": continuation,
                "payload": payload,
                "payload_sha256": digest(payload),
                "frozen_sha256": digest(frozen),
                "trajectory_head": report["head"],
            }
            name = f"{case['case_id']}-{snapshot}.json"
            write_json(root / name, job)
            jobs.append(name)
            if args.include_calibration and len(jobs) == 1:
                baseline = copy.deepcopy(job)
                baseline["calibration"] = True
                baseline["snapshot"] = "noop"
                baseline["drops"], baseline["reposition"] = {}, []
                baseline["payload"]["drop_message"] = {}
                if baseline["policy"] == POLICIES[0]:
                    baseline["payload"]["reposition"] = []
                baseline["payload_sha256"] = digest(baseline["payload"])
                write_json(root / "noop.json", baseline)
                jobs.insert(0, "noop.json")
    manifest = {
        "reference": "independent_event_model_forward",
        "jobs": jobs,
        "model": args.model,
        "frozen_sha256": digest(frozen),
    }
    write_json(root / "jobs.json", manifest)
    return manifest


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
    write_json(root / "source-coverage.json", coverage)
    require_coverage(coverage)
    frozen = {
        "schema": 1,
        "cases": args.cases,
        "entries": evidence,
        "tools_sha256": digest(tools),
        "model": args.model,
        "model_files": model_fingerprint(args.model),
        "numeric_floors": {"max_abs": 0.125, "mean_abs": 0.02, "p99_abs": 0.0625},
        "numeric_baseline_multiplier": 2,
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
        or model_fingerprint(args.model) != frozen["model_files"]
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


async def diagnostics(args):
    """Collect real fixed-token cold/hot/retry/eviction evidence separately."""
    import aiohttp
    import torch
    from serving_logits_probe import serialized_probe

    source = Path(args.jobs)
    manifest = json.loads(source.read_text())
    if args.model != manifest["model"]:
        raise ValueError("Diagnostic model differs from frozen reference jobs")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    processor = serialized_probe()
    rows = []
    head = await asyncio.to_thread(
        subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
    )
    report = {
        "comparisons": rows,
        "completed": False,
        "head": head.strip(),
    }
    if "noop.json" not in manifest["jobs"] and not args.baseline:
        raise ValueError("Freeze a no-event numerical baseline before candidates")
    if args.baseline:
        report["calibration"] = json.loads(Path(args.baseline).read_text())
    write_json(root / "diagnostics.json", report)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:

        async def flush():
            for endpoint in (args.prefill_url, args.url):
                if endpoint:
                    base = endpoint.split("/v1/")[0]
                    async with session.post(base + "/flush_cache") as response:
                        if response.status != 200:
                            raise RuntimeError("Diagnostic cache flush failed")

        async def call(job, name, *, payload=None, evict=0, check=True):
            payload = copy.deepcopy(payload if payload is not None else job["payload"])
            forced = [*job["continuation"], job["continuation"][-1]]
            spec = {
                "policy": job["policy"],
                "token_count": len(job["tokens"]),
                "drops": job["drops"],
                "reposition": job["reposition"],
            }
            payload.update(
                max_tokens=len(forced),
                ignore_eos=True,
                temperature=0,
                stream=False,
                custom_logit_processor=processor,
                return_meta_info=True,
                return_output_ids_in_sglext=True,
            )

            async def send(endpoint, role, count, offset):
                body = copy.deepcopy(payload)
                path = root / f"{name}-{role}.pt"
                body["custom_params"] = {
                    "context_trace_path": str(path),
                    "context_trace_count": count,
                    "context_forced_tokens": forced,
                    "context_forced_offset": offset,
                    "context_cache_policy_probe": check,
                    "context_layer_probe": check,
                    "context_cache_policy_evict_tokens": evict,
                    "context_policy_model": args.model,
                }
                if check:
                    body["custom_params"]["context_policy_spec"] = spec
                write_json(root / f"{name}-{role}.request.json", body)
                response = await infer(session, endpoint, body)
                write_json(root / f"{name}-{role}.response.json", response)
                return response, path

            if args.prefill_url:
                payload.update(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=args.bootstrap_port,
                    bootstrap_room=time.time_ns() % (1 << 52),
                )
                (_, p_path), (response, d_path) = await asyncio.gather(
                    send(args.prefill_url, "prefill", 1, 0),
                    send(args.url, "decode", len(forced) - 1, 1),
                )
                paths = [p_path, d_path]
            else:
                response, path = await send(args.url, "ordinary", len(forced), 0)
                paths = [path]
            if response["sglext"]["output_ids"] != [forced]:
                raise AssertionError("Candidate did not consume frozen continuation")
            logits = torch.cat([torch.load(p, weights_only=True) for p in paths])
            assert len(logits) == len(forced) and torch.isfinite(logits).all()
            torch.save(logits, root / f"{name}.pt")
            if not check:
                return {"paths": paths}
            stats = [
                json.loads(Path(str(p) + ".policy.json").read_text()) for p in paths
            ]
            reads = [
                json.loads(line)
                for p in paths
                for f in root.glob(p.name + ".reads.rank*.jsonl")
                for line in f.read_text().splitlines()
            ]
            assert reads, "Independent physical reader hooks produced no evidence"
            assert all(
                r["slots_ok"]
                and r["stored_k_ok"]
                and r["stored_v_ok"]
                and r["fp32_close"]
                for r in reads
            )
            assert all(s["structure_errors"] == [] for s in stats)
            result = {
                "case_id": job["case_id"],
                "snapshot": job["snapshot"],
                "policy": job["policy"],
                "deployment": "pd" if args.prefill_url else "ordinary",
                "candidate": f"{name}.pt",
                "payload_sha256": job["payload_sha256"],
                "same_payload": True,
                "same_fixed_continuation": True,
                "structure_errors": 0,
                "physical_reads_checked": True,
                "per_layer_checked": True,
                "checked_layers": sorted({r["layer"] for r in reads}),
                "retry_tokens": sum(s["retry_tokens"] for s in stats),
                "matched_tokens": sum(s["matched"] for s in stats),
                "resident_tokens": sum(s["resident"] for s in stats),
                "hole_tokens": sum(s["holes"] for s in stats),
                "recovered_queries": sum(
                    b - a for s in stats for a, b in s["recovery_intervals"]
                ),
            }
            # Every GPT-OSS layer must be checked, not just a convenient subset.
            assert result["checked_layers"] == list(range(36))
            if job["policy"] == POLICIES[1]:
                assert result["retry_tokens"] == 0
            return result

        for job_name in manifest["jobs"]:
            job = json.loads((source.parent / job_name).read_text())
            stem = Path(job_name).stem
            await flush()
            if job.get("calibration"):
                if not args.reference_output:
                    raise ValueError(
                        "Calibration requires independent --reference-output"
                    )
                await call(job, stem)
                candidate = torch.load(root / f"{stem}.pt", weights_only=True)
                expected = torch.load(
                    Path(args.reference_output) / f"{stem}.logits.pt", weights_only=True
                )
                assert (
                    candidate.shape == expected.shape and torch.isfinite(expected).all()
                )
                delta = (candidate.float() - expected.float()).abs()
                baseline = {
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean(-1).max()),
                    "p99_abs": float(torch.quantile(delta, 0.99, dim=-1).max()),
                }
                limits = {
                    name: max(floor, 2 * baseline[name])
                    for name, floor in (
                        ("max_abs", 0.125),
                        ("mean_abs", 0.02),
                        ("p99_abs", 0.0625),
                    )
                }
                calibration = {
                    "noop_baseline": baseline,
                    "limits": limits,
                    "reference": str(Path(args.reference_output) / f"{stem}.logits.pt"),
                    "candidate": str(root / f"{stem}.pt"),
                    "frozen_before_policy_candidates": True,
                }
                write_json(root / "numeric-baseline.json", calibration)
                report["calibration"] = calibration
                write_json(root / "diagnostics.json", report)
                continue
            for mode in ("cold", "hot"):
                row = await call(job, f"{stem}-{mode}")
                if mode == "hot":
                    assert row["matched_tokens"] > 0
                rows.append(dict(row, mode=mode))
                write_json(root / "diagnostics.json", report)
            await flush()
            if job["policy"] == POLICIES[0]:
                warm = copy.deepcopy(job["payload"])
                last = len(warm["messages"]) - 1
                repos = warm.get("reposition", [])
                warm["reposition"] = (
                    [r for r in repos if r != last] if last in repos else [*repos, last]
                )
                await call(job, f"{stem}-retry-source", payload=warm, check=False)
                row = await call(job, f"{stem}-retrigger")
                assert row["retry_tokens"] > 0
                rows.append(dict(row, mode="retrigger"))
                write_json(root / "diagnostics.json", report)
                await flush()
            from cache_policy_reference import simulate_compact

            ref = simulate_compact(
                len(job["tokens"]),
                {int(k): v for k, v in job["drops"].items()},
                job["reposition"],
                policy=job["policy"],
            )
            dead = int((ref.expiry < 2**63 - 1).sum())
            source_result = await call(
                job, f"{stem}-eviction-source", check=False, evict=dead + 256
            )
            evicted = sum(
                json.loads(Path(str(p) + ".eviction.json").read_text())[
                    "num_tokens_evicted"
                ]
                for p in source_result["paths"]
            )
            assert evicted > 0
            row = await call(job, f"{stem}-partial_eviction")
            assert row["recovered_queries"] > 0 and row["resident_tokens"] > 0
            rows.append(dict(row, mode="partial_eviction", evicted_pages=evicted))
            write_json(root / "diagnostics.json", report)
    report["completed"] = True
    write_json(root / "diagnostics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "run", "reference-jobs", "diagnostics")
    )
    parser.add_argument("--dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--template-kwargs", default='{"preserve_thinking_history": true}'
    )
    parser.add_argument("--cases", nargs=5, default=DEFAULT_CASES)
    parser.add_argument("--frozen")
    parser.add_argument("--report")
    parser.add_argument("--jobs")
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--reference-output")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--url")
    parser.add_argument("--prefill-url")
    parser.add_argument("--bootstrap-port", type=int)
    parser.add_argument("--gpu-mapping")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.action in ("preflight", "run") and not args.dataset:
        parser.error("preflight/run requires --dataset")
    if args.action == "reference-jobs":
        if not args.frozen or not args.report:
            parser.error("reference-jobs requires --frozen and --report")
        print(json.dumps(reference_jobs(args), ensure_ascii=False))
        return
    if args.action == "diagnostics":
        if (
            not args.jobs
            or not args.url
            or (args.prefill_url and not args.bootstrap_port)
        ):
            parser.error("diagnostics requires --jobs, --url and PD bootstrap port")
        print(json.dumps(asyncio.run(diagnostics(args)), ensure_ascii=False))
        return
    if args.action == "run" and not all(
        (args.frozen, args.policy, args.url, args.gpu_mapping)
    ):
        parser.error("run requires --frozen, --policy, --url and --gpu-mapping")
    result = preflight(args) if args.action == "preflight" else asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
