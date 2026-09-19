"""Join native P/D physical counters to the unchanged test_serving window."""

import argparse
import copy
import json
import re
from pathlib import Path

from test_serving import load_method


def read_counts(root, tp):
    counts = {}
    for mode in ("prefill", "decode"):
        for line in (root / f"{mode}.log").read_text(errors="replace").splitlines():
            if "pd_matrix_compute=" not in line:
                continue
            if "ReqTimeStats(rid=HEALTH_CHECK_" in line:
                continue
            room = re.search(r"bootstrap_room=(\d+)", line)
            if room is None:
                continue
            rank = re.search(r"\bTP(\d+)\]", line)
            rank = int(rank[1]) if rank else 0
            value = json.loads(line.split("pd_matrix_compute=", 1)[1].split(", r2_pd_timing=", 1)[0])
            key = (room[1], mode)
            old = counts.setdefault(key, {})
            if rank in old and old[rank] != value:
                raise ValueError(f"Conflicting counter records: {key} TP{rank}")
            old[rank] = value
    return counts


def join_counts(result, counts, tp):
    result = copy.deepcopy(result)
    for row in result["turns"]:
        if not row["success"]:
            continue
        room = str((row.get("server_metrics") or {}).get("pd_bootstrap_room"))
        total = {"pf": 0, "decode": 0}
        for mode in ("prefill", "decode"):
            ranks = counts.get((room, mode), {})
            if 0 not in ranks or not set(ranks).issubset(set(range(tp))):
                raise ValueError(f"Incomplete TP counters: room={room} mode={mode}: {ranks}")
            if any(value != ranks[0] for value in ranks.values()):
                raise ValueError(f"TP counts disagree: {room} {mode}: {ranks}")
            for k, v in ranks[0].items():
                if type(v) is not int or v < 0:
                    raise ValueError(f"Invalid physical count: {ranks}")
                total[k] += v
        if total["pf"] <= 0:
            raise ValueError(f"Expected at least one P forward: {room}")
        row.setdefault("server_metrics", {}).update(
            prefill_compute_tokens=total["pf"], decode_compute_tokens=total["decode"],
            context_stage_count=1,
        )
    return result


def rebuild_summaries(result, method):
    """Use the joined physical counts in every window, not only overall."""
    result["overall"] = method.summary(result["turns"], result["start"], result["cutoff"])
    if "rounds" not in result:
        # The SGLang adapter retains the mini scheduler's round boundaries,
        # but leaves materializing the cohort reports to this offline join.
        result["rounds"] = []
        previous = result["start"]
        for index, end in enumerate(result.get("round_ends", [])):
            tasks = [task for task in result["tasks"] if not task["filler"]
                     and previous < task.get("end_time", float("inf")) <= end]
            if not tasks:
                raise ValueError(f"Empty first-pass cohort for round {index + 1}")
            result["rounds"].append(dict(round=index + 1, tasks=tasks))
            previous = end
    previous = result["start"]
    for report in result.get("rounds", []):
        tasks = report["tasks"]
        end = max(task["end_time"] for task in tasks)
        own = [row for row in result["turns"] if row["instance"] in {t["instance"] for t in tasks}]
        report["window"] = method.summary(result["turns"], previous, end)
        report["cumulative"] = method.summary(result["turns"], result["start"], end)
        report["cohort"] = method.summary(own, min(t["start_time"] for t in tasks), end)
        report["cohort_task_lifetime_s"] = method.stats(
            [t["end_time"] - t["start_time"] for t in tasks])
        previous = end
    tasks = [task for task in result.get("tasks", [])
             if not task["filler"] and task.get("status") == "all_turns_completed"]
    result["user_latency"] = dict(
        definition="Completed non-filler BCP task end_time-start_time; excludes waiting for a client slot",
        seconds=method.stats([task["end_time"] - task["start_time"] for task in tasks]),
        tasks=[dict(instance=t["instance"], case_id=t["case_id"],
                    seconds=t["end_time"] - t["start_time"]) for t in tasks],
    )
    return result


def detailed_accounting(result, method, root=None):
    """Keep useful work from abnormal/cancelled turns without inventing usage."""
    start, end = result["start"], result["cutoff"]
    seconds = end - start
    rows = [r for r in result["turns"] if start <= r["start_time"] <= end]
    completed = [r for r in rows if r.get("raw_done_time") is not None
                 and r["raw_done_time"] <= end]
    known, unknown = [], []
    for row in rows:
        usage = row.get("last_received_usage")
        observed = row.get("usage_received_perf")
        if usage is not None and observed is not None and observed <= end:
            known.append((row, usage))
        else:
            unknown.append(row)

    def totals(items):
        prompt = sum(usage.get("prompt_tokens", 0) for _, usage in items)
        output = sum(usage.get("completion_tokens", 0) for _, usage in items)
        return dict(turns=len(items), logical_prompt_tokens=prompt, output_tokens=output,
                    logical_prefill_throughput=prompt / seconds, output_throughput=output / seconds,
                    logical_all_throughput=(prompt + output) / seconds,
                    output_tokens_per_gpu_hour=output / (8 * seconds / 3600))

    value = dict(duration_s=seconds, gpu_hours=8 * seconds / 3600,
                 all_known=totals(known), primary=totals([(r, u) for r, u in known if not r["filler"]]),
                 supplementary=totals([(r, u) for r, u in known if r["filler"]]),
                 abnormal_known=totals([(r, u) for r, u in known if not r["success"]]),
                 unknown_usage_turns=len(unknown), logical_counts_complete=not unknown,
                 user_latency_s=method.stats([r["latency"] for r in completed if r.get("latency") is not None]),
                 user_throughput=method.stats([r["output_len"] / r["latency"] for r in completed
                                               if r.get("latency", 0) and r.get("usage") is not None]),
                 cleanup_s=method.stats([r["cleanup_time_s"] for r in completed if r.get("cleanup_time_s") is not None]),
                 ttft_s=method.stats([r["ttft"] for r in completed if r.get("ttft") is not None]),
                 tpot_s=method.stats([r["tpot_s"] for r in completed if r.get("tpot_s") is not None]),
                 sse_chunk_gap_s=method.stats([x for r in completed for x in r.get("itl", [])]),
                 tbt_s=method.stats([x for r in completed for x in (r.get("tbt_s") or [])]),
                 tbt_available_turns=sum(r.get("tbt_s") is not None for r in completed),
                 definitions={"logical": "Prompt includes cache hits; known usage received by cutoff, including abnormal turns. Unknown work is a lower-bound gap, not zero work.",
                              "user": "Raw D DONE minus request start; cleanup excluded. Incomplete turns are censored, not latency samples.",
                              "sse": "SSE chunk gaps are not token-level TBT.",
                              "physical": "TP0 forward dispatches inside measurement window; includes partial turns; boundary-crossing GPU work is not time-prorated."})
    if root is not None:
        physical = dict(prefill_tokens=0, decode_tokens=0, stages={})
        for stage in ("prefill", "decode"):
            path = root / f"{stage}-forward.jsonl"
            if not path.exists():
                physical["stages"][stage] = dict(available=False)
                continue
            # A dispatch and its later CUDA-completed record describe one step.
            steps = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (row["pid"], row["start_perf"])
                steps[key] = {**steps.get(key, {}), **row}
            forwards = list(steps.values())
            forwards = [r for r in forwards if start <= r["start_perf"] <= end]
            for key in ("prefill_tokens", "decode_tokens"):
                physical[key] += sum(r[key] for r in forwards)
            physical["stages"][stage] = dict(available=True, steps=len(forwards),
                gpu_ms=method.stats([r["gpu_ms"] for r in forwards if "gpu_ms" in r]),
                cpu_ms=method.stats([r["cpu_ms"] for r in forwards]))
        physical["complete_stage_logs"] = all(v["available"] for v in physical["stages"].values())
        for stage in ("prefill", "decode"):
            physical[stage + "_throughput"] = physical[stage + "_tokens"] / seconds if physical["complete_stage_logs"] else None
        physical["all_throughput"] = (physical["prefill_tokens"] + physical["decode_tokens"]) / seconds if physical["complete_stage_logs"] else None
        value["physical_forward_dispatch"] = physical
    return value


def summarize(root, mini_root, tp=2):
    method = load_method(mini_root)
    result = join_counts(json.loads((root / "workload/result.json").read_text()), read_counts(root, tp), tp)
    rebuild_summaries(result, method)
    if result.get("args", {}).get("unique_cohort"):
        result["accounting"] = detailed_accounting(result, method, root)
    result["measurement"] = {
        "physical_count_source": "CPU ScheduleBatch lengths at successful run_batch dispatch, both P/D",
        "tp_policy": "native TP0 logs count once; any additional visible ranks must agree",
        "prebuilt": "zero model tokens; first sampled token is produced by P",
        "profiler": (root / "profile-command.json").exists(),
        "original_result_preserved": "workload/result.json",
        "failed_and_cutoff_work": "not in successful numerator; elapsed time remains in denominator",
        "window_assignment": "Complete successful turns assigned by client completion time, including filler; not a GPU-time integral",
        "tbt": "Only scheduler-observed token gaps if returned; SSE chunk gaps are separate, never substituted",
    }
    target = root / "counted-result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--tp", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(summarize(a.root, a.mini_root, a.tp)["overall"], ensure_ascii=False))
