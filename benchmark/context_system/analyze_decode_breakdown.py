"""Conservative per-rank decode accounting; never sum parallel TP replicas.

An unattributed graph kernel stays unknown. CPU NVTX duration is never used as
GPU component time. Graph-node attribution is used only when observed directly.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def exclusive_intervals(events):
    """Disjoint duration by active category set; overlaps appear exactly once."""
    endpoints = defaultdict(Counter)
    for row in events:
        if row["end"] < row["start"]:
            raise ValueError("Negative GPU interval")
        endpoints[row["start"]][row["category"]] += 1
        endpoints[row["end"]][row["category"]] -= 1
    active, durations = Counter(), Counter()
    previous = None
    for when, delta in sorted(endpoints.items()):
        labels = tuple(sorted(k for k, v in active.items() if v > 0))
        if previous is not None and labels:
            key = labels[0] if len(labels) == 1 else "overlap:" + "+".join(labels)
            durations[key] += when - previous
        active.update(delta)
        previous = when
    return dict(durations)


def conservative_category(name):
    name = name.lower()
    if "allreduce" in name or "all_reduce" in name or "nccl" in name:
        return "tp_communication"
    if "rope" in name or "rotary" in name:
        return "rope_or_fused_kv"
    if "rmsnorm" in name or "rms_norm" in name:
        return "norm_or_fused_residual"
    # Names alone cannot distinguish Full/SWA attention or attention/MoE GEMMs.
    return "unknown"


def read_nsys(path):
    db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    needed = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS", "StringIds"}
    if not needed <= tables:
        raise ValueError(f"Missing Nsight node/kernel tables: {needed - tables}")
    strings = dict(db.execute("select id,value from StringIds"))
    ranges = defaultdict(list)
    for r in db.execute("select * from NVTX_EVENTS"):
        row = dict(r)
        label = row.get("text") or strings.get(row.get("textId"), "")
        if row.get("end") is not None and label.startswith(("component:", "decode_step:", "benchmark:")):
            ranges[row["globalTid"]].append((row["start"], row["end"], label))
    runtime, component_nodes = {}, {}
    for r in db.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"):
        row = dict(r)
        pid = (row["globalTid"] >> 24) & 0xffffff
        enclosing = [v for v in ranges[row["globalTid"]]
                     if v[0] <= row["start"] and row["end"] <= v[1]]
        step = next((int(v[2].split(":")[1]) for v in enclosing if v[2].startswith("decode_step:")), None)
        components = sorted((v for v in enclosing if v[2].startswith("component:")), key=lambda v: v[1] - v[0])
        category = components[0][2].split(":", 2)[1] if components else None
        if any(v[2].startswith("benchmark:") for v in enclosing):
            category = "benchmark_control"
        runtime[pid, row["correlationId"]] = dict(step=step, category=category)
    kernels = []
    for r in db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        row = dict(r)
        pid = (row["globalPid"] >> 24) & 0xffffff
        api = runtime.get((pid, row["correlationId"]), {})
        name = strings.get(row.get("demangledName"), strings.get(row.get("shortName"), "unknown"))
        node = row.get("graphNodeId")
        if node and api.get("category"):
            key = (pid, node)
            if key in component_nodes and component_nodes[key] != api["category"]:
                raise ValueError("Conflicting component assignment for CUDA graph node")
            component_nodes[key] = api["category"]
        kernels.append(dict(pid=pid, start=row["start"], end=row["end"],
                            step=api.get("step"), name=name, category=api.get("category"),
                            graph_node_id=node, stream=row["streamId"], device=row["deviceId"]))
    for k in kernels:
        k["attribution"] = "runtime_nvtx" if k["category"] else "graph_node" if (k["pid"], k["graph_node_id"]) in component_nodes else "name_or_unknown"
        k["category"] = k["category"] or component_nodes.get((k["pid"], k["graph_node_id"])) or conservative_category(k["name"])
    db.close()
    return kernels


def summarize_steps(rows, expected_ranks=4):
    selected = [r for r in rows if r.get("kind") == "decode_step" and r.get("measured")]
    by_rank = defaultdict(dict)
    for r in selected:
        if r["step"] in by_rank[r["rank"]]:
            raise ValueError("Duplicate step/rank: do not combine repeats in one run")
        by_rank[r["rank"]][r["step"]] = r
    if set(by_rank) != set(range(expected_ranks)):
        raise ValueError("Incomplete TP rank timing coverage")
    ref = by_rank[0]
    for rank, steps in by_rank.items():
        if set(steps) != set(ref):
            raise ValueError("Mismatched measured steps across TP ranks")
        for step, r in steps.items():
            if r["request_ids"] != ref[step]["request_ids"] or r["batch_size"] != ref[step]["batch_size"]:
                raise ValueError("TP cohort mismatch")
            if r["gpu_ms"] <= 0:
                raise ValueError("Invalid CUDA event time")
    ranks = {}
    for rank, steps in by_rank.items():
        ms = [r["gpu_ms"] for r in steps.values()]
        ranks[rank] = dict(steps=len(ms), mean_gpu_ms=statistics.mean(ms),
                           median_gpu_ms=statistics.median(ms), p95_gpu_ms=percentile(ms, .95),
                           batch_sizes=sorted({r["batch_size"] for r in steps.values()}),
                           graph_batch_sizes=sorted({r["graph_batch_size"] for r in steps.values()}, key=str))
    # This is a bottleneck-rank statistic, not a cross-device synchronized
    # end-to-end stopwatch and not the sum of TP worker durations.
    max_ms = [max(by_rank[rank][s]["gpu_ms"] for rank in by_rank) for s in ref]
    return dict(ranks=ranks, max_rank_mean_gpu_ms=statistics.mean(max_ms),
                max_rank_p95_gpu_ms=percentile(max_ms, .95),
                boundary="ModelRunner.forward CUDA events; excludes sampling and client/network")


def analyze(args):
    rows = []
    for p in sorted(Path(args.timings).glob("decode-rank*.jsonl")):
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    summary = summarize_steps(rows)
    runtime = {r["rank"]: r for r in rows if r.get("kind") == "runtime"}
    verified = {r["rank"]: r for r in rows if r.get("kind") == "cohort_tokens_verified"}
    if set(runtime) != set(range(4)) or set(verified) != set(range(4)):
        raise ValueError("Missing runtime metadata or final token-path verification")
    for rank, r in runtime.items():
        config = r["config"]
        expected = set(range(config["warm_steps"], config["warm_steps"] + config["measure_steps"]))
        actual = [v for v in rows if v.get("kind") == "decode_step" and v.get("measured") and v["rank"] == rank]
        if {v["step"] for v in actual} != expected or not all(v["cuda_graph"] for v in actual):
            raise ValueError("Incomplete fixed measurement window or CUDA graph fallback")
        if verified[rank]["hashes"] != verified[0]["hashes"]:
            raise ValueError("TP ranks emitted different token paths")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    measured = {(r["pid"], r["step"]): r for r in rows if r.get("kind") == "decode_step" and r.get("measured")}
    if args.sqlite:
        kernels = [k for k in read_nsys(args.sqlite) if (k["pid"], k["step"]) in measured]
        grouped = defaultdict(list)
        for k in kernels:
            grouped[k["pid"], k["step"]].append(k)
        if set(grouped) != set(measured):
            raise ValueError("Nsight kernel-to-step coverage incomplete; no component result emitted")
        accounting = []
        for key, values in grouped.items():
            parts = exclusive_intervals(values)
            span = max(k["end"] for k in values) - min(k["start"] for k in values)
            busy = sum(parts.values())
            assert busy <= span
            accounting.append(dict(pid=key[0], rank=measured[key]["rank"], step=key[1],
                exclusive_gpu_ns=parts, kernel_span_ns=span, busy_ns=busy,
                inter_kernel_gap_ns=span - busy,
                unknown_fraction=sum(v for k, v in parts.items() if "unknown" in k) / busy))
        (out / "component_steps.json").write_text(json.dumps(accounting, indent=2))
        with (out / "kernels.jsonl").open("w") as f:
            for k in kernels:
                f.write(json.dumps(k) + "\n")
        summary["component_coverage_pass"] = all(r["unknown_fraction"] <= .05 for r in accounting)
        summary["component_warning"] = "GPU busy union excludes copies and graph gaps; compare kernel span with event time. Unknown kernels are not assigned speculative categories."
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "steps.csv").open("w") as f:
        keys = ["rank", "pid", "step", "batch_size", "graph_batch_size", "gpu_ms", "cpu_ms", "start_perf"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measured.values())
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timings", required=True)
    p.add_argument("--sqlite")
    p.add_argument("--output", required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
