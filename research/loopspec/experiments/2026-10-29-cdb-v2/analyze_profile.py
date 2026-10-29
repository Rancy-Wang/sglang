"""Summarize measured intervals without charging shared GPU work per lane."""

import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import statistics


def describe(values):
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "sum": sum(values)} if values else {"n": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.directory.glob("cohort-*.json"))]
    stages, paths = collections.defaultdict(list), collections.defaultdict(list)
    token_rows = []
    path_stages = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted((args.directory / "trace").glob("*-recurrent-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {row["id"]: row for row in rows}
        for row in rows:
            stage = row["stage"]
            if stage == "loopspec.graph":
                stage += "." + row["input"]["role"]
            stages[stage].append(row)
            parent = row
            while parent["stage"] != "loopspec.output" and parent["parent"] is not None:
                parent = by_id[parent["parent"]]
            if parent["stage"] == "loopspec.output":
                path_stages[str(parent["input"]["entry_depth"])][stage].append(row)
            if row["stage"] == "loopspec.output":
                before, after = row["input"], row["output"]
                paths[str(before["entry_depth"])].append(row)
                token_rows.append({**before, **after, "new_cycles": after["cycles"] - before["cycles"],
                                   "cpu_ms": row["cpu_ms"], "gpu_stream_ms": row.get("gpu_stream_ms")})
    summary = {
        "note": "CPU/GPU spans are inclusive intervals; nested/shared spans must not be summed. Profile latency is diagnostic only.",
        "requests": [{"question": q + 1,
                      "e2e_ms": describe([r["requests"][q]["e2e_ms"] for r in runs]),
                      "ttft_ms": describe([r["requests"][q]["ttft_ms"] for r in runs]),
                      "tokens": [len(r["requests"][q]["output_ids"]) for r in runs]}
                     for q in range(len(runs[0]["requests"]))] if runs else [],
        "makespan_ms": describe([r["makespan_ms"] for r in runs]),
        "stages": {name: {"cpu_ms": describe([r["cpu_ms"] for r in group]),
                          "gpu_stream_ms": describe([r["gpu_stream_ms"] for r in group if "gpu_stream_ms" in r])}
                   for name, group in stages.items()},
        "entry_depth": {str(depth): {"cpu_ms": describe([r["cpu_ms"] for r in paths[str(depth)]]),
                                     "new_cycles": describe([r["output"]["cycles"] - r["input"]["cycles"] for r in paths[str(depth)]]),
                                     "stages": {name: {"cpu_ms": describe([r["cpu_ms"] for r in group]),
                                                       "gpu_stream_ms": describe([r["gpu_stream_ms"] for r in group if "gpu_stream_ms" in r])}
                                                for name, group in path_stages[str(depth)].items()}}
                        for depth in range(4)},
    }
    kernels = collections.defaultdict(list)
    intervals = []
    for path in sorted((args.directory / "trace").glob("*.trace.json.gz")):
        with gzip.open(path, "rt") as stream:
            trace = json.load(stream)
        for event in trace.get("traceEvents", []):
            if event.get("cat") == "kernel" and "dur" in event:
                kernels[event["name"]].append(event["dur"] / 1000)
                intervals.append((event["ts"], event["ts"] + event["dur"]))
        del trace
    union_us, end = 0, float("-inf")
    for left, right in sorted(intervals):
        union_us += max(0, right - max(left, end))
        end = max(end, right)
    summary["gpu_kernel_busy_union_ms"] = union_us / 1000
    summary["kernels"] = {name: describe(values) for name, values in
                          sorted(kernels.items(), key=lambda item: -sum(item[1]))}
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.directory / "output-paths.json").write_text(json.dumps(token_rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
