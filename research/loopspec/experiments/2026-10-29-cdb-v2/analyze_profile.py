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
    shared_batches = []
    request_rows = collections.defaultdict(lambda: {"shared_batch_ids": [], "results": []})
    path_stages = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted((args.directory / "trace").glob("*-recurrent-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        by_id = {row["id"]: row for row in rows}
        output_stages = collections.defaultdict(lambda: collections.defaultdict(list))
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
                output_stages[parent["id"]][stage].append(row)
            metadata = row.get("input", {})
            if stage == "scheduler.receive":
                request_rows[metadata["rid"]]["receive_ns"] = row["start_ns"]
            if stage == "scheduler.forward":
                batch_id = len(shared_batches)
                shared_batches.append({"id": batch_id, **metadata,
                                       "start_ns": row["start_ns"], "end_ns": row["end_ns"],
                                       "cpu_ms": row["cpu_ms"]})
                for rid in metadata["rids"]:
                    request_rows[rid]["shared_batch_ids"].append(batch_id)
            if stage == "scheduler.result":
                for rid, length in zip(metadata["rids"], metadata["output_lengths"], strict=True):
                    request_rows[rid]["results"].append({"start_ns": row["start_ns"],
                        "end_ns": row["end_ns"], "cpu_ms": row["cpu_ms"],
                        "output_length_before_processing": length, "mode": metadata["mode"]})
        for row in rows:
            if row["stage"] == "loopspec.output":
                before, after = row["input"], row["output"]
                paths[str(before["entry_depth"])].append(row)
                token_rows.append({**before, **after, "new_cycles": after["cycles"] - before["cycles"],
                                   "cpu_ms": row["cpu_ms"], "gpu_stream_ms": row.get("gpu_stream_ms"),
                                   "stages": {name: {"calls": len(group),
                                       "cpu_ms": sum(r["cpu_ms"] for r in group),
                                       "gpu_stream_ms": sum(r.get("gpu_stream_ms", 0) for r in group)}
                                       for name, group in output_stages[row["id"]].items()}})
        # CDB spans multiple scheduler calls. Associate steps by native request
        # and output index; shared steps remain explicit and are not summed
        # across requests. bs=1 yields directly comparable stage totals.
        continuations = collections.defaultdict(list)
        for step in (row for row in rows if row["stage"] == "loopspec.step"):
            for item in step["input"]["requests"]:
                key = item["rid"], item["output_index"]
                continuations[key].append((step, item))
                if item["rid"] not in step["output"]["committed"]:
                    continue
                chain = continuations.pop(key)
                before = chain[0][1]
                depth = str(before["entry_depth"])
                ids = {entry[0]["id"] for entry in chain}
                groups = collections.defaultdict(list)
                for child in rows:
                    parent = child
                    while parent["id"] not in ids and parent["parent"] is not None:
                        parent = by_id[parent["parent"]]
                    if parent["id"] in ids:
                        groups[child["stage"]].append(child)
                cpu = sum(entry[0]["cpu_ms"] for entry in chain)
                gpu = sum(entry[0].get("gpu_stream_ms", 0) for entry in chain)
                after = {"cycles": item["cycles"] + 1}
                paths[depth].append({"input": before, "output": after, "cpu_ms": cpu})
                for stage, group in groups.items():
                    path_stages[depth][stage].extend(group)
                token_rows.append({**before, **after, "new_cycles": len(chain),
                    "cpu_ms": cpu, "gpu_stream_ms": gpu,
                    "shared_step_ids": sorted(ids),
                    "stages": {name: {"calls": len(group),
                        "cpu_ms": sum(r["cpu_ms"] for r in group),
                        "gpu_stream_ms": sum(r.get("gpu_stream_ms", 0) for r in group)}
                        for name, group in groups.items()}})
    for rid, request in request_rows.items():
        request["rid"] = rid
        if request["shared_batch_ids"]:
            first = shared_batches[request["shared_batch_ids"][0]]["start_ns"]
            request["first_forward_ns"] = first
            if "receive_ns" in request:
                request["receive_to_first_forward_ms"] = (first - request["receive_ns"]) / 1e6
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
    summary["stage_time_per_output"] = {
        str(depth): {stage: {key: describe([row["stages"].get(stage, {}).get(key, 0)
                                          for row in token_rows if row["entry_depth"] == depth])
                            for key in ("calls", "cpu_ms", "gpu_stream_ms")}
                    for stage in path_stages[str(depth)]}
        for depth in range(4)
    }
    summary["server_requests"] = list(request_rows.values())
    summary["shared_batch_note"] = (
        "shared-batches.json contains CPU submission intervals, not GPU duration. "
        "Scheduler GPU events recorded on a scheduling stream cannot time an overlapped forward. "
        "Per-output stage totals sum same-name calls only; nested stages must not be added together."
    )
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
    (args.directory / "shared-batches.json").write_text(json.dumps(shared_batches, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
