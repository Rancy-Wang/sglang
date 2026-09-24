"""Auditable interval/DAG accounting for PLAN-CS-20260924-E2E-BREAKDOWN-R1.

Unobserved intervals remain unknown. Coverage is NOT proof of causal attribution.
No percentile summation, GPU-rank summation, or treating CPU spans as GPU work.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3


def partition(start, end, intervals):
    """Exclusive union of labeled intervals, retaining overlap and gaps."""
    if not math.isfinite(start+end) or end < start:
        raise ValueError("Invalid window")
    changes = defaultdict(list)
    changes[start]
    changes[end]
    for row in intervals:
        a, b = row["start"], row["end"]
        if not math.isfinite(a+b) or b < a:
            raise ValueError("Invalid interval")
        a, b = max(start, a), min(end, b)
        if a < b:
            changes[a].append((row["category"], 1))
            changes[b].append((row["category"], -1))
    active, totals, pieces = Counter(), Counter(), []
    previous = start
    for now in sorted(changes):
        if now > previous:
            keys = sorted(k for k,v in active.items() if v > 0)
            label = keys[0] if len(keys) == 1 else "overlap:"+"+".join(keys) if keys else "unobserved"
            totals[label] += now-previous
            pieces.append(dict(start=previous, end=now, category=label))
        for key, delta in changes[now]:
            active[key] += delta
        previous = now
    return dict(totals), pieces


def critical_path(nodes, terminal):
    """Trace the latest enabling dependency. Dependencies must be explicit.

    Node intervals must not overlap their predecessors. Queue spans without
    an observed releasing dependency remain waiting, never labeled attention.
    """
    lookup = {n["id"]:n for n in nodes}
    if len(lookup) != len(nodes):
        raise ValueError("Duplicate node")
    visiting, done = set(), set()
    def check(key):
        if key in visiting:
            raise ValueError("Dependency cycle")
        if key in done:
            return
        visiting.add(key)
        n = lookup[key]
        if n["end"] < n["start"]:
            raise ValueError("Negative node duration")
        for pred in n.get("parents", []):
            check(pred)
            if lookup[pred]["end"] > n["start"]+1e-9:
                raise ValueError("Overlapping dependency; split the node")
        visiting.remove(key)
        done.add(key)
    check(terminal)
    path = []
    key = terminal
    while True:
        n = lookup[key]
        path.append(n)
        parents = n.get("parents", [])
        if not parents:
            break
        key = max(parents, key=lambda p:(lookup[p]["end"], p))
    path.reverse()
    intervals = [dict(start=n["start"], end=n["end"], category=n["category"]) for n in path]
    totals, pieces = partition(path[0]["start"], path[-1]["end"], intervals)
    return dict(nodes=[n["id"] for n in path], totals=totals, pieces=pieces,
                tie_rule="latest predecessor end; equal ends by node ID")


def contribution(no_drop, drop):
    delta = sum(no_drop.values())-sum(drop.values())
    return [dict(category=k, no_drop_s=no_drop.get(k,0), drop_s=drop.get(k,0),
                 saved_s=no_drop.get(k,0)-drop.get(k,0),
                 contribution_percent=100*(no_drop.get(k,0)-drop.get(k,0))/delta if delta else None)
            for k in sorted(no_drop.keys() | drop.keys())]


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def coverage(root):
    root = Path(root)
    forward, gpu, counts, ranks, identities, clocks = {}, {}, Counter(), defaultdict(set), set(), []
    stages = 0
    for path in (root/"timing").glob("*.jsonl"):
        for row in rows(path):
            counts[row["kind"]] += 1
            if row.get("rank") is not None:
                ranks[row["role"]].add(row["rank"])
            key = (row["pid"], row.get("step"))
            if row["kind"] == "forward":
                if key in forward:
                    raise ValueError("Duplicate forward")
                forward[key] = row
            elif row["kind"] == "gpu_completed":
                if key in gpu:
                    raise ValueError("Duplicate GPU completion")
                gpu[key] = row
            elif row["kind"] == "request_identity":
                identities.add((row["role"], row["rid"]))
            elif row["kind"] == "request_stage":
                stages += bool(row.get("rid"))
            elif row["kind"] == "clock":
                clocks.append(row["after_ns"]-row["before_ns"])
    missing_requests = sum(not r["requests"] for r in forward.values()
                           if "IDLE" not in r["mode"] and r["batch_size"])
    missing = sorted(set(forward)-set(gpu))
    result = dict(records=dict(counts), ranks={k:sorted(v) for k,v in ranks.items()},
                  forwards=len(forward), gpu_completions=len(gpu),
                  missing_gpu=missing, missing_request_batches=missing_requests,
                  identified_requests=len(identities), identified_stage_events=stages,
                  max_clock_bracket_ns=max(clocks, default=None), attribution_validated=False)
    result["capture_coverage_pass"] = (not missing and not missing_requests and bool(forward)
        and all(ranks[k] == set(range(4)) for k in ("prefill", "decode")) and stages > 0)
    return result


def export_kernels(sqlite_path, output):
    """Stream node-attributed kernels with per-process monotonic clock anchors.

    Nsight timestamp offsets can differ between independent P and D captures.
    Each is aligned by its own e2e_clock marker; discrepancy is reported.
    """
    from analyze_decode_breakdown import RangeIndex, conservative_category
    db = sqlite3.connect(f"file:{Path(sqlite_path).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("select id,value from StringIds"))
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    ranges, anchors = defaultdict(list), defaultdict(list)
    for r in db.execute("select * from NVTX_EVENTS"):
        r = dict(r)
        label = r.get("text") or strings.get(r.get("textId"), "")
        pid = (r["globalTid"] >> 24) & 0xffffff
        if label.startswith("e2e_clock:"):
            anchors[pid].append(int(label.split(":")[1])-r["start"])
        if r.get("end") is not None and label.startswith(("component:", "e2e_step:", "e2e_cpu:")):
            ranges[r["globalTid"]].append((r["start"], r["end"], label))
    indices = {tid:RangeIndex(values) for tid,values in ranges.items()}
    empty = RangeIndex([])
    def annotation(tid, a, b):
        spans = indices.get(tid, empty).enclosing(a,b)
        steps = [int(s[2].split(":")[1]) for s in spans if s[2].startswith("e2e_step:")]
        components = sorted([s for s in spans if s[2].startswith("component:")], key=lambda s:s[1]-s[0])
        cpu = sorted([s for s in spans if s[2].startswith("e2e_cpu:")], key=lambda s:s[1]-s[0])
        return dict(step=steps[0] if steps else None,
                    category=components[0][2].split(":",2)[1] if components else None,
                    cpu=cpu[0][2] if cpu else None)
    runtime, nodes, parents = {}, {}, {}
    for r in db.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"):
        pid = (r["globalTid"] >> 24) & 0xffffff
        runtime[pid,r["correlationId"]] = annotation(r["globalTid"],r["start"],r["end"])
    if "CUDA_GRAPH_NODE_EVENTS" in tables:
        for row in db.execute("select * from CUDA_GRAPH_NODE_EVENTS order by start"):
            r = dict(row)
            if strings.get(r.get("nameId")) != "Graph Node Creation":
                continue
            key = ((r["globalTid"] >> 24)&0xffffff,r["graphNodeId"])
            if r.get("originalGraphNodeId"):
                parents[key] = (key[0],r["originalGraphNodeId"])
            else:
                cat = annotation(r["globalTid"],r["start"],r["start"])["category"]
                if cat:
                    nodes[key] = cat
        for key in parents:
            seen, cur = set(), key
            while cur in parents and cur not in nodes:
                if cur in seen:
                    raise ValueError("Graph provenance cycle")
                seen.add(cur)
                cur = parents[cur]
            if cur in nodes:
                nodes[key] = nodes[cur]
    offsets = {pid:sorted(values)[len(values)//2] for pid,values in anchors.items()}
    counts = Counter()
    with Path(output).open("x") as out:
        for row in db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
            r = dict(row)
            pid = (r["globalPid"] >> 24)&0xffffff
            api = runtime.get((pid,r["correlationId"]),{})
            name = strings.get(r.get("demangledName"),strings.get(r.get("shortName"),"unknown"))
            cat = api.get("category") or nodes.get((pid,r.get("graphNodeId"))) or conservative_category(name)
            if conservative_category(name) == "tp_communication":
                cat = "tp_communication"
            counts[cat] += 1
            offset = offsets.get(pid)
            out.write(json.dumps(dict(pid=pid, start_ns=r["start"], end_ns=r["end"],
                start=(r["start"]+offset)/1e9 if offset is not None else None,
                end=(r["end"]+offset)/1e9 if offset is not None else None,
                step=api.get("step"), category=cat, name=name, stream=r["streamId"], device=r["deviceId"]))+"\n")
    db.close()
    return dict(kernel_counts=dict(counts), clock_offsets_ns=offsets,
                clock_spread_ns={pid:max(v)-min(v) for pid,v in anchors.items()},
                note="Kernel export only; host spans and copies must also be joined before E2E attribution")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument("--nsys-sqlite")
    p.add_argument("--kernels-output")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if bool(a.run) == bool(a.nsys_sqlite):
        p.error("Specify exactly one of --run / --nsys-sqlite")
    value = coverage(a.run) if a.run else export_kernels(a.nsys_sqlite,a.kernels_output)
    Path(a.output).write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2))


if __name__ == "__main__":
    main()
