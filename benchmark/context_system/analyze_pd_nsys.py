"""Audit a full PD Nsight SQLite export and draw all four CUDA worker lanes.

CPU API elapsed time, GPU execution coverage, and request state intervals are
different quantities. The output deliberately retains each separately.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from analyze_pd_timing import analyze, union_overlap


def read_cpu_samples(path, origin):
    samples, previous = [], {}
    if not path.exists():
        return samples
    for line in path.read_text().splitlines():
        row = json.loads(line)
        current = {}
        totals = defaultdict(lambda: [0.0, 0.0])
        for pid, tid, name, run, wait, slices in row["threads"]:
            key = (pid, tid)
            current[key] = (row["time"], run, wait)
            if key not in previous:
                continue
            start, old_run, old_wait = previous[key]
            if run < old_run or wait < old_wait or row["time"] <= start:
                continue
            totals[pid][0] += (run - old_run) / 1e9
            totals[pid][1] += (wait - old_wait) / 1e9
        if previous:
            start = next(iter(previous.values()))[0]
            for pid, (run, wait) in totals.items():
                samples.append(dict(pid=pid,start=start-origin,end=row["time"]-origin,
                                    cpu_running_seconds=run,runnable_wait_seconds=wait))
        previous = current
    return samples


def extract(root, database):
    c = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}

    def rows(query, parameters=()):
        return [dict(r) for r in c.execute(query, parameters)]

    clock = rows("select * from TARGET_INFO_SESSION_START_TIME")[0]
    origin = clock["systemClockNs"] / 1e9
    timing = analyze(root, 1.0)
    transfers, roles = [], {}
    for mode in ("prefill", "decode"):
        for line in (root / f"{mode}.log").read_text(errors="replace").splitlines():
            if "pd_matrix_transfer=" in line:
                row = json.loads(line.split("pd_matrix_transfer=", 1)[1])
                row["start"] -= origin
                row["end"] = row["start"] + row["seconds"]
                transfers.append(row)
                roles[row["pid"]] = mode
    forwards = []
    if "NVTX_EVENTS" in tables:
        for row in rows("""select (n.globalTid>>24)&16777215 pid,
                n.globalTid&16777215 tid,n.start/1e9 start,n.end/1e9 end,
                coalesce(n.text,s.value) label from NVTX_EVENTS n
                left join StringIds s on s.id=n.textId
                where coalesce(n.text,s.value) like 'pd_forward:%'"""):
            data = json.loads(row.pop("label").split(":", 1)[1])
            row.update(data)
            roles[row["pid"]] = "prefill" if row["pf"] else "decode"
            forwards.append(row)
    kernel = defaultdict(list)
    kernel_counts = defaultdict(int)
    gpu_compute_tables = [t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE") if t in tables]
    if gpu_compute_tables:
        query = " union all ".join(f"select (globalPid>>24)&16777215 pid,start/1e9 start,end/1e9 end from {t}" for t in gpu_compute_tables)
        for r in c.execute(query + " order by start"):
            pid, start, end = r
            kernel_counts[pid] += 1
            values = kernel[pid]
            if values and start <= values[-1][1]:
                values[-1][1] = max(values[-1][1], end)
            else:
                values.append([start, end])
    runtime, osrt = [], []
    for table, output in (("CUPTI_ACTIVITY_KIND_RUNTIME", runtime), ("OSRT_API", osrt)):
        if table not in tables:
            continue
        # Keep all long CPU calls. poll/condition waits are often ordinary idle
        # waits; their presence alone does not prove a transfer bottleneck.
        output.extend(rows(f"""select (a.globalTid>>24)&16777215 pid,
            a.globalTid&16777215 tid,a.start/1e9 start,a.end/1e9 end,
            s.value name,a.callchainId
            {',a.correlationId' if table.endswith('RUNTIME') else ''}
            from {table} a join StringIds s on s.id=a.nameId
            where a.end-a.start>=50000000 order by a.start"""))
    stacks = {}
    if "OSRT_CALLCHAINS" in tables:
        ids = {r["callchainId"] for r in osrt if "rwlock" in r["name"] and r["callchainId"] is not None}
        for ident in ids:
            stacks[ident] = rows("""select a.stackDepth,s.value symbol,m.value module
                from OSRT_CALLCHAINS a left join StringIds s on s.id=a.symbol
                left join StringIds m on m.id=a.module where a.id=? order by a.stackDepth""", (ident,))
    copies, copy_summary = [], []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copy_summary = rows("""select (globalPid>>24)&16777215 pid,copyKind,
            count(*) count,sum(bytes) bytes,sum(end-start)/1e9 sum_seconds,
            max(end-start)/1e9 max_seconds from CUPTI_ACTIVITY_KIND_MEMCPY group by pid,copyKind""")
        correlations = {(r["pid"], r["correlationId"]) for r in runtime}
        for row in c.execute("""select (globalPid>>24)&16777215 pid,
            correlationId,start/1e9 start,end/1e9 end,bytes,copyKind,streamId
            from CUPTI_ACTIVITY_KIND_MEMCPY"""):
            if (row["pid"], row["correlationId"]) in correlations:
                copies.append(dict(row))
    coverage = []
    for pid in sorted(set(roles) | set(kernel)):
        values = kernel[pid]
        coverage.append(dict(pid=pid, role=roles.get(pid, "unknown"),
                             kernel_count=kernel_counts[pid],
                             first=values[0][0] if values else None,
                             last=values[-1][1] if values else None,
                             gpu_execution_seconds=sum(b-a for a,b in values)))
    tails = sorted((r for r in timing["requests"] if r["stage_s"].get("p_finished_to_d_ready") is not None),
                   key=lambda r:r["stage_s"]["p_finished_to_d_ready"], reverse=True)[:10]
    evidence = []
    for req in tails:
        pspan = req["stage_intervals"]["p_forward_including_chunk_gaps"]
        dspan = req["stage_intervals"]["d_ready_wait"]
        if not pspan or not dspan:
            continue
        window = [pspan[1]-origin, dspan[0]-origin]
        if window[1] < window[0]:
            continue
        intersect = lambda r: r["start"] < window[1] and r["end"] > window[0]
        evidence.append(dict(room=req["room"],case_id=req["case_id"],turn=req["turn"],
                             window=window,seconds=window[1]-window[0],
                             gpu_execution_seconds_by_pid={pid:union_overlap(window, intervals) for pid,intervals in kernel.items()},
                             transfers=[r for r in transfers if intersect(r)],
                             runtime=[r for r in runtime if intersect(r)],
                             locks=[r for r in osrt if "rwlock" in r["name"] and intersect(r)]))
    expected = {role:sum(r["role"]==role and r["kernel_count"]>0 for r in coverage) for role in ("prefill","decode")}
    c.close()
    cpu_samples = read_cpu_samples(root / "cpu-sched.jsonl", origin)
    return dict(root=str(root),clock=clock,origin=origin,coverage=coverage,cpu_samples=cpu_samples,
                four_worker_gpu_trace=expected == {"prefill":2,"decode":2},
                roles=roles,kernel_intervals=kernel,forwards=forwards,transfers=transfers,
                runtime=runtime,osrt=osrt,stacks=stacks,copy_summary=copy_summary,
                correlated_copies=copies,request_timing=timing,longest_transfer_tails=evidence,
                limitations=["GPU execution coverage is not SM utilization.",
                    "Kernel intervals include whole CUDA Graph executions when graph-level tracing is used; graph-internal gaps are not resolved.",
                    "CPU API elapsed time is not GPU copy duration or CPU running time.",
                    "CPU run/runnable-wait counters are approximately 1 Hz /proc samples across threads; not a context-switch trace or stack sampler.",
                    "P finish to D ready includes host bookkeeping, polling, transfer queueing and actual copies.",
                    "Concurrent intervals overlap and must not be added as a wall-clock decomposition."])


def plot(data, destination, window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = sorted(data["roles"], key=lambda p:(data["roles"][p],p), reverse=True)
    if not pids:
        raise ValueError("No P/D workers in trace; cannot draw an empty GPU timeline")
    fig, axes = plt.subplots(len(pids)*4, 1, figsize=(18, max(5, len(pids)*4)), sharex=True, squeeze=False)
    for i, pid in enumerate(pids):
        specs = [(data["kernel_intervals"].get(pid, []), "GPU kernels / graphs", "#2a9d8f"),
                 ([(r["start"],r["end"]) for r in data["runtime"] if r["pid"]==pid], "CUDA API >=50 ms", "#e76f51"),
                 ([(r["start"],r["end"]) for r in data["transfers"] if r["pid"]==pid], "KV transfer calls", "#457b9d")]
        for j, (intervals,label,color) in enumerate(specs):
            ax=axes[i*4+j,0]
            visible=[(a,b-a) for a,b in intervals if window is None or (a<window[1] and b>window[0])]
            ax.broken_barh(visible,(0,1),facecolors=color)
            ax.set_yticks([])
            ax.set_ylabel(f"{data['roles'][pid]} PID {pid}\n{label}",rotation=0,ha="right",va="center",fontsize=8)
            ax.grid(axis="x",alpha=.2)
            if window: ax.set_xlim(*window)
        ax=axes[i*4+3,0]
        samples=[r for r in data["cpu_samples"] if r["pid"]==pid and (window is None or r["start"]<window[1] and r["end"]>window[0])]
        for field, color, label in (("cpu_running_seconds","#2a9d8f","Running"),("runnable_wait_seconds","#e9c46a","Runnable wait")):
            ax.step([r["end"] for r in samples],[r[field]/(r["end"]-r["start"]) for r in samples],where="pre",color=color,label=label)
        ax.set_ylabel(f"{data['roles'][pid]} PID {pid}\nCPU thread sum (cores)",rotation=0,ha="right",va="center",fontsize=8)
        ax.legend(loc="upper right",fontsize=7)
        ax.grid(axis="x",alpha=.2)
    axes[-1,0].set_xlabel("Seconds since Nsight session origin (same-host monotonic clock)")
    fig.suptitle(Path(data["root"]).name + " — GPU execution and CPU elapsed intervals",fontsize=13)
    fig.tight_layout(rect=(.03,0,1,.97))
    fig.savefig(destination,dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--sqlite",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    data=extract(a.root,a.sqlite)
    (a.output/"timeline.json").write_text(json.dumps(data,separators=(",",":")))
    plot(data,a.output/"overview.png")
    for i,tail in enumerate(data["longest_transfer_tails"][:3]):
        start,end=tail["window"]
        plot(data,a.output/f"long-tail-{i+1}.png",[start-1,end+1])
    print(json.dumps({k:data[k] for k in ("four_worker_gpu_trace","coverage","copy_summary")}))
