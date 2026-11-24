#!/usr/bin/env python3
"""Lower-overhead graph spans/gaps; never the primary throughput measurement."""
import collections
import json
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import time

import decode_sweep as sweep
import context_diagnostic as context
import a10_paperlong_pipeline as safe
import benchmark_throughput as bench

ROOT = sweep.ROOT / "light"


def analyze(path, expected=(8, 9)):
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    # Some Nsight graph tables contain warmups. Only runtime-correlated launches
    # recorded during the active cudaProfilerApi range belong in this result.
    graphs = conn.execute("""select g.start,g.end,g.correlationId
       from CUPTI_ACTIVITY_KIND_GRAPH_TRACE g where g.correlationId in
       (select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
        where s.value like 'cudaGraphLaunch%') order by g.start""").fetchall()
    assert len(graphs) in expected, f"Captured {len(graphs)} graphs, expected {expected}"
    assert len({r[2] for r in graphs}) == len(graphs), "Ambiguous nested/repeated graph correlations"
    spans = [(end-start)/1e6 for start,end,_ in graphs]
    gaps = [(b[0]-a[1])/1e6 for a,b in zip(graphs, graphs[1:])]
    assert all(x > 0 for x in spans) and all(x >= 0 for x in gaps)
    apis = collections.defaultdict(list)
    for name, start, end in conn.execute("""select s.value,r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r
          join StringIds s on s.id=r.nameId"""):
        apis[name].append((end-start)/1e6)
    return {"source": str(Path(path).resolve()), "decode_graphs": len(graphs),
            "graph_ms_mean": statistics.mean(spans), "graph_ms_median": statistics.median(spans),
            "graph_ms": spans, "gap_ms_mean": statistics.mean(gaps),
            "gap_ms_median": statistics.median(gaps), "gap_ms": gaps,
            "cpu_apis": {name: {"calls": len(v), "total_ms": sum(v), "mean_ms": statistics.mean(v),
                                "median_ms": statistics.median(v), "max_ms": max(v)} for name,v in apis.items()},
            "note": "Graph-only instrumented timeline. Gaps can include encoding and scheduling; CPU waits overlap GPU and must not be added to GPU time. Not unprofiled throughput."}


def run():
    assert os.environ.get("TMUX")
    assert safe.capture(["git", "-C", str(safe.FROZEN_REPO), "rev-parse", "HEAD"]) == safe.FROZEN_SHA
    original = sweep.ROOT
    ROOT.mkdir(parents=True, exist_ok=True)
    state = ROOT / "status.json"
    assert not state.exists()
    assert safe.read_json(original / "profile-recovery-status.json")["status"] == "completed"
    safe.new_json(ROOT / "input-2048.json", safe.read_json(original / "input-2048.json"))
    assert safe.sha256(ROOT / "input-2048.json") == safe.sha256(original / "input-2048.json")
    sweep.ROOT = context.ROOT = ROOT
    base = {"status": "running", "started_unix": time.time(), "inference_sha": safe.FROZEN_SHA,
            "helper_sha256": safe.sha256(__file__), "gpu_uuid": context.UUID, "bs": 4}
    safe.new_json(state, base)
    export_env = dict(os.environ, NSYS_TMPDIR=f"/tmp/loopvq-nsys-{os.getuid()}", TMPDIR=f"/tmp/loopvq-nsys-{os.getuid()}")
    try:
        for method in ("bf16", "loopvq"):
            identity, manifest, log, launcher = context.start(method, True,
                launch_script=Path(__file__).resolve().parent / "a10_throughput_server.sh",
                server_protocol="decode-sweep", graph_trace="graph")
            bench.atomic_json(state, {**base, "method": method, "server": identity})
            sweep.profile_batch(method, 4, manifest, log)
            context.stop(identity)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                reports = list(ROOT.glob(f"{method}-nodes.*.nsys-rep"))
                if len(reports) == 4 and all(p.stat().st_size > 0 for p in reports):
                    break
                time.sleep(10)
            assert len(reports) == 4, f"Incomplete graph export: {launcher}"
            for index, output in enumerate(sweep.OUTPUTS, 1):
                report = ROOT / f"{method}-nodes.{index}.nsys-rep"
                sql = ROOT / f"{method}-out{output}-bs4.sqlite"
                subprocess.run(["nsys", "export", "--type=sqlite", f"--output={sql}", str(report)], env=export_env, check=True)
                safe.new_json(ROOT / f"{method}-out{output}-bs4-timeline.json", analyze(sql))
        bench.atomic_json(state, {**base, "status": "completed", "ended_unix": time.time(), "gpu_released": True})
    except BaseException as exc:
        bench.atomic_json(state, {**safe.read_json(state), "status": "failed", "error_type": type(exc).__name__,
                                  "error": str(exc), "ended_unix": time.time()})
        raise


if __name__ == "__main__":
    run()
