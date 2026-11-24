#!/usr/bin/env python3
"""Attribute frozen Ouro R4 node traces by validated execution boundaries.

GPU kernel durations are exclusive category sums, not CPU wall time. Generic
RoPE pointwise/layout operations remain one honest category. No kernel rename
or modification of captured inference is needed.
"""
import argparse
import collections
import json
from pathlib import Path
import sqlite3
import statistics


def norm(name):
    return "rmsnorm" in name.lower() or "rms_norm" in name.lower()


def matrix(name):
    return any(x in name.lower() for x in ("gemm", "gemv", "cutlass", "cublas"))


def attention(name):
    for marker, category in (("vqg_stage1", "attention.quant"),
                             ("_official_hp_stage1", "attention.hp"),
                             ("_official_reduce", "attention.reduce"),
                             ("_fwd_kernel_stage1", "attention.bf16_stage1"),
                             ("_fwd_kernel_stage2", "attention.bf16_stage2")):
        if marker in name:
            return category
    return None


def classify_graph(rows):
    indices = [i for i, row in enumerate(rows) if norm(row[2])]
    assert len(indices) == 772, f"Expected 4*(48*4+1) norms, got {len(indices)}"
    assigned = {}
    counters = collections.Counter()

    def put(index, loop, category):
        assert index not in assigned
        assigned[index] = (loop, category)

    for loop in range(4):
        ns = indices[loop*193:(loop+1)*193]
        for layer in range(48):
            a, b, c, d, e = ns[layer*4:layer*4+5]
            for index in (a, b, c, d):
                put(index, loop, "norm.sandwich")
            attn = [i for i in range(a+1, b) if attention(rows[i][2])]
            assert len(attn) in (2, 3), f"Unrecognized attention sequence: {len(attn)}"
            for i in range(a+1, b):
                name = rows[i][2]
                cat = attention(name)
                if cat:
                    counters[(loop, cat)] += 1
                elif "_tier_" in name:
                    cat = "cache.metadata"
                elif "_write_hp" in name or "store_kvcache" in name:
                    cat = "cache.write"
                elif "FillFunctor<float>" in name:
                    cat = "attention.lse_reset"
                elif matrix(name):
                    cat = "projection.qkv" if i < attn[0] else "projection.o"
                else:
                    cat = "rope_layout_pointwise"
                put(i, loop, cat)
            for i in range(b+1, c):
                put(i, loop, "residual.add")
            activation = [i for i in range(c+1, d) if "silu" in rows[i][2].lower()]
            assert len(activation) == 1, "MLP SiLU boundary missing or ambiguous"
            for i in range(c+1, d):
                cat = ("mlp.gate_up" if i < activation[0] else "mlp.down") if matrix(rows[i][2]) else "mlp.activation"
                put(i, loop, cat)
            for i in range(d+1, e):
                put(i, loop, "residual.add")
        put(ns[-1], loop, "norm.loop_end")
        # Any inter-loop work is explicit, rather than dropped from accounting.
        end = indices[(loop+1)*193] if loop < 3 else ns[-1]+1
        for i in range(ns[-1]+1, end):
            put(i, loop, "loop_transition")
    for loop in range(4):
        counts = {cat: count for (l, cat), count in counters.items() if l == loop}
        assert counts and all(count == 48 for count in counts.values()), counts
    for i in range(len(rows)):
        if i not in assigned:
            put(i, -1, "embedding_lmhead_sampling")
    result = collections.defaultdict(float)
    for i, (loop, cat) in assigned.items():
        result[(loop, cat)] += (rows[i][1] - rows[i][0]) / 1e6
    assert abs(sum(result.values()) - sum((r[1]-r[0])/1e6 for r in rows)) < 1e-6
    return result


def analyze(path, expected):
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    # Runtime correlation restricts selection to actually recorded launches.
    query = """select k.correlationId,k.start,k.end,s.value
      from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
      where k.graphId is not null and k.correlationId in
       (select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds n on n.id=r.nameId
        where n.value like 'cudaGraphLaunch%') order by k.start"""
    groups = collections.defaultdict(list)
    for correlation, start, end, name in conn.execute(query):
        groups[correlation].append((start, end, name))
    assert groups and (expected is None or len(groups) == expected), f"Captured graph count {len(groups)}, expected {expected}"
    values = collections.defaultdict(list)
    spans = []
    for correlation, rows in groups.items():
        for key, ms in classify_graph(rows).items():
            values[key].append(ms)
        spans.append({"correlation": correlation, "start": min(r[0] for r in rows),
                      "end": max(r[1] for r in rows), "kernel_ms": sum((r[1]-r[0])/1e6 for r in rows)})
    spans.sort(key=lambda x: x["start"])
    offgraph = []
    for name, ns, count in conn.execute("""select s.value,sum(k.end-k.start),count(*)
       from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
       where k.graphId is null group by s.value order by sum(k.end-k.start) desc"""):
        offgraph.append({"kernel": name, "gpu_ms": ns/1e6, "calls": count})
    return {"source": str(Path(path).resolve()), "decode_graphs": len(groups),
            "units": "GPU kernel milliseconds per decoded batch step; exclusive categories, loop index 1..4",
            "note": "Node profiling perturbs scheduling. Offgraph includes prefill/encoding; not isolated CPU cost. Graph gaps can overlap offgraph work.",
            "categories": [{"loop": loop+1 if loop >= 0 else "outside", "category": cat,
                            "mean_ms": sum(ms)/len(groups), "observed_graphs": len(ms)}
                           for (loop, cat), ms in sorted(values.items())],
            "graph_kernel_ms_mean": statistics.mean(x["kernel_ms"] for x in spans),
            "graph_span_ms_mean": statistics.mean((x["end"]-x["start"])/1e6 for x in spans),
            "intergraph_gap_ms": [(b["start"]-a["end"])/1e6 for a,b in zip(spans, spans[1:])],
            "offgraph_kernel_totals": offgraph, "graphs": spans}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--expected-graphs", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.expected_graphs)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("source", "decode_graphs", "graph_kernel_ms_mean")}))
