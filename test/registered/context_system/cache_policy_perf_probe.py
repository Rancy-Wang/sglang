"""Fixed B request diagnostics, isolated from evaluator rollout records.

Manifest entries contain request and previous_request paths to saved request.json.gz.
CPU mode compares the production planner to its retained general implementation.
Serving mode preserves request budgets and records stochastic generation separately.
"""

import argparse
import gzip
import json
import statistics
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import torch


def expected_graph_slots(backend, logical_ids, window):
    """Translate independent oracle IDs for either shared or hybrid KV pools."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    pool = backend.token_to_kv_pool
    if window >= 0 and isinstance(pool, BaseSWAKVPool):
        return pool.translate_loc_from_full_to_swa(logical_ids)
    if not isinstance(pool, (MHATokenToKVPool, BaseSWAKVPool)):
        raise TypeError(f"Unsupported diagnostic KV pool: {type(pool).__name__}")
    return backend.kv_index_translator.translate_full_attn_ids(logical_ids)


def install_perf_graph_probe():
    """Inspect buffers outside capture, immediately before native graph replay."""
    from serving_logits_probe import independent_policy_decode_pairs
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    if getattr(DecodeCudaGraphRunner, "_cache_policy_perf_graph_probe", False):
        return
    original = DecodeCudaGraphRunner.load_batch

    def load_batch(runner, batch, *args, **kwargs):
        result = original(runner, batch, *args, **kwargs)
        params = getattr(getattr(batch, "sampling_info", None), "custom_params", None)
        if (not batch.forward_mode.is_decode() or not params
                or len(params) != 1 or not params[0]
                or not params[0].get("context_graph_read_probe")):
            return result
        import json
        from pathlib import Path

        import torch
        from sglang.srt.context_system.request_storage import request_row

        backend = runner._replay_attn_backend()
        param = params[0]
        generated = len(param.get("_context_probe_rows", []))
        if generated >= param["context_trace_count"]:
            return result  # Native overlap may execute a discarded final step.
        generated += param.get("context_forced_offset", 0)
        req = param["__req__"]
        position = int(batch.positions[0])
        assert generated >= 1
        assert int(batch.input_ids[0]) == param["context_forced_tokens"][generated - 1]
        row = request_row(backend.req_to_token_pool, req.kv.req_pool_idx)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        windows = [-1]
        if backend.sliding_window_size is not None and backend.sliding_window_size > 0:
            windows.append(backend.sliding_window_size)
        for window in windows:
            pairs = independent_policy_decode_pairs(param["context_policy_spec"], generated, position, window)
            raw = torch.tensor([i for i, _ in pairs], device=row.device, dtype=torch.long)
            wanted = row[raw]
            if window >= 0:
                wanted = expected_graph_slots(backend, wanted, window)
                indptr, indices = backend.window_kv_indptr, backend.cuda_graph_window_kv_indices
            else:
                wanted = expected_graph_slots(backend, wanted, window)
                indptr, indices = backend.kv_indptr, backend.cuda_graph_kv_indices
            actual = indices[int(indptr[0]):int(indptr[1])]
            report = {"rank": rank, "position": position, "generated": generated,
                      "window": window, "expected_count": len(pairs), "read_count": len(actual),
                      "slots_ok": torch.equal(actual, wanted), "source": "native-graph-replay-buffers"}
            with Path(param["context_trace_path"] + f".graph-reads.rank{rank}.jsonl").open("a") as out:
                out.write(json.dumps(report) + "\n")
            if not report["slots_ok"]:
                raise AssertionError(f"Graph physical reader mismatch: {report}")
        return result

    DecodeCudaGraphRunner.load_batch = load_batch
    DecodeCudaGraphRunner._cache_policy_perf_graph_probe = True


def serialized_perf_probe():
    """Use the existing forced-token observer with a pool-aware graph audit."""
    from serving_logits_probe import serialized_probe
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    base = serialized_probe(graph_metadata=False)

    class Probe(CustomLogitProcessor):
        def __init__(self):
            from cache_policy_perf_probe import install_perf_graph_probe

            install_perf_graph_probe()
            self.delegate = CustomLogitProcessor.from_str(base)

        def __call__(self, logits, custom_param_list):
            return self.delegate(logits, custom_param_list)

    return Probe.to_str()


def read_request(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def compile_request(body):
    from contextualize.harmony_encoding import HarmonyEncoder
    from sglang.srt.context_system.planner import compile_chat_program
    from sglang.srt.context_system.rules import parse_drop_rule

    assert body["drop_rule"] == "message_drop_without_repos"
    assert not body.get("reposition")
    ids, owners, _ = HarmonyEncoder("high").render_tokens(body["messages"], body.get("tools"))
    rule = parse_drop_rule(body["drop_rule"], body["messages"], legacy_drop_message=body.get("drop_message"))
    return compile_chat_program(body["messages"], SimpleNamespace(input_ids=ids, owners=owners), rule, None)


def cpu_probe(entry, repeats):
    from sglang.srt.context_system import recovery
    from sglang.srt.context_system.ir import ContextKeyData

    program = compile_request(read_request(entry["request"]))
    previous = compile_request(read_request(entry["previous_request"]))
    key, old = [ContextKeyData.from_layout(p.layout) for p in (program, previous)]
    n = len(program.input_ids)
    matched = min(n-1, key.match(old, 0, 0, min(n, len(previous.input_ids))))
    assert matched > 0
    output = []
    generator = torch.Generator().manual_seed(1118)
    for pressure in ("retained", "dropped_holes", "sparse_holes"):
        resident = torch.ones(matched, dtype=torch.bool)
        if pressure == "dropped_holes":
            resident &= program.layout.keep_mask[:matched]
        elif pressure == "sparse_holes":
            resident &= torch.rand(matched, generator=generator) > .1
        swa = resident & (torch.rand(matched, generator=generator) > .05)
        arguments = (resident, program.visible_until, program.layout, None, None,
                     swa, 128, torch.zeros(matched, dtype=torch.bool),
                     program.layout.positions[:matched])
        guard = recovery._history_layout
        results, times = {}, {}
        try:
            for mode in ("general", "fast"):
                recovery._history_layout = (lambda layout: False) if mode == "general" else guard
                # Exclude compilation and one warmup from timing.
                results[mode] = recovery._plan_insert_recovery(*arguments)
                times[mode] = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    results[mode] = recovery._plan_insert_recovery(*arguments)
                    times[mode].append(time.perf_counter()-start)
        finally:
            recovery._history_layout = guard
        a, b = results["general"], results["fast"]
        assert a.intervals == b.intervals
        assert torch.equal(a.required_prefix, b.required_prefix)
        assert torch.equal(a.reusable_prefix, b.reusable_prefix)
        output.append({"pressure": pressure, "prompt": n, "matched": matched, "times": times,
                           "speedup": statistics.median(times["general"])/statistics.median(times["fast"]),
                           "planned": sum(z-a for a, z in b.intervals)})
    return output


def serve_probe(entry, base_url):
    body = read_request(entry["request"])
    assert body["drop_rule"] == "message_drop_without_repos"
    assert not body.get("reposition")
    # Budget and sampling remain the original values. Stream framing is not part
    # of the model input; omit stream-only options for the ordinary JSON response.
    body["stream"] = False
    body.pop("stream_options", None)
    start = time.perf_counter()
    request = urllib.request.Request(base_url.rstrip("/")+"/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    assert not result.get("error")
    return {"seconds": time.perf_counter()-start, "usage": result.get("usage"),
                "finish_reasons": [c.get("finish_reason") for c in result.get("choices", [])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "serve"], required=True)
    parser.add_argument("--server", default="http://127.0.0.1:30182")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    report = {"completed": False, "mode": args.mode, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        value = cpu_probe(entry, args.repeats) if args.mode == "cpu" else serve_probe(entry, args.server)
        report["results"].append({"entry": entry, "result": value})
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"completed": len(report["results"]), "total": len(manifest)}), flush=True)
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
