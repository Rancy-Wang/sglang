#!/usr/bin/env python3
"""Fixed 2K input, 1K/2K/4K/8K NEW tokens; pure full-batch decode rate.

All primary rates exclude prefill and first-decode admission. The old HTTP
measurement JSON remains intact as auxiliary evidence. Run in A10 tmux.
"""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import benchmark_throughput as bench
import a10_paperlong_pipeline as safe
import context_diagnostic as context
import profile_loop_breakdown as loops

ROOT = safe.TASK / "results/decode-20261124-r12"
OUTPUTS = (1024, 2048, 4096, 8192)
BATCHES = (1, 2, 4, 8, 16)
INPUT_LEN = 2048
HERE = Path(__file__).resolve().parent
ORIGINAL_BATCH = bench.run_batch
ORIGINAL_STATS = bench.StreamStats


def fixture():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(safe.MODEL, trust_remote_code=True)
    marker = "LOOPVQ_CONTEXT_BODY"
    rendered = tok.apply_chat_template([{"role": "user", "content": marker}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    assert rendered.count(marker) == 1
    prefix, suffix = rendered.split(marker)
    p, s = (tok.encode(text, add_special_tokens=False) for text in (prefix, suffix))
    body = tok.encode("Read the following notes carefully. Each numbered observation describes a simple arithmetic example. " * 1000,
                      add_special_tokens=False)
    ids = p + body[:INPUT_LEN-len(p)-len(s)] + s
    assert len(ids) == INPUT_LEN
    safe.new_json(ROOT / "input-2048.json", {"input_ids": [ids]*16,
                  "construction": "Native chat delimiters; body truncated in token space, supplied as input_ids",
                  "chat_template": tok.chat_template, "rendered_template_with_marker": rendered,
                  "prefix_ids": p, "suffix_ids": s, "enable_thinking": True,
                  "model_path": str(safe.MODEL), "token_ids_sha256": bench.digest(ids)})


def decode_interval(events_path, bs, output_len):
    """Use common count barriers, never per-request first-token TPOT."""
    arrivals = [dict() for _ in range(bs)]
    with Path(events_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            index, count, seconds = row["index"], row["completion_tokens"], row["arrival_seconds"]
            assert 0 <= index < bs and 0 <= count <= output_len and seconds >= 0
            arrivals[index].setdefault(count, seconds)
    common = set.intersection(*(set(row) for row in arrivals))
    first = min(count for count in common if count >= 2)
    assert first <= 8 and output_len in common, "Missing early or final full-batch token barrier"
    barrier = lambda count: max(row[count] for row in arrivals)
    start, end = barrier(first), barrier(output_len)
    assert end > start
    boundaries = [first] + [x for x in range(1024, output_len, 1024) if x in common and x > first] + [output_len]
    windows = []
    for a, b in zip(boundaries, boundaries[1:]):
        elapsed = barrier(b) - barrier(a)
        assert elapsed > 0
        windows.append({"from_completion_count": a, "to_completion_count": b,
                        "tokens": bs*(b-a), "seconds": elapsed, "tokens_s": bs*(b-a)/elapsed})
    return {"metric": "full_batch_client_decode_tokens_s", "start_completion_count": first,
            "end_completion_count": output_len, "decode_tokens": bs*(output_len-first),
            "start_seconds": start, "end_seconds": end, "decode_seconds": end-start,
            "decode_tokens_s": bs*(output_len-first)/(end-start), "windows": windows,
            "definition": "Max first-arrival timestamp across requests at each common token-count barrier; prefill and first decode excluded",
            "events_sha256": safe.sha256(events_path)}


def short_warmup(args, inputs, phase, number, config, manifest, capture):
    effective = copy.copy(args)
    if phase == "warmup":
        effective.output_len = 32
    return ORIGINAL_BATCH(effective, inputs, phase, number, config, manifest, capture)


def arguments(method, output, bs, out, manifest, log):
    protocol = f"decode-output-{output}"
    bench.PROTOCOLS[protocol] = (INPUT_LEN, output, 8, 64)
    m = copy.deepcopy(manifest)
    m["protocol"] = protocol
    safe.new_json(out / "manifest.input", m)
    return argparse.Namespace(base_url=context.URL, method=method, protocol=protocol, bs=bs,
                              input_len=INPUT_LEN, output_len=output,
                              input_ids_json=str(ROOT / "input-2048.json"),
                              server_manifest=str(out / "manifest.input"), server_log=str(log),
                              out_dir=str(out), timeout=3600), m


def measure_cell(method, output, bs, manifest, log):
    out = ROOT / "timing" / method / str(output) / f"bs{bs}"
    assert not out.exists(), f"Preserve previous cell: {out}"
    out.mkdir(parents=True)
    args, m = arguments(method, output, bs, out, manifest, log)
    required = bs*((INPUT_LEN+output+7)//8*8)
    capacity = manifest["config"]["max_total_tokens"]
    if required > capacity:
        safe.new_json(out / "decode-summary.json", {"status": "capacity_limited_not_run", "method": method,
                      "input_len": INPUT_LEN, "output_len": output, "bs": bs, "server_manifest": m,
                      "required_tokens": required, "pool_upper_bound": capacity,
                      "requests_sent": 0, "decode_tokens_s_median": None})
        return
    bench.run_batch = short_warmup
    try:
        assert bench.measure(args) == 0, f"Cell failed: {out}"
    finally:
        bench.run_batch = ORIGINAL_BATCH
    auxiliary = safe.read_json(out / "summary.json")
    rates = []
    for i in range(1, 4):
        raw = safe.read_json(out / f"measure-{i:02d}.json")
        assert raw["status"] == "ok" and raw["batch_evidence"]["verified"]
        row = decode_interval(out / f"measure-{i:02d}.events.jsonl", bs, output)
        safe.new_json(out / f"decode-measure-{i:02d}.json", row)
        rates.append(row["decode_tokens_s"])
    safe.new_json(out / "decode-summary.json", {"status": "ok", "method": method,
                  "input_len": INPUT_LEN, "output_len": output, "bs": bs,
                  "primary_metric": "full_batch_client_decode_tokens_s", "server_manifest": m,
                  "server_config_readback": auxiliary["server_config_readback"],
                  "input_ids_sha256": auxiliary["input_ids_sha256"],
                  "input_fixture_sha256": auxiliary["input_fixture_sha256"],
                  "decode_tokens_s_median": statistics.median(rates),
                  "decode_tokens_s_min": min(rates), "decode_tokens_s_max": max(rates), "rates": rates,
                  "warmup_count": 2, "warmup_output_len": 32, "repeat_count": 3,
                  "actual_batch_fraction": auxiliary["min_target_batch_fraction"],
                  "auxiliary_http_summary": str(out / "summary.json"),
                  "prefill_in_primary_metric": False})
    print(json.dumps({"method": method, "output": output, "bs": bs, "decode_tokens_s": statistics.median(rates)}), flush=True)


def aggregate():
    rows = []
    for output in OUTPUTS:
        for bs in BATCHES:
            cells = {}
            for method in ("bf16", "loopvq"):
                path = ROOT / "timing" / method / str(output) / f"bs{bs}" / "decode-summary.json"
                cells[method] = safe.read_json(path) if path.exists() else {"status": "pending"}
            b, q = cells["bf16"], cells["loopvq"]
            both = b["status"] == q["status"] == "ok"
            if both:
                for key in ("input_len", "output_len", "bs", "input_ids_sha256", "input_fixture_sha256",
                            "warmup_count", "warmup_output_len", "repeat_count", "primary_metric"):
                    assert b[key] == q[key], f"Pair mismatch: {key}"
                for key in bench.IDENTITY_KEYS:
                    assert b["server_manifest"][key] == q["server_manifest"][key], f"Identity mismatch: {key}"
                for key in bench.PAIR_CONFIG_KEYS:
                    assert b["server_config_readback"].get(key) == q["server_config_readback"].get(key), key
            left, right = b.get("decode_tokens_s_median"), q.get("decode_tokens_s_median")
            rows.append({"input_tokens": INPUT_LEN, "max_new_tokens": output, "bs": bs,
                         "bf16_status": b["status"], "loopvq_status": q["status"],
                         "bf16_decode_tokens_s": left, "loopvq_decode_tokens_s": right,
                         "decode_speedup": right/left if both else None})
    report = {"primary_metric": "full_batch_client_decode_tokens_s", "prefill_included": False,
              "gpu_uuid": context.UUID, "inference_sha": safe.FROZEN_SHA,
              "complete": all(r["bf16_status"] in ("ok", "capacity_limited_not_run") and r["loopvq_status"] == "ok" for r in rows),
              "rows": rows, "updated_unix": time.time()}
    bench.atomic_json(ROOT / "decode-throughput.json", report)
    return report


class ProfileStats(ORIGINAL_STATS):
    triggers = None

    def feed(self, item, seconds):
        result = super().feed(item, seconds)
        threshold = (1024, 2048, 4096, 8160)[len(self.triggers)] if len(self.triggers) < 4 else None
        counts = [r["completion_tokens"] for r in self.requests]
        if threshold is not None and min(counts) >= threshold:
            before = time.time()
            response = context.request("/start_profile", {"activities": ["CUDA_PROFILER"],
                                       "num_steps": 8, "detailed_annotations": True})
            self.triggers.append({"target_completion": threshold, "observed_counts_before_start": counts,
                                  "start_unix": before, "response_unix": time.time(),
                                  "response": response.decode(errors="replace")})
        return result


def profile_batch(method, bs, manifest, log):
    out = ROOT / "profile" / method / f"bs{bs}"
    assert not out.exists()
    out.mkdir(parents=True)
    args, m = arguments(method, 8192, bs, out, manifest, log)
    cfg = bench.get_server_config(context.URL, 30)
    bench.validate_manifest(m, cfg, method, args.protocol, bs, 8, 64)
    inputs = safe.read_json(args.input_ids_json)["input_ids"][:bs]
    capture = bench.capture_from_log(log)
    for number in (1, 2):
        assert short_warmup(args, inputs, "warmup", number, cfg, m, capture)["status"] == "ok"
    ProfileStats.triggers = []
    bench.StreamStats = ProfileStats
    try:
        result = ORIGINAL_BATCH(args, inputs, "profile", 1, cfg, m, capture)
    finally:
        bench.StreamStats = ORIGINAL_STATS
    safe.new_json(out / "summary.json", {"status": result["status"], "method": method, "bs": bs,
                  "input_tokens": INPUT_LEN, "output_tokens": 8192, "triggers": ProfileStats.triggers,
                  "requested_steps_per_capture": 8, "stop_mode": "automatic num_steps; no redundant manual stop",
                  "not_a_primary_throughput_measurement": True})
    assert result["status"] == "ok" and len(ProfileStats.triggers) == 4


def export_profiles(method, launcher):
    deadline = time.monotonic() + 600
    reports = []
    while time.monotonic() < deadline:
        reports = list(ROOT.glob(f"{method}-nodes.*.nsys-rep"))
        if len(reports) == 8 and all(p.stat().st_size > 0 for p in reports):
            break
        time.sleep(10)
    assert len(reports) == 8, f"Incomplete asynchronous export: {launcher}"
    for index, (bs, output) in enumerate(((b, n) for b in (1, 4) for n in OUTPUTS), 1):
        report = ROOT / f"{method}-nodes.{index}.nsys-rep"
        sql = ROOT / f"{method}-out{output}-bs{bs}.sqlite"
        subprocess.run(["nsys", "export", "--type=sqlite", f"--output={sql}", str(report)], check=True)
        result = loops.analyze(sql, expected=None)
        assert 8 <= result["decode_graphs"] <= 9, "Unexpected CUDA profiler capture extent"
        safe.new_json(ROOT / f"{method}-out{output}-bs{bs}-loops.json", result)
        subprocess.run(["nsys", "stats", "--report=cuda_api_sum,cuda_gpu_mem_time_sum", "--format=csv",
                        f"--output={ROOT / f'{method}-out{output}-bs{bs}'}", str(report)], check=True)


def publish():
    report = aggregate()
    assert report["complete"], "Cannot publish an incomplete matrix as complete"
    code = """import json,sys,wandb
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
r=wandb.init(entity='529513043-tsinghua-university',project='loopvq-ouro-2p6b-thinking',name='decode-2k-input-1k-8k-output',config={k:v for k,v in x.items() if k!='rows'})
columns=list(x['rows'][0]);r.log({'decode_throughput':wandb.Table(columns=columns,data=[[row[k] for k in columns] for row in x['rows']])})
for row in x['rows']:
 for k in ('bf16_decode_tokens_s','loopvq_decode_tokens_s','decode_speedup'):
  if row[k] is not None:r.summary[f\"output{row['max_new_tokens']}/bs{row['bs']}/{k}\"]=row[k]
u=r.url;r.finish();(p.parent/'wandb.json').write_text(json.dumps({'url':u},indent=2))
"""
    subprocess.run([str(safe.BUILD_PYTHON), "-c", code, str(ROOT / "decode-throughput.json")], check=True)


def recover_profile():
    """Narrow recovery of candidate12's already-completed BF16 bs1 request."""
    assert os.environ.get("TMUX")
    context.ROOT = ROOT
    old = safe.read_json(ROOT / "profile-status.json")
    assert old["status"] == "failed" and old["method"] == "bf16" and old["bs"] == 1
    assert old["error"] == "HTTP Error 500: Internal Server Error"
    assert old["helper_sha256"] == "6bdb1af57b867c05c91a0170d66b36880a3efa94d709f722b97aa09c4280933c"
    assert old["inference_sha"] == safe.FROZEN_SHA
    assert safe.capture(["git", "-C", str(safe.FROZEN_REPO), "rev-parse", "HEAD"]) == safe.FROZEN_SHA
    candidates = list(ROOT.glob("context-bf16-nodes-*.manifest.json"))
    assert len(candidates) == 1
    manifest = safe.read_json(candidates[0])
    identity = manifest["process_identity"]
    safe.same_process(identity, safe.proc_snapshot(identity["pid"]))
    safe.validate_parent(identity, safe.Paths().target_python, port=context.PORT)
    owned = safe.process_tree(identity["pid"])
    assert context.gpu_pids() and context.gpu_pids() <= owned and safe.listener_owned(owned, context.PORT)
    assert all(value == 0 for rows in safe.idle_values(context.request("/metrics").decode()).values() for value in rows)
    label = candidates[0].name.removesuffix(".manifest.json")
    log = safe.TASK / "logs" / f"{label}.log"
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 32000))
        assert "Profiling is not in progress. Call /start_profile first." in stream.read().decode(errors="replace")
    out = ROOT / "profile/bf16/bs1"
    completed = safe.read_json(out / "profile-01.json")
    assert completed["status"] == "ok" and completed["output_tokens"] == 8192
    assert completed["batch_evidence"]["verified"] and completed["batch_evidence"]["target_batch_fraction"] == 1
    assert completed["requests"][0]["prompt_tokens"] == INPUT_LEN
    captures = list(ROOT.glob("bf16-nodes.*.nsys-rep"))
    assert len(captures) == 4 and all(p.stat().st_size > 0 for p in captures)
    state = ROOT / "profile-recovery-status.json"
    assert not state.exists() and not (out / "summary.json").exists()
    base = {"status": "running", "started_unix": time.time(), "inference_sha": safe.FROZEN_SHA,
            "helper_sha256": safe.sha256(__file__), "prior_failure": str(ROOT / "profile-status.json")}
    safe.new_json(state, base)
    safe.new_json(out / "summary.json", {"status": "ok", "method": "bf16", "bs": 1,
                  "input_tokens": INPUT_LEN, "output_tokens": 8192, "requested_steps_per_capture": 8,
                  "recovered_completed_request": True, "stop_mode": "automatic num_steps",
                  "thresholds_from_candidate12_source": [1024, 2048, 4096, 8160],
                  "callback_timestamps": None, "note": "Callback list was not persisted before the redundant-stop failure; no timestamps reconstructed",
                  "report_sha256": {p.name: safe.sha256(p) for p in captures},
                  "not_a_primary_throughput_measurement": True})
    try:
        bench.atomic_json(state, {**base, "method": "bf16", "bs": 4, "server": identity})
        profile_batch("bf16", 4, manifest, log)
        context.stop(identity)
        export_profiles("bf16", ROOT / f"{label}.launcher.log")
        identity, manifest, log, launcher = context.start("loopvq", True,
            launch_script=HERE / "a10_throughput_server.sh", server_protocol="decode-sweep")
        for bs in (1, 4):
            bench.atomic_json(state, {**base, "method": "loopvq", "bs": bs, "server": identity})
            profile_batch("loopvq", bs, manifest, log)
        context.stop(identity)
        export_profiles("loopvq", launcher)
        bench.atomic_json(state, {**base, "status": "completed", "ended_unix": time.time(), "gpu_released": True})
    except BaseException as exc:
        bench.atomic_json(state, {**safe.read_json(state), "status": "failed", "error_type": type(exc).__name__,
                                  "error": str(exc), "ended_unix": time.time()})
        raise


def run(profile):
    assert os.environ.get("TMUX") and hasattr(os, "pidfd_open")
    assert safe.capture(["git", "-C", str(safe.FROZEN_REPO), "rev-parse", "HEAD"]) == safe.FROZEN_SHA
    context.ROOT = ROOT
    mode = "profile" if profile else "timing"
    state = ROOT / f"{mode}-status.json"
    assert not state.exists()
    base = {"status": "running", "mode": mode, "started_unix": time.time(),
            "helper_sha256": safe.sha256(__file__), "inference_sha": safe.FROZEN_SHA, "gpu_uuid": context.UUID}
    bench.atomic_json(state, base)
    try:
        for method in ("bf16", "loopvq"):
            identity, manifest, log, launcher = context.start(method, profile,
                launch_script=HERE / "a10_throughput_server.sh", server_protocol="decode-sweep")
            if profile:
                for bs in (1, 4):
                    bench.atomic_json(state, {**base, "method": method, "bs": bs, "server": identity})
                    profile_batch(method, bs, manifest, log)
            else:
                for output in OUTPUTS:
                    for bs in BATCHES:
                        bench.atomic_json(state, {**base, "method": method, "output": output, "bs": bs, "server": identity})
                        measure_cell(method, output, bs, manifest, log)
                        aggregate()
            context.stop(identity)
            if profile:
                export_profiles(method, launcher)
        if not profile:
            publish()
        bench.atomic_json(state, {**base, "status": "completed", "ended_unix": time.time(), "gpu_released": True})
    except BaseException as exc:
        bench.atomic_json(state, {**safe.read_json(state), "status": "failed", "error_type": type(exc).__name__,
                                  "error": str(exc), "ended_unix": time.time(),
                                  "cleanup": "No blind process termination; inspect owned server identity and idle state"})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fixture", "timing", "profile", "recover-profile", "aggregate", "publish"))
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "fixture":
        fixture()
    elif args.mode == "aggregate":
        print(json.dumps(aggregate(), indent=2))
    elif args.mode == "publish":
        publish()
    elif args.mode == "recover-profile":
        recover_profile()
    else:
        run(args.mode == "profile")
