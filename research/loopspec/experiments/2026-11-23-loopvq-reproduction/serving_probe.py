#!/usr/bin/env python3
"""Full-R4 serving diagnostics, not benchmark accuracy or throughput claims.

Run inside tmux beside an exclusive server with --enable-metrics and
--decode-log-interval 1. Exit 0: protocol and requested comparisons passed;
2: numerical difference preserved; 1: protocol/infrastructure failure.
--cases selects individual cases instead of --suite; reference-dir compares
identical cases from the same model/codebook (e.g. eager versus full graph).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from urllib.request import Request, urlopen
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_probe as bp
import benchmark_throughput as bench

BOUNDARIES = [63, 64, 65, 319, 320, 321, 327, 328, 333, 383]
CORE = ["reuse-before", "long4096", "boundaries"] + [f"bs{n}" for n in (1, 2, 4, 8, 16)] + [f"pad{n}" for n in (3, 5, 15)]
LIFECYCLE = ["ring", "ragged", "abort"]
SUITES = {"short": ["hp"], "core": CORE + ["reuse-after"],
          "lifecycle": ["reuse-before"] + LIFECYCLE + ["reuse-after", "flush"],
          "all": ["hp"] + CORE + LIFECYCLE + ["reuse-after", "flush"]}
GAUGES = ("num_running_reqs", "num_queue_reqs", "num_used_tokens", "token_usage")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def exact_prompt(tokenizer, length, seed=0):
    """Adjust user content only; never truncate checkpoint template tokens."""
    for filler in (" x", " a", " 1"):
        count, seen = 0, set()
        for _ in range(12):
            if count in seen or count < 0:
                break
            seen.add(count)
            messages = [{"role": "user", "content": f"Note {seed}:" + filler * count + "\nWhat is 7 + 8?"}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if not rendered.rstrip().endswith("<think>"):
                raise ValueError("Checkpoint template did not produce a Thinking prefix")
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) == length:
                return {"messages": messages, "rendered_prompt": rendered, "input_ids": ids}
            count += length - len(ids)
    raise ValueError(f"Cannot construct native-template prompt of exactly {length} tokens")


def compare_rows(expected, actual):
    if len(expected) != len(actual):
        raise ValueError("Reference batch shape differs")
    comparisons = []
    for old, new in zip(expected, actual):
        if old["input_ids"] != new["input_ids"] or old["sampling"] != new["sampling"]:
            raise ValueError("Reference input IDs or sampling shape differs")
        divergence = bp.first_divergence(old["output_ids"], new["output_ids"])
        shared = min(len(old["top20"]), len(new["top20"]),
                     divergence["generated_index_zero_based"] + 1 if divergence else len(old["top20"]))
        steps = [bp.top20_comparison({"top20": [{"token_id": int(x[1]), "logprob": float(x[0])}
                                                for x in old["top20"][i]]}, new["top20"][i]) for i in range(shared)]
        comparisons.append({"first_divergence": divergence, "same_finish_reason": old["finish"] == new["finish"],
                            "same_prefix_top20": steps})
    return comparisons


def metrics_values(raw):
    values = {key: [] for key in GAUGES}
    for line in raw.splitlines():
        match = re.match(r"^sglang:(\w+)(?:\{[^}]*\})?\s+(\S+)", line)
        if match and match[1] in values:
            values[match[1]].append(float(match[2]))
    if any(not v or any(not math.isfinite(x) for x in v) for v in values.values()):
        raise ValueError("Missing/nonfinite idle metrics; server requires --enable-metrics")
    return values


def validate_identity(manifest, config, model, template):
    declared = manifest.get("config", manifest)
    if declared.get("total_ut_steps") != 4 or json.loads((model / "config.json").read_text()).get("total_ut_steps") != 4:
        raise ValueError("Manifest and unmodified checkpoint must attest full R4")
    if Path(config.get("model_path") or "").resolve() != model.resolve() or config.get("dtype") not in ("bf16", "bfloat16"):
        raise ValueError("Actual model path/dtype differs from the BF16 Thinking checkpoint")
    for key, value in {"tp_size": 1, "pp_size": 1, "dp_size": 1, "disable_radix_cache": True,
                       "chunked_prefill_size": -1, "speculative_algorithm": None, "page_size": 8,
                       "decode_log_interval": 1, "attention_backend": "triton"}.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported server setting {key}: {config.get(key)!r}")
    graph = config.get("cuda_graph_config", {})
    if graph.get("prefill", {}).get("backend") != "disabled" or graph.get("decode", {}).get("backend") not in ("full", "disabled"):
        raise ValueError("Require disabled prefill graph and eager/full decode")
    if manifest.get("exclusive_server") is not True or not all(manifest.get(k) for k in ("model_sha256", "sglang_sha")):
        raise ValueError("Manifest needs exclusive_server, model_sha256 and sglang_sha")
    method = manifest.get("method")
    profile = config.get("ouro_decode_profile", "hf_reference")
    if profile not in ("hf_reference", "optimized") or declared.get("ouro_decode_profile", "hf_reference") != profile:
        raise ValueError("Manifest Ouro decode profile differs from the running service")
    if method not in ("bf16", "loopvq") or bool(config.get("loopvq_codebook_dir")) != (method == "loopvq"):
        raise ValueError("Manifest method differs from actual LoopVQ activation")
    if method == "loopvq" and (not manifest.get("bundle_sha256") or
            (config.get("loopvq_prefix_tokens"), config.get("loopvq_recent_tokens")) != (64, 256)):
        raise ValueError("Require immutable VQ bundle hash and prefix64/recent256")
    if method == "loopvq" and declared.get("loopvq_codebook_dir") != config["loopvq_codebook_dir"]:
        raise ValueError("Manifest codebook directory differs from the running service")
    return {"method": method, "model_sha256": manifest["model_sha256"], "sglang_sha": manifest["sglang_sha"],
            "ouro_decode_profile": profile,
            "effective_decode": ("loopvq_official_cuda" if method == "loopvq" else "bf16_generic_split")
                if profile == "optimized" else ("loopvq_bf16_reference" if method == "loopvq" else "ouro_bf16_reference"),
            "bundle_sha256": manifest.get("bundle_sha256"), "template_sha256": digest(template),
            "model_config_sha256": bp.sha256(model / "config.json"), "prefix": 64, "recent": 256}


class Probe:
    def __init__(self, args, tokenizer, summary):
        self.a, self.t, self.summary = args, tokenizer, summary
        self.out, self.log = args.out_dir, args.server_log
        self.reference = json.loads((args.reference_dir / "summary.json").read_text()) if args.reference_dir else None
        self.capture = bench.capture_from_log(self.log)

    def http(self, path, stem, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(self.a.url.rstrip("/") + path, body, {"Content-Type": "application/json"})
        evidence = {"path": path, "request": payload}
        try:
            with urlopen(request, timeout=min(60, self.a.timeout)) as response:
                evidence.update(status=response.status, raw=response.read().decode())
            if path == "/server_info":
                actual = json.loads(evidence["raw"])
                evidence["raw"] = json.dumps({k: actual.get(k) for k in (*bench.CONFIG_KEYS, "model_path")})
                evidence["allowlist_only"] = True
            return evidence["raw"]
        finally:
            bp.save_json(self.out / f"{stem}-http.json", evidence)

    def idle(self, stem):
        deadline = time.monotonic() + self.a.idle_timeout
        observations = []
        while True:
            values = metrics_values(self.http("/metrics", stem))
            observations.append({"unix": time.time(), "values": values})
            bp.save_json(self.out / f"{stem}-idle.json", observations)
            if all(x == 0 for rows in values.values() for x in rows):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Server did not release all requests/KV: {values}")
            time.sleep(1)

    def log_evidence(self, name, start):
        end = self.log.stat().st_size
        with self.log.open("rb") as source:
            source.seek(start)
            text = source.read(end - start).decode(errors="replace")
        (self.out / f"{name}-server.log").write_text(text)
        return text, {"path": str(self.log), "start": start, "end": end}

    def generate(self, name, lengths, counts, natural=False):
        prompts = [exact_prompt(self.t, length, i) for i, length in enumerate(lengths)]
        eos, hf = [], None
        if name == "hp" and self.a.hf_reference:
            hm = json.loads((self.a.hf_reference / "manifest.json").read_text())
            item = hm["prompts"][0]
            path = self.a.hf_reference / item["file"]
            if not hm.get("complete") or hm["total_ut_steps"] != 4 or bp.sha256(path) != item["sha256"]:
                raise ValueError("Invalid/incomplete HF reference")
            if hm["chat_template"] != self.t.chat_template or hm["model_files"]["config.json"] != bp.sha256(self.a.model / "config.json"):
                raise ValueError("HF template/checkpoint config differs")
            hf = json.loads(path.read_text())
            prompts, counts, eos = [hf], [hm["max_new_tokens"]], hm["eos_token_ids"]
            if len(hf["input_ids"]) > 64 or len(hf["input_ids"]) + counts[0] > 320:
                raise ValueError("HF short probe exceeds the all-HP regime")
            natural = True
        samplings = [bp.generation_payload(p["input_ids"], n, eos)["sampling_params"] for p, n in zip(prompts, counts)]
        for sampling in samplings:
            sampling["ignore_eos"] = not natural
        payload = bp.generation_payload(prompts[0]["input_ids"], counts[0], eos)
        payload.update(input_ids=[p["input_ids"] for p in prompts], sampling_params=samplings)
        bp.save_json(self.out / f"{name}-prompts.json", prompts)
        start = self.log.stat().st_size
        try:
            response = bp.request_generation(self.a.url, payload, self.out, name, self.a.timeout)
            if not isinstance(response, list) or len(response) != len(prompts):
                raise ValueError("Server batch response shape differs")
            rows = []
            for p, n, sampling, result in zip(prompts, counts, samplings, response):
                ids, top, finish = bp.unpack_response(result, len(p["input_ids"]), n)
                if not natural and (len(ids) != n or finish != "length"):
                    raise ValueError("Fixed-length diagnostic stopped early or was clamped")
                rows.append({"input_ids": p["input_ids"], "sampling": sampling, "output_ids": ids, "top20": top, "finish": finish})
            bp.save_json(self.out / f"{name}-rows.json", rows)
            self.idle(name)
        finally:
            text, offsets = self.log_evidence(name, start)
        record = {"rows": f"{name}-rows.json", "log": offsets}
        if not natural and len(set(counts)) == 1:
            record["decode"] = bench.decode_evidence(text, len(prompts), counts[0], self.summary["config"], self.capture)
            if name.startswith("pad") and record["decode"]["decode_graph_backend"] == "full":
                buckets = self.capture.get("capture_bs") or []
                larger = [b for b in buckets if b > len(prompts)]
                if len(prompts) in buckets or not larger:
                    raise ValueError("Padding case did not select a larger actually captured bucket")
                record["decode"]["expected_padding_from_actual_capture"] = min(larger) - len(prompts)
            if not record["decode"]["verified"]:
                bp.save_json(self.out / f"{name}-decode.json", record["decode"])
                raise ValueError(f"Decode/graph/batch evidence failed: {record['decode']['reasons']}")
        expected = None
        if self.reference:
            if name not in self.reference["cases"]:
                raise ValueError(f"Reference did not run case {name}")
            expected = json.loads((self.a.reference_dir / self.reference["cases"][name]["rows"]).read_text())
        if expected is not None:
            record["comparison"] = compare_rows(expected, rows)
        if hf:
            hf_rows = [{**rows[0], "output_ids": hf["output_ids"], "finish": hf["stop"]["finish_reason"],
                        "top20": [[[x["logprob"], x["token_id"]] for x in step["top20"]] for step in hf["steps"]]}]
            record["hf_comparison"] = compare_rows(hf_rows, rows)
        if name == "reuse-after":
            record["reuse"] = compare_rows(json.loads((self.out / "reuse-before-rows.json").read_text()), rows)
            if any(x["first_divergence"] or not x["same_finish_reason"] for x in record["reuse"]):
                bp.save_json(self.out / "reuse-failure.json", record)
                raise ValueError("Identical request changed after lifecycle interference")
        return record

    def abort(self):
        rid = "loopvq-probe-" + uuid.uuid4().hex
        payload = bp.generation_payload(exact_prompt(self.t, 384)["input_ids"], 600, [])
        payload.update(rid=rid, stream=True)
        payload["sampling_params"]["ignore_eos"] = True
        bp.save_json(self.out / "abort-request.json", payload)
        request = Request(self.a.url.rstrip("/") + "/generate", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        sent, finish, start = False, None, self.log.stat().st_size
        try:
            with urlopen(request, timeout=self.a.timeout) as response, (self.out / "abort-sse.txt").open("w") as raw:
                for line in response:
                    line = line.decode()
                    raw.write(line)
                    raw.flush()
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    event = json.loads(line[5:])
                    meta = event.get("meta_info", {})
                    finish = meta.get("finish_reason") or finish
                    if not sent and meta.get("completion_tokens", 0) >= 16 and not finish:
                        self.http("/abort_request", "abort-owned", {"rid": rid, "abort_all": False})
                        sent = True
            if not sent or not isinstance(finish, dict) or finish.get("type") != "abort":
                raise ValueError(f"Owned running request did not finish by abort: sent={sent}, finish={finish}")
        finally:
            _, offsets = self.log_evidence("abort", start)
        self.idle("abort")
        return {"rid": rid, "finish": finish, "log": offsets}

    def run_case(self, name):
        if name == "abort":
            return self.abort()
        if name == "flush":
            self.idle("before-flush")
            self.http("/flush_cache?timeout=5", "flush", {})
            self.idle("after-flush")
            return self.generate("flush", [384], [32])
        if name == "hp":
            return self.generate(name, [64], [32])
        if name == "boundaries":
            return self.generate(name, BOUNDARIES, [17] * len(BOUNDARIES))
        if name == "ragged":
            return self.generate(name, [333, 383], [17, 600])
        batch = int(re.search(r"\d+$", name)[0]) if name.startswith(("bs", "pad")) else 1
        length, output = (4096, 32) if name == "long4096" else (384, 600 if name == "ring" else 64)
        return self.generate(name, [length] * batch, [output] * batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "manifest", "out-dir", "server-log"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--hf-reference", type=Path)
    parser.add_argument("--suite", choices=SUITES, default="short")
    parser.add_argument("--cases", nargs="+", choices=SUITES["all"])
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=60)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "cases": {}, "started": time.time(), "source_sha256": bp.sha256(__file__), "argv": sys.argv}
    try:
        if not os.environ.get("TMUX"):
            raise RuntimeError("Run serving probes in the task's tmux window")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        if not isinstance(tokenizer.chat_template, str) or "enable_thinking" not in tokenizer.chat_template:
            raise ValueError("Require the checkpoint-owned Thinking chat template")
        summary.update(manifest=json.loads(args.manifest.read_text()), chat_template=tokenizer.chat_template)
        probe = Probe(args, tokenizer, summary)
        actual = json.loads(probe.http("/server_info", "server-config"))
        summary["config"] = actual
        summary["identity"] = validate_identity(summary["manifest"], summary["config"], args.model, tokenizer.chat_template)
        if probe.reference and probe.reference["identity"] != summary["identity"]:
            raise ValueError("Reference model/codebook/source/template identity differs")
        if probe.reference and probe.reference.get("status") not in ("ok", "numerical_difference"):
            raise ValueError("Reference run did not complete")
        selected = args.cases or SUITES[args.suite]
        if "reuse-after" in selected and ("reuse-before" not in selected or selected.index("reuse-before") > selected.index("reuse-after")):
            raise ValueError("reuse-after requires reuse-before in this run")
        probe.http("/health", "health")
        probe.idle("initial")
        for name in selected:
            summary["active_case"] = name
            bp.save_json(args.out_dir / "summary.json", summary)
            summary["cases"][name] = probe.run_case(name)
            bp.save_json(args.out_dir / "summary.json", summary)
            print(json.dumps({"completed": name}), flush=True)
        different = any(x["first_divergence"] or not x["same_finish_reason"] for c in summary["cases"].values()
                        for key in ("comparison", "hf_comparison") for x in c.get(key, []))
        summary.update(status="numerical_difference" if different else "ok", ended=time.time())
        return 2 if different else 0
    except BaseException as error:
        summary.update(status="failed", error=repr(error), traceback=traceback.format_exc(), ended=time.time())
        return 1
    finally:
        bp.save_json(args.out_dir / "summary.json", summary)


if __name__ == "__main__":
    sys.exit(main())
