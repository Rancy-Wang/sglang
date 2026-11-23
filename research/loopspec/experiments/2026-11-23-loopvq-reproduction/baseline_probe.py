#!/usr/bin/env python3
"""Independent, short full-R4 BF16 diagnostics; never an accuracy/performance test.

HF (build environment, Transformers 4.56.2):
  python baseline_probe.py hf --model /absolute/Ouro-2.6B-Thinking --output /results/hf
Reference or target server (already running, BF16, no speculation):
  python baseline_probe.py server --reference /results/hf --url http://127.0.0.1:30000 \
    --label target-bf16 --server-manifest /results/target-server.json \
    --output /results/target --teacher-forced

HF recomputes every prefix without KV cache, using the checkpoint's four-loop
backbone and the final-depth LM head, bypassing the adaptive/mixture head. The
server receives exactly the saved token IDs. --teacher-forced additionally
requests one token at each HF prefix, so numerical differences after a greedy
divergence are compared on identical contexts. No numerical pass threshold is
chosen here. A difference is recorded, not hidden or treated as protocol failure.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROMPTS = (
    "What is 7 + 8? Give the answer briefly.",
    "Name the capital of France in one sentence.",
    "Write a Python function that returns the square of an integer.",
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def stop_position(ids, input_length, eos_ids, finish_reason):
    return {"finish_reason": finish_reason, "generated_tokens": len(ids),
            "last_generated_index_zero_based": len(ids) - 1 if ids else None,
            "last_absolute_position_zero_based": input_length + len(ids) - 1 if ids else None,
            "last_token_id": ids[-1] if ids else None,
            "last_token_is_eos": bool(ids and ids[-1] in eos_ids)}


def first_divergence(expected, actual):
    for index in range(max(len(expected), len(actual))):
        hf_id = expected[index] if index < len(expected) else None
        server_id = actual[index] if index < len(actual) else None
        if hf_id != server_id:
            return {"generated_index_zero_based": index, "hf_token_id": hf_id, "server_token_id": server_id}
    return None


def top20_comparison(hf_step, server_top):
    hf = {row["token_id"]: row["logprob"] for row in hf_step["top20"]}
    actual = {int(row[1]): float(row[0]) for row in server_top}
    shared = sorted(hf.keys() & actual.keys())
    deltas = [{"token_id": token, "hf_logprob": hf[token], "server_logprob": actual[token],
               "delta": actual[token] - hf[token]} for token in shared]
    return {"intersection_size": len(shared), "shared_tokens": deltas,
            "max_abs_logprob_delta_on_intersection": max((abs(x["delta"]) for x in deltas), default=None)}


def hf_run(args, manifest):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != "4.56.2":
        raise RuntimeError(f"Expected build environment transformers==4.56.2; got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("HF baseline requires the remote CUDA build environment; no CPU fallback")
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    template = tokenizer.chat_template
    if not isinstance(template, str) or "enable_thinking" not in template:
        raise ValueError("Expected the checkpoint-owned Ouro-Thinking chat template")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    if model.config.total_ut_steps != 4:
        raise ValueError(f"Checkpoint total_ut_steps must already be 4, got {model.config.total_ut_steps}")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("Model parameters are not BF16")
    eos = model.generation_config.eos_token_id
    eos_source = "checkpoint_generation_config"
    if eos is None:
        eos, eos_source = tokenizer.eos_token_id, "checkpoint_tokenizer"
    eos_ids = [int(eos)] if isinstance(eos, int) else [int(x) for x in eos or []]
    if not eos_ids:
        raise ValueError("No checkpoint EOS token IDs")
    manifest.update(model=str(model_path), total_ut_steps=4, dtype="bfloat16", attention="eager",
                    use_cache=False, head="final_depth_4_lm_head", transformers=transformers.__version__,
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    chat_template=template, chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                    eos_token_ids=eos_ids, eos_source=eos_source, max_new_tokens=args.max_new_tokens,
                    model_files={p.name: sha256(p) for p in sorted(model_path.iterdir()) if p.is_file()
                                 and (p.suffix in (".json", ".py", ".safetensors") or "tokenizer" in p.name)},
                    prompts=[])
    save_json(args.output / "manifest.json", manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for index, text in enumerate(PROMPTS):
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
        if not rendered.rstrip().endswith("<think>"):
            raise ValueError("Checkpoint template did not produce the Thinking prefix")
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        record = {"index": index, "messages": messages, "rendered_prompt": rendered,
                  "input_ids": input_ids, "output_ids": [], "steps": []}
        logits_rows, logprob_rows = [], []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(args.max_new_tokens):
                ids = torch.tensor([input_ids + record["output_ids"]], dtype=torch.long, device="cuda")
                _, depth_hidden, _ = model.model(input_ids=ids, use_cache=False)
                if len(depth_hidden) != 4:
                    raise ValueError(f"Actual backbone returned {len(depth_hidden)} depths, expected 4")
                raw = model.lm_head(depth_hidden[-1][:, -1, :])[0]
                if not torch.isfinite(raw).all().item():
                    raise ValueError(f"Nonfinite HF logits at prompt {index}, step {step}")
                logprobs = raw.float().log_softmax(-1)
                chosen = int(raw.argmax().item())
                top = logprobs.topk(20)
                record["steps"].append({"step": step, "chosen_token_id": chosen,
                                        "chosen_logprob": float(logprobs[chosen].item()),
                                        "raw_logit_dtype": str(raw.dtype),
                                        "top20": [{"token_id": int(token), "logprob": float(lp)}
                                                  for lp, token in zip(top.values.tolist(), top.indices.tolist())]})
                logits_rows.append(raw.cpu())
                logprob_rows.append(logprobs.cpu())
                record["output_ids"].append(chosen)
                del depth_hidden, raw, logprobs, ids
                if chosen in eos_ids:
                    break
        torch.cuda.synchronize()
        record["elapsed_seconds"] = time.perf_counter() - started
        record["stop"] = stop_position(record["output_ids"], len(input_ids), eos_ids,
                                        "stop" if record["output_ids"][-1] in eos_ids else "length")
        record["output_text_with_special_tokens"] = tokenizer.decode(record["output_ids"], skip_special_tokens=False)
        scores_path = args.output / f"prompt-{index}-scores.pt"
        temporary = scores_path.with_suffix(".partial")
        torch.save({"raw_logits": torch.stack(logits_rows), "logprobs_fp32": torch.stack(logprob_rows),
                    "input_ids": input_ids, "output_ids": record["output_ids"]}, temporary)
        temporary.replace(scores_path)
        record.update(scores_file=scores_path.name, scores_sha256=sha256(scores_path))
        save_json(args.output / f"prompt-{index}.json", record)
        manifest["prompts"].append({"file": f"prompt-{index}.json", "sha256": sha256(args.output / f"prompt-{index}.json")})
        save_json(args.output / "manifest.json", manifest)
        print(json.dumps({"prompt": index, "generated_tokens": len(record["output_ids"]),
                          "stop": record["stop"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    manifest["complete"] = True
    save_json(args.output / "manifest.json", manifest)


def request_generation(url, payload, output, stem, timeout):
    save_json(output / f"{stem}-request.json", payload)
    started = time.perf_counter()
    evidence = {"url": url.rstrip("/") + "/generate", "payload": payload}
    try:
        request = Request(evidence["url"], json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            evidence["http_status"] = response.status
            evidence["raw_response_text"] = response.read().decode("utf-8")
        result = json.loads(evidence["raw_response_text"])
        # Preserve the wire text even when a malformed model response contains
        # nonstandard NaN/Infinity values that strict JSON evidence cannot encode.
        return result
    except HTTPError as error:
        evidence.update(http_status=error.code, raw_response_text=error.read().decode("utf-8", errors="replace"),
                        error=repr(error), traceback=traceback.format_exc())
        raise
    except Exception as error:
        evidence.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        evidence["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / f"{stem}-http.json", evidence)


def generation_payload(input_ids, max_new_tokens, eos_ids):
    return {"input_ids": input_ids, "sampling_params": {
        "temperature": 0.0, "max_new_tokens": max_new_tokens, "ignore_eos": False,
        "stop_token_ids": eos_ids, "skip_special_tokens": False},
        "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 20, "stream": False}


def unpack_response(response, prompt_tokens, max_new_tokens):
    meta = response["meta_info"]
    logs, tops = meta["output_token_logprobs"], meta["output_top_logprobs"]
    ids = [int(row[1]) for row in logs]
    if "output_ids" in response and response["output_ids"] != ids:
        raise ValueError("Server output_ids disagree with output_token_logprobs IDs")
    if not 0 < len(ids) <= max_new_tokens or meta["completion_tokens"] != len(ids):
        raise ValueError("Invalid server output token count")
    if meta["prompt_tokens"] != prompt_tokens:
        raise ValueError("Server prompt_tokens differ from exact supplied input_ids")
    if len(tops) != len(ids) or any(len(row) != 20 for row in tops):
        raise ValueError("Missing per-step top20 logprobs")
    if any(not math.isfinite(float(row[0])) for row in logs):
        raise ValueError("Nonfinite selected-token logprobs")
    if any(not math.isfinite(float(row[0])) for top in tops for row in top):
        raise ValueError("Nonfinite top20 logprobs")
    reason = meta["finish_reason"]
    kind = reason["type"] if isinstance(reason, dict) else reason
    if kind not in ("stop", "length"):
        raise ValueError(f"Unexpected server finish reason: {reason}")
    return ids, tops, kind


def server_run(args, manifest):
    reference = json.loads((args.reference / "manifest.json").read_text())
    server = json.loads(args.server_manifest.read_text())
    if (not reference.get("complete") or reference.get("total_ut_steps") != 4
            or len(reference.get("prompts", [])) != 3):
        raise ValueError("HF reference must contain three completed full-R4 prompts")
    if server.get("total_ut_steps") != 4 or server.get("method") != "bf16":
        raise ValueError("Server manifest must attest method=bf16 and total_ut_steps=4")
    manifest.update(label=args.label, url=args.url, server=server, reference=str(args.reference.resolve()),
                    reference_manifest_sha256=sha256(args.reference / "manifest.json"),
                    chat_template_sha256=reference["chat_template_sha256"], total_ut_steps=4,
                    teacher_forced=args.teacher_forced, comparisons=[])
    save_json(args.output / "manifest.json", manifest)
    budget, eos_ids = reference["max_new_tokens"], reference["eos_token_ids"]
    for index, item in enumerate(reference["prompts"]):
        path = args.reference / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"HF reference changed: {path}")
        hf = json.loads(path.read_text())
        payload = generation_payload(hf["input_ids"], budget, eos_ids)
        response = request_generation(args.url, payload, args.output, f"prompt-{index}-greedy", args.timeout)
        ids, tops, reason = unpack_response(response, len(hf["input_ids"]), budget)
        divergence = first_divergence(hf["output_ids"], ids)
        # The distribution producing the first differing token still has the same prefix.
        shared_steps = min(len(ids), len(hf["output_ids"]),
                           divergence["generated_index_zero_based"] + 1 if divergence else budget)
        comparison = {"prompt": index, "input_ids": hf["input_ids"], "output_ids": ids,
                      "first_divergence": divergence, "identical_generated_ids": divergence is None,
                      "hf_stop": hf["stop"], "server_stop": stop_position(ids, len(hf["input_ids"]), eos_ids, reason),
                      "same_prefix_top20": [{"step": step, **top20_comparison(hf["steps"][step], tops[step])}
                                             for step in range(shared_steps)]}
        if args.teacher_forced:
            comparison["teacher_forced"] = []
            for step, hf_step in enumerate(hf["steps"]):
                forced = hf["input_ids"] + hf["output_ids"][:step]
                result = request_generation(args.url, generation_payload(forced, 1, eos_ids), args.output,
                                            f"prompt-{index}-forced-{step}", args.timeout)
                chosen, top, _ = unpack_response(result, len(forced), 1)
                comparison["teacher_forced"].append({"step": step, "hf_token_id": hf_step["chosen_token_id"],
                                                       "server_token_id": chosen[0],
                                                       **top20_comparison(hf_step, top[0])})
        save_json(args.output / f"prompt-{index}-comparison.json", comparison)
        manifest["comparisons"].append({"prompt": index, "first_divergence": divergence,
                                         "hf_stop": hf["stop"], "server_stop": comparison["server_stop"]})
        save_json(args.output / "manifest.json", manifest)
        print(json.dumps(manifest["comparisons"][-1]), flush=True)
    manifest["complete"] = True
    save_json(args.output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    hf = sub.add_parser("hf")
    hf.add_argument("--model", type=Path, required=True)
    hf.add_argument("--max-new-tokens", type=int, default=32)
    server = sub.add_parser("server")
    server.add_argument("--reference", type=Path, required=True)
    server.add_argument("--url", required=True)
    server.add_argument("--label", required=True)
    server.add_argument("--server-manifest", type=Path, required=True)
    server.add_argument("--teacher-forced", action="store_true")
    server.add_argument("--timeout", type=int, default=1800)
    for command in (hf, server):
        command.add_argument("--output", type=Path, required=True, help="new directory outside Git")
    args = parser.parse_args()
    if args.mode == "hf" and not 1 <= args.max_new_tokens <= 32:
        parser.error("HF max-new-tokens must be between 1 and 32")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"mode": args.mode, "complete": False, "host": platform.node(), "python": sys.executable,
                "argv": sys.argv, "source_sha256": sha256(__file__), "started_unix": time.time()}
    try:
        (hf_run if args.mode == "hf" else server_run)(args, manifest)
    except BaseException as error:
        save_json(args.output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
