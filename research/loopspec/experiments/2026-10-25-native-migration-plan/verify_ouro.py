"""Independent HF reference and standard SGLang request acceptance harness.

Run reference in a transformers==4.55.0 environment, and native in the pinned
SGLang environment. Results go outside the repository with --output.
"""

import argparse
import json
import time
from pathlib import Path

PROMPT = "The capital of France is"
SAMPLING = {"temperature": 0, "max_new_tokens": 32, "ignore_eos": False}


def reference(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    assert model.config.total_ut_steps == 4
    tokens = tokenizer.encode(PROMPT)
    input_ids = torch.tensor([tokens], device="cuda")
    output_ids = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    prefill = None
    with torch.inference_mode():
        for _ in range(SAMPLING["max_new_tokens"]):
            # Bypass the adaptive/mixture LM head: this is the fixed R=4 target.
            outputs, hidden_by_depth, _ = model.model(
                input_ids=input_ids, use_cache=False
            )
            assert len(hidden_by_depth) == 4
            logits = model.lm_head(hidden_by_depth[-1][:, -1, :]).float()
            next_id = int(logits.argmax(-1).item())
            if prefill is None:
                torch.cuda.synchronize()
                prefill = time.perf_counter() - start
            output_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            input_ids = torch.tensor([tokens + output_ids], device="cuda")
    torch.cuda.synchronize()
    return {
        "mode": "hf_fixed_depth", "input_ids": tokens, "output_ids": output_ids,
        "text": tokenizer.decode(output_ids), "prefill_seconds": prefill,
        "generation_seconds": time.perf_counter() - start,
        "finish_reason": "stop" if output_ids[-1] == tokenizer.eos_token_id else "length",
    }


def native(model_path):
    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path, dtype="bfloat16", attention_backend="triton",
        max_running_requests=1, page_size=1, disable_overlap_schedule=True,
        disable_radix_cache=True, chunked_prefill_size=-1, disable_cuda_graph=True,
        mem_fraction_static=0.7, context_length=2048,
    )
    try:
        start = time.perf_counter()
        result = engine.generate(PROMPT, SAMPLING, return_logprob=True)
        ids = [entry[1] for entry in result["meta_info"]["output_token_logprobs"]]
        return {
            "mode": "native_ar", "output_ids": ids, "response": result,
            "generation_seconds": time.perf_counter() - start,
            "finish_reason": result["meta_info"]["finish_reason"]["type"],
        }
    finally:
        engine.shutdown()


def http_request(url):
    import urllib.request

    start = time.perf_counter()
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps({"text": PROMPT, "sampling_params": SAMPLING,
                         "return_logprob": True, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    ids = [entry[1] for entry in result["meta_info"]["output_token_logprobs"]]
    return {"mode": "http", "output_ids": ids, "response": result,
            "generation_seconds": time.perf_counter() - start,
            "finish_reason": result["meta_info"]["finish_reason"]["type"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "native", "http", "compare"])
    parser.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    parser.add_argument("--url", default="http://127.0.0.1:30126")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "compare":
        results = [json.loads(path.read_text()) for path in args.results]
        assert len(results) >= 2
        expected = results[0]
        for actual in results[1:]:
            assert actual["output_ids"] == expected["output_ids"], (expected, actual)
            assert actual["finish_reason"] == expected["finish_reason"]
        print(json.dumps({"matched": len(results), "tokens": expected["output_ids"]}))
        return
    result = (http_request(args.url) if args.mode == "http" else
              reference(args.model) if args.mode == "reference" else native(args.model))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
