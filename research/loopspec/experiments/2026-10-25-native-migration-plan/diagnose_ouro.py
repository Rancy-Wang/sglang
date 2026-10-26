"""Independent HF versus native fixed-R=4 per-depth hidden/logit diagnostics.

Run HF in the independent transformers 4.55 environment. Native uses the
upstream one_batch loader and request builder (not a replacement server).
Files contain tensors; keep them under X5 /results, never in git.
"""

import argparse
from array import array
import json
from pathlib import Path

import torch

from verify_ouro import PROMPT


def reference(model_path, steps):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    assert model.config.total_ut_steps == 4
    ids = tokenizer.encode(PROMPT)
    records = []
    with torch.inference_mode():
        for _ in range(steps):
            _, depths, _ = model.model(
                input_ids=torch.tensor([ids], device="cuda"), use_cache=False
            )
            hidden = torch.stack([h[0, -1] for h in depths])
            logits = model.lm_head(hidden).float()
            records.append(
                {
                    "input_ids": ids.copy(),
                    "hidden": hidden.cpu(),
                    "logits": logits.cpu(),
                }
            )
            ids.append(int(logits[-1].argmax()))
    return {"mode": "independent_hf_fixed_r4", "records": records}


def native(model_path, inputs, compact):
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs

    options = dict(
        model_path=model_path,
        dtype="bfloat16",
        attention_backend="triton",
        max_running_requests=1,
        page_size=1,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        disable_cuda_graph=True,
        context_length=2048,
        mem_fraction_static=0.7,
        bf16_gemm_backend="torch",
    )
    if compact:
        options.update(speculative_algorithm="LOOPSPEC", speculative_draft_depth=[1, 2])
    args = ServerArgs(**options)
    ports = PortArgs.init_new(args)
    publish(args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    bench, _ = load_model(args, ports, 0, 0)
    runner = bench.torch_runner
    if compact:
        from sglang.srt.mem_cache.timeline_pool import TimelineRuntime

        runner.model.timeline_runtime = TimelineRuntime(
            runner.token_to_kv_pool,
            runner.token_to_kv_pool_allocator,
            runner.req_to_token_pool,
            runner,
        )
    captured = []
    handle = runner.model.model.norm.register_forward_hook(
        lambda module, args, output: captured.append(output[-1].detach().clone())
    )
    records = []
    with torch.inference_mode():
        for index, sample in enumerate(inputs["records"]):
            bench.clear()
            captured.clear()
            ids = sample["input_ids"]
            req = Req(
                rid=str(index),
                origin_input_text="",
                origin_input_ids=array("q", ids),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.extend_end = len(ids)
            _, actual_logits, _ = bench.extend([req])
            assert len(captured) == 4, len(captured)
            hidden = torch.stack(captured)
            logits = torch.nn.functional.linear(
                hidden, runner.model.lm_head.weight
            ).float()
            torch.testing.assert_close(logits[-1], actual_logits[0], rtol=0, atol=0)
            records.append(
                {"input_ids": ids, "hidden": hidden.cpu(), "logits": logits.cpu()}
            )
    handle.remove()
    return {
        "mode": "native_compact_prefill" if compact else "native_ar_prefill",
        "records": records,
    }


def compare(reference_path, candidate_path):
    ref = torch.load(reference_path, map_location="cpu", weights_only=True)
    actual = torch.load(candidate_path, map_location="cpu", weights_only=True)
    assert len(ref["records"]) == len(actual["records"])
    report = []
    passed = True
    for index, (a, b) in enumerate(zip(ref["records"], actual["records"])):
        assert a["input_ids"] == b["input_ids"]
        row = {"prefix": index, "depths": []}
        for d in range(4):
            metrics = {"depth": d + 1}
            for key, maximum, rms in (("hidden", 0.125, 0.02), ("logits", 0.5, 0.1)):
                x, y = a[key][d].float(), b[key][d].float()
                delta = x - y
                error = {
                    "max_abs": float(delta.abs().max()),
                    "rmse": float(delta.square().mean().sqrt()),
                }
                metrics[key] = error
                passed &= error["max_abs"] <= maximum and error["rmse"] <= rms
            logits = a["logits"][d].float()
            top = logits.topk(2).values
            metrics["reference_top1_margin"] = float(top[0] - top[1])
            metrics["argmax_equal"] = int(logits.argmax()) == int(
                b["logits"][d].argmax()
            )
            passed &= metrics["argmax_equal"]
            row["depths"].append(metrics)
        report.append(row)
    return {
        "passed": bool(passed),
        "thresholds": {
            "hidden_max": 0.125,
            "hidden_rmse": 0.02,
            "logits_max": 0.5,
            "logits_rmse": 0.1,
            "all_depth_argmax_equal": True,
        },
        "records": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "native", "compare"])
    parser.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "compare":
        result = compare(args.reference, args.candidate)
        args.output.write_text(json.dumps(result, indent=2))
        assert result["passed"], result
    else:
        result = (
            reference(args.model, args.steps)
            if args.mode == "reference"
            else native(
                args.model,
                torch.load(args.reference, weights_only=True, map_location="cpu"),
                args.compact,
            )
        )
        torch.save(result, args.output)


if __name__ == "__main__":
    main()
