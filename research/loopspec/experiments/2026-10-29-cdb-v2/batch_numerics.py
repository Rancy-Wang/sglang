"""Teacher-forced batch-shape diagnostics using native one_batch execution.

This is an eager numerical probe, never a serving latency benchmark. Every
decode receives the same reference token, even after its own argmax diverges.
"""

import argparse
from array import array
import json
from pathlib import Path

import torch


def error(a, b):
    delta = a.float() - b.float()
    return {"max_abs": float(delta.abs().max()),
            "rmse": float(delta.square().mean().sqrt()),
            "exact": bool(torch.equal(a, b))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--invariant", action="store_true")
    p.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    p.add_argument("--dataset", type=Path,
                   default=Path("/home/ruoxi/repo/test/datasets/aime.jsonl"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()][:8]
    prompts = [tokenizer.encode(row["problem"]) for row in rows]
    ref = json.loads((args.reference / "cohort-0.json").read_text())["requests"]
    assert len(ref) == len(prompts) == 8
    options = ServerArgs(
        model_path=args.model, dtype="bfloat16", attention_backend="triton",
        bf16_gemm_backend="auto", max_running_requests=8, page_size=1,
        disable_overlap_schedule=True, disable_radix_cache=True,
        chunked_prefill_size=-1, disable_cuda_graph=True,
        context_length=2048, mem_fraction_static=0.7,
    )
    ports = PortArgs.init_new(options)
    publish(options, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    bench, _ = load_model(options, ports, 0, 0)
    if args.invariant:
        # Diagnostic variant only; do not change the serving configuration.
        from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
    runner = bench.torch_runner
    captured, selection = [], []
    hook = runner.model.model.norm.register_forward_hook(
        lambda module, inputs, output: captured.append(output[selection].detach().clone())
    )
    projections, probe_step = [], [False]
    projection = runner.model.model.layers[0].self_attn.qkv_proj

    def capture_projection(module, inputs):
        if probe_step[0]:
            projections.append(inputs[0][selection[-1]:selection[-1] + 1].detach().clone())

    projection_hook = projection.register_forward_pre_hook(capture_projection)

    def run(indices):
        bench.clear()
        reqs, offset = [], 0
        selection.clear()
        for index in indices:
            ids = prompts[index]
            req = Req(rid=f"q{index+1}", origin_input_text="",
                      origin_input_ids=array("q", ids),
                      sampling_params=SamplingParams(temperature=0, max_new_tokens=args.steps+1))
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.extend_end = len(ids)
            reqs.append(req)
            offset += len(ids)
            selection.append(offset - 1)
        records = []
        for step in range(args.steps + 1):
            captured.clear()
            probe_step[0] = indices == [4] and step == min(11, args.steps)
            if step == 0:
                _, logits, batch = bench.extend(reqs)
            else:
                selection[:] = range(len(indices))
                forced = torch.tensor([ref[i]["output_ids"][step - 1] for i in indices],
                                      device=runner.device, dtype=torch.int64)
                _, logits = bench.decode(forced, batch)
            assert len(captured) == 4
            records.append({"hidden": torch.stack(captured).cpu(),
                            "logits": logits.detach().cpu()})
        probe_step[0] = False
        return records

    with torch.inference_mode():
        singles = [run([i]) for i in range(8)]
        batched = run(list(range(8)))
        projection_report = []
        for depth, x in enumerate(projections):
            for name, value, weight in (
                ("layer0.qkv", x, projection.weight),
                ("lm_head", singles[4][min(11, args.steps)]["hidden"][depth].to(runner.device),
                 runner.model.lm_head.weight),
            ):
                single = torch.nn.functional.linear(value, weight)
                for width in (2, 4, 8):
                    batch = torch.nn.functional.linear(value.repeat(width, 1), weight)
                    projection_report.append({"operator": name, "depth": depth + 1,
                                              "batch_width": width, **error(single[0], batch[0])})
    hook.remove()
    projection_hook.remove()
    torch.save({"single": singles, "batch": batched}, args.output / "tensors.pt")
    report = []
    for step, bat in enumerate(batched):
        for q in range(8):
            single = singles[q][step]
            a, b = single["logits"][0], bat["logits"][q]
            report.append({"step": step, "question": q + 1,
                           "hidden": [error(single["hidden"][d, 0], bat["hidden"][d, q])
                                      for d in range(4)],
                           "logits": error(a, b),
                           "argmax_single": int(a.argmax()), "argmax_batch": int(b.argmax()),
                           "reference_token": ref[q]["output_ids"][step],
                           "single_margin": float(a.topk(2).values.diff().abs()[0]),
                           "batch_margin": float(b.topk(2).values.diff().abs()[0])})
    result = {"invariant": args.invariant, "eager": True, "steps": args.steps,
              "projection_same_input": projection_report, "records": report}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "invariant": args.invariant,
                      "argmax_mismatches": sum(r["argmax_single"] != r["argmax_batch"] for r in report),
                      "projection": projection_report}), flush=True)


if __name__ == "__main__":
    main()
