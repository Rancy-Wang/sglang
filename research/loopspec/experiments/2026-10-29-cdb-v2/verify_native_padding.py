"""Calibrate padding with native full-R forward, without CDB execution code.

Independent dummy requests supply padding rows. For M=32 their new decode KV
is recycled after each step, retaining their original prefill; the eight real
requests keep their complete, growing positions/prefixes. For M=8 only prefill
is padded.
No production operator, model weight or decision threshold is modified.
"""

import argparse
from array import array
import json
from pathlib import Path

import torch

from batch_numerics import error
from verify_recurrent import check_envelope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=496)
    parser.add_argument("--decode-rows", type=int, choices=(8, 32), default=8)
    parser.add_argument("--original-norm", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.models.ouro import OuroForCausalLM
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs

    # Diagnostic-only: select the original ordinary KV/whole-R model path.
    OuroForCausalLM.supports_recurrent_execution = False
    model = "/home/ruoxi/models/Ouro-2.6B"
    options = ServerArgs(
        model_path=model, dtype="bfloat16", attention_backend="triton",
        bf16_gemm_backend="auto", max_running_requests=max(9, args.decode_rows), page_size=1,
        disable_overlap_schedule=True, disable_radix_cache=True,
        chunked_prefill_size=-1, disable_cuda_graph=True,
        context_length=2048, mem_fraction_static=.6,
    )
    ports = PortArgs.init_new(options)
    publish(options, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    bench, _ = load_model(options, ports, 0, 0)
    runner = bench.torch_runner
    assert runner.recurrent_runner is None
    assert not hasattr(runner.model, "timeline_runtime")
    if args.original_norm:
        from sglang.srt.layers.layernorm import RMSNorm
        for module in runner.model.modules():
            if isinstance(module, RMSNorm):
                module._forward_method = module.forward_native
    tokenizer = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(x) for x in Path("/home/ruoxi/repo/test/datasets/aime.jsonl").read_text().splitlines()][:8]
    prompts = [tokenizer.encode(r["problem"]) for r in rows]
    count = sum(map(len, prompts))
    bucket = 1 << (count - 1).bit_length()
    assert bucket > count
    dummy_rows = max(1, args.decode_rows - 8)
    padding, remainder = divmod(bucket - count, dummy_rows)
    assert padding > 0
    prompts.extend([0] * (padding + int(i < remainder)) for i in range(dummy_rows))
    requests, selection, offset = [], [], 0
    for i, ids in enumerate(prompts):
        req = Req(rid=f"padding-control-{i}", origin_input_text="",
                  origin_input_ids=array("q", ids),
                  sampling_params=SamplingParams(temperature=0, max_new_tokens=args.steps+1))
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = len(ids)
        requests.append(req)
        offset += len(ids)
        if i < 8:
            selection.append(offset - 1)
    captured = []
    hook = runner.model.model.norm.register_forward_hook(
        lambda module, inputs, output: captured.append(output[selection].detach().cpu().clone())
    )
    reference = torch.load(args.reference / "tensors.pt", weights_only=True)
    tokens = json.loads((args.tokens / "cohort-0.json").read_text())["requests"]
    records = []
    # Bound independent dummy history without truncating any real request.
    required_slots = bucket + 8 * args.steps + max(0, args.decode_rows - 8)
    assert runner.max_total_num_tokens >= required_slots, required_slots
    with torch.inference_mode():
        for step in range(args.steps + 1):
            captured.clear()
            if step == 0:
                _, logits, batch = bench.extend(requests)
                batch.filter_batch(keep_indices=list(range(args.decode_rows)))
            else:
                selection[:] = range(8)
                forced = torch.tensor([r["output_ids"][step-1] for r in tokens] + [0] * (args.decode_rows - 8), device=runner.device)
                _, logits = bench.decode(forced, batch)
                if args.decode_rows > 8:
                    dummy_slots = batch.out_cache_loc[8:]
                    assert dummy_slots.unique().numel() == args.decode_rows - 8
                    runner.token_to_kv_pool_allocator.free(dummy_slots)
                    batch.req_to_token_pool.req_to_token[
                        batch.req_pool_indices[8:], batch.seq_lens[8:] - 1
                    ] = 0
                    batch.seq_lens[8:] -= 1
                    batch.seq_lens_cpu[8:] -= 1
                    batch.orig_seq_lens[8:] -= 1
                    batch.seq_lens_sum = None
                    for req in batch.reqs[8:]:
                        req.kv.kv_allocated_len -= 1
                        req.kv.kv_committed_len -= 1
                    assert batch.seq_lens_cpu[8:].tolist() == [len(p) for p in prompts[8:]]
                assert batch.seq_lens_cpu[:8].tolist() == [len(p) + step for p in prompts[:8]]
            assert len(captured) == 4
            logits = logits[:8].cpu()
            for q in range(8):
                source = reference["single"][q][step]
                a, b = source["logits"][0], logits[q]
                records.append({"question": q+1, "step": step,
                    "hidden": [error(source["hidden"][d, 0], captured[d][q]) for d in range(4)],
                    "logits": error(a, b), "argmax_reference": int(a.argmax()),
                    "argmax_candidate": int(b.argmax()),
                    "reference_margin": float(a.topk(2).values.diff().abs()[0])})
    hook.remove()
    limits = json.loads(Path(__file__).with_name("numerical_envelope.json").read_text())
    failures = check_envelope(records, limits)
    report = {"original_prefill_tokens": count, "padded_tokens": bucket,
              "native_full_depth": True, "recurrent_scheduler": False,
              "decode_rows": args.decode_rows, "original_norm": args.original_norm,
              "dummy_history": "prefill retained; decode slot recycled each step",
              "required_slots": required_slots,
              "steps": args.steps, "records": records, "failures": failures}
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
