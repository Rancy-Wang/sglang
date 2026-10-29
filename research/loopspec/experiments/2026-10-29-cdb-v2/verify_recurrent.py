"""One-loop/mixed-plane execution checks, not a serving performance test."""

import argparse
from array import array
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from batch_numerics import error


def check_envelope(records, limits):
    failures = []

    def check(values, bound, label):
        for metric in ("max_abs", "rmse"):
            ordered = sorted(v[metric] for v in values)
            if not all(math.isfinite(v) for v in ordered):
                failures.append(label + ": nonfinite")
            for quantile, value in (("p99", ordered[math.ceil(len(ordered) * .99) - 1]),
                                    ("max", ordered[-1])):
                if value > bound[quantile][metric]:
                    failures.append(f"{label}.{quantile}.{metric}={value}")

    for depth in range(4):
        check([r["hidden"][depth] for r in records], limits["hidden"][depth], f"depth{depth}")
    check([r["logits"] for r in records], limits["logits"], "logits")
    flips = [r for r in records if r["argmax_reference"] != r["argmax_candidate"]]
    if len(flips) / len(records) > limits["max_argmax_flip_fraction"]:
        failures.append(f"argmax flip fraction {len(flips)}/{len(records)}")
    for row in flips:
        if row["reference_margin"] > limits["protected_reference_margin"]:
            failures.append(f"high margin flip q{row['question']} step{row['step']}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=63)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    parser.add_argument("--dataset", type=Path, default=Path("/home/ruoxi/repo/test/datasets/aime.jsonl"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    limits = json.loads(Path(__file__).with_name("numerical_envelope.json").read_text())
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs, _TorchBenchRunner
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.timeline_pool import TimelineTokenToKVPool, TimelineTokenAllocator
    from sglang.srt.model_executor.recurrent_runner import RecurrentRunner, RecurrentState, RecurrentCoreGraph
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs

    options = ServerArgs(
        model_path=args.model, dtype="bfloat16", attention_backend="triton",
        bf16_gemm_backend="auto", max_running_requests=8, page_size=1,
        disable_overlap_schedule=True, disable_radix_cache=True,
        chunked_prefill_size=-1, disable_cuda_graph=True,
        context_length=2048, mem_fraction_static=0.4,
    )
    ports = PortArgs.init_new(options)
    publish(options, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    baseline, _ = load_model(options, ports, 0, 0)
    runner = copy.copy(baseline.torch_runner)
    # Isolated, native timeline pool. The production allocator selection is
    # deliberately unchanged until ordinary Scheduler integration is validated.
    runner.token_to_kv_pool = TimelineTokenToKVPool(
        6000, recurrent_config=runner.model.config, fixed_depth_slots=True, page_size=1,
        dtype=runner.kv_cache_dtype, head_num=runner.model_config.get_num_kv_heads(1),
        head_dim=runner.model_config.head_dim, v_head_dim=runner.model_config.v_head_dim,
        layer_num=runner.model.config.num_hidden_layers, device=runner.device,
        enable_memory_saver=False,
    )
    runner.token_to_kv_pool_allocator = TimelineTokenAllocator(
        6000, runner.kv_cache_dtype, runner.device, runner.token_to_kv_pool, False
    )
    runner.req_to_token_pool = ReqToTokenPool(8, 2048, runner.device, False)
    runner.kv_index_translator = KVIndexTranslator(
        req_to_token=runner.req_to_token_pool.req_to_token,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        token_to_kv_pool=runner.token_to_kv_pool, page_size=1, device=runner.device,
    )
    runner.attn_backend = TritonAttnBackend(runner)
    recurrent = RecurrentRunner(runner, max_rows=8)
    with torch.inference_mode():
        graph = RecurrentCoreGraph(recurrent, 8) if args.graph else None
        if args.graph:
            recurrent.finish_graphs[8] = RecurrentCoreGraph(recurrent, 8, role="finish")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()][:8]
    ref_tokens = json.loads((args.tokens / "cohort-0.json").read_text())["requests"]
    reference = torch.load(args.reference / "tensors.pt", weights_only=True)
    assert reference["questions"] == list(range(1, 9))
    assert args.steps < min(len(r["output_ids"]) for r in ref_tokens)
    saved, mixed_checks, step = [], [], [0]
    prefill_graphs = {}

    @torch.inference_mode()
    def forward(batch):
        pool = runner.token_to_kv_pool
        slots = batch.out_cache_loc.tolist()
        for depth in range(4):
            pool.timeline.ensure(slots, "core", depth)
        token_rows = None
        if batch.forward_mode.is_extend():
            token_rows = torch.repeat_interleave(
                torch.arange(batch.batch_size, device=runner.device), batch.extend_seq_lens,
                output_size=len(batch.input_ids),
            )
            selection = torch.cumsum(batch.extend_seq_lens, 0).long() - 1
        else:
            selection = torch.arange(batch.batch_size, device=runner.device)
        if args.graph and batch.forward_mode.is_extend():
            size = 1 << (len(batch.input_ids) - 1).bit_length()
            prefill_graphs[size] = RecurrentCoreGraph(recurrent, 8, tokens=size)
        state = recurrent.prepare(batch, token_rows=token_rows)
        states = [state.hidden.clone()]
        for depth in range(4):
            depths = torch.full_like(batch.req_pool_indices, depth)
            if args.graph and batch.forward_mode.is_extend():
                state = prefill_graphs[size].replay(batch, state, depths)
            elif graph and batch.forward_mode.is_decode():
                state = graph.replay(batch, state, depths)
            else:
                state = recurrent.advance(batch, state, depths, token_rows=token_rows)
            # Caller-owned carried state: graph output cannot outlive replay.
            state = RecurrentState(state.hidden.clone(), state.injected)
            states.append(state.hidden)
        hidden = torch.stack([value[selection].cpu() for value in states[1:]])
        logits = recurrent.finish(batch, state, depths, token_rows=token_rows)
        if batch.forward_mode.is_decode() and step[0] in (1, 2, 11, args.steps):
            # Mixed depths, duplicate source request identities and reordered
            # execution rows must all select the correct independent KV plane.
            for order in ([7, 2, 5, 0, 6, 1, 4, 3], [2, 2, 2, 2, 6, 6, 6, 6]):
                index = torch.tensor(order, device=runner.device)
                ds = torch.arange(8, device=runner.device) % 4
                mixed = copy.copy(batch)
                for name in ("input_ids", "req_pool_indices", "seq_lens", "out_cache_loc", "positions"):
                    setattr(mixed, name, getattr(batch, name)[index])
                initial = torch.stack([states[d][i] for d, i in zip(ds.tolist(), order)])
                expected = torch.stack([states[d + 1][i] for d, i in zip(ds.tolist(), order)])
                actual = (graph.replay(mixed, RecurrentState(initial), ds) if graph else
                          recurrent.advance(mixed, RecurrentState(initial), ds)).hidden
                delta = error(expected, actual)
                # Same M and same token-prefix math: permit a BF16 rounding
                # difference, while the broader calibrated AR test runs below.
                assert delta["max_abs"] <= .125 and delta["rmse"] <= .02, delta
                mixed_checks.append({"step": step[0], "order": order, **delta})
        saved.append((hidden, logits.next_token_logits.cpu().clone()))
        return SimpleNamespace(logits_output=logits, can_run_graph=bool(graph and batch.forward_mode.is_decode()))

    runner.forward = forward
    bench = _TorchBenchRunner(runner)
    requests = []
    for i, row in enumerate(rows):
        ids = tokenizer.encode(row["problem"])
        request = Req(rid=f"q{i+1}", origin_input_text="", origin_input_ids=array("q", ids),
                      sampling_params=SamplingParams(temperature=0, max_new_tokens=args.steps+1))
        request.full_untruncated_fill_ids = request.origin_input_ids
        request.logprob_start_len = -1
        request.extend_end = len(ids)
        requests.append(request)
    with torch.inference_mode():
        _, _, batch = bench.extend(requests)
        for s in range(1, args.steps + 1):
            step[0] = s
            forced = torch.tensor([r["output_ids"][s-1] for r in ref_tokens], device=runner.device)
            bench.decode(forced, batch)
    records = []
    for s, (hidden, logits) in enumerate(saved):
        for q in range(8):
            # Independent unchanged native B1 reference, same prefix.
            source = reference["single"][q][s]
            a, b = source["logits"][0], logits[q]
            records.append({"question": q+1, "step": s,
                "hidden": [error(source["hidden"][d, 0], hidden[d, q]) for d in range(4)],
                "logits": error(a, b), "argmax_reference": int(a.argmax()),
                "argmax_candidate": int(b.argmax()),
                "reference_margin": float(a.topk(2).values.diff().abs()[0])})
    failures = check_envelope(records, limits)
    result = {"graph": args.graph, "steps": args.steps, "records": records,
              "mixed_checks": mixed_checks, "failures": failures,
              "calibration": limits, "success": not failures}
    (args.output / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"success": not failures, "positions": len(records),
                      "mixed_checks": mixed_checks, "failures": failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
