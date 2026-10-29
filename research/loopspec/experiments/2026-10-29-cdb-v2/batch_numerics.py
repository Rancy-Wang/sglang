"""Teacher-forced batch-shape diagnostics using native one_batch execution.

This is a numerical probe, never a serving latency benchmark. Every
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
    p.add_argument("--graph", choices=("none", "decode", "all"), default="none")
    p.add_argument("--prefill", choices=("joint", "serial"), default="joint")
    p.add_argument("--linear", choices=("native", "no-reduced", "invariant", "rowwise-decode"),
                   default="native")
    p.add_argument("--norm", choices=("native", "rowwise-decode"), default="native")
    p.add_argument("--head", choices=("native", "match-linear"), default="native")
    p.add_argument("--chunked-prefill-size", type=int, default=-1)
    p.add_argument("--probe-question", type=int, choices=range(1, 9))
    p.add_argument("--probe-step", type=int, default=2)
    p.add_argument("--questions", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--model", default="/home/ruoxi/models/Ouro-2.6B")
    p.add_argument("--dataset", type=Path,
                   default=Path("/home/ruoxi/repo/test/datasets/aime.jsonl"))
    args = p.parse_args()
    if len(set(args.questions)) != len(args.questions) or not all(1 <= q <= 8 for q in args.questions):
        p.error("questions must be distinct integers from 1 to 8")
    if args.invariant and args.linear != "native":
        p.error("isolate the global invariant and linear variants")
    if args.graph != "none" and (args.invariant or args.linear != "native" or args.norm != "native" or args.probe_question):
        p.error("graph routing is a separate diagnostic with unchanged operators")
    if args.head == "match-linear" and args.linear not in ("invariant", "rowwise-decode"):
        p.error("matching head requires an invariant or rowwise linear variant")
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
    graph_options = {}
    if args.graph == "decode":
        from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
        graph_options["cuda_graph_config"] = CudaGraphConfig(
            decode=PhaseConfig(backend="full", max_bs=8),
            prefill=PhaseConfig(backend="disabled"),
        )
    options = ServerArgs(
        model_path=args.model, dtype="bfloat16", attention_backend="triton",
        bf16_gemm_backend="auto", max_running_requests=8, page_size=1,
        disable_overlap_schedule=True, disable_radix_cache=True,
        chunked_prefill_size=args.chunked_prefill_size, disable_cuda_graph=args.graph == "none",
        cuda_graph_max_bs_decode=8,
        context_length=2048, mem_fraction_static=0.7,
        **graph_options,
    )
    if args.linear == "no-reduced":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ports = PortArgs.init_new(options)
    publish(options, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    bench, _ = load_model(options, ports, 0, 0)
    if args.invariant:
        # Diagnostic variant only; do not change the serving configuration.
        from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
        enable_batch_invariant_mode()
    runner = bench.torch_runner
    native_linear = torch.nn.functional.linear
    if args.linear in ("invariant", "rowwise-decode"):
        def diagnostic_linear(x, weight, bias=None):
            if x.ndim == 2 and x.is_cuda and x.dtype == torch.bfloat16:
                if args.linear == "invariant":
                    from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent
                    return matmul_persistent(x, weight.t(), bias=bias)
                if 1 < len(x) <= 8:
                    return torch.cat([native_linear(row, weight, bias) for row in x.split(1)])
            return native_linear(x, weight, bias)
        torch.nn.functional.linear = diagnostic_linear
    if args.head == "match-linear":
        native_head = runner.model.logits_processor._compute_lm_head

        def diagnostic_head(hidden, head, embedding_bias=None):
            if args.linear == "invariant":
                return torch.nn.functional.linear(hidden, head.weight, embedding_bias)
            if 1 < len(hidden) <= 8:
                return torch.cat([native_head(row, head, embedding_bias) for row in hidden.split(1)])
            return native_head(hidden, head, embedding_bias)

        runner.model.logits_processor._compute_lm_head = diagnostic_head
    if args.norm == "rowwise-decode":
        from sglang.srt.layers.layernorm import RMSNorm

        def wrap_norm(forward):
            def rowwise_norm(x, *a, **kw):
                if x.ndim == 2 and 1 < len(x) <= 8:
                    return torch.cat([forward(row, *a, **kw) for row in x.split(1)])
                return forward(x, *a, **kw)
            return rowwise_norm

        for module in runner.model.modules():
            if isinstance(module, RMSNorm):
                module.forward = wrap_norm(module.forward)

    layer_rows, layer_hooks, active_probe = {}, [], [None, None]
    if args.probe_question:
        def layer_hook(name):
            def capture(module, inputs, output):
                label, row = active_probe
                if label is None:
                    return
                value = output[0] if isinstance(output, tuple) else output
                layer_rows.setdefault(label, []).append({
                    "name": name, "input": inputs[0][row].detach().clone(),
                    "output": value[row].detach().clone(),
                })
            return capture

        for name, module in runner.model.model.named_modules():
            if "layernorm" in name or name == "norm" or name.endswith(("qkv_proj", "o_proj", "gate_up_proj", "down_proj")):
                layer_hooks.append(module.register_forward_hook(layer_hook(name)))
    graph_calls = []
    native_forward = runner.forward

    def observe_forward(batch, *a, **kw):
        result = native_forward(batch, *a, **kw)
        graph_calls.append({"mode": batch.forward_mode.name,
                            "requests": batch.batch_size,
                            "tokens": len(batch.input_ids),
                            "can_run_graph": result.can_run_graph})
        return result

    runner.forward = observe_forward
    captured, selection = [], []
    hook = runner.model.model.norm.register_forward_hook(
        lambda module, inputs, output: captured.append(output[selection].detach().clone())
    ) if args.graph == "none" else None
    projections, probe_step = [], [False]
    projection = runner.model.model.layers[0].self_attn.qkv_proj

    def capture_projection(module, inputs):
        if probe_step[0]:
            projections.append(inputs[0][selection[-1]:selection[-1] + 1].detach().clone())

    projection_hook = projection.register_forward_pre_hook(capture_projection) if hook else None

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
            probe_step[0] = indices == [4] and step == min(11, args.steps) and not projections
            active_probe[:] = [None, None]
            if args.probe_question and args.probe_question - 1 in indices and step == args.probe_step and step > 0:
                active_probe[:] = ["single" if len(indices) == 1 else "batch",
                                   indices.index(args.probe_question - 1)]
            if step == 0:
                if args.prefill == "serial" and len(reqs) > 1:
                    parts, states, hidden = [], [], []
                    for req in reqs:
                        captured.clear()
                        selection[:] = [len(req.origin_input_ids) - 1]
                        _, part, state = bench.extend([req])
                        parts.append(part.detach().clone())
                        states.append(state)
                        if hook:
                            assert len(captured) == 4
                            hidden.append(torch.stack(captured))
                    batch = states[0]
                    for state in states[1:]:
                        batch.merge_batch(state)
                    logits = torch.cat(parts)
                    if hook:
                        captured[:] = list(torch.cat(hidden, dim=1).unbind())
                else:
                    _, logits, batch = bench.extend(reqs)
            else:
                selection[:] = range(len(indices))
                forced = torch.tensor([ref[i]["output_ids"][step - 1] for i in indices],
                                      device=runner.device, dtype=torch.int64)
                _, logits = bench.decode(forced, batch)
            record = {"logits": logits.detach().cpu().clone()}
            if hook:
                assert len(captured) == 4
                record["hidden"] = torch.stack(captured).cpu()
            records.append(record)
        active_probe[:] = [None, None]
        probe_step[0] = False
        return records

    with torch.inference_mode():
        indices = [q - 1 for q in args.questions]
        singles = [run([i]) for i in indices]
        batched = run(indices)
        projection_report = []
        for depth, x in enumerate(projections):
            for name, value, weight in (
                ("layer0.qkv", x, projection.weight),
                ("lm_head", singles[indices.index(4)][min(11, args.steps)]["hidden"][depth].to(runner.device),
                 runner.model.lm_head.weight),
            ):
                single = torch.nn.functional.linear(value, weight)
                for width in (2, 4, 8):
                    batch = torch.nn.functional.linear(value.repeat(width, 1), weight)
                    projection_report.append({"operator": name, "depth": depth + 1,
                                              "batch_width": width, **error(single[0], batch[0])})
    if hook:
        hook.remove()
        projection_hook.remove()
    for handle in layer_hooks:
        handle.remove()
    torch.nn.functional.linear = native_linear
    torch.save({"questions": args.questions, "single": singles, "batch": batched}, args.output / "tensors.pt")
    layer_report = []
    if "single" in layer_rows and "batch" in layer_rows:
        assert len(layer_rows["single"]) == len(layer_rows["batch"])
        for a, b in zip(layer_rows["single"], layer_rows["batch"], strict=True):
            assert a["name"] == b["name"]
            layer_report.append({"name": a["name"], "input": error(a["input"], b["input"]),
                                 "output": error(a["output"], b["output"])})
    if layer_rows:
        torch.save({label: [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in row.items()}
                            for row in rows] for label, rows in layer_rows.items()}, args.output / "layers.pt")
    report = []
    for step, bat in enumerate(batched):
        for row, q in enumerate(indices):
            single = singles[row][step]
            a, b = single["logits"][0], bat["logits"][row]
            report.append({"step": step, "question": q + 1,
                           "hidden": [error(single["hidden"][d, 0], bat["hidden"][d, row])
                                      for d in range(4)] if hook else None,
                           "logits": error(a, b),
                           "argmax_single": int(a.argmax()), "argmax_batch": int(b.argmax()),
                           "reference_token": ref[q]["output_ids"][step],
                           "single_top2": {"ids": a.topk(2).indices.tolist(), "values": a.topk(2).values.tolist()},
                           "batch_top2": {"ids": b.topk(2).indices.tolist(), "values": b.topk(2).values.tolist()},
                           "single_margin": float(a.topk(2).values.diff().abs()[0]),
                           "batch_margin": float(b.topk(2).values.diff().abs()[0])})
    from sglang.srt.runtime_context import get_exec
    result = {"invariant": args.invariant, "eager": args.graph == "none", "steps": args.steps,
              "resolved_graph": get_exec().graph.cuda_graph_config.to_dict(),
              "graph": args.graph, "prefill": args.prefill, "linear": args.linear,
              "norm": args.norm, "head": args.head, "chunked_prefill_size": args.chunked_prefill_size,
              "layer_probe": {"question": args.probe_question, "step": args.probe_step, "records": layer_report},
              "questions": args.questions, "graph_calls": graph_calls,
              "projection_same_input": projection_report, "records": report}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "invariant": args.invariant,
                      "argmax_mismatches": sum(r["argmax_single"] != r["argmax_batch"] for r in report),
                      "projection": projection_report}), flush=True)


if __name__ == "__main__":
    main()
