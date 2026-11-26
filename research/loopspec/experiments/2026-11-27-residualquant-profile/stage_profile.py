"""Matched CUDA controls and perturbed CTA stage clocks; no NCU attribution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import torch
from run_isolated import bootstrap, OLD, ROOT

bootstrap()
sys.path.insert(0, str(OLD))
import diagnose_paged as reference
from sglang.kernels.ops.loopvq.residual_attention_paged_cuda import _SRC, PagedResidualAttention


def replace_once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)


def compile_control(kind, instrument, threads=512):
    source = _SRC
    if kind == "dense":
        start = source.index("__device__ __forceinline__ void row4(")
        stop = source.index("__device__ __forceinline__ void copy_table", start)
        signature = source[start:source.index("{", start)]
        source = source[:start] + signature + r'''{
    const auto* row = hp + ((long)loc * heads + head) * 128 + group * 4;
    uint2 value = *reinterpret_cast<const uint2*>(row);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(&value);
    #pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(v[j]);
}
''' + source[stop:]
        source = replace_once(source, "float kl = bf(low[j] * c[j] - high[j] * s[j]);", "float kl = low[j];")
        source = replace_once(source, "float kh = bf(high[j] * c[j] + low[j] * s[j]);", "float kh = high[j];")
    tensors = 16
    if instrument:
        tensors = 17
        source = replace_once(source, "float* OUT, float* LSE,", "float* OUT, float* LSE, unsigned long long* CLOCKS,")
        source = replace_once(source, "torch::Tensor lse, int64_t offset", "torch::Tensor lse, torch::Tensor clocks, int64_t offset")
        source = replace_once(source, "lse.data_ptr<float>(), q.size(1)",
                              "lse.data_ptr<float>(), reinterpret_cast<unsigned long long*>(clocks.data_ptr<int64_t>()), q.size(1)")
        source = replace_once(source, "    // Two tokens per warp,", "    if (!tid) CLOCKS[dst * 4] = clock64();\n    // Two tokens per warp,")
        source = replace_once(source, "    __syncthreads();\n#if VQPLUS_ASYNC\n    // No QK", "    __syncthreads();\n    if (!tid) CLOCKS[dst * 4 + 1] = clock64();\n#if VQPLUS_ASYNC\n    // No QK")
        source = replace_once(source, "    // Four tokens per warp,", "    if (!tid) CLOCKS[dst * 4 + 2] = clock64();\n    // Four tokens per warp,")
        source = replace_once(source, "        OUT[dst * 128 + tid] = total / stats[1];\n    }", "        OUT[dst * 128 + tid] = total / stats[1];\n    }\n    if (!tid) CLOCKS[dst * 4 + 3] = clock64();")
    from torch.utils.cpp_extension import load_inline
    extension = load_inline(
        name=f"vqrq_control_{kind}_clock{int(instrument)}_t{threads}_r2",
        cpp_sources="#include <torch/extension.h>\nvoid residual_init();\nvoid residual_cuda(" + ",".join(["torch::Tensor"] * tensors) + ",int64_t,double,int64_t);",
        cuda_sources=source, functions=["residual_init", "residual_cuda"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false", "-Xptxas=-v",
            f"-DVQPLUS_THREADS={threads}", "-DVQPLUS_UNROLL=1", "-DVQPLUS_GLOBAL=1", "-DVQPLUS_ASYNC=0",
            "-DVQPAGE_LAYOUT=2", "-DVQPAGE_COOP=0", f"-DVQPAGE_SCALAR={int(kind == 'scalar')}",
            "-gencode=arch=compute_80,code=sm_80"])
    extension.residual_init()
    return extension


def make_reader(pool, batch, kind, instrument, k, v):
    selected = copy.copy(pool)
    if kind == "dense":
        selected.get_hp_key_buffer = lambda loop: k
        selected.get_hp_value_buffer = lambda loop: v
    reader = PagedResidualAttention(selected, max_batch=batch, layout=2, cooperative=False)
    extension = compile_control(kind, instrument)
    clocks = torch.zeros(batch, 8, 32, 4, device="cuda", dtype=torch.int64)
    if instrument:
        def launch(*args):
            return extension.residual_cuda(*args[:16], clocks, *args[16:])
        reader.extension = SimpleNamespace(residual_cuda=launch)
    else:
        reader.extension = extension
    return reader, clocks


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    n = manifest["length"]
    rows = []
    for item in sorted((r for r in manifest["files"] if r["layer"] == args.layer), key=lambda r: r["request"])[:args.batch]:
        path = args.snapshot / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        rows.append(torch.load(path, map_location="cpu", weights_only=True))
    assert len(rows) == args.batch
    tables = {}
    for kind, name in (("k", "codebook.pt"), ("v", "vq_v_codebook.pt")):
        path = args.book / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["book_hashes"][name]
        blob = torch.load(path, map_location="cpu", weights_only=True)
        h0 = manifest["tp_rank"] * 8
        tables[kind] = reference.build_coupled_codebook(blob["codebooks"], 48, 4, 4, 4, 16, 128)[args.layer, h0:h0+8].contiguous()
        del blob
    pool, raw, queries, ptr, loc = reference.make_pool(rows, tables)
    page, _, _, _, _ = reference.make_pool(rows, tables, paged=2)
    rope = torch.load(args.snapshot / "rope.pt", weights_only=True).cuda()
    q = queries[args.loop]
    k, v = reference.materialize(pool, raw, args.loop, rope, loc, n)
    expected = reference.oracle(q, k, v, ptr, loc)
    scalar_pool = copy.copy(pool)
    for kind in ("k", "v"):
        setattr(scalar_pool, kind+"_book", SimpleNamespace(cb16=torch.zeros_like(getattr(pool, kind+"_book").cb16)))
    sk, sv = reference.materialize(scalar_pool, raw, args.loop, rope, loc, n)
    scalar_expected = reference.oracle(q, sk, sv, ptr, loc)
    baseline = PagedResidualAttention(page, max_batch=args.batch, layout=2, cooperative=False)
    native = reference.NativeAttention(k, v, q, ptr, loc, max_splits=8, dynamic=True)
    calls = {"production_page": lambda: baseline.forward(q, args.loop, ptr, loc, rope, 128**-.5), "native_serving": native}
    instruments = {}
    correctness = {}
    for kind in ("full", "dense", "scalar"):
        for instrument in (False, True):
            reader, clocks = make_reader(page, args.batch, kind, instrument, k, v)
            name = kind + ("_clock" if instrument else "")
            fn = lambda reader=reader: reader.forward(q, args.loop, ptr, loc, rope, 128**-.5)
            got = fn()
            target = scalar_expected if kind == "scalar" else expected
            torch.testing.assert_close(got, target, atol=.002, rtol=.01)
            correctness[name] = reference.metrics(got, target)
            calls[name] = fn
            if instrument:
                instruments[kind] = (fn, clocks)
    timing = reference.paired(calls, args.repeats, 32)
    cycle_samples = {}
    for kind, (fn, clocks) in instruments.items():
        samples = []
        for _ in range(args.repeats):
            clocks.zero_()
            fn()
            stamps = clocks.cpu()
            active = stamps[..., 3] > 0
            delta = stamps[..., 1:] - stamps[..., :-1]
            values = delta[active].double()
            assert (values > 0).all()
            samples.append({"active_ctas": int(active.sum()), "mean_cycles": values.mean(0).tolist(),
                            "median_cycles": values.median(0).values.tolist(), "max_cycles": values.max(0).values.tolist()})
        cycle_samples[kind] = samples
    result = dict(source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        manifest=manifest, batch=args.batch, layer=args.layer, loop=args.loop, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), correctness=correctness, timing=timing, stage_order=["QK", "softmax", "PV_and_partial_store"],
        cta_cycle_samples=cycle_samples, notes=["CTA residence cycles are not additive device latency.",
            "Clock instrumentation can change registers and scheduling; quantify overhead.",
            "Dense is a matched CUDA mapping control; native_serving is the production BF16 control.",
            "Scalar removes centers from existing residuals; it is not an alternative quality-certified codec.",
            "Materialization and cache encoding excluded from all reader timings."], passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"timing_us": {k: v["median_ms"] * 1000 for k, v in timing.items()},
                      "mean_stage_cycles": {k: [statistics.median(s["mean_cycles"][i] for s in v) for i in range(3)] for k, v in cycle_samples.items()}}), flush=True)


if __name__ == "__main__":
    main()
