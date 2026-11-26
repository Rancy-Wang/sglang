"""Existing-kernel split sweep with independent output and LSE checks."""
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
    parser.add_argument("--targets", type=int, nargs="+", default=[1,2,3,4,5,6,8,12,16,24,32])
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
    expected_lse = []
    for b in range(args.batch):
        slots = loc[int(ptr[b]):int(ptr[b+1])].long()
        scores = torch.einsum("hd,nhd->hn", q[b].float(), k[slots].float()) * 128**-.5
        expected_lse.append(torch.logsumexp(scores, -1))
    expected_lse = torch.stack(expected_lse)
    baseline = PagedResidualAttention(page, max_batch=args.batch, layout=2, cooperative=False)
    native = reference.NativeAttention(k, v, q, ptr, loc, max_splits=8, dynamic=True)
    calls = {"page_auto": lambda: baseline.forward(q,args.loop,ptr,loc,rope,128**-.5), "native": native}
    correctness = {}
    effective = {}
    for target in args.targets:
        assert 1 <= target <= 32
        reader = PagedResidualAttention(page,max_batch=args.batch,layout=2,cooperative=False)
        extension = reader.extension
        def launch(*xs, extension=extension, target=target):
            return extension.residual_cuda(*xs[:-1], target)
        reader.extension = SimpleNamespace(residual_cuda=launch)
        fn = lambda reader=reader: reader.forward(q,args.loop,ptr,loc,rope,128**-.5)
        got = fn()
        torch.testing.assert_close(got,expected,atol=.002,rtol=.01)
        lse = torch.logsumexp(reader.lse[:args.batch],dim=-1)
        torch.testing.assert_close(lse,expected_lse,atol=2e-5,rtol=2e-5)
        name = f"target{target}"
        correctness[name] = dict(output=reference.metrics(got,expected), max_lse_error=float((lse-expected_lse).abs().max()))
        effective[name] = min(32,max((n+1023)//1024,min(target,max(1,(n+63)//64))))
        calls[name] = fn
    for name in ("page_auto","native"):
        torch.testing.assert_close(calls[name](),expected,atol=.002,rtol=.01)
    timing = reference.paired(calls,args.repeats,32)
    result = dict(source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        manifest=manifest,batch=args.batch,layer=args.layer,loop=args.loop,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),correctness=correctness,effective_splits=effective,timing=timing,passed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"timing_us":{k:v["median_ms"]*1000 for k,v in timing.items()},"passed":True}),flush=True)


if __name__ == "__main__":
    main()
