"""Actual production dispatch: paired original/payload/native real-input replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from run_isolated import bootstrap, OLD, ROOT

bootstrap()
sys.path.insert(0, str(OLD))
import diagnose_paged as reference
from sglang.kernels.ops.loopvq.residual_attention import ResidualAttention


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 15, 16])
    parser.add_argument("--loops", type=int, nargs="+", default=[0])
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.snapshot / "manifest.json").read_text())
    n = manifest["length"]
    rows = []
    for item in sorted((r for r in manifest["files"] if r["layer"] == args.layer), key=lambda r: r["request"])[:max(args.batches)]:
        path = args.snapshot / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        rows.append(torch.load(path, map_location="cpu", weights_only=True))
    assert len(rows) == max(args.batches)
    tables = {}
    for kind, name in (("k", "codebook.pt"), ("v", "vq_v_codebook.pt")):
        path = args.book / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["book_hashes"][name]
        blob = torch.load(path, map_location="cpu", weights_only=True)
        h0 = manifest["tp_rank"] * 8
        tables[kind] = reference.build_coupled_codebook(blob["codebooks"], 48, 4, 4, 4, 16, 128)[args.layer, h0:h0 + 8].contiguous()
        del blob
    rope = torch.load(args.snapshot / "rope.pt", weights_only=True).cuda()
    result = dict(source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  manifest=manifest, layer=args.layer, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), cases=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b in args.batches:
        pool, raw, queries, ptr, loc = reference.make_pool(rows[:b], tables)
        page, _, _, _, _ = reference.make_pool(rows[:b], tables, paged=2)
        readers = {
            name: ResidualAttention(page, max_batch=b, max_context_len=32768,
                                    global_table=True, layout=2, payload=enabled)
            for name, enabled in (("original", False), ("payload", True))
        }
        for loop in args.loops:
            q = queries[loop]
            k, v = reference.materialize(pool, raw, loop, rope, loc, n)
            expected = reference.oracle(q, k, v, ptr, loc)
            expected_lse = []
            for req in range(b):
                slots = loc[int(ptr[req]):int(ptr[req + 1])].long()
                scores = torch.einsum("hd,nhd->hn", q[req].float(), k[slots].float()) * 128**-.5
                expected_lse.append(torch.logsumexp(scores, -1))
            expected_lse = torch.stack(expected_lse)
            calls, checks = {}, {}
            for name, reader in readers.items():
                fn = lambda reader=reader: reader.forward(q, loop, ptr, loc, rope, 128**-.5)
                partial = reader.group_partial if reader.use_grouped(b) else reader.partial
                lse = reader.group_lse if reader.use_grouped(b) else reader.lse
                partial.fill_(float("nan"))
                lse.fill_(float("nan"))
                got = fn()
                torch.testing.assert_close(got, expected, atol=.002, rtol=.01)
                actual_lse = lse[:b].logsumexp(-1)
                torch.testing.assert_close(actual_lse, expected_lse, atol=2e-5, rtol=2e-5)
                checks[name] = dict(output=reference.metrics(got, expected),
                                    lse_max_abs=float((actual_lse - expected_lse).abs().max()),
                                    grouped=reader.use_grouped(b), payload=reader.use_payload(b),
                                    resident_capacity=reader.payload_capacity)
                calls[name] = fn
            native = reference.NativeAttention(k, v, q, ptr, loc, max_splits=8, dynamic=True)
            torch.testing.assert_close(native(), expected, atol=.002, rtol=.01)
            calls["native"] = native
            timing = reference.paired(calls, args.repeats, 32)
            result["cases"].append(dict(batch=b, loop=loop, correctness=checks, timing=timing))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(dict(batch=b, loop=loop, timing_us={k: v["median_ms"] * 1000 for k, v in timing.items()}, correctness=checks)), flush=True)
        del readers, pool, raw, queries, page
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
