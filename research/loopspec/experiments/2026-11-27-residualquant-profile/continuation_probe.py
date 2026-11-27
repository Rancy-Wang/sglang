"""Real-cache ablations; R13 control, independent FP32 oracle and paired timing."""
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
from continuation_source import prepare


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--book', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[16])
    parser.add_argument('--loops', type=int, nargs='+', default=[0])
    parser.add_argument('--modes', nargs='+', default=['grid8', 'grid16', 'grid32'])
    parser.add_argument('--length', type=int)
    parser.add_argument('--layer', type=int, default=23)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--loc-bits', type=int, choices=[32, 64], default=64)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.snapshot / 'manifest.json').read_text())
    n = args.length or manifest['length']
    assert 80 <= n <= manifest['length'] and n % 8 == 0
    rows = []
    for item in sorted((x for x in manifest['files'] if x['layer'] == args.layer),
                       key=lambda x: x['request'])[:max(args.batches)]:
        path = args.snapshot / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        row = torch.load(path, map_location='cpu', weights_only=True)
        if args.length:
            row = dict(row)
            for kind in ('k', 'v'):
                row[kind] = row[kind][:, :n].contiguous()
        rows.append(row)
    assert len(rows) == max(args.batches)
    tables = {}
    for kind, name in (('k', 'codebook.pt'), ('v', 'vq_v_codebook.pt')):
        path = args.book / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['book_hashes'][name]
        blob = torch.load(path, map_location='cpu', weights_only=True)
        h0 = manifest['tp_rank'] * 8
        tables[kind] = reference.build_coupled_codebook(
            blob['codebooks'], 48, 4, 4, 4, 16, 128)[args.layer, h0:h0+8].contiguous()
        del blob
    rope = torch.load(args.snapshot / 'rope.pt', weights_only=True)[:n].cuda()
    extensions = {mode: prepare(mode) for mode in args.modes}
    result = dict(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                     text=True).strip(),
                  manifest=manifest, length=n, derived_prefix=bool(args.length),
                  loc_bits=args.loc_bits, torch=torch.__version__, cuda=torch.version.cuda,
                  modes=args.modes, capacities={m: e.residual_capacity() for m,e in extensions.items()},
                  cases=[], passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b in args.batches:
        pool, raw, queries, ptr, loc = reference.make_pool(rows[:b], tables)
        page, _, _, _, _ = reference.make_pool(rows[:b], tables, paged=2)
        loc = loc.long() if args.loc_bits == 64 else loc
        readers = {mode: ResidualAttention(page, max_batch=b, max_context_len=32768,
                                          global_table=True, layout=2)
                   for mode in ['r13', *args.modes]}
        for mode in args.modes:
            readers[mode].payload_extension = extensions[mode]
            readers[mode].payload_capacity = 324
        for loop in args.loops:
            q = queries[loop]
            k,v = reference.materialize(pool, raw, loop, rope, loc, n)
            expected = reference.oracle(q, k, v, ptr, loc)
            lses = []
            for request in range(b):
                slots = loc[int(ptr[request]):int(ptr[request+1])].long()
                scores = torch.einsum('hd,nhd->hn', q[request].float(), k[slots].float()) * 128**-.5
                lses.append(torch.logsumexp(scores, -1))
            expected_lse = torch.stack(lses)
            calls, checks = {}, {}
            for mode, reader in readers.items():
                fn = lambda reader=reader: reader.forward(q, loop, ptr, loc, rope, 128**-.5)
                reader.partial.fill_(float('nan')); reader.lse.fill_(float('nan'))
                got = fn()
                torch.testing.assert_close(got, expected, atol=.002, rtol=.01)
                torch.testing.assert_close(reader.lse[:b].logsumexp(-1), expected_lse,
                                           atol=2e-5, rtol=2e-5)
                assert torch.isfinite(reader.partial[:b]).all()
                checks[mode] = dict(output=reference.metrics(got, expected),
                                    max_lse_error=float((reader.lse[:b].logsumexp(-1)-expected_lse).abs().max()))
                calls[mode] = fn
            native = reference.NativeAttention(k,v,q,ptr,loc,max_splits=8,dynamic=True)
            torch.testing.assert_close(native(), expected, atol=.002, rtol=.01)
            calls['native'] = native
            timing = reference.paired(calls,args.repeats,32)
            row = dict(batch=b, loop=loop, checks=checks, timing=timing)
            result['cases'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(batch=b, loop=loop, timing_us={m:t['median_ms']*1000 for m,t in timing.items()})),flush=True)
        del readers,pool,raw,queries,page
    result['passed'] = True
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
