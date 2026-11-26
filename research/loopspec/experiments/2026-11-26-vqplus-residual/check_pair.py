"""Interleaved table-copy ablation with identical resident KV and warmed clocks."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import check_residual as test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare', choices=['async', 'global'], default='async')
    args = parser.parse_args()
    torch.manual_seed(20261126)
    theta = torch.arange(8192, device='cuda').float()[:, None] / (
        10000 ** (torch.arange(64, device='cuda').float()[None] / 64))
    rope = torch.cat((theta.cos(), theta.sin()), -1).contiguous()
    results = []
    for batch, length in ((1, 1024), (1, 2048), (1, 8192), (2, 2048),
                          (4, 1024), (4, 8192), (8, 1024), (16, 2048)):
        pool = test.make_pool(cap=batch * 8192 + 8, direct=True, hp_size=batch * 80)
        variants = {mode: test.ResidualAttention(
            pool, max_batch=16, max_context_len=8192, **({'async_table': mode} if args.compare == 'async' else {'global_table': mode}))
            for mode in (False, True)}
        q = torch.randn(batch, 8, 128, device='cuda').bfloat16()
        ptr, loc = test.locations([length] * batch, pool.hp_global_offset,
                                  layout='page8', disjoint=True)
        calls = {mode: lambda mode=mode: variants[mode].forward(q, 0, ptr, loc, rope, 128**-.5)
                 for mode in variants}
        torch.testing.assert_close(calls[False](), calls[True](), atol=0, rtol=0)
        # Both modules compile before measuring; alternate order every repeat.
        for _ in range(3):
            for mode in (False, True): test.bench(calls[mode], inner=16)
        times = {False: [], True: []}
        for repeat in range(8):
            for mode in ((False, True) if repeat % 2 == 0 else (True, False)):
                times[mode].append(test.bench(calls[mode], inner=16))
        row = dict(compare=args.compare, batch=batch, length=length, baseline_ms=times[False], candidate_ms=times[True],
                   speedup=statistics.median(times[False]) / statistics.median(times[True]))
        results.append(row)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(batch, length, row['speedup'], flush=True)


if __name__ == '__main__':
    main()
