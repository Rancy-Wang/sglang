"""Independent packed-code oracle and mixed-cache GPU tests; no accuracy samples."""
import argparse
import importlib.util
import json
import math
import sys
import subprocess
import inspect
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--kernel-root', type=Path, default=ROOT)
KERNEL_ROOT = bootstrap.parse_known_args()[0].kernel_root
SRC = KERNEL_ROOT / 'python/sglang/kernels/ops/loopvq'
for name in ('sglang', 'sglang.kernels', 'sglang.kernels.ops', 'sglang.kernels.ops.loopvq'):
    module = types.ModuleType(name)
    module.__path__ = [str(SRC)] if name.endswith('loopvq') else []
    sys.modules[name] = module
from sglang.kernels.ops.loopvq.residual_encode import encode_residual_steps, encode_residual_from_hp
from sglang.kernels.ops.loopvq.residual_attention import ResidualAttention


def reference_pack(actual, base):
    error = (actual.float() - base.float()).reshape(*actual.shape[:-1], 4, 32)
    scale = (error.abs().amax(-1) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    codes = (error / scale.float().unsqueeze(-1)).round().clamp(-7, 7).to(torch.int32).reshape_as(actual)
    return ((codes[..., ::2] & 15) | ((codes[..., 1::2] & 15) << 4)).byte(), scale


def unpack(packed, scale):
    codes = torch.stack((packed & 15, packed >> 4), -1).to(torch.int32).flatten(-2)
    codes = torch.where(codes >= 8, codes - 16, codes).float()
    return codes * scale.float().repeat_interleave(32, -1)


def centers(indices, cb):
    h = torch.arange(indices.shape[1], device='cuda')[None, :, None]
    c = torch.arange(128, device='cuda')[None, None, :]
    codes = indices.long().repeat_interleave(4, -1)
    return cb[h, c // 4, codes, c % 4].float()


def codec_test():
    lb, h, n, cap = 2, 8, 19, 64
    enc = torch.randn(lb, h, 32, 256, 16, device='cuda').half()
    cb = torch.stack([enc[b, ..., u::4] for u in range(4) for b in range(lb)]).bfloat16()
    idx = torch.randint(256, (lb, cap, h, 32), device='cuda', dtype=torch.uint8)
    hp = torch.randn(lb * 4, cap, h, 128, device='cuda').bfloat16()
    src = torch.randperm(cap, device='cuda')[:n]
    dst = torch.randperm(cap, device='cuda')[:n]
    valid = torch.arange(n, device='cuda') % 5 != 0
    packed = torch.full((lb * 4, cap, h, 64), 165, device='cuda', dtype=torch.uint8)
    scales = torch.full((lb * 4, cap, h, 4), -1., device='cuda', dtype=torch.float16)
    encode_residual_from_hp(hp, idx, cb, packed, scales, src, dst, valid)
    for slot in range(lb * 4):
        expected, scale = reference_pack(hp[slot, src[valid]], centers(idx[slot % lb, dst[valid]], cb[slot]))
        torch.testing.assert_close(packed[slot, dst[valid]], expected, rtol=0, atol=0)
        torch.testing.assert_close(scales[slot, dst[valid]], scale, rtol=0, atol=0)
    assert (packed[:, dst[~valid]] == 165).all()
    assert (scales[:, dst[~valid]] == -1).all()
    # Prefill: non-contiguous inputs, HP destinations outside quant arena.
    loc = torch.cat((dst, torch.tensor([cap + 8, cap + 9], device='cuda')))
    steps = [torch.randn(n + 2, h * 3, 128, device='cuda').bfloat16()[:, :h] for _ in range(4)]
    encode_residual_steps(steps, idx[1], cb, packed, scales, loc, 1)
    for u in range(4):
        slot = u * lb + 1
        expected, scale = reference_pack(steps[u][:n], centers(idx[1, dst], cb[slot]))
        torch.testing.assert_close(packed[slot, dst], expected, rtol=0, atol=0)
        torch.testing.assert_close(scales[slot, dst], scale, rtol=0, atol=0)
    # All-zero errors must have zero codes and finite minimum scale.
    steps = [centers(idx[1, dst], cb[u * lb + 1]).bfloat16() for u in range(4)]
    encode_residual_steps(steps, idx[1], cb, packed, scales, dst, 1)
    assert not packed[1::lb, dst].any()
    assert (scales[1::lb, dst] == torch.finfo(torch.float16).tiny).all()
    # Exact half-integer ties with scale=1, signed codes, and tiny-scale floor.
    cb.zero_()
    values = torch.tensor([7., -7., .5, -.5, 1.5, -1.5, 2.5, -2.5] * 16,
                          device='cuda', dtype=torch.bfloat16)
    for factor in (1., 2.**-20):
        steps = [(values * factor).expand(n, h, 128).contiguous() for _ in range(4)]
        encode_residual_steps(steps, idx[1], cb, packed, scales, dst, 1)
        expected, scale = reference_pack(steps[0], torch.zeros_like(steps[0]))
        for u in range(4):
            torch.testing.assert_close(packed[u * lb + 1, dst], expected, rtol=0, atol=0)
            torch.testing.assert_close(scales[u * lb + 1, dst], scale, rtol=0, atol=0)
    return {'passed': True, 'cases': ['flush', 'valid_mask', 'prefill_strided', 'loop_layout', 'zero_residual', 'round_even_ties', 'tiny_scale']}


def invalid_codec(case, output):
    # Run in a separate process: a device assertion intentionally poisons CUDA.
    value = {'nan': float('nan'), 'inf': float('inf'), 'overflow': 1.e10}[case]
    steps = [torch.full((1, 1, 128), value, dtype=torch.bfloat16, device='cuda') for _ in range(4)]
    idx = torch.zeros((1, 1, 32), dtype=torch.uint8, device='cuda')
    cb = torch.zeros((4, 1, 32, 256, 4), dtype=torch.bfloat16, device='cuda')
    packed = torch.empty((4, 1, 1, 64), dtype=torch.uint8, device='cuda')
    scales = torch.empty((4, 1, 1, 4), dtype=torch.float16, device='cuda')
    loc = torch.zeros(1, dtype=torch.int32, device='cuda')
    try:
        encode_residual_steps(steps, idx, cb, packed, scales, loc, 0)
        torch.cuda.synchronize()
    except RuntimeError as error:
        if 'device-side assert' not in str(error): raise
        Path(output).write_text(json.dumps({'invalid_case': case, 'rejected': True}))
        return
    raise AssertionError('Invalid residual input was silently accepted')


def make_pool(cap=32776, heads=8, direct=False, hp_size=80):
    pool = types.SimpleNamespace(head_num=heads, head_dim=128, residual_bits=4, hp_global_offset=cap)
    for kind in ('k', 'v'):
        cb = torch.randn(1, heads, 32, 256, 4, device='cuda').bfloat16()
        idx = torch.randint(256, (cap, heads, 32), device='cuda', dtype=torch.uint8)
        if direct:
            lo = torch.randint(0, 15, (cap, heads, 64), device='cuda', dtype=torch.uint8)
            hi = torch.randint(0, 15, lo.shape, device='cuda', dtype=torch.uint8)
            packed = (lo + (lo >= 8).byte()) | ((hi + (hi >= 8).byte()) << 4)
            scale = torch.full((cap, heads, 4), .02, device='cuda', dtype=torch.float16)
        else:
            base = centers(idx, cb[0])
            raw = base + torch.randn_like(base) * 0.15
            packed, scale = reference_pack(raw, base)
        setattr(pool, kind + '_book', types.SimpleNamespace(cb16=cb))
        setattr(pool, kind + '_indices', idx)
        setattr(pool, kind + '_residual', packed.unsqueeze(0))
        setattr(pool, kind + '_residual_scale', scale.unsqueeze(0))
        setattr(pool, 'hp_' + kind, torch.randn(1, hp_size, heads, 128, device='cuda').bfloat16())
    pool.get_key_buffer = lambda layer: pool.k_indices
    pool.get_value_buffer = lambda layer: pool.v_indices
    pool.get_hp_key_buffer = lambda layer: pool.hp_k[layer]
    pool.get_hp_value_buffer = lambda layer: pool.hp_v[layer]
    return pool


def locations(lengths, cap, layout="random-token", disjoint=False):
    ptr, rows = [0], []
    region = ((cap - 8) // len(lengths) // 8) * 8 if disjoint else cap - 8
    for request, length in enumerate(lengths):
        pre = min(8, length)
        recent = min(64, length - pre)
        quant = length - pre - recent
        if layout == "page8":
            pages = torch.randperm(region // 8, device='cuda')[:math.ceil(quant / 8)] + 1
            quant_locs = (pages[:, None] * 8 + torch.arange(8, device='cuda')[None]).flatten()[:quant]
        else:
            quant_locs = torch.randperm(region, device='cuda')[:quant] + 8
        quant_locs = quant_locs + (request * region if disjoint else 0)
        hp_base = cap + (request * 80 if disjoint else 0)
        rows.append(torch.cat((torch.arange(pre, device='cuda') + hp_base,
                               quant_locs,
                               torch.arange(recent, device='cuda') + hp_base + 8)))
        ptr.append(ptr[-1] + length)
    return torch.tensor(ptr, dtype=torch.int32, device='cuda'), torch.cat(rows).int()


def oracle(q, pool, ptr, loc, rope, return_lse=False):
    outputs, lses = [], []
    for b in range(q.shape[0]):
        slots = loc[int(ptr[b]):int(ptr[b+1])].long()
        if not slots.numel():
            outputs.append(torch.zeros_like(q[b])); lses.append(torch.full((pool.head_num,), -float('inf'), device='cuda')); continue
        quant = slots < pool.hp_global_offset
        values = []
        for kind in ('k', 'v'):
            x = torch.empty(slots.numel(), pool.head_num, 128, device='cuda')
            ids = slots[quant]
            base = centers(getattr(pool, kind + '_indices')[ids], getattr(pool, kind + '_book').cb16[0])
            x[quant] = base + unpack(getattr(pool, kind + '_residual')[0, ids], getattr(pool, kind + '_residual_scale')[0, ids])
            x[~quant] = getattr(pool, 'hp_' + kind)[0, slots[~quant] - pool.hp_global_offset].float()
            values.append(x)
        k, v = values
        c, s = rope[:slots.numel()].chunk(2, -1)
        c, s = c.repeat(1, 2)[:, None], s.repeat(1, 2)[:, None]
        k = (k * c + torch.cat((-k[..., 64:], k[..., :64]), -1) * s).bfloat16().float()
        scores = torch.einsum('hd,nhd->hn', q[b].float(), k) / math.sqrt(128)
        lses.append(scores.logsumexp(-1))
        outputs.append(torch.einsum('hn,nhd->hd', scores.softmax(-1), v.bfloat16().float()).bfloat16())
    return (torch.stack(outputs), torch.stack(lses)) if return_lse else torch.stack(outputs)


def bench(fn, inner=1):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner): result = fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(30): graph.replay()
    end.record(); end.synchronize()
    ms = start.elapsed_time(end) / (30 * inner)
    graph.reset()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--kernel-root', type=Path, default=ROOT)
    ap.add_argument('--implementation', choices=['cuda', 'triton'], default='cuda')
    ap.add_argument('--threads', type=int, choices=[128, 256, 512], default=512)
    ap.add_argument('--fixed-splits', action='store_true')
    ap.add_argument('--unroll', type=int, choices=[1,2,4], default=1)
    ap.add_argument('--group-size', type=int, choices=[0,1,2], default=0)
    ap.add_argument('--graph-inner', type=int, choices=[1,16], default=16)
    ap.add_argument('--layout', choices=['random-token','page8'], default='page8')
    ap.add_argument('--async-table', action='store_true')
    ap.add_argument('--invalid-codec', choices=['nan', 'inf', 'overflow'])
    args = ap.parse_args()
    if args.invalid_codec:
        invalid_codec(args.invalid_codec, args.output)
        return
    torch.manual_seed(20261126)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'codec': codec_test(), 'attention': [], 'timings': [], 'tolerance': {'atol': .002, 'rtol': .01}}
    result['environment'] = {'torch': torch.__version__, 'kernel_root': str(KERNEL_ROOT),
        'source_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=KERNEL_ROOT,text=True).strip(),
        'gpu': torch.cuda.get_device_name()}
    pool = make_pool()
    theta = torch.arange(32768, device='cuda').float()[:, None] / (10000 ** (torch.arange(64, device='cuda').float()[None] / 64))
    rope = torch.cat((theta.cos(), theta.sin()), -1).contiguous()
    options = dict(implementation=args.implementation, threads=args.threads, dynamic_splits=not args.fixed_splits)
    if 'unroll' in inspect.signature(ResidualAttention).parameters:
        options['unroll'] = args.unroll
    else:
        assert args.unroll == 1, 'Baseline does not implement token unroll'
    if 'group_size' in inspect.signature(ResidualAttention).parameters:
        options['group_size'] = args.group_size
    else:
        assert args.group_size in (0, 1)
    if 'async_table' in inspect.signature(ResidualAttention).parameters:
        options['async_table'] = args.async_table
    else:
        assert not args.async_table
    attn = ResidualAttention(pool, max_batch=16, max_context_len=32768, **options)
    for length in (0, 8, 64, 72, 79, 80, 127, 128, 1024, 2048, 4096, 8192, 16384, 32768):
        q = torch.randn(2, 24, 128, device='cuda').bfloat16()[:, :8]
        ptr, loc = locations([length, length // 2], pool.hp_global_offset)
        # Stale scratch must not leak even for fully empty rows.
        attn.partial.fill_(float('nan')); attn.lse.fill_(float('nan'))
        if getattr(attn, 'grouped', None) is not None:
            attn.group_partial.fill_(float('nan')); attn.group_lse.fill_(float('nan'))
        expected, expected_lse = oracle(q, pool, ptr, loc, rope, return_lse=True)
        actual = attn.forward(q, 0, ptr, loc, rope, 128**-.5)
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.01)
        torch.testing.assert_close((attn.group_lse if getattr(attn, 'grouped', None) is not None else attn.lse)[:2].logsumexp(-1), expected_lse, atol=2e-5, rtol=2e-5)
        result['attention'].append({'length': length, 'max_abs': (actual.float()-expected.float()).abs().max().item()})
    # Capture once, then change physical addresses and request lengths in place.
    q = torch.randn(2, 8, 128, device='cuda').bfloat16()
    ptr, loc = locations([128, 128], pool.hp_global_offset)
    for _ in range(3): attn.forward(q, 0, ptr, loc, rope, 128**-.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attn.forward(q, 0, ptr, loc, rope, 128**-.5)
    for length in (80, 8, 128):
        p2, l2 = locations([length, length], pool.hp_global_offset)
        ptr.copy_(p2); loc[:l2.numel()].copy_(l2)
        attn.partial.fill_(float('nan')); attn.lse.fill_(float('nan'))
        if getattr(attn, 'grouped', None) is not None:
            attn.group_partial.fill_(float('nan')); attn.group_lse.fill_(float('nan'))
        graph.replay()
        torch.testing.assert_close(captured, oracle(q, pool, ptr, loc, rope), atol=.002, rtol=.01)
    # Nonzero descriptor length with all-zero locations is graph padding.
    loc.zero_(); graph.replay()
    assert not captured.any()
    graph.reset()
    result['graph_dynamic_and_padding'] = True
    for odd in (1, 3, 5, 15, 16):
        q = torch.randn(odd, 8, 128, device='cuda').bfloat16()
        ptr, loc = locations([128 if i % 2 == 0 else 8 for i in range(odd)], pool.hp_global_offset)
        actual = attn.forward(q, 0, ptr, loc, rope, 128**-.5)
        torch.testing.assert_close(actual, oracle(q, pool, ptr, loc, rope), atol=.002, rtol=.01)
    result['unequal_and_odd_batches'] = True
    for b in (1, 2, 4, 8, 16):
        pool = make_pool(cap=b * 32768 + 8, direct=True, hp_size=b * 80)
        attn = ResidualAttention(pool, max_batch=16, max_context_len=32768, **options)
        q = torch.randn(b, 24, 128, device='cuda').bfloat16()[:, :8]
        for length in (1024, 2048, 4096, 8192, 16384, 32768):
            ptr, loc = locations([length] * b, pool.hp_global_offset, args.layout, disjoint=True)
            ms = bench(lambda: attn.forward(q, 0, ptr, loc, rope, 128**-.5), args.graph_inner)
            result['timings'].append({'batch': b, 'length': length, 'splits': attn.splits, 'threads': args.threads, 'dynamic_splits': not args.fixed_splits, 'unroll': args.unroll, 'layout': args.layout, 'graph_inner': args.graph_inner, 'disjoint_request_addresses': True, 'group_size': args.group_size, 'async_table': args.async_table, 'selected_group': 2 if getattr(attn, 'use_grouped', lambda b: args.group_size == 2 and b > 1)(b) else 1, 'graph_ms': ms})
    result['passed'] = True
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
