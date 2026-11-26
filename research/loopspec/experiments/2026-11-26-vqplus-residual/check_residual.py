"""Independent packed-code oracle and mixed-cache GPU tests; no accuracy samples."""
import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'python/sglang/kernels/ops/loopvq'
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
    return {'passed': True, 'cases': ['flush', 'valid_mask', 'prefill_strided', 'loop_layout', 'zero_residual']}


def make_pool(cap=32776, heads=8):
    pool = types.SimpleNamespace(head_num=heads, head_dim=128, residual_bits=4, hp_global_offset=cap)
    for kind in ('k', 'v'):
        cb = torch.randn(1, heads, 32, 256, 4, device='cuda').bfloat16()
        idx = torch.randint(256, (cap, heads, 32), device='cuda', dtype=torch.uint8)
        base = centers(idx, cb[0])
        raw = base + torch.randn_like(base) * 0.15
        packed, scale = reference_pack(raw, base)
        setattr(pool, kind + '_book', types.SimpleNamespace(cb16=cb))
        setattr(pool, kind + '_indices', idx)
        setattr(pool, kind + '_residual', packed.unsqueeze(0))
        setattr(pool, kind + '_residual_scale', scale.unsqueeze(0))
        setattr(pool, 'hp_' + kind, torch.randn(1, 80, heads, 128, device='cuda').bfloat16())
    pool.get_key_buffer = lambda layer: pool.k_indices
    pool.get_value_buffer = lambda layer: pool.v_indices
    pool.get_hp_key_buffer = lambda layer: pool.hp_k[layer]
    pool.get_hp_value_buffer = lambda layer: pool.hp_v[layer]
    return pool


def locations(lengths, cap):
    ptr, rows = [0], []
    for length in lengths:
        pre = min(8, length)
        recent = min(64, length - pre)
        quant = length - pre - recent
        rows.append(torch.cat((torch.arange(pre, device='cuda') + cap,
                               torch.randperm(cap - 8, device='cuda')[:quant] + 8,
                               torch.arange(recent, device='cuda') + cap + 8)))
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


def bench(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result = fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(30): graph.replay()
    end.record(); end.synchronize()
    ms = start.elapsed_time(end) / 30
    graph.reset()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--implementation', choices=['cuda', 'triton'], default='cuda')
    ap.add_argument('--threads', type=int, choices=[128, 256, 512], default=512)
    ap.add_argument('--fixed-splits', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(20261126)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'codec': codec_test(), 'attention': [], 'timings': [], 'tolerance': {'atol': .002, 'rtol': .01}}
    pool = make_pool()
    theta = torch.arange(32768, device='cuda').float()[:, None] / (10000 ** (torch.arange(64, device='cuda').float()[None] / 64))
    rope = torch.cat((theta.cos(), theta.sin()), -1).contiguous()
    attn = ResidualAttention(pool, max_batch=16, max_context_len=32768, implementation=args.implementation, threads=args.threads, dynamic_splits=not args.fixed_splits)
    for length in (0, 8, 64, 72, 79, 80, 127, 128, 1024, 2048, 4096, 8192, 16384, 32768):
        q = torch.randn(2, 24, 128, device='cuda').bfloat16()[:, :8]
        ptr, loc = locations([length, length // 2], pool.hp_global_offset)
        # Stale scratch must not leak even for fully empty rows.
        attn.partial.fill_(float('nan')); attn.lse.fill_(float('nan'))
        expected, expected_lse = oracle(q, pool, ptr, loc, rope, return_lse=True)
        actual = attn.forward(q, 0, ptr, loc, rope, 128**-.5)
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.01)
        torch.testing.assert_close(attn.lse[:2].logsumexp(-1), expected_lse, atol=2e-5, rtol=2e-5)
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
        graph.replay()
        torch.testing.assert_close(captured, oracle(q, pool, ptr, loc, rope), atol=.002, rtol=.01)
    # Nonzero descriptor length with all-zero locations is graph padding.
    loc.zero_(); graph.replay()
    assert not captured.any()
    graph.reset()
    result['graph_dynamic_and_padding'] = True
    for b in (1, 2, 4, 8, 16):
        q = torch.randn(b, 24, 128, device='cuda').bfloat16()[:, :8]
        for length in (1024, 2048, 4096, 8192, 16384, 32768):
            ptr, loc = locations([length] * b, pool.hp_global_offset)
            ms = bench(lambda: attn.forward(q, 0, ptr, loc, rope, 128**-.5))
            result['timings'].append({'batch': b, 'length': length, 'splits': attn.splits, 'threads': args.threads, 'dynamic_splits': not args.fixed_splits, 'graph_ms': ms})
    result['passed'] = True
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
