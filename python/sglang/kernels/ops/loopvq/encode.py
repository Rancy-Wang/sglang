# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Fused G16/c4 nearest-centroid assignment and BF16 HP writes.

The channel-major / step-minor layout follows LoopVQ d5d58c83.
No full dequantized historical cache is materialized.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _write_hp(K, V, HK, HV, LOC, n: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              KS: tl.constexpr, VS: tl.constexpr, OFFSET: tl.constexpr,
              HP_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(LOC + row) - OFFSET
    ok = (slot >= 0) & (slot < HP_SIZE) & (col < H * D)
    k = tl.load(K + row * KS + col, ok, 0)
    v = tl.load(V + row * VS + col, ok, 0)
    tl.store(HK + slot * H * D + col, k, ok)
    tl.store(HV + slot * H * D + col, v, ok)


def write_hp(k, v, hk, hv, loc, hp_offset):
    if not loc.numel():
        return
    if k.stride(-1) != 1 or k.stride(-2) != k.shape[-1]:
        k = k.contiguous()
    if v.stride(-1) != 1 or v.stride(-2) != v.shape[-1]:
        v = v.contiguous()
    _write_hp[(loc.numel(), triton.cdiv(k.shape[1] * k.shape[2], 256))](
        k, v, hk, hv, loc, loc.numel(), k.shape[1], k.shape[2],
        k.stride(0), v.stride(0), hp_offset, hk.shape[0], 256,
    )


@triton.jit
def _encode(K0, K1, K2, K3, CB, OUT, LOC, SRC, VALID,
            N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
            LSTRIDE: tl.constexpr, OUTSIZE: tl.constexpr,
            FROM_HP: tl.constexpr, HAS_VALID: tl.constexpr, PAGED: tl.constexpr,
            M: tl.constexpr = 16):
    rows = tl.program_id(0) * M + tl.arange(0, M)
    hg = tl.program_id(1)
    layer = tl.program_id(2).to(tl.int64)
    ng = D // 4
    h, group = hg // ng, hg % ng
    cols = tl.arange(0, 16)
    dest = tl.load(LOC + rows, rows < N, -1).to(tl.int64)
    valid = (rows < N) & (dest >= 0) & (dest < OUTSIZE)
    if HAS_VALID:
        valid = valid & (tl.load(VALID + rows, rows < N, 0) != 0)
    if FROM_HP:
        src = tl.load(SRC + rows, rows < N, 0)
        valid = valid & (src >= 0)
        src = tl.maximum(src, 0)
    else:
        src = rows
    channel = h * D + group * 4 + cols // 4
    common = layer * LSTRIDE + channel[None, :]
    mask = valid[:, None]
    x0 = tl.load(K0 + src[:, None] * S0 + common, mask & (cols[None, :] % 4 == 0), 0)
    x1 = tl.load(K1 + src[:, None] * S1 + common, mask & (cols[None, :] % 4 == 1), 0)
    x2 = tl.load(K2 + src[:, None] * S2 + common, mask & (cols[None, :] % 4 == 2), 0)
    x3 = tl.load(K3 + src[:, None] * S3 + common, mask & (cols[None, :] % 4 == 3), 0)
    x = (x0.to(tl.float32) + x1.to(tl.float32) + x2.to(tl.float32) + x3.to(tl.float32)).to(tl.float16)
    ids = tl.arange(0, 256)
    table = tl.load(CB + ((layer * H + h) * ng + group) * 256 * 16
                    + ids[None, :] * 16 + cols[:, None])
    norm = tl.sum(table.to(tl.float32) * table.to(tl.float32), 0) * 0.5
    score = tl.dot(x, table) - norm[None, :]
    code = tl.argmax(score, 1, tie_break_left=True)
    if PAGED:
        where = layer * OUTSIZE * H * ng + (((dest // 8 * H + h) * 4 + group // 8) * 8 + dest % 8) * 8 + group % 8
    else:
        where = ((layer * OUTSIZE + dest) * H + h) * ng + group
    tl.store(OUT + where, code, valid)


def encode_steps(steps, table, output, loc, *, paged=False):
    """Four [n,H,D] BF16 inputs -> scatter uint8 indices [slots,H,D/4]."""
    n, h, d = steps[0].shape
    if len(steps) != 4 or any(x.shape != steps[0].shape for x in steps):
        raise ValueError("LoopVQ encode needs four identically shaped steps")
    if paged and (d != 128 or output.shape[0] % 8):
        raise ValueError("Page encoding requires D128 and page8 capacity")
    steps = [x if x.stride(-2) == d and x.stride(-1) == 1 else x.contiguous()
             for x in steps]
    if n:
        _encode[(triton.cdiv(n, 16), h * (d // 4), 1)](
            *steps, table, output, loc, loc, loc, n, h, d,
            *(x.stride(0) for x in steps), 0, output.shape[0], False, False, paged,
        )


def encode_from_hp(hp, table, output, src, dst, valid, *, paged=False):
    """Encode all base layers together when demoting one or more complete pages."""
    lb = table.shape[0]
    _, _, h, d = hp.shape
    n = src.numel()
    if hp.shape[0] != lb * 4:
        raise ValueError("LoopVQ flush requires every R4 slot")
    if paged and (d != 128 or output.shape[1] % 8):
        raise ValueError("Page encoding requires D128 and page8 capacity")
    if n:
        _encode[(triton.cdiv(n, 16), h * (d // 4), lb)](
            hp[0], hp[lb], hp[2 * lb], hp[3 * lb], table, output,
            dst, src, valid, n, h, d,
            hp.stride(1), hp.stride(1), hp.stride(1), hp.stride(1),
            hp.stride(0), output.shape[1], True, True, paged,
        )
