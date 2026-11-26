# SPDX-License-Identifier: Apache-2.0
"""Group32 INT4 residual sidecars; centers are decoded in BF16.

Run after joint index assignment and before publishing the destination page.
Each program owns one token/head/channel-group/loop, so its scale reduction
does not depend on other programs. No float history is materialized.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _encode_residual(X0, X1, X2, X3, IDX, CB, OUT, SCALE, DST, SRC, VALID,
                     N: tl.constexpr, H: tl.constexpr, LB: tl.constexpr,
                     BASE_OFFSET: tl.constexpr, QSIZE: tl.constexpr,
                     S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                     S3: tl.constexpr, LSTRIDE: tl.constexpr,
                     FROM_HP: tl.constexpr, PAGED: tl.constexpr):
    row, hg, ls = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    base, step = (ls // 4).to(tl.int64), ls % 4
    head, group = hg // 4, hg % 4
    slot = step * LB + base + BASE_OFFSET
    dst = tl.load(DST + row).to(tl.int64)
    ok = (dst >= 0) & (dst < QSIZE)
    src = row
    if FROM_HP:
        src = tl.load(SRC + row)
        ok = ok & (src >= 0) & tl.load(VALID + row).to(tl.int1)
    c = group * 32 + tl.arange(0, 32)
    offset = base * LSTRIDE + head * 128 + c
    if step == 0:
        x = tl.load(X0 + src * S0 + offset, ok, 0).to(tl.float32)
    elif step == 1:
        x = tl.load(X1 + src * S1 + offset, ok, 0).to(tl.float32)
    elif step == 2:
        x = tl.load(X2 + src * S2 + offset, ok, 0).to(tl.float32)
    else:
        x = tl.load(X3 + src * S3 + offset, ok, 0).to(tl.float32)
    if PAGED == 1:
        pg = (((dst // 8) * H + head) * 4 + group) * 8 + dst % 8
        io = base * QSIZE * H * 32 + pg * 8 + (c // 4) % 8
    elif PAGED == 2:
        page_row = ((dst // 8 * H + head) * 8 + dst % 8)
        io = base * QSIZE * H * 32 + page_row * 32 + c // 4
    else:
        io = ((base * QSIZE + dst) * H + head) * 32 + c // 4
    code = tl.load(IDX + io, ok, 0).to(tl.int32)
    center = tl.load(CB + (((slot * H + head) * 32 + c // 4) * 256 + code) * 4 + c % 4,
                     ok, 0).to(tl.bfloat16).to(tl.float32)
    error = x - center
    scale = (tl.max(tl.abs(error), 0) / 7.0).to(tl.float16)
    scale = tl.maximum(scale.to(tl.float32), 0.00006103515625).to(tl.float16)
    # Match training's finite-scale rejection; this executes only when encoding
    # a page, without a host synchronization or extra decode-time kernel.
    tl.device_assert((~ok) | (tl.sum((tl.abs(error) < float('inf')).to(tl.int32), 0) == 32),
                     "VQ+ residual input must be finite")
    tl.device_assert((~ok) | (scale.to(tl.float32) < float('inf')),
                     "VQ+ residual scale exceeds FP16")
    z = tl.minimum(7., tl.maximum(-7., libdevice.rint(tl.div_rn(error, scale.to(tl.float32))))).to(tl.int32)
    pairs = tl.reshape(z & 15, (16, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
    if PAGED == 1:
        ro = slot * QSIZE * H * 64 + pg * 16 + tl.arange(0, 16)
        so = slot * QSIZE * H * 4 + pg
    elif PAGED == 2:
        ro = slot * QSIZE * H * 64 + page_row * 64 + group * 16 + tl.arange(0, 16)
        so = slot * QSIZE * H * 4 + page_row * 4 + group
    else:
        ro = ((slot * QSIZE + dst) * H + head) * 64 + group * 16 + tl.arange(0, 16)
        so = ((slot * QSIZE + dst) * H + head) * 4 + group
    tl.store(OUT + ro, packed, ok)
    tl.store(SCALE + so, scale, ok)


def encode_residual_steps(steps, indices, table, packed, scales, loc, base, *, paged=False):
    """Four live prefill tensors, full slot-decode table, shared index row."""
    n, h, d = steps[0].shape
    if d != 128 or len(steps) != 4 or any(x.shape != steps[0].shape for x in steps):
        raise ValueError("Residual encoding requires four [tokens,heads,128] tensors")
    if paged and packed.shape[1] % 8:
        raise ValueError("Page residual encoding requires page8 capacity")
    steps = [x.contiguous() if x.stride(-1) != 1 or x.stride(-2) != 128 else x for x in steps]
    if n:
        _encode_residual[(n, h * 4, 4)](
            *steps, indices, table, packed, scales, loc, loc, loc,
            n, h, table.shape[0] // 4, base, packed.shape[1],
            *(x.stride(0) for x in steps), 0, False, paged,
            num_warps=1, enable_fp_fusion=False, debug=True,
        )


def encode_residual_from_hp(hp, indices, table, packed, scales, src, dst, valid, *, paged=False):
    lb = hp.shape[0] // 4
    if paged and packed.shape[1] % 8:
        raise ValueError("Page residual encoding requires page8 capacity")
    if src.numel():
        _encode_residual[(src.numel(), hp.shape[2] * 4, lb * 4)](
            hp[0], hp[lb], hp[2 * lb], hp[3 * lb], indices, table,
            packed, scales, dst, src, valid,
            src.numel(), hp.shape[2], lb, 0, packed.shape[1],
            hp.stride(1), hp.stride(1), hp.stride(1), hp.stride(1), hp.stride(0), True, paged,
            num_warps=1, enable_fp_fusion=False, debug=True,
        )
