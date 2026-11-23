# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Mixed BF16 / LoopVQ decode with the target Ouro BF16 arithmetic profile.

Index layout follows LoopVQ d5d58c83. Arithmetic follows this repository's
ouro_attention: QK, scaled scores, and globally normalized probabilities round
to BF16. This two-pass kernel is deliberately an auditable first implementation,
not a claim of speed parity with the upstream CUDA kernels. It reads codebook
entries in-place and never expands the full historical K/V cache.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _load_rows(IDX, CB, HP, slots, dims, head, valid,
               H: tl.constexpr, D: tl.constexpr, OFFSET: tl.constexpr):
    quant = slots < OFFSET
    code = tl.load(IDX + (slots[None, :] * H + head) * (D // 4)
                   + dims[:, None] // 4, valid[None, :] & quant[None, :], 0).to(tl.int32)
    value = tl.load(CB + ((head * (D // 4) + dims[:, None] // 4) * 256 + code) * 4
                    + dims[:, None] % 4, valid[None, :] & quant[None, :], 0)
    full = tl.load(HP + ((slots[None, :] - OFFSET) * H + head) * D + dims[:, None],
                   valid[None, :] & ~quant[None, :], 0)
    return tl.where(quant[None, :], value.to(HP.dtype.element_ty), full)


@triton.jit
def _decode(Q, KI, VI, KC, VC, HK, HV, PTR, LOC, ROPE, O,
            QS: tl.constexpr, QH: tl.constexpr, OS: tl.constexpr, OH: tl.constexpr,
            H: tl.constexpr, D: tl.constexpr, OFFSET: tl.constexpr,
            SCALE: tl.constexpr, N: tl.constexpr = 32):
    request, head = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(PTR + request), tl.load(PTR + request + 1)
    length = end - begin
    rows = tl.arange(0, 16)
    dims = tl.arange(0, D)
    query = tl.load(Q + request * QS + head * QH + dims[None, :]
                    + tl.zeros((16, 1), tl.int32), rows[:, None] == 0, 0)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    result = tl.zeros((16, D), tl.float32)
    partner = (dims + D // 2) % D
    sign = tl.where(dims < D // 2, -1.0, 1.0)
    for stage in tl.static_range(2):
        for start in range(0, length, N):
            positions = start + tl.arange(0, N)
            valid = positions < length
            slots = tl.load(LOC + begin + positions, valid, 0)
            raw = _load_rows(KI, KC, HK, slots, dims, head, valid, H, D, OFFSET)
            paired = _load_rows(KI, KC, HK, slots, partner, head, valid, H, D, OFFSET)
            cos = tl.load(ROPE + positions[None, :] * D + (dims[:, None] % (D // 2)),
                          valid[None, :], 0).to(query.dtype)
            sin = tl.load(ROPE + positions[None, :] * D + D // 2 + (dims[:, None] % (D // 2)),
                          valid[None, :], 0).to(query.dtype)
            # Match OuroRotaryEmbedding.forward_native's two rounded products.
            a = (raw.to(tl.float32) * cos.to(tl.float32)).to(query.dtype)
            b = (paired.to(tl.float32) * sign[:, None] * sin.to(tl.float32)).to(query.dtype)
            keys = (a.to(tl.float32) + b.to(tl.float32)).to(query.dtype)
            scores = tl.dot(query, keys).to(query.dtype).to(tl.float32)
            scores = (scores * SCALE).to(query.dtype).to(tl.float32)
            scores = tl.where(valid[None, :], scores, -float("inf"))
            if stage == 0:
                next_max = tl.maximum(maximum, tl.max(scores, 1))
                denominator = denominator * libdevice.exp(maximum - next_max) + tl.sum(
                    libdevice.exp(scores - next_max[:, None]), 1)
                maximum = next_max
            else:
                p = tl.div_rn(libdevice.exp(scores - maximum[:, None]), denominator[:, None]).to(query.dtype)
                values = _load_rows(VI, VC, HV, slots, dims, head, valid, H, D, OFFSET)
                result += tl.dot(p, tl.trans(values).to(query.dtype))
    tl.store(O + request * OS + head * OH + dims[None, :]
             + tl.zeros((16, 1), tl.int32), result.to(O.dtype.element_ty), rows[:, None] == 0)


def loopvq_decode_attention(q, pool, layer_id, kv_indptr, kv_indices,
                            cos_sin_cache, sm_scale, out=None, num_splits=16):
    """Read complete position-ordered mixed slot lists, one decode token/request.

    num_splits is accepted for the backend contract; this arithmetic-preserving
    first kernel has a global normalization and does not partition the sequence.
    """
    if q.ndim != 3 or q.shape[1:] != (pool.head_num, pool.head_dim):
        raise ValueError("LoopVQ decode requires [batch, MHA heads, head_dim]")
    if q.dtype != torch.bfloat16 or pool.head_dim != 128:
        raise ValueError("LoopVQ decode currently requires BF16 Ouro D128")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != pool.head_dim:
        raise ValueError("LoopVQ requires the NeoX [position, cos|sin] RoPE table")
    if out is None:
        out = torch.empty_like(q)
    out3 = out.view_as(q)
    _decode[(q.shape[0], pool.head_num)](
        q, pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id),
        pool.k_book.cb16[layer_id], pool.v_book.cb16[layer_id],
        pool.get_hp_key_buffer(layer_id), pool.get_hp_value_buffer(layer_id),
        kv_indptr, kv_indices, cos_sin_cache, out3,
        q.stride(0), q.stride(1), out3.stride(0), out3.stride(1),
        pool.head_num, pool.head_dim, pool.hp_global_offset, float(sm_scale),
        enable_fp_fusion=False, num_warps=4,
    )
    return out
