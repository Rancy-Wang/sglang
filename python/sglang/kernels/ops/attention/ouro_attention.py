# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Ouro eager-checkpoint attention arithmetic over native paged KV.

QK, scaled scores, and *normalized* probabilities round to input dtype.
Two passes avoid a quadratic score buffer while preserving these boundaries.
Only the arithmetic differs from generic Triton attention; KV ownership and
request/depth indexing remain with the native backend.
"""

import triton
import triton.language as tl


@triton.jit
def _ouro_attention(
    Q,
    K,
    V,
    KC,
    VC,
    O,
    KP,
    KI,
    QP,
    qs: tl.constexpr,
    qh: tl.constexpr,
    ks: tl.constexpr,
    kh: tl.constexpr,
    vs: tl.constexpr,
    vh: tl.constexpr,
    kcs: tl.constexpr,
    kch: tl.constexpr,
    vcs: tl.constexpr,
    vch: tl.constexpr,
    os: tl.constexpr,
    oh: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    DECODE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    request, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head // GROUP
    begin = tl.load(KP + request)
    prefix = tl.load(KP + request + 1) - begin
    if DECODE:
        qbegin, qlen, total = request, 1, prefix
    else:
        qbegin = tl.load(QP + request)
        qlen = tl.load(QP + request + 1) - qbegin
        total = prefix + qlen
    rows = tile * M + tl.arange(0, M)
    dims = tl.arange(0, D)
    query = tl.load(
        Q + (qbegin + rows[:, None]) * qs + head * qh + dims[None, :],
        rows[:, None] < qlen,
        other=0,
    )
    if DECODE:
        visible = tl.full((M,), prefix, tl.int32)
    else:
        visible = prefix + rows + 1
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    result = tl.zeros((M, D), tl.float32)
    # Recompute scores after knowing the global normalizer. Rounding online
    # unnormalized probabilities before PV does not match the checkpoint.
    for stage in tl.static_range(2):
        for start in range(0, total, N):
            cols = start + tl.arange(0, N)
            slots = tl.load(KI + begin + cols, cols < prefix, other=0)
            keys = tl.load(
                KC + slots[None, :] * kcs + kv_head * kch + dims[:, None],
                cols[None, :] < prefix,
                other=0,
            )
            if not DECODE:
                current = qbegin + cols - prefix
                new_keys = tl.load(
                    K + current[None, :] * ks + kv_head * kh + dims[:, None],
                    (cols[None, :] >= prefix) & (cols[None, :] < total),
                    other=0,
                )
                keys = tl.where(cols[None, :] < prefix, keys, new_keys)
            scores = tl.dot(query, keys).to(query.dtype).to(tl.float32)
            scores = (scores * SCALE).to(query.dtype).to(tl.float32)
            scores = tl.where(
                (cols[None, :] < visible[:, None]) & (cols[None, :] < total),
                scores,
                -float("inf"),
            )
            if stage == 0:
                next_max = tl.maximum(maximum, tl.max(scores, 1))
                denominator = denominator * tl.exp(maximum - next_max) + tl.sum(
                    tl.exp(scores - next_max[:, None]), 1
                )
                maximum = next_max
            else:
                probability = (
                    tl.exp(scores - maximum[:, None]) / denominator[:, None]
                ).to(query.dtype)
                values = tl.load(
                    VC + slots[:, None] * vcs + kv_head * vch + dims[None, :],
                    cols[:, None] < prefix,
                    other=0,
                )
                if not DECODE:
                    new_values = tl.load(
                        V + current[:, None] * vs + kv_head * vh + dims[None, :],
                        (cols[:, None] >= prefix) & (cols[:, None] < total),
                        other=0,
                    )
                    values = tl.where(cols[:, None] < prefix, values, new_values)
                result += tl.dot(probability, values)
    tl.store(
        O + (qbegin + rows[:, None]) * os + head * oh + dims[None, :],
        result.to(O.dtype.element_ty),
        rows[:, None] < qlen,
    )


def ouro_attention(
    q,
    k,
    v,
    kc,
    vc,
    out,
    kv_indptr,
    kv_indices,
    qo_indptr,
    batch_size,
    max_extend_len,
    scale,
    *,
    decode=False,
):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert triton.next_power_of_2(q.shape[-1]) == q.shape[-1]
    _ouro_attention[(batch_size, q.shape[1], triton.cdiv(max_extend_len, 16))](
        q,
        k,
        v,
        kc,
        vc,
        out,
        kv_indptr,
        kv_indices,
        qo_indptr,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kc.stride(0),
        kc.stride(1),
        vc.stride(0),
        vc.stride(1),
        out.stride(0),
        out.stride(1),
        q.shape[1] // k.shape[1],
        q.shape[-1],
        scale,
        decode,
        16,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
