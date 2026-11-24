# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Adapted from mscheong01/LoopVQ d5d58c83fe5bc9a4ce470ae96c69b55f085e2977.
"""Identity HP pre-RoPE stage 1 from upstream decode_attention.py.

Retains official arithmetic: FP32 RoPE, FP16 QK MMA, online softmax with
unnormalized probabilities converted to V dtype. This is not HF BF16 math.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _official_hp_stage1(Q, K_Buffer, V_Buffer, CosSin, Full_Seq_Len, sm_scale, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh, stride_cs_pos, stride_mid_ob, stride_mid_oh, stride_mid_os, kv_group_num: tl.constexpr, q_head_num: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, MIN_BLOCK_KV: tl.constexpr, L: tl.constexpr, LV: tl.constexpr, PREFIX_TOKENS: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < LV
    offs_half = tl.arange(0, BLOCK_D // 2)
    mask_half = offs_half < L // 2
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    pos_offset = tl.load(Full_Seq_Len + cur_batch) - cur_batch_seq_len
    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    if split_kv_end > split_kv_start:
        offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_half[None, :]
        offs_q2 = offs_q1 + L // 2
        q1 = tl.load(Q + offs_q1, mask=mask_h[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
        q2 = tl.load(Q + offs_q2, mask=mask_h[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(kv_indices + cur_batch_kv_start_idx + offs_n, mask=mask_n, other=0).to(tl.int64)
            k_base = kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh
            abs_pos = tl.where(offs_n < PREFIX_TOKENS, offs_n, pos_offset + offs_n)
            k1 = tl.load(K_Buffer + k_base + offs_half[None, :], mask=mask_n[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
            k2 = tl.load(K_Buffer + k_base + (L // 2 + offs_half)[None, :], mask=mask_n[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
            cs_row = tl.load(CosSin + abs_pos[:, None] * stride_cs_pos + offs_half[None, :], mask=mask_n[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
            sn_row = tl.load(CosSin + abs_pos[:, None] * stride_cs_pos + (L // 2 + offs_half)[None, :], mask=mask_n[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
            o1 = k1 * cs_row - k2 * sn_row
            o2 = k2 * cs_row + k1 * sn_row
            qk = tl.dot(q1.to(tl.float16), tl.trans(o1).to(tl.float16))
            qk += tl.dot(q2.to(tl.float16), tl.trans(o2).to(tl.float16))
            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float('-inf'))
            v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :], mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        offs_mid_o = cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv[None, :]
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os) // LV
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def official_hp_attention(q, hp_k, hp_v, cos_sin_cache, full_seq_len,
                          att_out, att_lse, kv_indptr, kv_indices, num_splits,
                          *, max_splits=12, sm_scale, prefix_tokens):
    """Fill official HP split scratch. Indices are local HP arena coordinates.

    HP rows must retain sequence order: protected prefix, then contiguous recent
    tail. Full lengths include the current decode token. Empty splits keep the
    caller's -inf LSE; scratch output values for them are never consumed.
    """
    b = q.shape[0]
    if q.ndim != 3 or q.shape[1] not in (8, 16) or q.shape[2] != 128 or q.dtype != torch.bfloat16:
        raise ValueError("Official HP attention requires BF16 [B,local_heads,128]")
    if hp_k.shape != hp_v.shape or hp_k.shape[1:] != q.shape[1:]:
        raise ValueError("Official HP K/V geometry mismatch")
    if hp_k.dtype != torch.bfloat16 or hp_v.dtype != torch.bfloat16:
        raise ValueError("Official HP K/V must be BF16")
    if max_splits != 12 or prefix_tokens < 0:
        raise ValueError("Official HP profile requires 12 split slots and a nonnegative prefix")
    if att_out.shape != (b, q.shape[1], max_splits, 128) or att_lse.shape != (b, q.shape[1], max_splits):
        raise ValueError("Official HP scratch geometry mismatch")
    if att_out.dtype != torch.float32 or att_lse.dtype != torch.float32:
        raise ValueError("Official HP scratch must be FP32")
    if tuple(s // 128 for s in att_out.stride()[:3]) != att_lse.stride():
        raise ValueError("HP output/LSE scratch strides must describe the same splits")
    if (cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 128
            or cos_sin_cache.dtype != torch.float32):
        raise ValueError("Official HP requires an FP32 NeoX RoPE table")
    if kv_indptr.shape != (b + 1,) or num_splits.shape != (b,) or full_seq_len.shape != (b,):
        raise ValueError("Official HP request metadata geometry mismatch")
    if kv_indptr.dtype != torch.int32 or num_splits.dtype != torch.int32:
        raise ValueError("Official HP pointers/splits must be int32")
    operands = (hp_k, hp_v, cos_sin_cache, full_seq_len, att_out, att_lse,
                kv_indptr, kv_indices, num_splits)
    if any(t.device != q.device for t in operands):
        raise ValueError("Official HP operands must share a device")
    if any(t.stride(-1) != 1 for t in (q, hp_k, hp_v, cos_sin_cache, full_seq_len,
                                     att_out, att_lse, kv_indptr, kv_indices, num_splits)):
        raise ValueError("Official HP requires contiguous innermost dimensions")
    _official_hp_stage1[(b, q.shape[1], max_splits)](
        q, hp_k, hp_v, cos_sin_cache, full_seq_len, sm_scale,
        kv_indptr, kv_indices, att_out, att_lse, num_splits,
        q.stride(0), q.stride(1), hp_k.stride(0), hp_k.stride(1),
        hp_v.stride(0), hp_v.stride(1), cos_sin_cache.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=1, q_head_num=q.shape[1], BLOCK_D=128, BLOCK_DV=128,
        BLOCK_N=32, BLOCK_H=4, MIN_BLOCK_KV=32, L=128, LV=128,
        PREFIX_TOKENS=int(prefix_tokens), num_warps=4, num_stages=1,
    )
