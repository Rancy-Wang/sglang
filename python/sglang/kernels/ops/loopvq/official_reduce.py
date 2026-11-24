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
"""Official FP32 LSE merge of HP and quantized split outputs (no sinks)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _official_reduce_kernel(Mid_O, Mid_O_1, O, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_obs, stride_oh, TOTAL_SPLITS: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_DV: tl.constexpr, Lv: tl.constexpr):
    """Stage 2 of the unified decode: merges the bf16-tier and quant-tier splits by LSE.

    All LSEs are one vector load, the softmax weights come from one max / exp / sum, and the
    valid rows are one masked [BLOCK_S, BLOCK_DV] load reduced over splits.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < TOTAL_SPLITS
    base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    lse = tl.load(Mid_O_1 + base // Lv + offs_s * (stride_mid_os // Lv), mask=mask_s, other=-float('inf'))
    valid = lse > -float('inf')
    e_max = tl.max(lse, axis=0)
    e_max_safe = tl.where(e_max > -float('inf'), e_max, 0.0)
    w = tl.where(valid, tl.exp(lse - e_max_safe), 0.0)
    e_sum = tl.sum(w, axis=0)
    tv = tl.load(Mid_O + base + offs_s[:, None] * stride_mid_os + offs_d[None, :], mask=valid[:, None] & mask_d[None, :], other=0.0)
    acc = tl.sum(tv * w[:, None], axis=0)
    safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out = tl.where(e_sum > 0.0, acc / safe_e_sum * 1.0, 0.0)
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, out, mask=mask_d)


def official_reduce(att_out, att_lse, out):
    """Merge normalized split outputs, ignoring every -inf-LSE split."""
    if out.ndim != 3 or out.shape[1] not in (8, 16) or out.shape[2] != 128 or out.dtype != torch.bfloat16:
        raise ValueError("Official LoopVQ reduction requires BF16 [B,local_heads,128] output")
    b, h, d = out.shape
    if att_out.ndim != 4 or att_out.shape[:2] != (b, h) or att_out.shape[-1] != d:
        raise ValueError("Official reduction scratch geometry mismatch")
    if att_lse.shape != att_out.shape[:3]:
        raise ValueError("Official reduction LSE geometry mismatch")
    if att_out.dtype != torch.float32 or att_lse.dtype != torch.float32:
        raise ValueError("Official reduction scratch must be FP32")
    if tuple(s // d for s in att_out.stride()[:3]) != att_lse.stride():
        raise ValueError("Output/LSE scratch strides must describe the same splits")
    if att_out.device != out.device or att_lse.device != out.device:
        raise ValueError("Official reduction operands must share a device")
    if out.stride(-1) != 1 or att_out.stride(-1) != 1 or att_lse.stride(-1) != 1:
        raise ValueError("Official reduction requires contiguous innermost dimensions")
    total = att_out.shape[2]
    _official_reduce_kernel[(b, h)](
        att_out, att_lse, out, att_out.stride(0), att_out.stride(1),
        att_out.stride(2), out.stride(0), out.stride(1),
        TOTAL_SPLITS=total, BLOCK_S=triton.next_power_of_2(total),
        BLOCK_DV=128, Lv=128, num_warps=4, num_stages=2,
    )
    return out
