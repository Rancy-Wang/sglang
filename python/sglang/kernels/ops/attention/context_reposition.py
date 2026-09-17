# MIT License
#
# Copyright (c) 2026 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import triton
import triton.language as tl


@triton.jit
def _reposition_layers_kernel(
    k_layout,
    k_data_ptrs,
    v_data_ptrs,
    source_slots,
    destination_slots,
    position_pairs,
    cos_sin_cache,
    k_stride_page,
    k_stride_slot,
    k_stride_head,
    v_stride_page,
    v_stride_slot,
    v_stride_head,
    position_stride_token,
    rope_stride_position,
    PAGE_SIZE: tl.constexpr,
    NEOX_STYLE: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    # Production KV buffers can exceed 2**31 elements even when every individual
    # stride fits in int32. Promote the grid coordinates before multiplying by
    # those strides so high-layer addresses cannot wrap around.
    token = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)

    source = tl.load(source_slots + token).to(tl.int64)
    destination = tl.load(destination_slots + token).to(tl.int64)
    position_row = position_pairs + token * position_stride_token
    old_position = tl.load(position_row).to(tl.int64)
    new_position = tl.load(position_row + 1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim

    k_buffer = tl.load(k_data_ptrs + layer).to(
        tl.pointer_type(k_layout.dtype.element_ty)
    )
    v_buffer = tl.load(v_data_ptrs + layer).to(
        tl.pointer_type(k_layout.dtype.element_ty)
    )
    source_k = (
        k_buffer
        + (source // PAGE_SIZE) * k_stride_page
        + (source % PAGE_SIZE) * k_stride_slot
        + head * k_stride_head
    )
    destination_k = (
        k_buffer
        + (destination // PAGE_SIZE) * k_stride_page
        + (destination % PAGE_SIZE) * k_stride_slot
        + head * k_stride_head
    )
    first_offsets = offsets if NEOX_STYLE else 2 * offsets
    second_offsets = half_dim + offsets if NEOX_STYLE else 2 * offsets + 1
    first = tl.load(source_k + first_offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(source_k + second_offsets, mask=mask, other=0.0).to(tl.float32)

    old_rope = cos_sin_cache + old_position * rope_stride_position
    new_rope = cos_sin_cache + new_position * rope_stride_position
    old_cos = tl.load(old_rope + offsets, mask=mask, other=0.0).to(tl.float32)
    old_sin = tl.load(old_rope + half_dim + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    new_cos = tl.load(new_rope + offsets, mask=mask, other=0.0).to(tl.float32)
    new_sin = tl.load(new_rope + half_dim + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    scale_squared = old_cos * old_cos + old_sin * old_sin
    delta_cos = (new_cos * old_cos + new_sin * old_sin) / scale_squared
    delta_sin = (new_sin * old_cos - new_cos * old_sin) / scale_squared
    # A different final Radix branch can need its own page at the same position.
    # In that case copy the source bits instead of round-tripping through RoPE.
    same_position = old_position == new_position
    rotated_first = first * delta_cos - second * delta_sin
    rotated_second = second * delta_cos + first * delta_sin
    tl.store(
        destination_k + first_offsets,
        tl.where(same_position, first, rotated_first),
        mask=mask,
    )
    tl.store(
        destination_k + second_offsets,
        tl.where(same_position, second, rotated_second),
        mask=mask,
    )

    source_v = (
        v_buffer
        + (source // PAGE_SIZE) * v_stride_page
        + (source % PAGE_SIZE) * v_stride_slot
        + head * v_stride_head
    )
    destination_v = (
        v_buffer
        + (destination // PAGE_SIZE) * v_stride_page
        + (destination % PAGE_SIZE) * v_stride_slot
        + head * v_stride_head
    )
    value_first = tl.load(source_v + offsets, mask=mask, other=0.0)
    value_second = tl.load(source_v + half_dim + offsets, mask=mask, other=0.0)
    tl.store(destination_v + offsets, value_first, mask=mask)
    tl.store(destination_v + half_dim + offsets, value_second, mask=mask)


def reposition_kv_layers(
    k_data_ptrs: torch.Tensor,
    v_data_ptrs: torch.Tensor,
    k_layout: torch.Tensor,
    v_layout: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    position_pairs: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox_style: bool = True,
) -> None:
    """Rotate K and copy V across native per-layer pools in one GPU launch.

    Pointer tables are constructed once by the native KV pool. Each table's
    layers must have the exemplar's dtype and strides. Supports native NHD
    [slot, head, dim] and HND [page, head, slot-in-page, dim]. SWA and full pools
    invoke this separately with their own physical slot mappings.

    The caller owns fresh, distinct destination slots disjoint from all source
    slots, validates CPU positions/slots before H2D, and holds the source KV and
    metadata until the stream finishes. This wrapper does no GPU-to-CPU reads
    and no synchronization. It does not allocate or release any KV page.
    """
    tensors = (
        k_data_ptrs,
        v_data_ptrs,
        k_layout,
        v_layout,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
    )
    if not k_layout.is_cuda or any(t.device != k_layout.device for t in tensors):
        raise ValueError("Reposition tensors must be on one CUDA device")
    if k_layout.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Context Reposition requires unquantized floating-point KV")
    if (
        k_layout.ndim not in (3, 4)
        or v_layout.shape != k_layout.shape
        or v_layout.dtype != k_layout.dtype
    ):
        raise ValueError("Reposition requires matching native NHD or HND K/V layouts")
    if k_layout.stride(-1) != 1 or v_layout.stride(-1) != 1:
        raise ValueError("Reposition requires contiguous head dimensions")
    if (
        k_data_ptrs.ndim != 1
        or v_data_ptrs.shape != k_data_ptrs.shape
        or k_data_ptrs.dtype not in (torch.uint64, torch.int64)
        or v_data_ptrs.dtype not in (torch.uint64, torch.int64)
        or not k_data_ptrs.is_contiguous()
        or not v_data_ptrs.is_contiguous()
    ):
        raise ValueError("Reposition requires contiguous 64-bit layer pointer tables")
    count = len(source_slots)
    if (
        source_slots.ndim != 1
        or destination_slots.shape != source_slots.shape
        or source_slots.dtype != torch.int32
        or destination_slots.dtype != torch.int32
        or not source_slots.is_contiguous()
        or not destination_slots.is_contiguous()
        or position_pairs.shape != (count, 2)
        or position_pairs.dtype != torch.int32
        or position_pairs.stride(-1) != 1
    ):
        raise ValueError("Reposition requires int32 slot vectors and [N, 2] positions")
    head_dim = k_layout.shape[-1]
    if (
        head_dim % 2
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != head_dim
        or cos_sin_cache.stride(-1) != 1
        or not cos_sin_cache.is_floating_point()
    ):
        raise ValueError("Reposition requires a full-head native cos/sin RoPE cache")
    if count == 0 or len(k_data_ptrs) == 0:
        return
    if k_layout.ndim == 3:
        page_size = 1
        k_page, k_slot, k_head = k_layout.stride(0), 0, k_layout.stride(1)
        v_page, v_slot, v_head = v_layout.stride(0), 0, v_layout.stride(1)
    else:
        page_size = k_layout.shape[2]
        k_page, k_slot, k_head = (
            k_layout.stride(0),
            k_layout.stride(2),
            k_layout.stride(1),
        )
        v_page, v_slot, v_head = (
            v_layout.stride(0),
            v_layout.stride(2),
            v_layout.stride(1),
        )
    _reposition_layers_kernel[(count, len(k_data_ptrs), k_layout.shape[1])](
        k_layout,
        k_data_ptrs,
        v_data_ptrs,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
        k_page,
        k_slot,
        k_head,
        v_page,
        v_slot,
        v_head,
        position_pairs.stride(0),
        cos_sin_cache.stride(0),
        PAGE_SIZE=page_size,
        NEOX_STYLE=is_neox_style,
        head_dim=head_dim,
        half_dim=head_dim // 2,
        BLOCK_HALF=triton.next_power_of_2(head_dim // 2),
        num_warps=4,
    )
