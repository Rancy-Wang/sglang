# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Official-style CUDA quant + Triton HP split decode, independent of LoopSpec.

Only attention arithmetic changes under the optimized profile. The model's
native query RoPE, normalization, MLP and full-prefill behavior remain intact.
"""
import torch

from .official_hp import official_hp_attention
from .official_metadata import OfficialMetadata
from .official_quant_cuda import official_quant_attention, prepare_official_quant
from .official_reduce import official_reduce


class OfficialAttention:
    def __init__(self, pool, *, max_batch, max_context_len):
        if pool.head_num not in (8, 16) or (pool.head_dim, pool.layer_num) != (128, 192):
            raise ValueError("Official LoopVQ decode requires the full Ouro-2.6B R4 geometry")
        device = pool.hp_k.device
        self.pool = pool
        self.metadata = OfficialMetadata(
            max_batch=max_batch, max_context_len=max_context_len,
            prefix_tokens=pool.hp_prefix_tokens, recent_tokens=pool.hp_recent_tokens,
            hp_offset=pool.hp_global_offset, device=device,
            head_num=pool.head_num, sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        )
        # Compile and load both R1/R2 modules before any CUDA graph capture.
        prepare_official_quant(device=device, max_context_len=max_context_len, max_splits=48)
        self.partial = torch.empty((max_batch, pool.head_num, 60, 128), dtype=torch.float32, device=device)
        self.lse = torch.empty((max_batch, pool.head_num, 60), dtype=torch.float32, device=device)
        self.k_books, self.v_books = list(pool.k_book.cb16.unbind(0)), list(pool.v_book.cb16.unbind(0))

    def forward(self, q, layer_id, kv_indptr, kv_indices, cos_sin_cache, sm_scale):
        batch = q.shape[0]
        if q.ndim != 3 or q.shape[1:] != (self.pool.head_num, 128) or q.dtype != torch.bfloat16:
            raise ValueError("Official LoopVQ decode requires BF16 [batch,local_heads,128] queries")
        if not 1 <= batch <= self.metadata.max_batch or not 0 <= layer_id < 192:
            raise ValueError("Official LoopVQ batch/slot exceeds its persistent capacity")
        if cos_sin_cache.dtype != torch.float32 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 128:
            raise ValueError("Official LoopVQ requires the checkpoint FP32 NeoX RoPE cache")
        if cos_sin_cache.shape[0] < self.metadata.max_context_len:
            raise ValueError("Official LoopVQ context exceeds the model RoPE table")
        if layer_id == 0:
            self.metadata.update(kv_indptr, kv_indices, batch)
        m, pool = self.metadata, self.pool
        partial, lse = self.partial[:batch], self.lse[:batch]
        # Empty tiers and unused splits must not reuse last layer's LSE. The
        # reducer masks their uninitialized output cells by this -inf sentinel.
        lse.fill_(-float("inf"))
        official_quant_attention(
            q, pool.get_key_buffer(layer_id), self.k_books[layer_id],
            pool.get_value_buffer(layer_id), self.v_books[layer_id],
            m.quant_indptr[:batch + 1], m.quant_indices, m.quant_splits[:batch],
            partial[:, :, 12:], lse[:, :, 12:], cos_sin_cache, sm_scale,
            pos_offset=pool.hp_prefix_tokens, group_size=1 if batch == 1 else 2, max_splits=48,
        )
        official_hp_attention(
            q, pool.get_hp_key_buffer(layer_id), pool.get_hp_value_buffer(layer_id),
            cos_sin_cache, m.full_seq_len[:batch], partial[:, :, :12], lse[:, :, :12],
            m.hp_indptr[:batch + 1], m.hp_indices, m.hp_splits[:batch],
            max_splits=12, sm_scale=sm_scale, prefix_tokens=pool.hp_prefix_tokens,
        )
        out = torch.empty_like(q)
        official_reduce(partial, lse, out)
        return out
