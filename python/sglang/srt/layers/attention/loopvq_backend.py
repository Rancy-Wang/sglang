# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Standalone coupled-KV dispatch; compressed indices never enter BF16 kernels."""

import torch


def forward_extend(backend, q, k, v, layer, batch, *, save_kv_cache):
    from sglang.kernels.ops.attention.ouro_attention import ouro_attention

    if save_kv_cache:
        raise ValueError("LoopVQ prefill must store raw K after live RoPE attention.")
    if any(batch.extend_prefix_lens_cpu):
        raise ValueError("LoopVQ prefill does not support a cached prefix.")
    if batch.spec_info is not None:
        raise ValueError("LoopVQ does not support speculative prefill.")
    metadata = backend.forward_metadata
    q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
    out = torch.empty_like(q)
    # All prefixes are empty. Pass live K/V as the masked cache operands,
    # keeping the same checkpoint rounding as the existing BF16 baseline.
    ouro_attention(
        q, k, v, k, v, out,
        metadata.kv_indptr, metadata.kv_indices, metadata.qo_indptr,
        batch.batch_size, metadata.max_extend_len, layer.scaling,
        decode=False,
    )
    return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


def forward_decode(backend, q, k, v, layer, batch, *, save_kv_cache, cos_sin_cache):
    from sglang.kernels.ops.loopvq.attention import loopvq_decode_attention

    if cos_sin_cache is None:
        raise ValueError("LoopVQ decode requires the model's RoPE cache.")
    if batch.spec_info is not None:
        raise ValueError("LoopVQ does not support speculative decode.")
    pool = backend.token_to_kv_pool
    if save_kv_cache:
        pool.set_kv_buffer(layer, batch.out_cache_loc, k, v, is_decode=True)
    metadata = backend.forward_metadata
    if getattr(backend, "ouro_decode_profile", "hf_reference") == "optimized":
        if backend.loopvq_official is None:
            raise RuntimeError("Official LoopVQ state was not prepared before graph capture")
        out = backend.loopvq_official.forward(
            q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            layer.layer_id, metadata.kv_indptr, metadata.kv_indices,
            cos_sin_cache, layer.scaling,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
    out = loopvq_decode_attention(
        q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
        pool, layer.layer_id, metadata.kv_indptr, metadata.kv_indices,
        cos_sin_cache, layer.scaling,
    )
    return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
