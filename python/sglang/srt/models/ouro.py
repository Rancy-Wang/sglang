# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Ouro with shared recurrent weights and independent KV at each depth.

Checkpoint names and the four sandwich norms follow the Ouro reference model.
The early-exit head is loaded, but the target always executes total_ut_steps.
"""

from typing import Iterable

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaAttention, LlamaMLP
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


def _norm(config):
    # HF Ouro rounds normalized activations before multiplying by the weight.
    return RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        cast_x_before_out_mul=True,
        # Match the checkpoint reduction order at BF16 rounding boundaries.
        force_native=True,
    )


class OuroRotaryEmbedding(nn.Module):
    """Keep checkpoint BF16 product rounding without changing shared RoPE."""

    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, positions, query, key):
        # The native torch path casts cos/sin and each product to input dtype.
        # CUDA's generic fused RoPE instead keeps the cache/products in FP32.
        return self.rotary.forward_native(positions, query, key)


class OuroMLP(LlamaMLP):
    def forward(self, hidden, forward_batch=None):
        gate_up, _ = self.gate_up_proj(hidden)
        gate, up = gate_up.chunk(2, dim=-1)
        # HF rounds SiLU to BF16 before multiplying by up. Neither fused
        # SiLU-and-mul nor dual-GEMM preserves that intermediate rounding.
        product = torch.nn.functional.silu(gate) * up
        return self.down_proj(product)[0]


class OuroAttention(LlamaAttention):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.rotary_emb = OuroRotaryEmbedding(self.rotary_emb)
        self.attn = nn.ModuleList(
            RadixAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                layer_id=depth * config.num_hidden_layers + layer_id,
                quant_config=quant_config,
                prefix=add_prefix(f"attn.{depth}", prefix),
            )
            for depth in range(config.total_ut_steps)
        )

        for attention in self.attn:
            attention.ouro_bf16_rounding = True

    def forward(self, positions, hidden_states, forward_batch, depth):
        q, k, v = self.forward_prepare_native(positions, hidden_states)
        output = self.attn[depth](q, k, v, forward_batch)
        return self.o_proj(output)[0]


class OuroDecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        self.self_attn = OuroAttention(
            config, layer_id, quant_config, add_prefix("self_attn", prefix)
        )
        self.mlp = OuroMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = _norm(config)
        self.input_layernorm_2 = _norm(config)
        self.post_attention_layernorm = _norm(config)
        self.post_attention_layernorm_2 = _norm(config)

    def forward(self, positions, hidden, forward_batch, depth):
        attention = self.self_attn(
            positions, self.input_layernorm(hidden), forward_batch, depth
        )
        hidden = hidden + self.input_layernorm_2(attention)
        mlp = self.mlp(self.post_attention_layernorm(hidden), forward_batch)
        return hidden + self.post_attention_layernorm_2(mlp)


class OuroModel(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            OuroDecoderLayer(config, i, quant_config, add_prefix(f"layers.{i}", prefix))
            for i in range(config.num_hidden_layers)
        )
        self.norm = _norm(config)
        self.early_exit_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=True,
            prefix=add_prefix("early_exit_gate", prefix),
        )

    def advance_loop(self, hidden, positions, forward_batch, depth):
        """Materialize one recurrence; never read KV from another depth."""
        if not 0 <= depth < self.config.total_ut_steps:
            raise ValueError(f"Invalid Ouro recurrence: {depth}")
        for layer in self.layers:
            hidden = layer(positions, hidden, forward_batch, depth)
        return self.norm(hidden)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        for depth in range(self.config.total_ut_steps):
            hidden = self.advance_loop(hidden, positions, forward_batch, depth)
        return hidden

    def advance_batch(self, hidden, positions, groups):
        """One core invocation, shared projections across recurrent depths.

        groups contains (depth, start, end, forward_batch, attention_backend).
        Attention remains separate because every depth owns independent KV.
        """
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        for layer in self.layers:
            normalized = layer.input_layernorm(hidden)
            q, k, v = layer.self_attn.forward_prepare_native(positions, normalized)
            parts = []
            for depth, start, end, batch, backend in groups:
                with forward_context(ForwardContext(backend)):
                    parts.append(
                        layer.self_attn.attn[depth](
                            q[start:end], k[start:end], v[start:end], batch
                        )
                    )
            attention = layer.self_attn.o_proj(torch.cat(parts, dim=0))[0]
            hidden = hidden + layer.input_layernorm_2(attention)
            mlp = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = hidden + layer.post_attention_layernorm_2(mlp)
        return self.norm(hidden)


class OuroForCausalLM(nn.Module):
    has_injection = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if get_parallel().pp_size != 1:
            raise ValueError("Ouro currently requires pipeline-parallel-size=1")
        if quant_config is not None:
            raise ValueError("Ouro quantized checkpoint loading is not implemented")
        self.config = config
        # Physical range: the runner expands this by total_ut_steps for KV.
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.model = OuroModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = (
            self.model.embed_tokens
            if config.tie_word_embeddings
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("lm_head", prefix),
            )
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if get_embedding:
            raise ValueError("Ouro embedding serving is not implemented")
        if hasattr(self, "timeline_runtime"):
            from sglang.srt.model_executor.forward_context import (
                ForwardContext,
                forward_context,
            )

            runtime = self.timeline_runtime
            runtime.ensure_prefill(
                forward_batch.out_cache_loc, {self.config.total_ut_steps}
            )
            hidden = (
                self.model.embed_tokens(input_ids)
                if input_embeds is None
                else input_embeds
            )
            for depth in range(self.config.total_ut_steps):
                batch, backend = runtime.target_batch(forward_batch, "core", depth)
                with forward_context(ForwardContext(backend)):
                    hidden = self.model.advance_loop(hidden, positions, batch, depth)
        else:
            hidden = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_recurrent(self, input_ids, positions, batch):
        return self.model.embed_tokens(input_ids), None

    def advance_recurrent(self, hidden, positions, groups, injected=None):
        return self.model.advance_batch(hidden, positions, groups)

    def readout_recurrent(self, hidden, positions, groups):
        from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

        return bf16_gemm_dispatch(hidden, self.lm_head.weight, None).float()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        shards = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", 0),
            (".up_proj.", ".gate_up_proj.", 1),
        )
        required = set()
        for name in parameters:
            parts = [src for src, dst, _ in shards if dst in name]
            if parts:
                for src, dst, _ in shards:
                    if dst in name:
                        required.add(name.replace(dst, src))
            else:
                required.add(name)
        seen, loaded = set(), set()
        for name, tensor in weights:
            if name.endswith(".rotary_emb.inv_freq"):
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            if name not in required or name in seen:
                raise ValueError(f"Unexpected or duplicate Ouro weight: {name}")
            seen.add(name)
            target, shard = name, None
            for src, dst, index in shards:
                if src in name:
                    target, shard = name.replace(src, dst), index
                    break
            param = parameters[target]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard)
            loaded.add(target)
        if missing := required - seen:
            raise ValueError(f"Missing Ouro checkpoint weights: {sorted(missing)}")
        return loaded


EntryClass = OuroForCausalLM
