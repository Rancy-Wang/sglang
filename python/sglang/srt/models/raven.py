"""Fixed-depth Raven, with persistent input injection and separate post timelines.

Checkpoint conventions follow kaist-flexml-lab/loopspec at 41aa710e,
modeling/raven.py and modeling/components.py. No external runtime is imported.
"""

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer
from sglang.srt.models.olmo2 import Olmo2DecoderLayer
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import add_prefix


class RavenForCausalLM(nn.Module):
    has_injection = True

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if (
            get_parallel().tp_size != 1
            or get_parallel().pp_size != 1
            or quant_config is not None
        ):
            raise ValueError(
                "Native Raven currently requires TP=PP=1, unquantized weights"
            )
        self.config = config
        self.start_layer, self.end_layer = 0, config.logical_attention_layers
        self.post_norm = config.head_dim == 128
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.adapter = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=config.bias,
            gather_output=True,
            prefix=add_prefix("adapter", prefix),
        )
        block_class = Olmo2DecoderLayer if self.post_norm else LlamaDecoderLayer

        def build(count, name, offset, stride, repeats):
            blocks = nn.ModuleList()
            for index in range(count):
                block = block_class(
                    config=config,
                    layer_id=0,
                    prefix=add_prefix(f"{name}.{index}", prefix),
                )
                original = block.self_attn.attn
                block.self_attn.attn = nn.ModuleList(
                    deepcopy(original) for _ in range(repeats)
                )
                for depth, attn in enumerate(block.self_attn.attn):
                    attn.layer_id = offset + depth * stride + index
                blocks.append(block)
            return blocks

        p, c, o, r = (
            config.n_layers_in_prelude,
            config.n_layers_in_recurrent_block,
            config.n_layers_in_coda,
            config.mean_recurrence,
        )
        self.pre = build(p, "pre", 0, p, 1)
        self.core = build(c, "core", p, c, r)
        self.post = build(o, "post", p + c * r, o, r)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.embed_tokens)
        self.logits_processor = LogitsProcessor(config)

    @property
    def model(self):
        return self

    @property
    def layers(self):
        return list(self.pre) + list(self.core) + list(self.post)

    def _blocks(self, blocks, hidden, positions, groups):
        for block in blocks:
            attn = block.self_attn
            source = hidden if self.post_norm else block.input_layernorm(hidden)
            if self.post_norm:
                packed, _ = attn.qkv_proj(source)
                q, k, v = packed.split(
                    [attn.q_size, attn.kv_size, attn.kv_size], dim=-1
                )
                q, k = attn._apply_qk_norm(q, k)
                q, k = attn.rotary_emb(positions, q, k)
            else:
                q, k, v = attn.forward_prepare_native(positions, source)
            parts = []
            for depth, start, end, batch, backend in groups:
                with (
                    forward_context(ForwardContext(backend))
                    if backend is not None
                    else nullcontext()
                ):
                    parts.append(
                        attn.attn[depth](
                            q[start:end], k[start:end], v[start:end], batch
                        )
                    )
            output = attn.o_proj(torch.cat(parts))[0]
            hidden = hidden + (
                block.post_attention_layernorm(output) if self.post_norm else output
            )
            output = block.mlp(
                hidden if self.post_norm else block.post_attention_layernorm(hidden)
            )
            hidden = hidden + (
                block.post_feedforward_layernorm(output) if self.post_norm else output
            )
        return hidden

    def prepare_recurrent(self, input_ids, positions, batch, input_embeds=None):
        injected = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        scale = float(self.config.init_values["embed_scale"])
        injected = self._blocks(
            self.pre, injected * scale, positions, [(0, 0, len(input_ids), batch, None)]
        )
        hidden = torch.randn_like(injected)
        std = float(self.config.init_values["std"])
        if std > 0:
            nn.init.trunc_normal_(hidden, mean=0, std=std, a=-3 * std, b=3 * std)
        else:
            hidden.zero_()
        return hidden * scale, injected

    def advance_recurrent(self, hidden, positions, groups, injected=None):
        if injected is None:
            raise ValueError("Raven requires persistent input injection")
        hidden = self.adapter(torch.cat((hidden, injected), dim=-1))[0]
        return self._blocks(self.core, hidden, positions, groups)

    def post_hidden(self, hidden, positions, groups):
        return self.norm(self._blocks(self.post, hidden, positions, groups))

    def readout_recurrent(self, hidden, positions, groups):
        hidden = self.post_hidden(hidden, positions, groups)
        return bf16_gemm_dispatch(hidden, self.lm_head.weight, None).float()

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        endpoints = (
            set(get_spec().speculative_draft_depth or ())
            if get_spec().speculative_algorithm == "LOOPSPEC"
            else set()
        )
        final = self.config.mean_recurrence - 1
        runtime = getattr(self, "timeline_runtime", None)
        if runtime is not None:
            runtime.ensure_prefill(forward_batch.out_cache_loc, endpoints | {final + 1})
            batch, backend = runtime.target_batch(forward_batch, "pre", 0)
        else:
            batch, backend = forward_batch, None
        with (
            forward_context(ForwardContext(backend))
            if backend is not None
            else nullcontext()
        ):
            hidden, injected = self.prepare_recurrent(
                input_ids, positions, batch, input_embeds
            )
        for depth in range(final + 1):
            batch, backend = (
                runtime.target_batch(forward_batch, "core", depth)
                if runtime is not None
                else (forward_batch, None)
            )
            groups = [(depth, 0, len(input_ids), batch, backend)]
            hidden = self.advance_recurrent(hidden, positions, groups, injected)
            if depth + 1 in endpoints or depth == final:
                batch, backend = (
                    runtime.target_batch(forward_batch, "post", depth)
                    if runtime is not None
                    else (forward_batch, None)
                )
                groups = [(depth, 0, len(input_ids), batch, backend)]
                post = self.post_hidden(hidden, positions, groups)
        return self.logits_processor(input_ids, post, self.lm_head, forward_batch)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_embed_and_head(self):
        return self.embed_tokens.weight, self.lm_head.weight

    def map_weight(self, name):
        roots = {
            "transformer.prelude.": "pre.",
            "transformer.core_block.": "core.",
            "transformer.coda.": "post.",
            "transformer.wte.": "embed_tokens.",
            "transformer.adapter.": "adapter.",
            "transformer.ln_f.": "norm.",
            "lm_head.": "lm_head.",
        }
        for source, target in roots.items():
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        else:
            raise ValueError(f"Unknown Raven checkpoint key: {name}")
        replacements = {
            ".attn.Wqkv.": ".self_attn.qkv_proj.",
            ".attn.proj.": ".self_attn.o_proj.",
            ".attn.q_norm.": ".self_attn.q_norm.",
            ".attn.k_norm.": ".self_attn.k_norm.",
            ".mlp.fc.": ".mlp.gate_up_proj.",
            ".mlp.proj.": ".mlp.down_proj.",
            ".norm_1.": ".post_attention_layernorm."
            if self.post_norm
            else ".input_layernorm.",
            ".norm_2.": ".post_feedforward_layernorm."
            if self.post_norm
            else ".post_attention_layernorm.",
        }
        for source, target in replacements.items():
            name = name.replace(source, target)
        return name

    def load_weights(self, weights):
        params = dict(self.named_parameters(remove_duplicate=False))
        required = {id(param) for param in params.values()}
        loaded, seen = set(), set()
        for name, value in weights:
            if name.endswith("rotary_emb.inv_freq"):
                continue
            target = self.map_weight(name)
            if target not in params or target in seen:
                raise ValueError(f"Unexpected or duplicate Raven weight: {name}")
            param = params[target]
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            loaded.add(id(param))
            seen.add(target)
        if missing := required - loaded:
            raise ValueError(
                f"Missing Raven weights: {[n for n, p in params.items() if id(p) in missing]}"
            )
        return seen


EntryClass = RavenForCausalLM
