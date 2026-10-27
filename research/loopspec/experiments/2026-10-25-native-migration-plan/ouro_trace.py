"""Read-only layer hooks and same-input BF16 operator diagnostics (not timing)."""

import argparse
import json
from pathlib import Path

import torch


class OuroTrace:
    def __init__(self, model, native, prefix):
        self.active = False
        self.prefix = prefix
        self.data = {}
        self.counts, self.depths = {}, {}
        self.handles = []
        self.config = model.config

        def save(key, value):
            if not self.active:
                return
            if isinstance(value, tuple):
                value = value[0]
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            self.data[key] = value.detach().cpu().clone()

        self.save = save

        def key(i, stage):
            return f"d{self.depths[i]:02}/l{i:02}/{stage}"

        def output(module, i, stage):
            def hook(module, args, out):
                if self.active:
                    save(key(i, stage), out)

            self.handles.append(module.register_forward_hook(hook))

        def input_(module, i, stage):
            def hook(module, args):
                if self.active:
                    save(key(i, stage), args[0])

            self.handles.append(module.register_forward_pre_hook(hook))

        for i, layer in enumerate(model.model.layers):

            def enter(module, args, kwargs, i=i):
                if self.active:
                    self.depths[i] = self.counts.get(i, 0)
                    self.counts[i] = self.depths[i] + 1
                    hidden = (
                        args[1]
                        if native
                        else kwargs.get("hidden_states", args[0] if args else None)
                    )
                    save(key(i, "input"), hidden)

            self.handles.append(
                layer.register_forward_pre_hook(enter, with_kwargs=True)
            )
            output(layer, i, "output")
            for stage, name in enumerate(
                (
                    "input_layernorm",
                    "input_layernorm_2",
                    "post_attention_layernorm",
                    "post_attention_layernorm_2",
                ),
                1,
            ):
                norm = getattr(layer, name)
                input_(norm, i, f"norm{stage}.in")
                output(norm, i, f"norm{stage}.out")
                self.data[f"weight/l{i:02}/norm{stage}"] = (
                    norm.weight.detach().cpu().clone()
                )
            attn, mlp = layer.self_attn, layer.mlp
            input_(attn.o_proj, i, "attention")
            output(attn.o_proj, i, "attention_out")
            input_(mlp.down_proj, i, "product")
            output(mlp.down_proj, i, "mlp_out")
            if native:

                def qkv(module, args, out, i=i, attn=attn):
                    if self.active:
                        for name, tensor in zip(
                            ("q", "k", "v"),
                            out[0].split(
                                (attn.q_size, attn.kv_size, attn.kv_size), dim=-1
                            ),
                        ):
                            save(key(i, name), tensor)

                self.handles.append(attn.qkv_proj.register_forward_hook(qkv))

                def gate_up(module, args, out, i=i):
                    if self.active:
                        for name, tensor in zip(
                            ("gate", "up"), out[0].chunk(2, dim=-1)
                        ):
                            save(key(i, name), tensor)

                self.handles.append(mlp.gate_up_proj.register_forward_hook(gate_up))

                def rope(module, args, out, i=i):
                    if self.active:
                        save(key(i, "q_rope"), out[0])
                        save(key(i, "k_rope"), out[1])

                self.handles.append(attn.rotary_emb.register_forward_hook(rope))
            else:
                for name in ("q", "k", "v"):
                    output(getattr(attn, name + "_proj"), i, name)
                for name in ("gate", "up"):
                    output(getattr(mlp, name + "_proj"), i, name)
        if not native:

            def rope_cache(module, args, out):
                save("cos", out[0])
                save("sin", out[1])

            self.handles.append(
                model.model.rotary_emb.register_forward_hook(rope_cache)
            )

    def begin(self, prefix):
        self.active = prefix == self.prefix
        self.counts, self.depths = {}, {}

    def finish(self, path):
        for handle in self.handles:
            handle.remove()
        torch.save(
            {"prefix": self.prefix, "eps": self.config.rms_norm_eps, "data": self.data},
            path,
        )


def error(a, b):
    delta = a.float() - b.float()
    return {
        "max_abs": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "equal": bool(torch.equal(a, b)),
    }


def compare(reference, candidate):
    ref = torch.load(reference, map_location="cpu", weights_only=True)
    actual = torch.load(candidate, map_location="cpu", weights_only=True)
    assert ref["prefix"] == actual["prefix"]
    a, b = ref["data"], actual["data"]
    order = (
        "input",
        "norm1.out",
        "q",
        "k",
        "v",
        "attention",
        "attention_out",
        "norm2.out",
        "norm3.in",
        "norm3.out",
        "gate",
        "up",
        "product",
        "mlp_out",
        "norm4.out",
        "output",
    )
    common, isolated = [], []
    cos, sin = a["cos"].cuda()[:, None], a["sin"].cuda()[:, None]
    head_dim = cos.shape[-1]
    for base in sorted(key[: -len("input")] for key in b if key.endswith("/input")):
        for stage in order:
            name = base + stage
            common.append({"key": name, **error(a[name], b[name])})
        depth, layer, _ = base.split("/")
        for stage in range(1, 5):
            x = b[base + f"norm{stage}.in"].cuda()
            weight = b[f"weight/{layer}/norm{stage}"].cuda()
            y = weight * (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + ref["eps"])
            ).to(x.dtype)
            isolated.append(
                {
                    "key": base + f"norm{stage}",
                    **error(y.cpu(), b[base + f"norm{stage}.out"]),
                }
            )
        gate, up = b[base + "gate"].cuda(), b[base + "up"].cuda()
        y = torch.nn.functional.silu(gate) * up
        isolated.append(
            {"key": base + "silu_product", **error(y.cpu(), b[base + "product"])}
        )
        for name in ("q", "k"):
            x = (
                b[base + name]
                .cuda()
                .reshape(-1, b[base + name].shape[-1] // head_dim, head_dim)
            )
            x1, x2 = x.chunk(2, dim=-1)
            y = x * cos + torch.cat((-x2, x1), dim=-1) * sin
            isolated.append(
                {
                    "key": base + name + "_rope",
                    **error(y.flatten(1).cpu(), b[base + name + "_rope"]),
                }
            )
        q, k, v = [
            b[base + name]
            .cuda()
            .reshape(-1, b[base + name].shape[-1] // head_dim, head_dim)
            .transpose(0, 1)
            for name in ("q_rope", "k_rope", "v")
        ]
        k, v = [t.repeat_interleave(q.shape[0] // k.shape[0], dim=0) for t in (k, v)]
        scores = torch.matmul(q, k.transpose(1, 2)) * head_dim**-0.5
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device="cuda").triu(1)
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
        p = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        y = torch.matmul(p, v).transpose(0, 1).flatten(1)
        isolated.append(
            {"key": base + "attention", **error(y.cpu(), b[base + "attention"])}
        )
    return {
        "prefix": ref["prefix"],
        "common": common,
        "same_input_hf_arithmetic_vs_native": isolated,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare(args.reference, args.candidate), indent=2)
    )
