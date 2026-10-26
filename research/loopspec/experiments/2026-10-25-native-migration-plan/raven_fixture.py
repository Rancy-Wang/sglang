"""Generate small, explicitly synthetic Raven checkpoints for integration tests.

Uses zero initial hidden to make AR and speculative executions comparable
without assuming that different schedules consume a shared RNG in the same order.
Never changes a released model/config. Fixture output is not model-quality evidence.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def build(directory, post_norm, tokenizer):
    directory.mkdir(parents=True, exist_ok=True)
    h = 256 if post_norm else 128
    config = dict(
        model_type="huginn_raven",
        architectures=["RavenForCausalLM"],
        n_embd=h,
        n_heads=2,
        n_layers=3,
        head_dim=h // 2,
        num_key_value_heads=2,
        vocab_size=49152,
        intermediate_size=2 * h,
        n_layers_in_prelude=1,
        n_layers_in_recurrent_block=1,
        n_layers_in_coda=1,
        mean_recurrence=4,
        norm_eps=1e-6,
        tie_embeddings=False,
        init_values={"std": 0.0, "embed_scale": 1.0},
        max_position_embeddings=2048,
        torch_dtype="bfloat16",
        rope_base=10000.0,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    torch.manual_seed(11907)
    weights = {}

    def matrix(name, rows, cols):
        weights[name] = (torch.randn(rows, cols) * 0.02).bfloat16()

    matrix("transformer.wte.weight", 49152, h)
    matrix("transformer.adapter.weight", h, 2 * h)
    for stage in ("prelude", "core_block", "coda"):
        root = f"transformer.{stage}.0."
        matrix(root + "attn.Wqkv.weight", 3 * h, h)
        matrix(root + "attn.proj.weight", h, h)
        matrix(root + "mlp.fc.weight", 4 * h, h)
        matrix(root + "mlp.proj.weight", h, 2 * h)
        for name in ("norm_1.weight", "norm_2.weight"):
            weights[root + name] = torch.ones(h, dtype=torch.bfloat16)
        if post_norm:
            weights[root + "attn.q_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
            weights[root + "attn.k_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    weights["transformer.ln_f.weight"] = torch.ones(h, dtype=torch.bfloat16)
    head = torch.zeros(49152, h, dtype=torch.bfloat16)
    for index in range(4):
        head[100 + 2 * index, index] = 4
        head[101 + 2 * index, index] = -4
    weights["lm_head.weight"] = head
    save_file(weights, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    for source in tokenizer.iterdir():
        if source.name in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        }:
            dest = directory / source.name
            if not dest.exists():
                dest.symlink_to(source)
    return {"path": str(directory), "weights": len(weights), "post_norm": post_norm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/home/ruoxi/models/Ouro-2.6B")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            [
                build(args.output / name, flag, args.tokenizer)
                for name, flag in [("raven-pre", False), ("raven-post", True)]
            ]
        )
    )
