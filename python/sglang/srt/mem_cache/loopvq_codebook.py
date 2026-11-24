# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Strict identity LoopVQ bundle reader.

Layout adapted from mscheong01/LoopVQ at d5d58c83 (vq_codebook.py).
This first serving contract is G16 / four loop steps / 256 FP16 centroids.
Transform-bearing TaSQ bundles are rejected instead of silently misread.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.srt.mem_cache.loopvq_pack import (
    build_coupled_codebook,
    code_layer_map,
    slot_codebooks,
)


@dataclass
class LoopVQCodebook:
    cb16_enc: torch.Tensor
    cb16: torch.Tensor
    base_layers: int
    total_steps: int
    head_num: int
    head_dim: int
    code_layer: list[int]
    steps: int = 4
    channels: int = 4
    codebook_size: int = 256
    group_dim: int = 4

    @property
    def num_groups(self):
        return self.head_dim // self.channels

    @property
    def n_code_layers(self):
        return self.cb16_enc.shape[0]


def load_loopvq_codebook(path, *, layer_num, head_num, head_dim, device, tp_rank=0, tp_size=1):
    """Validate on CPU before allocating device tables; never mutate the bundle."""
    b = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(b, dict):
        raise ValueError("LoopVQ bundle must be a tensor dictionary")
    lc = b.get("loop_coupled", {})
    if lc.get("loops") != 4 or lc.get("channels") != 4 or lc.get("plain_slots"):
        raise ValueError("LoopVQ serving requires four-step, four-channel coupled bundles")
    lb = int(lc["base_layers"])
    if layer_num != lb * 4 or head_dim % 16:
        raise ValueError("LoopVQ bundle/model slot geometry mismatch (requires full R4)")
    for name in ("pertoken_norm", "pool_heads_scale", "perm_rope"):
        if b.get(name, False):
            raise ValueError(f"LoopVQ identity profile rejects {name}")
    for name in ("loop_scale", "chan_weight", "freq_idx"):
        if b.get(name) is not None:
            raise ValueError(f"LoopVQ identity profile rejects {name}")
    if tp_size not in (1, 2) or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid LoopVQ tensor-parallel rank")
    global_heads = head_num * tp_size
    shape = (layer_num, global_heads, head_dim, head_dim)
    identity = torch.eye(head_dim)
    for name in ("forward", "inverse"):
        t = b.get(name)
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
            raise ValueError(f"Invalid {name} tensor in LoopVQ bundle")
        if not torch.equal(t, identity.to(t.dtype).expand(shape)):
            raise ValueError(f"LoopVQ identity profile requires identity {name}")
    mean = b.get("mean")
    if (not isinstance(mean, torch.Tensor)
            or tuple(mean.shape) != shape[:3] or bool(mean.any())):
        raise ValueError("LoopVQ identity profile requires zero mean")
    bounds = [(g * 16, (g + 1) * 16, 8) for g in range(head_dim // 16)]
    if list(b.get("bounds", [])) != bounds:
        raise ValueError("LoopVQ serving requires uniform G16 8-bit code indices")
    books = b.get("codebooks", {})
    expected = {(s, h) for s in range(layer_num) for h in range(global_heads)}
    if set(books) != expected:
        raise ValueError("LoopVQ codebook slot/head coverage does not match the model")
    for key, groups in books.items():
        if len(groups) != head_dim // 16:
            raise ValueError(f"LoopVQ group count mismatch at {key}")
        for table in groups:
            if (not isinstance(table, torch.Tensor) or table.shape != (256, 16)
                    or table.dtype != torch.float16 or not bool(torch.isfinite(table).all())):
                raise ValueError(f"Invalid LoopVQ FP16 centroid table at {key}")
    local_books = {(s, h): books[s, tp_rank * head_num + h]
                   for s in range(layer_num) for h in range(head_num)}
    enc = build_coupled_codebook(local_books, lb, 4, 4, 4, head_num, head_dim)
    dec = slot_codebooks(enc, lb, 4, 4)
    return LoopVQCodebook(
        enc.to(device), dec.to(device), lb, 4, head_num, head_dim,
        code_layer_map(lb, 4, 4),
    )
