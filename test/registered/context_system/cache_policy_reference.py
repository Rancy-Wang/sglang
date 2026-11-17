"""Slow specification, intentionally independent of production IR/compiler.

Every query records (logical token, key position) in causal order. A Drop in
text mode discards the old query graph and recomputes the entire live prefix;
in KV mode it recomputes full history at frozen-death/current-live positions.
"""

from dataclasses import dataclass


@dataclass
class Reference:
    active: list
    positions: list
    compute_positions: list
    readers: list
    next_position: int


def simulate(tokens, drops, reposition=(), *, policy):
    alive, positions, birth, readers = [], [], [], []
    expiry, cursor = [], 0
    for boundary in range(len(tokens) + 1):
        remove = {i for a, b in drops.get(boundary, ()) for i in range(a, b)}
        effective = [i for i in alive if i in remove]
        if effective:
            for i in effective:
                expiry[i] = boundary
            alive = [i for i in alive if i not in remove]
            if policy == "text_drop_with_pe":
                for rank, i in enumerate(alive):
                    birth[i] = positions[i]
                    readers[i] = [(j, positions[j]) for j in alive[: rank + 1]]
            else:
                for rank, i in enumerate(alive):
                    positions[i] = rank
                cursor = len(alive)
                for i in range(boundary):
                    birth[i] = positions[i]
                    readers[i] = [
                        (j, positions[j]) for j in range(i + 1) if i < expiry[j]
                    ]
        if boundary in reposition:
            if policy != "text_drop_with_pe":
                raise ValueError("KV policy has no reposition")
            if not alive:
                raise ValueError("No live tokens to reposition")
            # Match the existing definition of an effective R: a changed K.
            if any(positions[i] != rank for rank, i in enumerate(alive)):
                for rank, i in enumerate(alive):
                    positions[i] = rank
                cursor = len(alive)
        if boundary < len(tokens):
            i = boundary
            positions.append(cursor)
            birth.append(cursor)
            cursor += 1
            expiry.append(2**31 - 1)
            alive.append(i)
            readers.append([(j, positions[j]) for j in alive])
    return Reference(alive, positions, birth, readers, cursor)


def hidden_fingerprints(tokens, reference, layers=3):
    """Symbolic multilayer dependency graph, includes query/key PE separately."""
    hidden = list(tokens)
    for _ in range(layers):
        hidden = [
            (
                hidden[i],
                reference.compute_positions[i],
                tuple((hidden[j], p) for j, p in read),
            )
            for i, read in enumerate(reference.readers)
        ]
    return hidden


def reference_attention_forward(model, reference, projection):
    """Slow real-model oracle: native projections, independent FP32 attention.

    This replaces RadixAttention.forward only inside an opt-in test. It reads
    neither production Context metadata nor a KV pool. Positions and readers
    come exclusively from the event simulator above. Intended for bounded
    numerical tests; do not materialize this quadratic graph for long BCP runs.
    """
    import torch

    rotary = {
        module.attn.layer_id: module.rotary_emb
        for module in model.modules()
        if hasattr(module, "attn") and hasattr(module, "rotary_emb")
    }
    inverse = {raw: i for i, raw in enumerate(projection)}
    rows = [[(inverse[j], p) for j, p in reference.readers[i]] for i in projection]
    births = [reference.compute_positions[i] for i in projection]

    def forward(layer, q, k, v, batch, *args, **kwargs):
        if len(q) != len(projection) or k is None or v is None:
            raise ValueError("Reference requires one complete, bounded prefill")
        if batch.positions.tolist() != births:
            raise ValueError(
                "Reference model positions differ from independent simulator"
            )
        rope = rotary[layer.layer_id]
        cache = rope.cos_sin_cache.float()
        width = rope.rotary_dim
        q = q.reshape(len(q), layer.tp_q_head_num, layer.qk_head_dim).float()
        k = k.reshape(len(q), layer.tp_k_head_num, layer.qk_head_dim)
        v = v.reshape(len(q), layer.tp_v_head_num, layer.v_head_dim)
        groups = layer.tp_q_head_num // layer.tp_k_head_num
        window = layer.sliding_window_size
        results = []
        for i, row in enumerate(rows):
            chosen = row
            if window is not None and window >= 0:
                chosen = [pair for pair in row if pair[1] >= births[i] - window][
                    -(window + 1) :
                ]
            raw = torch.tensor([j for j, _ in chosen], device=q.device)
            old = torch.tensor([births[j] for j, _ in chosen], device=q.device)
            new = torch.tensor([p for _, p in chosen], device=q.device)
            keys = k[raw].float().clone()
            # Complex division gives the RoPE rotation between positions,
            # including a scaled cosine/sine cache (e.g. YaRN).
            co, so = cache[old].chunk(2, -1)
            cn, sn = cache[new].chunk(2, -1)
            denom = co.square() + so.square()
            cos = ((cn * co + sn * so) / denom)[:, None, :]
            sin = ((sn * co - cn * so) / denom)[:, None, :]
            if rope.is_neox_style:
                left, right = keys[..., :width].chunk(2, -1)
                rotated = torch.cat(
                    (left * cos - right * sin, right * cos + left * sin), -1
                )
            else:
                left, right = keys[..., :width:2], keys[..., 1:width:2]
                rotated = torch.stack(
                    (left * cos - right * sin, right * cos + left * sin), -1
                ).flatten(-2)
            changed = old != new
            keys[changed, :, :width] = rotated[changed]
            keys = keys.to(k.dtype).float().repeat_interleave(groups, 1)
            values = v[raw].float().repeat_interleave(groups, 1)
            scores = torch.einsum("hd,khd->hk", q[i], keys) * layer.scaling
            if layer.logit_cap:
                scores = layer.logit_cap * torch.tanh(scores / layer.logit_cap)
            sinks = kwargs.get("sinks")
            if sinks is not None:
                scores = torch.cat((scores, sinks.float().reshape(-1, 1)), -1)
            weights = scores.softmax(-1)
            if sinks is not None:
                weights = weights[:, :-1]
            results.append(torch.einsum("hk,khd->hd", weights, values))
        return torch.stack(results).to(v.dtype).reshape(len(q), -1)

    return forward
