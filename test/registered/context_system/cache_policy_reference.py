"""Slow specification, intentionally independent of production IR/compiler.

Every query records (logical token, key position) in causal order. A Drop in
text mode discards the old query graph and recomputes the entire live prefix;
in KV mode it recomputes full history at frozen-death/current-live positions.
"""

from dataclasses import dataclass
from itertools import pairwise


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


@dataclass
class CompactReference:
    """Linear token state plus R boundaries; no materialized query/key graph.

    All arrays use execution order. ``raw`` maps back to logical history, and
    ``expiry`` is a logical boundary, independent of position encodings.
    """

    raw: object
    positions: object
    compute_positions: object
    expiry: object
    rotation_cuts: tuple
    next_position: int

    def key_positions(self, query):
        import numpy as np

        positions = self.compute_positions.copy()
        for cut in self.rotation_cuts:
            if query < cut:
                break
            positions[:cut] = np.arange(cut)
        return positions

    def row(self, query, window=-1):
        """Inspection only: one row, in execution coordinates."""
        import numpy as np

        positions = self.key_positions(query)
        mask = (self.raw <= self.raw[query]) & (self.raw[query] < self.expiry)
        if window >= 0:
            mask &= positions >= self.compute_positions[query] - window
        indices = np.flatnonzero(mask)
        if window >= 0:
            indices = indices[-(window + 1) :]
        return [(int(i), int(positions[i])) for i in indices]


def simulate_compact(token_count, drops, reposition=(), *, policy):
    """Independent event replay in O(N * events) time and O(N + events) memory.

    Unlike the production compiler, this just mutates flat arrays at events.
    Unlike ``simulate``, it never constructs the quadratic reader graph. Token
    IDs do not affect attention geometry. R boundaries before a text reset are
    erased only from the computation graph, after their position effects apply.
    """
    import numpy as np

    if policy not in ("text_drop_with_pe", "message_drop_without_repos"):
        raise ValueError("Unknown reference policy")
    if policy == "message_drop_without_repos" and reposition:
        raise ValueError("KV policy has no reposition")
    n = token_count
    positions = np.zeros(n, dtype=np.int64)
    birth = np.zeros(n, dtype=np.int64)
    alive = np.zeros(n, dtype=bool)
    expiry = np.full(n, 2**63 - 1, dtype=np.int64)
    previous = cursor = 0
    rotations = []
    repos = set(reposition)
    for boundary in sorted({0, n, *drops, *repos}):
        if not 0 <= boundary <= n:
            raise ValueError("Reference event outside token history")
        count = boundary - previous
        positions[previous:boundary] = np.arange(cursor, cursor + count)
        birth[previous:boundary] = positions[previous:boundary]
        alive[previous:boundary] = True
        cursor += count
        remove = np.zeros(n, dtype=bool)
        for start, end in drops.get(boundary, ()):
            if not 0 <= start <= end <= boundary:
                raise ValueError("Drop must refer to already born tokens")
            remove[start:end] = True
        remove &= alive
        if remove.any():
            expiry[remove] = boundary
            alive[remove] = False
            if policy == "text_drop_with_pe":
                birth[alive] = positions[alive]
                rotations.clear()
            else:
                positions[alive] = np.arange(alive.sum())
                birth[:boundary] = positions[:boundary]
                cursor = int(alive.sum())
        if boundary in repos:
            if not alive.any():
                raise ValueError("No live tokens to reposition")
            compact = np.arange(alive.sum())
            if not np.array_equal(positions[alive], compact):
                positions[alive] = compact
                cursor = len(compact)
                rotations.append(boundary)
        previous = boundary
    raw = np.flatnonzero(alive) if policy == "text_drop_with_pe" else np.arange(n)
    return CompactReference(
        raw,
        positions[raw],
        birth[raw],
        expiry[raw],
        tuple(int(np.searchsorted(raw, boundary)) for boundary in rotations),
        cursor,
    )


def chunked_reference_attention_forward(
    model, reference, *, query_block=128, key_block=2048, streaming=False
):
    """Full-prefill FP32 oracle with bounded attention scratch memory.

    Uses only the independent flat event replay above and native Q/K/V. It
    holds one layer's projected KV, O(query_block * N) visibility, and bounded
    score tiles; it never reads a production KV pool, compiler, or occurrence
    plan. Full attention still takes quadratic arithmetic. This is a numerical
    diagnostic, not a performance baseline or a serving replacement. Streaming
    mode owns independent per-layer K/V arrays and consumes consecutive chunks
    (including fixed-continuation decode) from this immutable reference tape.
    """
    import torch

    if query_block <= 0 or key_block <= 0:
        raise ValueError("Reference block sizes must be positive")
    rotary = {
        module.attn.layer_id: module.rotary_emb
        for module in model.modules()
        if hasattr(module, "attn") and hasattr(module, "rotary_emb")
    }
    n = len(reference.raw)
    state = {}

    def attention(layer, q, k, v, batch, *args, **kwargs):
        count = len(q)
        offset = state.get(layer.layer_id, (0,))[0] if streaming else 0
        end = offset + count
        if not count or end > n or k is None or v is None:
            raise ValueError("Reference requires nonempty consecutive token chunks")
        if not streaming and count != n:
            raise ValueError("Non-streaming reference requires a complete prefill")
        if batch.positions.tolist() != reference.compute_positions[offset:end].tolist():
            raise ValueError("Reference model positions differ from event replay")
        rope = rotary[layer.layer_id]
        device = q.device
        births = torch.as_tensor(reference.compute_positions[:end], device=device)
        raw = torch.as_tensor(reference.raw[:end], device=device)
        expiry = torch.as_tensor(reference.expiry[:end], device=device)
        q = q.reshape(count, layer.tp_q_head_num, layer.qk_head_dim).float()
        k = k.reshape(count, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.reshape(count, layer.tp_v_head_num, layer.v_head_dim)
        if streaming:
            if layer.layer_id not in state:
                saved_k = k.new_empty(n, *k.shape[1:])
                saved_v = v.new_empty(n, *v.shape[1:])
            else:
                _, saved_k, saved_v = state[layer.layer_id]
            saved_k[offset:end].copy_(k)
            saved_v[offset:end].copy_(v)
            state[layer.layer_id] = (end, saved_k, saved_v)
            k, v = saved_k[:end], saved_v[:end]
        groups = layer.tp_q_head_num // layer.tp_k_head_num
        window = layer.sliding_window_size
        window = -1 if window is None else window
        cache, width = rope.cos_sin_cache.float(), rope.rotary_dim
        sinks = kwargs.get("sinks")
        result = torch.empty(
            count,
            layer.tp_q_head_num,
            layer.v_head_dim,
            device=device,
            dtype=torch.float32,
        )
        cuts = sorted(
            {offset, end, *(c for c in reference.rotation_cuts if offset < c < end)}
        )
        for begin, stage_end in pairwise(cuts):
            positions = torch.as_tensor(
                reference.key_positions(begin)[:end], device=device
            )
            keys = k.float().clone()
            co, so = cache[births].chunk(2, -1)
            cn, sn = cache[positions].chunk(2, -1)
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
            changed = births != positions
            keys[changed, :, :width] = rotated[changed]
            keys = keys.to(k.dtype)
            for start in range(begin, stage_end, query_block):
                stop = min(start + query_block, stage_end)
                visible = (raw[None] <= raw[start:stop, None]) & (
                    raw[start:stop, None] < expiry[None]
                )
                if window >= 0:
                    visible &= positions[None] >= births[start:stop, None] - window
                    rank = visible.cumsum(-1, dtype=torch.int32)
                    visible &= rank > rank[:, -1:] - (window + 1)
                shape = (layer.tp_q_head_num, stop - start)
                maximum = torch.full(
                    shape, -torch.inf, device=device, dtype=torch.float32
                )
                denominator = torch.zeros(shape, device=device, dtype=torch.float32)
                total = torch.zeros(
                    *shape, layer.v_head_dim, device=device, dtype=torch.float32
                )
                if sinks is not None:
                    maximum[:] = sinks.float().reshape(-1, 1)
                    denominator.fill_(1)
                queries = q[start - offset : stop - offset].transpose(0, 1)
                for left in range(0, stop, key_block):
                    right = min(left + key_block, stop)
                    mask = visible[:, left:right]
                    if not bool(mask.any()):
                        continue
                    kk = keys[left:right].float().repeat_interleave(groups, 1)
                    vv = v[left:right].float().repeat_interleave(groups, 1)
                    scores = torch.einsum("hqd,khd->hqk", queries, kk) * layer.scaling
                    if layer.logit_cap:
                        scores = layer.logit_cap * torch.tanh(scores / layer.logit_cap)
                    scores.masked_fill_(~mask[None], -torch.inf)
                    updated = torch.maximum(maximum, scores.max(-1).values)
                    safe = torch.where(torch.isfinite(updated), updated, 0)
                    prior = (maximum - safe).exp()
                    weights = (scores - safe[..., None]).exp()
                    total = total * prior[..., None] + torch.einsum(
                        "hqk,khd->hqd", weights, vv
                    )
                    denominator = denominator * prior + weights.sum(-1)
                    maximum = updated
                if not bool((denominator > 0).all()):
                    raise AssertionError("Independent reference produced an empty row")
                result[start - offset : stop - offset] = (
                    total / denominator[..., None]
                ).transpose(0, 1)
        return result.to(v.dtype).reshape(count, -1)

    @torch.no_grad()
    def forward(layer, q, k, v, batch, *args, **kwargs):
        # Model setup can enable TF32 globally. The oracle's dot products must
        # retain FP32 precision; restore the caller's setting even on failure.
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            return attention(layer, q, k, v, batch, *args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    return forward


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


def run_reference_model(
    wrapper,
    tokens,
    drops,
    reposition,
    *,
    policy,
    continuation=(),
    chunk_tokens=2048,
    audit=None,
):
    """Model logits with independent positions, readers and owned streaming KV.

    ``continuation`` is an externally frozen sequence of input tokens. Return
    the prompt's next-token logits followed by one row per continuation token.
    Native plain-request allocation is only scaffolding: no Context compiler,
    Radix match, occurrence plan or production KV contents feed the oracle.
    """
    from array import array
    from unittest.mock import patch

    import numpy as np
    import torch
    from sglang.benchmark.one_batch import TreeCacheNamespace
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if chunk_tokens <= 0:
        raise ValueError("Reference chunk size must be positive")
    full = list(tokens) + list(continuation)
    reference = simulate_compact(len(full), drops, reposition, policy=policy)
    projected = [full[i] for i in reference.raw]
    prompt_end = int(np.searchsorted(reference.raw, len(tokens)))
    if prompt_end == 0:
        raise ValueError("Reference needs a nonempty active prompt")
    ends = [*range(chunk_tokens, prompt_end, chunk_tokens), prompt_end]
    ends.extend(range(prompt_end + 1, len(projected) + 1))
    runner = wrapper.torch_runner
    wrapper.clear()
    attention = chunked_reference_attention_forward(
        runner.model, reference, query_block=256, key_block=4096, streaming=True
    )
    offsets = {}

    def oracle(layer, q, k, v, batch, *args, **kwargs):
        start = offsets.get(layer.layer_id, 0)
        result = attention(layer, q, k, v, batch, *args, **kwargs)
        if audit is not None:
            for i in range(max(prompt_end - 1, start), start + len(q)):
                audit[(layer.layer_id, i)] = {
                    name: tensor[i - start].detach().clone()
                    for name, tensor in (
                        ("q", q),
                        ("k", k),
                        ("v", v),
                        ("output", result),
                    )
                }
        offsets[layer.layer_id] = start + len(q)
        return result

    prefix = torch.empty(0, dtype=torch.int64, device=runner.device)
    results = []
    with (
        torch.no_grad(),
        patch.object(RadixAttention, "forward", oracle),
        patch.object(runner, "prefill_cuda_graph_runner", None),
    ):
        for end in ends:
            req = Req(
                "independent-reference",
                "",
                array("q", projected),
                SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.prefix_indices = prefix
            req.set_extend_range(len(prefix), end)
            batch = ScheduleBatch.init_new(
                reqs=[req],
                req_to_token_pool=runner.req_to_token_pool,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
                tree_cache=TreeCacheNamespace(
                    page_size=1,
                    device=runner.device,
                    token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
                ),
                model_config=runner.model_config,
                enable_overlap=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            batch.prepare_for_extend()
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device)
            batch.prefill_input_ids_cpu = None
            fb = ForwardBatch.init_new(
                batch, runner, return_hidden_states_before_norm=False
            )
            fb.positions = torch.as_tensor(
                reference.compute_positions[len(prefix) : end],
                dtype=fb.positions.dtype,
                device=runner.device,
            )
            logits = runner.forward(fb).logits_output.next_token_logits
            if end >= prompt_end:
                results.append(logits.detach().clone())
            prefix = torch.cat((prefix, batch.out_cache_loc))
            runner.req_to_token_pool.free(req)
    return torch.cat(results)
