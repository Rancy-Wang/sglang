# SPDX-License-Identifier: Apache-2.0
"""Split-KV VQ+ attention. Packed correction, RoPE and PV stay in registers.

All splits overwrite their partial/LSE, including empty and graph-padding rows.
The same stage handles HP and quantized locations using one ordered page list.
"""
import torch
import triton
import triton.language as tl

from .official_reduce import official_reduce


@triton.jit
def _rows(IDX, CB, RES, SCALE, HP, slots, c, head, valid,
          H: tl.constexpr, OFFSET: tl.constexpr):
    quant = (slots < OFFSET) & (slots >= 0)
    mask = valid[:, None] & quant[:, None]
    code = tl.load(IDX + (slots[:, None] * H + head) * 32 + c[None, :] // 4, mask, 0).to(tl.int32)
    base = tl.load(CB + ((head * 32 + c[None, :] // 4) * 256 + code) * 4 + c[None, :] % 4, mask, 0).to(tl.float32)
    byte = tl.load(RES + (slots[:, None] * H + head) * 64 + c[None, :] // 2, mask, 0).to(tl.int32)
    z = (byte >> (4 * (c[None, :] % 2))) & 15
    z = tl.where(z >= 8, z - 16, z).to(tl.float32)
    scale = tl.load(SCALE + (slots[:, None] * H + head) * 4 + c[None, :] // 32, mask, 0).to(tl.float32)
    hp = tl.load(HP + ((slots[:, None] - OFFSET) * H + head) * 128 + c[None, :],
                 valid[:, None] & ~quant[:, None], 0).to(tl.float32)
    return tl.where(quant[:, None], base + z * scale, hp)


@triton.jit
def _stage1(Q, KI, VI, KC, VC, KR, VR, KS, VS, HK, HV, PTR, LOC, ROPE, OUT, LSE,
            H: tl.constexpr, OFFSET: tl.constexpr, QS: tl.constexpr, QH: tl.constexpr,
            SPLITS: tl.constexpr, SCALE: tl.constexpr, N: tl.constexpr):
    req, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin, end = tl.load(PTR + req), tl.load(PTR + req + 1)
    length = end - begin
    first = tl.load(LOC + begin, length > 0, 0)
    length = tl.where(first != 0, length, 0)
    per = tl.cdiv(length, SPLITS)
    lo, hi = split * per, tl.minimum((split + 1) * per, length)
    c = tl.arange(0, 128)
    pair = (c + 64) % 128
    sign = tl.where(c < 64, -1., 1.)
    q = tl.load(Q + req * QS + head * QH + c).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    denom = tl.full((), 0., tl.float32)
    acc = tl.full((128,), 0., tl.float32)
    for start in range(lo, hi, N):
        pos = start + tl.arange(0, N)
        valid = pos < hi
        slots = tl.load(LOC + begin + pos, valid, 0)
        raw = _rows(KI, KC, KR, KS, HK, slots, c, head, valid, H, OFFSET)
        paired = _rows(KI, KC, KR, KS, HK, slots, pair, head, valid, H, OFFSET)
        cos = tl.load(ROPE + pos[:, None] * 128 + c[None, :] % 64, valid[:, None], 0)
        sin = tl.load(ROPE + pos[:, None] * 128 + 64 + c[None, :] % 64, valid[:, None], 0)
        k = (raw * cos + paired * sign[None, :] * sin).to(tl.bfloat16).to(tl.float32)
        logits = tl.sum(k * q[None, :], 1) * SCALE
        logits = tl.where(valid, logits, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(logits, 0))
        alpha = tl.exp(maximum - new_max)
        prob = tl.exp(logits - new_max)
        v = _rows(VI, VC, VR, VS, HV, slots, c, head, valid, H, OFFSET).to(tl.bfloat16).to(tl.float32)
        acc = acc * alpha + tl.sum(prob[:, None] * v, 0)
        denom = denom * alpha + tl.sum(prob, 0)
        maximum = new_max
    where = (req * H + head) * SPLITS + split
    tl.store(OUT + where * 128 + c, acc / tl.maximum(denom, 1.e-30))
    tl.store(LSE + where, tl.where(denom > 0, maximum + tl.log(denom), -float('inf')))


class ResidualAttention:
    def __init__(self, pool, *, max_batch, max_context_len, splits=None, block=32, implementation="cuda", threads=512, dynamic_splits=True, unroll=1, group_size=0, async_table=None, global_table=False, layout=0, payload=True, adaptive=False):
        if layout not in (0, 2) or (layout and (implementation != 'cuda' or not global_table or threads != 512 or unroll != 1)):
            raise ValueError('Head pages require the standard SM80 global CUDA kernels')
        if implementation not in ("cuda", "triton"):
            raise ValueError("Unknown residual attention implementation")
        if splits is None:
            splits = 32 if implementation == "cuda" else 16
        if global_table and async_table:
            raise ValueError("Global VQ+ codebook lookup cannot prefetch a shared table")
        self.r1_async = (not global_table) if async_table is None else bool(async_table)
        self.r2_async = False if async_table is None else bool(async_table)
        self.extension = None
        if implementation == "cuda":
            if splits != 32 or not 0 < max_context_len <= 32768:
                raise ValueError("VQ+ shared CUDA requires 32 splits and context <= 32768")
            if layout:
                from .residual_attention_paged_cuda import prepare
                self.extension = prepare(pool.hp_k.device, layout=layout, cooperative=False)
            else:
                from .residual_attention_cuda import prepare
                self.extension = prepare(pool.hp_k.device, threads, unroll, self.r1_async, global_table)
        self.dynamic_splits = dynamic_splits
        self.sm_count = torch.cuda.get_device_properties(pool.hp_k.device).multi_processor_count
        if pool.head_dim != 128 or not pool.residual_bits == 4:
            raise ValueError("VQ+ requires D128 and INT4 group32")
        if group_size not in (0, 1, 2) or (group_size == 2 and implementation != 'cuda'):
            raise ValueError('Grouped VQ+ requires the CUDA implementation')
        self.group_size = group_size
        self.layout = layout
        self.grouped = None
        if implementation == 'cuda' and group_size != 1 and max_batch >= 2:
            from .residual_attention_grouped_cuda import prepare_grouped
            self.grouped = prepare_grouped(pool.hp_k.device, self.r2_async, global_table, layout=layout)
            self.group_partial = torch.empty((max_batch, pool.head_num, 64, 128), dtype=torch.float32, device=pool.hp_k.device)
            self.group_lse = torch.empty((max_batch, pool.head_num, 64), dtype=torch.float32, device=pool.hp_k.device)
        self.pool, self.splits, self.block = pool, splits, block
        self.max_batch, self.max_context_len = max_batch, max_context_len
        self.partial = torch.empty((max_batch, pool.head_num, splits, 128), dtype=torch.float32, device=pool.hp_k.device)
        self.lse = torch.empty((max_batch, pool.head_num, splits), dtype=torch.float32, device=pool.hp_k.device)
        self.payload_extension, self.payload_capacity = None, 0
        # Limit automatic promotion to the measured hardware/head shape. Other
        # configurations retain the existing reader and grouped dispatch.
        if (payload and layout == 2 and group_size == 0 and dynamic_splits
                and max_batch >= 4 and pool.head_num == 8 and self.sm_count == 108):
            from .residual_attention_payload_cuda import prepare
            extension, capacity = prepare(pool.hp_k.device)
            if capacity == 324:
                self.payload_extension, self.payload_capacity = extension, capacity
        self.adaptive_extension = None
        if payload and adaptive and layout == 2 and group_size == 0 and dynamic_splits:
            from .residual_attention_adaptive_cuda import prepare
            self.adaptive_extension = prepare(pool.hp_k.device, max_batch, pool.head_num)

    def use_grouped(self, batch):
        # Capture-static dispatch, no GPU lengths copied to the host.
        # Token pages retain their existing R2 policy. Head-local pages measured
        # faster with R1 at large B, while B2 still benefits from R2 at long N.
        return self.grouped is not None and (batch > 1 if self.group_size == 2 else batch == 2 or (not self.layout and batch >= 16))

    def use_payload(self, batch):
        return self.payload_extension is not None and 4 <= batch <= 16

    def use_adaptive(self, batch):
        return (self.adaptive_extension is not None and not self.use_grouped(batch)
                and self.adaptive_extension.supports(batch))

    def forward(self, q, layer_id, kv_indptr, kv_indices, cos_sin_cache, sm_scale):
        pool, b = self.pool, q.shape[0]
        if q.shape != (b, pool.head_num, 128) or q.dtype != torch.bfloat16 or b > self.max_batch:
            raise ValueError("VQ+ requires BF16 [batch,local_heads,128] within capacity")
        out = torch.empty_like(q)
        if self.extension is not None:
            grouped = self.use_grouped(b)
            launch = self.grouped.grouped_cuda if grouped else self.extension.residual_cuda
            partial, lse = (self.group_partial, self.group_lse) if grouped else (self.partial, self.lse)
            groups = (b + 1) // 2 if grouped else b
            target = max(1, (2 * self.sm_count + groups * pool.head_num - 1) // (groups * pool.head_num)) if self.dynamic_splits else (64 if grouped else 0)
            if self.use_adaptive(b):
                launch = self.adaptive_extension.residual_cuda
                target = -self.adaptive_extension.capacity_for(b)
            elif self.use_payload(b):
                launch = self.payload_extension.residual_cuda
                # Lengths remain on the device, including during graph replay.
                target = -self.payload_capacity
            launch(
                q, pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id),
                pool.k_book.cb16[layer_id], pool.v_book.cb16[layer_id],
                pool.k_residual[layer_id], pool.v_residual[layer_id],
                pool.k_residual_scale[layer_id], pool.v_residual_scale[layer_id],
                pool.get_hp_key_buffer(layer_id), pool.get_hp_value_buffer(layer_id),
                kv_indptr, kv_indices, cos_sin_cache, partial, lse,
                pool.hp_global_offset, float(sm_scale),
                target,
            )
            return official_reduce(partial[:b], lse[:b], out)
        _stage1[(b, pool.head_num, self.splits)](
            q, pool.get_key_buffer(layer_id), pool.get_value_buffer(layer_id),
            pool.k_book.cb16[layer_id], pool.v_book.cb16[layer_id],
            pool.k_residual[layer_id], pool.v_residual[layer_id],
            pool.k_residual_scale[layer_id], pool.v_residual_scale[layer_id],
            pool.get_hp_key_buffer(layer_id), pool.get_hp_value_buffer(layer_id),
            kv_indptr, kv_indices, cos_sin_cache, self.partial, self.lse,
            pool.head_num, pool.hp_global_offset, q.stride(0), q.stride(1),
            self.splits, float(sm_scale), self.block,
            enable_fp_fusion=False, num_warps=4,
        )
        return official_reduce(self.partial[:b], self.lse[:b], out)
