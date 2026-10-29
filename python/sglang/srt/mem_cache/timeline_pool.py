"""Compact recurrent KV with logical scheduler slots and lazy depth timelines.

One physical core layer stores every recurrence in distinct token slots. A
logical token owns only the role/depth slots that have actually been allocated.
The scheduler capacity stays conservative: N complete target tokens, not R*N.
"""

import copy
import logging

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, MHATokenToKVPool


class TimelineSlots:
    def __init__(self, size, repeats, roles, device):
        self.size, self.repeats, self.roles, self.device = size, repeats, roles, device
        self.maps = {
            role: torch.zeros(
                (1 if role == "pre" else repeats, size + 1),
                dtype=torch.int64,
                device=device,
            )
            for role, count in roles.items()
            if count
        }
        self.clear()

    def clear(self):
        self.entries = {}
        self.free_ids = {
            role: list(range(1, self.size * (1 if role == "pre" else self.repeats) + 1))
            for role in self.maps
        }
        for table in self.maps.values():
            table.zero_()

    def ensure(self, logical_ids, role, depth):
        if role not in self.maps:
            return
        values = (
            logical_ids.tolist()
            if isinstance(logical_ids, torch.Tensor)
            else logical_ids
        )
        logical_new, physical_new = [], []
        for logical in values:
            if logical == 0:
                continue
            if not 0 < logical <= self.size:
                raise ValueError("Timeline logical slot outside allocator")
            key = (role, depth, logical)
            if key in self.entries:
                continue
            if not self.free_ids[role]:
                raise RuntimeError("Timeline physical slots exhausted")
            physical = self.free_ids[role].pop()
            self.entries[key] = physical
            logical_new.append(logical)
            physical_new.append(physical)
        if logical_new:
            self.maps[role][depth, torch.tensor(logical_new, device=self.device)] = (
                torch.tensor(physical_new, device=self.device)
            )

    def release(self, logical_ids):
        values = (
            logical_ids.tolist()
            if isinstance(logical_ids, torch.Tensor)
            else logical_ids
        )
        for role, table in self.maps.items():
            for depth in range(table.shape[0]):
                released = []
                for logical in values:
                    if (
                        physical := self.entries.pop((role, depth, logical), None)
                    ) is not None:
                        released.append(logical)
                        self.free_ids[role].append(physical)
                if released:
                    table[depth, torch.tensor(released, device=self.device)] = 0

    @property
    def live_slots(self):
        return len(self.entries)


class TimelineTokenToKVPool(MHATokenToKVPool):
    def __init__(self, size, *, recurrent_config, **kwargs):
        self.logical_size = size
        self.repeats = recurrent_config.total_ut_steps
        if recurrent_config.model_type == "ouro":
            p, c, o = 0, recurrent_config.num_hidden_layers, 0
        else:
            p, c, o = (
                recurrent_config.n_layers_in_prelude,
                recurrent_config.n_layers_in_recurrent_block,
                recurrent_config.n_layers_in_coda,
            )
        self.roles = {"pre": p, "core": c, "post": o}
        self.logical_to_physical = (
            list(range(p))
            + [p + i for _ in range(self.repeats) for i in range(c)]
            + [p + c + i for _ in range(self.repeats) for i in range(o)]
        )
        self.capacities = [size] * p + [size * self.repeats] * (c + o)
        if (
            kwargs.get("post_capture_active")
            or kwargs.get("quant_method") is not None
            or kwargs["page_size"] != 1
        ):
            raise ValueError(
                "Timeline KV requires ordinary unquantized page-size-1 storage"
            )
        kwargs.update(
            layer_num=p + c + o,
            start_layer=0,
            end_layer=p + c + o,
            enable_kv_cache_copy=False,
            kv_cache_layout="nhd",
        )
        super().__init__(size * self.repeats, **kwargs)
        self.timeline = TimelineSlots(size, self.repeats, self.roles, self.device)
        logging.getLogger(__name__).info(
            "LoopSpec timeline KV: physical_layers=%d logical_layers=%d full_depth_capacity=%d core_slots=%d",
            p + c + o,
            len(self.logical_to_physical),
            size,
            size * self.repeats,
        )

    def _create_buffers_normal(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (cap + 1, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for cap in self.capacities
            ]
            self.v_buffer = [
                torch.zeros(
                    (cap + 1, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for cap in self.capacities
            ]

    def _build_kv_buffer_descs(self):
        return [
            KvBufferDesc(
                f"{kind}{i}",
                tuple(buf.shape),
                row_bytes=buf.stride(0) * buf.element_size(),
                tokens_per_row=1,
            )
            for kind, buffers in (("k", self.k_buffer), ("v", self.v_buffer))
            for i, buf in enumerate(buffers)
        ]

    def get_key_buffer(self, layer_id):
        return super().get_key_buffer(self.logical_to_physical[layer_id])

    def get_value_buffer(self, layer_id):
        return super().get_value_buffer(self.logical_to_physical[layer_id])

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        **kwargs,
    ):
        logical = layer.layer_id if layer_id_override is None else layer_id_override
        return super().set_kv_buffer(
            layer,
            loc_info,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=self.logical_to_physical[logical],
            **kwargs,
        )

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("Timeline KV does not support host offload")

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("Timeline KV does not support host offload")

    def move_kv_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "Timeline KV ownership is managed by the recurrent worker"
        )


class TimelineTokenAllocator(TokenToKVPoolAllocator):
    def clear(self):
        super().clear()
        self._kvcache.timeline.clear()

    def free(self, free_index):
        # Deferred scheduler frees release physical ownership only at group_end.
        if self.free_group is None:
            self._kvcache.timeline.release(free_index)
        super().free(free_index)


class TimelineRuntime:
    def __init__(self, pool, allocator, request_pool, runner):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.pool, self.backend = pool, runner.attn_backend
        self.source = request_pool.req_to_token
        self.tables, self.translators, self.backends = {}, {}, {}
        for role, maps in pool.timeline.maps.items():
            for depth in range(maps.shape[0]):
                key = (role, depth)
                table = torch.zeros_like(self.source)
                self.tables[key] = table
                self.translators[key] = KVIndexTranslator(
                    req_to_token=table,
                    token_to_kv_pool_allocator=allocator,
                    token_to_kv_pool=pool,
                    page_size=1,
                    device=pool.device,
                )
                view = copy.copy(runner)
                view.req_to_token_pool = copy.copy(request_pool)
                view.req_to_token_pool.req_to_token = table
                view.kv_index_translator = self.translators[key]
                self.backends[key] = TritonAttnBackend(view)

    def ensure_prefill(self, slots, endpoints):
        values = slots.tolist()
        self.pool.timeline.ensure(values, "pre", 0)
        for depth in range(self.pool.repeats):
            self.pool.timeline.ensure(values, "core", depth)
            if depth + 1 in endpoints:
                self.pool.timeline.ensure(values, "post", depth)

    def bind(self, batch, role, depth):
        if role not in self.pool.timeline.maps:
            return batch
        key = (role, depth)
        mapping = self.pool.timeline.maps[role][depth]
        self.tables[key].copy_(mapping[self.source.long()])
        mapped = copy.copy(batch)
        mapped.out_cache_loc = mapping[batch.out_cache_loc]
        self.translators[key].bind_own_plan(mapped)
        return mapped

    def target_batch(self, batch, role, depth):
        if role not in self.pool.timeline.maps:
            return batch, self.backend
        mapped = self.bind(batch, role, depth)
        backend = self.backends[(role, depth)]
        backend.init_forward_metadata(mapped)
        return mapped, backend


class MixedTimelineRuntime:
    """A compact execution view, with one KV plane per row.

    Like vllm-rlt's MetadataBuilder, row identity and recurrence are inputs,
    not graph layout. The native allocator still owns logical tokens; this
    object only translates already-reserved timelines. Different branch rows
    may name the same source request without overwriting each other's view.
    """

    def __init__(self, runner, request_pool, max_rows):
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        self.pool = runner.token_to_kv_pool
        if not isinstance(self.pool, TimelineTokenToKVPool):
            raise TypeError("Mixed recurrence requires timeline KV storage")
        self.source = request_pool.req_to_token
        self.max_rows = max_rows
        self.rows = torch.arange(1, max_rows + 1, device=self.source.device)
        self.columns = torch.arange(self.source.shape[1], device=self.source.device)
        self.views = {}
        for role in self.pool.timeline.maps:
            table = torch.zeros(
                (max_rows + 1, self.source.shape[1]),
                dtype=self.source.dtype,
                device=self.source.device,
            )
            translator = KVIndexTranslator(
                req_to_token=table,
                token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
                token_to_kv_pool=self.pool,
                page_size=1,
                device=runner.device,
            )
            view = copy.copy(runner)
            view.req_to_token_pool = copy.copy(request_pool)
            view.req_to_token_pool.req_to_token = table
            view.kv_index_translator = translator
            self.views[role] = table, translator, TritonAttnBackend(view)
        self.default_backend = runner.attn_backend

    def target_batch(self, batch, role, depths, token_rows=None):
        """Bind active rows only; no host readback of depth/length is needed.

        ``depths`` has one entry per sequence. For extend, ``token_rows`` maps
        packed query tokens to those sequences. All referenced role/depth
        slots must have been reserved by the owner before submission.
        """
        if batch.batch_size > self.max_rows:
            raise ValueError("Recurrent execution exceeds row capacity")
        if role not in self.views:
            return batch, self.default_backend
        table, translator, backend = self.views[role]
        count = batch.batch_size
        if depths.shape != (count,):
            raise ValueError("One recurrent depth is required per sequence")
        mapping = self.pool.timeline.maps[role]
        if role == "pre":
            depths = torch.zeros_like(depths)
        # The scratch rows are execution-row identities, not source Req ids.
        # Mask stale cells beyond the current prefix before using them as ids.
        logical = self.source[batch.req_pool_indices.long()].long()
        valid = self.columns[None, :] < batch.seq_lens[:, None]
        logical = torch.where(valid, logical, 0)
        table[1 : count + 1].copy_(mapping[depths[:, None], logical])
        if token_rows is None:
            if not batch.forward_mode.is_decode_or_idle():
                raise ValueError("Packed extend requires token-to-row indices")
            write_depths = depths
        else:
            write_depths = depths[token_rows]
        mapped = copy.copy(batch)
        mapped.req_pool_indices = self.rows[:count]
        # A row plan can repeat source requests (different depths/branches).
        # Its parent's CPU length sum is not a valid CSR allocation bound.
        mapped.seq_lens_sum = count * self.source.shape[1]
        mapped.out_cache_loc = mapping[write_depths, batch.out_cache_loc.long()]
        translator.bind_own_plan(mapped)
        backend.init_forward_metadata(mapped)
        return mapped, backend
