# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Independent identity LoopVQ KV pool, adapted from LoopVQ d5d58c83.

The quant arena has one uint8 row per base layer, shared by its four recurrent
slots. The HP arena keeps independent BF16 rows at every depth. Only fresh,
unchunked prefill and one-token decode are supported by this first contract.
"""
import torch

from sglang.kernels.ops.loopvq.encode import encode_from_hp, encode_steps, write_hp
from sglang.srt.mem_cache.loopvq_codebook import load_loopvq_codebook
from sglang.srt.mem_cache.memory_pool import KVCache


def loopvq_codebook_bytes(layer_num, head_num, head_dim):
    # K/V, encode/decode, FP16 centroids, 256 entries per coordinate.
    return 2 * 2 * layer_num * head_num * head_dim * 256 * 2


def loopvq_hp_bytes(layer_num, head_num, head_dim, max_req_slots, prefix, recent):
    slots = 8 + 2 * max_req_slots * prefix + max_req_slots * (recent + 8)
    return slots * layer_num * head_num * head_dim * 2 * 2


class LoopVQKVPool(KVCache):
    loopvq_enabled = True
    vq_enabled = True
    vq_v_enabled = True
    shadow_enabled = False

    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device,
                 *, codebook_path, v_codebook_path, max_req_slots,
                 hp_prefix_tokens=64, hp_recent_tokens=256, num_hp_prefix_slots=None,
                 enable_memory_saver=False, start_layer=0, end_layer=None,
                 **kwargs):
        if page_size != 8 or dtype != torch.bfloat16 or start_layer != 0:
            raise ValueError("LoopVQ requires page_size=8, BF16, and all layers on one rank")
        if hp_prefix_tokens < 0 or hp_recent_tokens < 8 or hp_prefix_tokens % 8 or hp_recent_tokens % 8:
            raise ValueError("LoopVQ windows must be eight-aligned, with recent >= 8")
        if head_dim != 128 or max_req_slots < 1:
            raise ValueError("LoopVQ requires Ouro D128 and positive request capacity")
        if kwargs.get("v_head_dim", head_dim) != head_dim:
            raise ValueError("LoopVQ requires equal K/V dimensions")
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver,
                         start_layer=start_layer, end_layer=end_layer)
        self.head_num, self.head_dim, self.v_head_dim = head_num, head_dim, head_dim
        self.hp_dtype = dtype
        self.N_Q = self.flush_interval = 8
        self.N_H = 1
        self.num_quant_pages = (int(size) + 7) // 8 + 1
        self.quant_size = self.hp_global_offset = self.num_quant_pages * 8
        self.max_req_slots = int(max_req_slots)
        self.hp_prefix_tokens, self.hp_recent_tokens = hp_prefix_tokens, hp_recent_tokens
        self.hp_recent_ring_size = hp_recent_tokens + 8
        self.num_hp_prefix_slots = (8 + 2 * max_req_slots * hp_prefix_tokens
                                    if num_hp_prefix_slots is None else int(num_hp_prefix_slots))
        if self.num_hp_prefix_slots < 8 or self.num_hp_prefix_slots % 8:
            raise ValueError("LoopVQ requires a reserved, eight-aligned HP dummy page")
        self.hp_prefix_pool_slots = self.num_hp_prefix_slots
        self.hp_recent_base = self.num_hp_prefix_slots  # local HP arena coordinate
        self.hp_size = self.num_hp_prefix_slots + max_req_slots * self.hp_recent_ring_size
        self._hp_offset, self._hp_recent_base = self.hp_global_offset, self.hp_recent_base
        self.k_book = load_loopvq_codebook(codebook_path, layer_num=layer_num,
                                           head_num=head_num, head_dim=head_dim, device=device)
        self.v_book = load_loopvq_codebook(v_codebook_path, layer_num=layer_num,
                                           head_num=head_num, head_dim=head_dim, device=device)
        self._vq, self._vq_v = self.k_book, self.v_book
        self.base_layers = self.k_book.base_layers
        self._next_slab_offset = torch.zeros(max_req_slots, dtype=torch.int32, device=device)
        self._flush_counter = torch.zeros_like(self._next_slab_offset)
        self._pending_forward_done = None
        self._loop_stash = {}
        with self.memory_saver_adapter.region("kv_cache"):
            self.k_indices = torch.zeros((self.base_layers, self.quant_size, head_num, head_dim // 4),
                                         dtype=torch.uint8, device=device)
            self.v_indices = torch.zeros_like(self.k_indices)
            self.hp_k = torch.empty((layer_num, self.hp_size, head_num, head_dim), dtype=dtype, device=device)
            self.hp_v = torch.empty_like(self.hp_k)
            self.hp_k[:, :8].zero_()
            self.hp_v[:, :8].zero_()
        self.k_buffer = [self.k_indices[c] for c in self.k_book.code_layer]
        self.v_buffer = [self.v_indices[c] for c in self.v_book.code_layer]
        self.hp_k_buffer = list(self.hp_k.unbind(0))
        self.hp_v_buffer = list(self.hp_v.unbind(0))
        self._finalize_allocation_log(size)

    def mixed_kv_enabled(self):
        return True

    def quant_format(self):
        return "loopvq"

    def stash_pending_forward(self, event):
        self._pending_forward_done = event

    def wait_pending_forward(self):
        event = self._pending_forward_done
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
            self._pending_forward_done = None

    def release_req_slab(self, req_pool_idx):
        self.wait_pending_forward()
        self._next_slab_offset[req_pool_idx] = 0
        self._flush_counter[req_pool_idx] = 0

    def get_key_buffer(self, layer_id):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_hp_key_buffer(self, layer_id):
        return self.hp_k_buffer[layer_id]

    def get_hp_value_buffer(self, layer_id):
        return self.hp_v_buffer[layer_id]

    def get_kv_size_bytes(self):
        return tuple(sum(t.numel() * t.element_size() for t in tensors) for tensors in (
            (self.k_indices, self.hp_k, self.k_book.cb16, self.k_book.cb16_enc),
            (self.v_indices, self.hp_v, self.v_book.cb16, self.v_book.cb16_enc)))

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None,
                      layer_id_override=None, is_decode=False, **kwargs):
        if not loc.numel():
            return
        slot = layer.layer_id if layer_id_override is None else layer_id_override
        k = cache_k.view(-1, self.head_num, self.head_dim)
        v = cache_v.view_as(k)
        write_hp(k, v, self.hp_k[slot], self.hp_v[slot], loc, self.hp_global_offset)
        if is_decode:
            return
        step, base = divmod(slot, self.base_layers)
        held = self._loop_stash.setdefault(base, [])
        if len(held) != step:
            raise RuntimeError("LoopVQ prefill writes must arrive once in loop-major order")
        # Compact copies release otherwise-pinned QKV projection storage.
        if step < 3:
            held.append((k.clone(memory_format=torch.contiguous_format),
                         v.clone(memory_format=torch.contiguous_format)))
        else:
            encode_steps([x[0] for x in held] + [k], self.k_book.cb16_enc[base], self.k_indices[base], loc)
            encode_steps([x[1] for x in held] + [v], self.v_book.cb16_enc[base], self.v_indices[base], loc)
            del self._loop_stash[base]

    def vq_flush_k(self, plan):
        encode_from_hp(self.hp_k, self.k_book.cb16_enc, self.k_indices,
                       plan.src_hp_slot, plan.dst_quant_slots, plan.valid_mask)

    def vq_flush_v(self, plan):
        encode_from_hp(self.hp_v, self.v_book.cb16_enc, self.v_indices,
                       plan.src_hp_slot, plan.dst_quant_slots, plan.valid_mask)

    def move_kv_cache(self, tgt_loc, src_loc):
        raise NotImplementedError("LoopVQ prefix reuse / cache relocation is not supported")

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("LoopVQ CPU KV offload is not supported")

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError("LoopVQ CPU KV offload is not supported")
