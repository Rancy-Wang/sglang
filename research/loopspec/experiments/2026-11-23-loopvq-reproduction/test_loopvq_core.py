"""Synthetic engineering checks; no model accuracy or performance claim.

CPU (Torch only): python test_loopvq_core.py
CUDA (Torch + Triton): python test_loopvq_core.py --cuda
Modules are loaded by path to avoid importing an unrelated installed SGLang.
"""
import argparse
import contextlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "python" / "sglang"
CUDA = "--cuda" in sys.argv
sys.argv = [a for a in sys.argv if a != "--cuda"]


def package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module
    return module


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRC / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.kernels",
             "sglang.kernels.ops", "sglang.kernels.ops.loopvq"):
    package(name)
base = load("sglang.srt.mem_cache.allocator.base", "srt/mem_cache/allocator/base.py")
package("sglang.srt.mem_cache.allocator").BaseTokenToKVPoolAllocator = base.BaseTokenToKVPoolAllocator
pack = load("sglang.srt.mem_cache.loopvq_pack", "srt/mem_cache/loopvq_pack.py")
reader = load("sglang.srt.mem_cache.loopvq_codebook", "srt/mem_cache/loopvq_codebook.py")
alloc = load("sglang.srt.mem_cache.loopvq_allocator", "srt/mem_cache/loopvq_allocator.py")
# Only the request-slot admission hook is replaced; all tier allocation,
# remapping and free logic exercised below comes from the real new module.
package("sglang.srt.mem_cache.allocation").alloc_req_slots = lambda pool, reqs, tree: pool.alloc(reqs)


def bundle(lb=2, heads=2, dim=128):
    generator = torch.Generator().manual_seed(17)
    return {
        "loop_coupled": {"loops": 4, "channels": 4, "base_layers": lb},
        "forward": torch.eye(dim).expand(lb * 4, heads, dim, dim).clone(),
        "inverse": torch.eye(dim).expand(lb * 4, heads, dim, dim).clone(),
        "mean": torch.zeros(lb * 4, heads, dim),
        "bounds": [(g * 16, (g + 1) * 16, 8) for g in range(dim // 16)],
        "codebooks": {(s, h): [torch.randn(256, 16, generator=generator).half()
                                  for _ in range(dim // 16)]
                      for s in range(lb * 4) for h in range(heads)},
    }


class LayoutTests(unittest.TestCase):
    def test_grouping_matches_channel_major_step_minor(self):
        x = torch.arange(8 * 3 * 2 * 128).reshape(8, 3, 2, 128)
        grouped = pack.regroup_for_encode(x, 2, 4, 4, 4)
        for layer in range(2):
            for group in range(32):
                for channel in range(4):
                    for step in range(4):
                        torch.testing.assert_close(grouped[layer, :, :, group, channel * 4 + step],
                                                   x[step * 2 + layer, :, :, group * 4 + channel])
        torch.testing.assert_close(pack.ungroup_after_decode(grouped, 2, 4, 4, 4), x)

    def test_reader_layout_and_reject_transforms(self):
        data = bundle()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.pt"
            torch.save(data, path)
            book = reader.load_loopvq_codebook(path, layer_num=8, head_num=2, head_dim=128, device="cpu")
            self.assertEqual(tuple(book.cb16_enc.shape), (2, 2, 32, 256, 16))
            self.assertEqual(tuple(book.cb16.shape), (8, 2, 32, 256, 4))
            self.assertEqual(book.code_layer, [0, 1] * 4)
            for slot in range(8):
                for group in range(32):
                    source = data["codebooks"][(slot % 2 + (group // 8) * 2, 1)][group % 8]
                    torch.testing.assert_close(book.cb16[slot, 1, group], source[:, slot // 2::4])
            for key, value in (("loop_scale", torch.ones(8, 2, 128)),
                               ("chan_weight", torch.ones(8, 2, 128)),
                               ("pertoken_norm", True)):
                invalid = dict(data, **{key: value})
                torch.save(invalid, path)
                with self.assertRaises(ValueError):
                    reader.load_loopvq_codebook(path, layer_num=8, head_num=2, head_dim=128, device="cpu")


class FakePool:
    hp_prefix_tokens = 64
    hp_recent_tokens = 256
    hp_recent_ring_size = 264
    max_req_slots = 17
    num_hp_prefix_slots = 8 + 2 * max_req_slots * hp_prefix_tokens
    num_quant_pages = 513
    hp_global_offset = num_quant_pages * 8

    def __init__(self):
        self._next_slab_offset = torch.zeros(self.max_req_slots, dtype=torch.int32)
        self._flush_counter = torch.zeros_like(self._next_slab_offset)
        self.values = {}
        self.flushes = 0

    def wait_pending_forward(self):
        pass

    def release_req_slab(self, ids):
        self._next_slab_offset[ids] = 0
        self._flush_counter[ids] = 0

    def vq_flush_k(self, plan):
        self.flushes += 1
        for src, dst in zip(plan.src_hp_slot.tolist(), plan.dst_quant_slots.tolist()):
            self.values[dst] = self.values[src + self.hp_global_offset]

    def vq_flush_v(self, plan):
        pass


class FakeReqPool:
    def __init__(self):
        self.req_to_token = torch.full((17, 4096), -1, dtype=torch.int32)

    def alloc(self, requests):
        for index, req in enumerate(requests, 1):
            req.kv.req_pool_idx = index
        return list(range(1, len(requests) + 1))

    def write(self, where, values):
        self.req_to_token[where] = values

    def alloc_aux_to_lengths(self, **kwargs):
        pass


def make_allocator(pool):
    return alloc.LoopVQKVAllocator(pool.num_quant_pages, 8, 64, 256, 264, 17,
                                  pool.num_hp_prefix_slots, torch.uint8, torch.bfloat16,
                                  "cpu", pool, False)


class LifecycleTests(unittest.TestCase):
    def test_ragged_prefill_flush_ring_free_reuse(self):
        pool = FakePool()
        allocator = make_allocator(pool)
        capacity = allocator.available_size()
        lengths = [1, 7, 8, 63, 64, 65, 255, 256, 257, 319, 320, 321, 327, 328, 333, 383]
        row_pool = FakeReqPool()
        for cycle in range(2):
            batch = types.SimpleNamespace(
                token_to_kv_pool_allocator=allocator, req_to_token_pool=row_pool,
                prefix_lens=[0] * len(lengths), seq_lens_cpu=torch.tensor(lengths),
                extend_lens=lengths, device="cpu", tree_cache=None,
                reqs=[types.SimpleNamespace(kv=types.SimpleNamespace()) for _ in lengths],
            )
            out, ids, ids_cpu = alloc.alloc_for_extend(batch)
            batch.req_pool_indices, batch.req_pool_indices_cpu = ids, ids_cpu
            batch.seq_lens = batch.seq_lens_cpu.clone()
            for rid, n in zip(ids.tolist(), lengths):
                for pos, loc in enumerate(row_pool.req_to_token[rid, :n].tolist()):
                    pool.values[loc] = (rid, pos, cycle)
            for _ in range(80):
                previous = batch.seq_lens_cpu.tolist()
                out = alloc.alloc_for_decode(batch, 1)
                for rid, pos, loc in zip(ids.tolist(), previous, out.tolist()):
                    pool.values[loc] = (rid, pos, cycle)
                batch.seq_lens_cpu += 1
                batch.seq_lens += 1
                live = []
                for rid, n in zip(ids.tolist(), batch.seq_lens_cpu.tolist()):
                    locs = row_pool.req_to_token[rid, :n].tolist()
                    live.extend(locs)
                    self.assertEqual([pool.values[x] for x in locs], [(rid, p, cycle) for p in range(n)])
                self.assertEqual(len(live), len(set(live)), "requests alias a live physical slot")
            allocator.free_group_begin()
            for rid, n in zip(ids.tolist(), batch.seq_lens_cpu.tolist()):
                # Deliberately split mid-page: release must deduplicate physical pages.
                row = row_pool.req_to_token[rid, :n]
                allocator.free_segments([(row[:67], 0), (row[67:], 67)])
                allocator.release_request(rid)
            allocator.free_group_end()
            self.assertEqual(allocator.available_size(), capacity)
            self.assertEqual(len(allocator.free_pages), len(torch.unique(allocator.free_pages)))
            self.assertEqual(len(allocator.hp_prefix_free_pages), len(torch.unique(allocator.hp_prefix_free_pages)))
            self.assertEqual(allocator.request_state, {})
            self.assertEqual(int(pool._next_slab_offset.sum()), 0)
        self.assertGreater(pool.flushes, 0)


@unittest.skipUnless(CUDA, "pass --cuda to run synthetic GPU kernel checks")
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda requested but CUDA is unavailable")
        cls.encode = load("sglang.kernels.ops.loopvq.encode", "kernels/ops/loopvq/encode.py")
        cls.attention = load("sglang.kernels.ops.loopvq.attention", "kernels/ops/loopvq/attention.py")

    def test_assignment_and_flush(self):
        torch.manual_seed(23)
        device, lb, n, h, d = "cuda", 2, 33, 2, 128
        table = torch.randn(lb, h, d // 4, 256, 16, device=device).half()
        codes = torch.randint(0, 256, (lb, n, h, d // 4), device=device)
        vectors = torch.empty(lb, n, h, d // 4, 16, device=device)
        for layer in range(lb):
            for head in range(h):
                for group in range(d // 4):
                    vectors[layer, :, head, group] = table[layer, head, group, codes[layer, :, head, group]]
        steps = pack.ungroup_after_decode(vectors, lb, 4, 4, 4).bfloat16()
        loc = torch.arange(n, device=device) + 8
        indices = torch.full((lb, n + 16, h, d // 4), 255, device=device, dtype=torch.uint8)
        for layer in range(lb):
            self.encode.encode_steps([steps[step * lb + layer] for step in range(4)],
                                     table[layer], indices[layer], loc)
        torch.testing.assert_close(indices[:, loc].long(), codes)
        torch.testing.assert_close(indices[:, :8], torch.full_like(indices[:, :8], 255))
        hp = torch.zeros(lb * 4, n + 16, h, d, device=device, dtype=torch.bfloat16)
        hp[:, loc] = steps
        actual = torch.full_like(indices, 255)
        valid = torch.ones(n, device=device, dtype=torch.bool)
        valid[-1] = False
        self.encode.encode_from_hp(hp, table, actual, loc, loc, valid)
        torch.testing.assert_close(actual[:, loc[:-1]].long(), codes[:, :-1])
        torch.testing.assert_close(actual[:, loc[-1]], torch.full_like(actual[:, loc[-1]], 255))

    def test_mixed_attention_and_graph_batches(self):
        torch.manual_seed(29)
        device, h, d, slots = "cuda", 2, 128, 4096
        enc = torch.randn(1, h, 32, 256, 16, device=device).half()
        dec = pack.slot_codebooks(enc, 1, 4, 4)
        idx = torch.randint(0, 256, (slots, h, 32), device=device, dtype=torch.uint8)
        vidx = torch.randint(0, 256, (slots, h, 32), device=device, dtype=torch.uint8)
        hp = torch.randn(slots, h, d, device=device).bfloat16()
        hv = torch.randn_like(hp)
        pool = types.SimpleNamespace(head_num=h, head_dim=d, hp_global_offset=slots,
                                     k_book=types.SimpleNamespace(cb16=dec),
                                     v_book=types.SimpleNamespace(cb16=dec * 0.5),
                                     get_key_buffer=lambda layer: idx,
                                     get_value_buffer=lambda layer: vidx,
                                     get_hp_key_buffer=lambda layer: hp,
                                     get_hp_value_buffer=lambda layer: hv)
        freq = torch.outer(torch.arange(256, device=device), 10000 ** (-torch.arange(0, d, 2, device=device).float() / d))
        rope = torch.cat((freq.cos(), freq.sin()), -1)
        worst = 0.0
        for batch_size in (1, 2, 4, 8, 16):
            for all_hp in (False, True):
                lengths = [1, 8, 31, 32, 33, 64, 65, 97] * 2
                lengths = lengths[:batch_size]
                positions = [torch.arange(n, device=device) for n in lengths]
                local = [p + i * 128 for i, p in enumerate(positions)]
                locs = [x + torch.where((p < 8) | (p >= n - 16) | all_hp, slots, 0)
                        for x, p, n in zip(local, positions, lengths)]
                loc = torch.cat(locs).int()
                ptr = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)
                q = torch.randn(batch_size, h, d, device=device).bfloat16()
                refs = []
                for request, (where, n) in enumerate(zip(locs, lengths)):
                    raw = torch.empty(n, h, d, device=device, dtype=torch.bfloat16)
                    values = torch.empty_like(raw)
                    for head in range(h):
                        for channel in range(d):
                            quant = where < slots
                            qi = where.clamp(max=slots - 1)
                            hi = (where - slots).clamp(min=0)
                            raw[:, head, channel] = torch.where(quant,
                                dec[3, head, channel // 4, idx[qi, head, channel // 4].long(), channel % 4].bfloat16(),
                                hp[hi, head, channel])
                            values[:, head, channel] = torch.where(quant,
                                pool.v_book.cb16[3, head, channel // 4, vidx[qi, head, channel // 4].long(), channel % 4].bfloat16(),
                                hv[hi, head, channel])
                    cos, sin = rope[:n].bfloat16().chunk(2, -1)
                    cos, sin = torch.cat((cos, cos), -1)[:, None], torch.cat((sin, sin), -1)[:, None]
                    rotated = raw * cos + torch.cat((-raw[..., d // 2:], raw[..., :d // 2]), -1) * sin
                    scores = (q[request, :, None] @ rotated.permute(1, 2, 0)) * (d ** -0.5)
                    probabilities = torch.softmax(scores.float(), -1).bfloat16()
                    refs.append((probabilities @ values.transpose(0, 1)).squeeze(1))
                expected = torch.stack(refs)
                out = self.attention.loopvq_decode_attention(q, pool, 3, ptr, loc, rope, d ** -0.5)
                worst = max(worst, float((out.float() - expected.float()).abs().max()))
                torch.testing.assert_close(out, expected, atol=0.0078125, rtol=0.015625)
                fixed = torch.empty_like(q)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.attention.loopvq_decode_attention(q, pool, 3, ptr, loc, rope, d ** -0.5, out=fixed)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.attention.loopvq_decode_attention(q, pool, 3, ptr, loc, rope, d ** -0.5, out=fixed)
                q.mul_(0.75)
                graph.replay()
                eager = self.attention.loopvq_decode_attention(q, pool, 3, ptr, loc, rope, d ** -0.5)
                torch.testing.assert_close(fixed, eager, atol=0, rtol=0)
        print(f"synthetic mixed attention worst_abs_error={worst:.8g}; graph replay matched eager")


if __name__ == "__main__":
    unittest.main(verbosity=2)
