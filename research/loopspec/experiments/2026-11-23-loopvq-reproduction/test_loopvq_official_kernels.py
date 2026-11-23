"""GPU comparison against the unchanged official modules, using final centroids.

Example in the isolated target CUDA environment:
  python test_loopvq_official_kernels.py --reference-root /path/to/LoopVQ \
      --bundle-dir /path/to/ouro_2p6b_think_loopvq8b_g16_lvgqe2e

This uses real exported codebooks, seeded Q/HP values and seeded uint8 codes.
It is an operator/graph test, not a model accuracy or throughput benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch
import triton

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--reference-root", type=Path, required=True)
PARSER.add_argument("--bundle-dir", type=Path, required=True)
ARGS = PARSER.parse_args()
TARGET = Path(__file__).resolve().parents[4] / "python/sglang"
REFERENCE = ARGS.reference_root / "python/sglang/srt/layers/attention/triton_ops"
UPSTREAM_SHA = "d5d58c83fe5bc9a4ce470ae96c69b55f085e2977"


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), allow_nan=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.kernels",
             "sglang.kernels.ops", "sglang.kernels.ops.loopvq"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
# Only platform/env discovery is supplied. Reference math comes from the
# original file, checked byte-for-byte against the requested upstream commit.
utils = types.ModuleType("sglang.srt.utils")
utils.is_hip = lambda: False
utils.get_bool_env_var = lambda name, default="false": os.environ.get(name, default).lower() in ("true", "1")
sys.modules[utils.__name__] = utils
load("sglang.srt.mem_cache.loopvq_pack", TARGET / "srt/mem_cache/loopvq_pack.py")
reader = load("_target_book_reader", TARGET / "srt/mem_cache/loopvq_codebook.py")
quant = load("_target_official_quant", TARGET / "kernels/ops/loopvq/official_quant_cuda.py")
hp = load("_target_official_hp", TARGET / "kernels/ops/loopvq/official_hp.py")
reduce = load("_target_official_reduce", TARGET / "kernels/ops/loopvq/official_reduce.py")
metadata = load("_target_official_metadata", TARGET / "kernels/ops/loopvq/official_metadata.py")


class OfficialKernelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This validation requires CUDA; it must not silently skip")
        torch.manual_seed(211)
        for filename in ("vqwide_g.py", "decode_attention.py"):
            path = REFERENCE / filename
            original = subprocess.check_output([
                "git", "-C", str(ARGS.reference_root), "show",
                f"{UPSTREAM_SHA}:python/sglang/srt/layers/attention/triton_ops/{filename}",
            ])
            if original != path.read_bytes():
                raise RuntimeError(f"Reference file changed from {UPSTREAM_SHA}: {path}")
            emit("reference_source", path=str(path), sha256=sha256(path), commit=UPSTREAM_SHA)
        # Freeze exactly the upstream defaults for plain G16 (not tuning).
        for key, value in {"R": 2, "CAP": 2048, "THR": 512, "UA1": 2,
                           "UAG": 1, "UC": 8, "PC": 1}.items():
            os.environ[f"SGLANG_VQWIDE_G_{key}"] = str(value)
        for key, value in {"BLOCK_N": 32, "BLOCK_H": 4, "NUM_WARPS": 4, "NUM_STAGES": 1}.items():
            os.environ[f"SGL_PREROPE_{key}"] = str(value)
        cls.ref_quant = load("_original_loopvq_grouped", REFERENCE / "vqwide_g.py")
        cls.ref_decode = load("_original_loopvq_decode", REFERENCE / "decode_attention.py")
        quant.prepare_official_quant(device="cuda", max_context_len=40960)
        cls.extension = cls.ref_quant.ext(32, 256, 128, 32, 256)
        cls.books = {}
        for kind, filename in (("k", "codebook.pt"), ("v", "vq_v_codebook.pt")):
            path = ARGS.bundle_dir / filename
            emit("real_bundle", kind=kind, path=str(path), sha256=sha256(path))
            book = reader.load_loopvq_codebook(path, layer_num=192, head_num=16,
                                               head_dim=128, device="cpu")
            cls.books[kind] = [book.cb16[slot].cuda() for slot in (0, 95, 191)]
            del book
        cls.unit = torch.ones(1, 1, 1, device="cuda", dtype=torch.float16)
        cls.dummy_f = torch.zeros(1, 1, device="cuda", dtype=torch.float32)
        cls.dummy_i = torch.zeros(1, 1, device="cuda", dtype=torch.int64)
        freq = torch.outer(torch.arange(40960, device="cuda"),
                           10000 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128))
        cls.rope = torch.cat((freq.cos(), freq.sin()), -1)
        emit("environment", torch=torch.__version__, triton=triton.__version__,
             cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), context=40960)

    def test_capacity_rejection(self):
        with self.assertRaises(ValueError):
            quant.prepare_official_quant(device="cuda", max_context_len=49153)
        with self.assertRaises(ValueError):
            quant.prepare_official_quant(device="cuda", max_context_len=40960, max_splits=8)

    def test_metadata_cpu_oracle_and_changed_length_graph(self):
        """Check canonical layouts independently of the attention kernels.

        Quant lengths deliberately include non-page-aligned 13 and 1025. A
        dummy row has one loc=0 in the source but zero effective tier lengths.
        Graph replay changes lengths, prefix sums and physical page ordering.
        """
        prefix, recent, offset, cap = 64, 256, 1_000_000, 40960
        sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        for b in (1, 2, 3, 4, 5, 8, 15, 16):
            group = 1 if b == 1 else 2
            meta = metadata.OfficialMetadata(
                max_batch=b, max_context_len=cap, prefix_tokens=prefix,
                recent_tokens=recent, hp_offset=offset, device="cuda", sm_count=sm)
            ptr = torch.empty(b + 1, device="cuda", dtype=torch.int32)
            loc = torch.empty(b * cap, device="cuda", dtype=torch.int64)

            def install(phase):
                rows, qrows, hrows, lengths = [], [], [], []
                choices = (0, 1, 13, 1025, 8192, 35000, -1, 320)
                for i in range(b):
                    n = choices[(i + phase) % len(choices)]
                    if n < 0:
                        rows.append([0])
                        qrows.append([])
                        hrows.append([])
                        lengths.append(0)
                        continue
                    hn = 17 if n == 0 else prefix + recent + (i + phase) % 8
                    # Unique non-contiguous locations make order mistakes
                    # observable, while keeping all HP locations local.
                    qs = [8 + 3 * j + i for j in range(n)]
                    hs = [8 + 2 * j + i for j in range(hn)]
                    if phase:
                        qs.reverse()
                        hs.reverse()
                    p = min(hn, prefix)
                    rows.append([offset + v for v in hs[:p]] + qs +
                                [offset + v for v in hs[p:]])
                    qrows.append(qs)
                    hrows.append(hs)
                    lengths.append(n + hn)
                source_ptr = [0]
                qptr, hptr = [0], [0]
                for row, qs, hs in zip(rows, qrows, hrows):
                    source_ptr.append(source_ptr[-1] + len(row))
                    qptr.append(qptr[-1] + len(qs))
                    hptr.append(hptr[-1] + len(hs))
                flat = [v for row in rows for v in row]
                ptr.copy_(torch.tensor(source_ptr, device="cuda", dtype=torch.int32))
                loc[:len(flat)].copy_(torch.tensor(flat, device="cuda", dtype=torch.int64))
                qlens = [len(row) for row in qrows]
                expected_splits = []
                occupancy = 2 * sm // (((b + group - 1) // group) * 16)
                for start in range(0, b, group):
                    longest = max(qlens[start:start + group])
                    count = max(1, (longest + 2048 // group - 1) // (2048 // group))
                    if occupancy > 1:
                        count = max(count, min(occupancy, max(1, (longest + 255) // 256)))
                    expected_splits.extend([count] * min(group, b - start))
                return dict(qptr=qptr, hptr=hptr, qlens=qlens,
                            hlens=[len(row) for row in hrows], lengths=lengths,
                            qloc=[v for row in qrows for v in row],
                            hloc=[v for row in hrows for v in row], splits=expected_splits)

            def check(expected, phase):
                pairs = ((meta.quant_indptr, "qptr"), (meta.hp_indptr, "hptr"),
                         (meta.quant_lens, "qlens"), (meta.hp_lens, "hlens"),
                         (meta.full_seq_len, "lengths"), (meta.quant_splits, "splits"))
                for value, key in pairs:
                    self.assertEqual(value.tolist(), expected[key], (b, phase, key))
                self.assertEqual(meta.hp_splits.tolist(), [12] * b)
                self.assertEqual(meta.quant_indices[:len(expected["qloc"])].tolist(), expected["qloc"])
                self.assertEqual(meta.hp_indices[:len(expected["hloc"])].tolist(), expected["hloc"])
                emit("metadata", batch=b, phase=phase, quant_lengths=expected["qlens"],
                     full_lengths=expected["lengths"], exact=True)

            expected = install(0)
            meta.update(ptr, loc, b)
            check(expected, "eager")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    meta.update(ptr, loc, b)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                meta.update(ptr, loc, b)
            expected = install(1)
            graph.replay()
            check(expected, "graph_changed_lengths_and_locations")

    def run_case(self, quant_lengths, *, group_size, slot_index, name):
        b = len(quant_lengths)
        h, d, prefix, recent = 16, 128, 64, 256
        # Negative length denotes a graph-padding row with zero effective length.
        hp_lengths = [0 if n < 0 else (17 if n == 0 else prefix + recent + i % 8)
                      for i, n in enumerate(quant_lengths)]
        qlens = [max(n, 0) for n in quant_lengths]
        full_lengths = [q + p for q, p in zip(qlens, hp_lengths)]
        q = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
        qptr = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
        hptr = torch.tensor([0] + list(torch.tensor(hp_lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
        qn, hn = sum(qlens), sum(hp_lengths)
        qloc = (torch.randperm(qn, device="cuda") + 8).int()
        hloc = (torch.randperm(hn, device="cuda") + 8).int()
        ki = torch.randint(0, 256, (qn + 16, h, 32), device="cuda", dtype=torch.uint8)
        vi = torch.randint(0, 256, ki.shape, device="cuda", dtype=torch.uint8)
        hk = torch.randn(hn + 16, h, d, device="cuda", dtype=torch.bfloat16)
        hv = torch.randn_like(hk)
        full = torch.tensor(full_lengths, device="cuda", dtype=torch.int32)
        splits = []
        sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        nb = (b + group_size - 1) // group_size
        occupancy = 2 * sm // (nb * h)
        for i in range(0, b, group_size):
            longest = max(qlens[i:i + group_size])
            count = max(1, (longest + 2048 // group_size - 1) // (2048 // group_size))
            if occupancy > 1:
                count = max(count, min(occupancy, max(1, (longest + 255) // 256)))
            self.assertLessEqual(count, 48)
            splits.extend([count] * min(group_size, b - i))
        qs = torch.tensor(splits, device="cuda", dtype=torch.int32)
        hs = torch.full((b,), 12, device="cuda", dtype=torch.int32)
        kcb, vcb = self.books["k"][slot_index], self.books["v"][slot_index]
        actual_o = torch.full((b, h, 60, d), float("nan"), device="cuda")
        actual_l = torch.full((b, h, 60), -float("inf"), device="cuda")
        reference_o, reference_l = torch.full_like(actual_o, float("nan")), torch.full_like(actual_l, -float("inf"))
        actual = torch.empty_like(q)
        expected = torch.empty_like(q)

        def target_forward():
            actual_l.fill_(-float("inf"))
            hp.official_hp_attention(q, hk, hv, self.rope, full, actual_o[:, :, :12], actual_l[:, :, :12],
                                     hptr, hloc, hs, max_splits=12, sm_scale=d ** -0.5, prefix_tokens=prefix)
            quant.official_quant_attention(q, ki, kcb, vi, vcb, qptr, qloc, qs,
                                            actual_o[:, :, 12:], actual_l[:, :, 12:], self.rope, d ** -0.5,
                                            pos_offset=prefix, group_size=group_size)
            reduce.official_reduce(actual_o, actual_l, actual)

        def reference_forward():
            reference_l.fill_(-float("inf"))
            self.ref_decode._decode_grouped_att_m_fwd_prerope(
                q, hk, hv, self.rope, full, reference_o[:, :, :12], reference_l[:, :, :12],
                hptr, hloc, hs, 12, d ** -0.5, 0.0, prefix_tokens=prefix)
            self.extension.vqwide_v2_cuda(
                q, ki, kcb, vi, vcb, self.unit.expand(qn + 16, h, 2), self.unit.expand(qn + 16, h, 2),
                self.rope, self.dummy_i, self.dummy_f, qptr, qloc, qs,
                reference_o[:, :, 12:], reference_l[:, :, 12:], 48, d ** -0.5, prefix,
                17 if group_size == 2 else 1)
            self.ref_decode._unified_stage2(reference_o, reference_l, expected, 60)

        def check(label):
            finite = torch.isfinite(reference_l)
            torch.testing.assert_close(torch.isfinite(actual_l), finite)
            torch.testing.assert_close(actual_l[finite], reference_l[finite], atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(actual_o[finite], reference_o[finite], atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            err = float((actual_o[finite] - reference_o[finite]).abs().max()) if bool(finite.any()) else 0.0
            emit("case", name=name, phase=label, batch=b, group=group_size,
                 lengths=full_lengths, quant_lengths=qlens, splits=splits,
                 max_split_output_abs_error=err,
                 merged_bf16_exact=True, real_bundle_slot=(0, 95, 191)[slot_index])

        target_forward()
        reference_forward()
        check("eager")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                target_forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            target_forward()
        q.mul_(0.75)
        if qn:
            qloc.copy_(qloc.flip(0))
        graph.replay()
        reference_forward()
        check("graph_changed_query_and_locations")

    def test_real_centroids_mixed_attention_and_graph(self):
        for i, b in enumerate((1, 2, 3, 4, 5, 8, 15, 16)):
            with self.subTest(batch=b):
                boundary = (1, 1023, 1024, 1025, 2047, 2048, 2049, 31)
                self.run_case([boundary[j % len(boundary)] for j in range(b)],
                              group_size=1 if b == 1 else 2,
                              slot_index=i % 3, name=f"split-boundary-b{b}")
        for length in (2048, 8192, 35000):
            for b in (1, 2, 16):
                with self.subTest(length=length, batch=b):
                    self.run_case([length - j % 8 for j in range(b)],
                                  group_size=1 if b == 1 else 2,
                                  slot_index=(length // 2048) % 3,
                                  name=f"long-{length}-b{b}")
        for active, padded in ((3, 4), (5, 8), (15, 16)):
            with self.subTest(active=active, padded=padded):
                self.run_case([2051 - j for j in range(active)] + [-1] * (padded - active),
                              group_size=2, slot_index=2, name=f"padding-{active}-to-{padded}")
        self.run_case([0, 0, 0, -1], group_size=2, slot_index=0, name="empty-quant")
        self.run_case([-1, -1], group_size=2, slot_index=1, name="all-padding")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
