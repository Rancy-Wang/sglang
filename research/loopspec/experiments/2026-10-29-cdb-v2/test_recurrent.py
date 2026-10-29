"""Strict addressing contracts, independent of floating-point tolerances."""

from types import SimpleNamespace
import unittest

import torch

from sglang.srt.mem_cache.timeline_pool import MixedTimelineRuntime, TimelineSlots
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class TestMixedTimeline(unittest.TestCase):
    def setUp(self):
        self.slots = TimelineSlots(8, 4, {"pre": 1, "core": 1, "post": 1}, "cpu")
        for role, table in self.slots.maps.items():
            for depth in range(len(table)):
                self.slots.ensure([1, 2, 3, 4], role, depth)
        runtime = object.__new__(MixedTimelineRuntime)
        runtime.pool = SimpleNamespace(timeline=self.slots)
        runtime.source = torch.tensor([[0, 0, 0, 0], [1, 2, 7, 7], [3, 4, 7, 7]])
        runtime.max_rows = 4
        runtime.rows = torch.arange(1, 5)
        runtime.columns = torch.arange(4)
        runtime.views = {}
        for role in self.slots.maps:
            translator = SimpleNamespace(bind_own_plan=lambda batch: None)
            backend = SimpleNamespace(init_forward_metadata=lambda batch: None)
            runtime.views[role] = torch.zeros(5, 4, dtype=torch.long), translator, backend
        self.runtime = runtime

    def batch(self, *, extend=False):
        return SimpleNamespace(
            batch_size=4, forward_mode=ForwardMode.EXTEND if extend else ForwardMode.DECODE,
            req_pool_indices=torch.tensor([1, 2, 1, 0]), seq_lens=torch.tensor([2, 1, 2, 1]),
            out_cache_loc=torch.tensor([2, 3, 2, 0]),
        )

    def test_duplicate_request_at_different_depth_and_padding(self):
        original = self.batch()
        mapped, _ = self.runtime.target_batch(original, "core", torch.tensor([0, 3, 2, 0]))
        table = self.runtime.views["core"][0]
        planes = self.slots.maps["core"]
        self.assertEqual(table[1].tolist(), [planes[0, 1], planes[0, 2], 0, 0])
        self.assertEqual(table[2].tolist(), [planes[3, 3], 0, 0, 0])
        self.assertEqual(table[3].tolist(), [planes[2, 1], planes[2, 2], 0, 0])
        self.assertEqual(table[4].tolist(), [0, 0, 0, 0])
        self.assertEqual(mapped.out_cache_loc.tolist(), [planes[0, 2], planes[3, 3], planes[2, 2], 0])
        self.assertEqual(mapped.req_pool_indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(original.req_pool_indices.tolist(), [1, 2, 1, 0])
        self.assertEqual(original.out_cache_loc.tolist(), [2, 3, 2, 0])

    def test_rebind_overwrites_stale_row_identity(self):
        batch = self.batch()
        self.runtime.target_batch(batch, "core", torch.tensor([0, 1, 2, 3]))
        batch.req_pool_indices = torch.tensor([2, 1, 2, 0])
        batch.seq_lens = torch.tensor([1, 2, 1, 1])
        batch.out_cache_loc = torch.tensor([3, 2, 3, 0])
        self.runtime.target_batch(batch, "core", torch.tensor([3, 2, 1, 0]))
        table = self.runtime.views["core"][0]
        self.assertEqual(table[1].tolist(), [self.slots.maps["core"][3, 3], 0, 0, 0])
        self.assertEqual(table[2, :2].tolist(), self.slots.maps["core"][2, [1, 2]].tolist())

    def test_packed_extend_uses_token_owner_depth(self):
        batch = self.batch(extend=True)
        batch.out_cache_loc = torch.tensor([1, 2, 3, 2, 0])
        mapped, _ = self.runtime.target_batch(batch, "core", torch.tensor([0, 3, 2, 0]),
                                               torch.tensor([0, 0, 1, 2, 3]))
        planes = self.slots.maps["core"]
        self.assertEqual(mapped.out_cache_loc.tolist(), [planes[0, 1], planes[0, 2], planes[3, 3], planes[2, 2], 0])
        mapped, _ = self.runtime.target_batch(batch, "pre", torch.tensor([0, 3, 2, 0]),
                                               torch.tensor([0, 0, 1, 2, 3]))
        self.assertEqual(mapped.out_cache_loc.tolist(), self.slots.maps["pre"][0, [1, 2, 3, 2, 0]].tolist())


if __name__ == "__main__":
    unittest.main()
