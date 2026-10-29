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


class TestRecurrentAdmission(unittest.TestCase):
    def test_admit_at_loop_boundary_and_do_not_reprepare_active_token(self):
        from unittest.mock import patch
        from sglang.srt.managers.schedule_batch import ScheduleBatch, NextBatchPlan
        from sglang.srt.managers.scheduler_components.recurrent import RecurrentScheduler

        class Request:
            def finished(self):
                return False

        a, b = Request(), Request()
        batches = [ScheduleBatch(reqs=[a], forward_mode=ForwardMode.EXTEND),
                   ScheduleBatch(reqs=[b], forward_mode=ForwardMode.EXTEND)]
        occupancy = []
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(
                recurrent_runner=object(), model=SimpleNamespace(config=SimpleNamespace(total_ut_steps=4)))),
            chunked_req=None,
        )

        def admit(view):
            occupancy.append(len(view.reqs))
            return NextBatchPlan(batch_to_run=batches.pop(0) if batches else None, running_batch=view)

        scheduler.get_new_batch_prefill = admit
        component = RecurrentScheduler(scheduler)
        running = ScheduleBatch(reqs=[])
        seen = []
        with patch('sglang.srt.managers.scheduler_components.recurrent.set_schedule_time_batch'):
            for _ in range(3):
                plan = component.next_batch(running)
                work = plan.batch_to_run.recurrent_plan[0]
                seen.append((work.batch.reqs[0], work.depth))
                work.depth += 1
        self.assertEqual(occupancy, [0, 1, 2])
        self.assertEqual(seen, [(a, 0), (a, 1), (b, 0)])
        self.assertEqual(len(component.pending), 2)
        self.assertEqual(component.pending[0].depth, 2)

class TestFixedPlanes(unittest.TestCase):
    def test_distinct_planes_and_stable_logical_reuse(self):
        from sglang.srt.mem_cache.timeline_pool import FixedDepthSlots
        planes = FixedDepthSlots(8, 4, {"pre": 1, "core": 1, "post": 0}, "cpu")
        core = planes.maps["core"]
        self.assertEqual(core[:, 0].tolist(), [0, 0, 0, 0])
        self.assertEqual(core[:, 3].tolist(), [3, 11, 19, 27])
        self.assertEqual(len(set(core[:, 1:].flatten().tolist())), 32)
        before = core.clone()
        planes.release(torch.tensor([3]))
        planes.ensure(torch.tensor([3]), "core", 2)
        planes.clear()
        self.assertTrue(torch.equal(before, core))
        self.assertEqual(planes.maps["pre"].shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
