"""Role/depth ownership and deferred-free contracts, runnable on CPU."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.timeline_pool import TimelineSlots, TimelineTokenAllocator


class TestTimeline(unittest.TestCase):
    def test_depths_roles_and_rejected_branch(self):
        slots = TimelineSlots(8, 4, {"pre": 1, "core": 1, "post": 1}, "cpu")
        # Parent prefix is full depth. A child has only a shallow core.
        slots.ensure([1], "pre", 0)
        for d in range(4):
            slots.ensure([1], "core", d)
            slots.ensure([1], "post", d)
        slots.ensure([2], "pre", 0)
        slots.ensure([2], "core", 0)
        self.assertEqual(slots.live_slots, 11)
        self.assertEqual(len(set(slots.maps["core"][:, 1].tolist())), 4)
        self.assertEqual(slots.maps["core"][3, 2], 0)
        before = slots.maps["core"][:, 1].clone()
        slots.release([2])
        self.assertEqual(slots.live_slots, 9)
        torch.testing.assert_close(before, slots.maps["core"][:, 1])
        slots.ensure([2], "core", 0)
        slots.ensure([2], "core", 0)
        self.assertEqual(slots.live_slots, 10)
        slots.release([1, 2])
        self.assertEqual(slots.live_slots, 0)
        for role, ids in slots.free_ids.items():
            self.assertEqual(len(ids), len(set(ids)), role)

    def test_scheduler_deferred_free_owns_timelines_until_group_end(self):
        slots = TimelineSlots(8, 4, {"pre": 0, "core": 2, "post": 0}, "cpu")
        allocator = TimelineTokenAllocator(
            8, torch.bfloat16, "cpu", SimpleNamespace(timeline=slots), False
        )
        ids = allocator.alloc(2)
        slots.ensure(ids, "core", 0)
        allocator.free_group_begin()
        allocator.free(ids)
        self.assertEqual(slots.live_slots, 2)
        allocator.free_group_end()
        self.assertEqual(slots.live_slots, 0)
        self.assertEqual(allocator.available_size(), 8)
        allocator.clear()
        self.assertEqual(len(slots.free_ids["core"]), 32)


if __name__ == "__main__":
    unittest.main()
