"""CPU checks for pairing, admission and overlapping TP timeline accounting."""

import copy
from types import SimpleNamespace
import unittest

from analyze_decode_breakdown import exclusive_intervals, summarize_steps
from build_decode_breakdown_fixture import PLAN, digest, paired_payloads
from profile_decode_breakdown import cohort_ready


class AccountingTests(unittest.TestCase):
    def test_overlaps_are_not_counted_twice(self):
        events = [dict(start=0, end=10, category="attention"),
                  dict(start=2, end=8, category="attention"),
                  dict(start=5, end=15, category="communication")]
        self.assertEqual(exclusive_intervals(events), {
            "attention": 5, "overlap:attention+communication": 5, "communication": 5})

    def test_wait_for_all_kv_ready(self):
        r = lambda name: SimpleNamespace(rid=name)
        self.assertFalse(cohort_ready([r("a")], [], ["a", "b"]))
        self.assertTrue(cohort_ready([r("a"), r("b")], [], ["a", "b"]))
        with self.assertRaises(ValueError):
            cohort_ready([r("a"), r("a")], [], ["a", "b"])
        with self.assertRaises(ValueError):
            cohort_ready([r("foreign")], [], ["a", "b"])

    def test_tp_replicas_not_summed_and_missing_rejected(self):
        rows = [dict(kind="decode_step", measured=True, rank=r, step=s,
                     gpu_ms=10 + r, graph_batch_size=8, batch_size=8, request_ids=["a"])
                for r in range(4) for s in range(3)]
        summary = summarize_steps(rows)
        self.assertEqual(summary["max_rank_mean_gpu_ms"], 13)
        with self.assertRaises(ValueError):
            summarize_steps(rows[:-1])
        with self.assertRaises(ValueError):
            summarize_steps(rows + rows[:1])

    def test_identical_prefixes_and_only_context_controls_differ(self):
        ids, tokens, messages = [1, 2, 3], [4, 5, 6, 7], [{"role": "user", "content": "x"}]
        fixture = dict(plan_id=PLAN, batch_size=1, require_repos=False, max_new_tokens=2,
            model="test", tools=[], template_kwargs={}, requests=[dict(
                rid="a", case_id="case", messages=messages, messages_sha256=digest(messages),
                input_ids=ids, input_sha256=digest(ids), replay_tokens=tokens,
                replay_sha256=digest(tokens), drop_state=dict(active_tokens=2,
                    drop_message={"2": [0]}, reposition=[]))])
        nd = paired_payloads(fixture, "no_drop")[0]
        drop = paired_payloads(fixture, "drop")[0]
        self.assertEqual({k: v for k, v in drop.items() if k not in ("drop_message", "reposition")}, nd)
        changed = copy.deepcopy(fixture)
        changed["requests"][0]["input_ids"][0] = 99
        with self.assertRaises(AssertionError):
            paired_payloads(changed, "drop")


if __name__ == "__main__":
    unittest.main()
