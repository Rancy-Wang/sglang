"""CPU checks for the physical counting boundary; no model imports."""

from types import SimpleNamespace as NS
import unittest

from launch_pd_counted import batch_work
from summarize_pd_matrix import join_counts


def batch(mode, lengths):
    return NS(forward_mode=NS(**{f"is_{m}": (lambda m=m: m == mode) for m in ("prebuilt", "idle", "decode", "extend")}),
              reqs=[NS(time_stats=NS()) for _ in lengths], extend_lens=lengths)


class Counters(unittest.TestCase):
    def test_real_work_excludes_prebuilt(self):
        self.assertEqual(batch_work(batch("prebuilt", [70000])), [])
        self.assertEqual([x[1:] for x in batch_work(batch("extend", [8192, 731]))], [(8192, 0), (731, 0)])
        self.assertEqual([x[1:] for x in batch_work(batch("decode", [70000, 90000]))], [(0, 1), (0, 1)])

    def test_join_not_tp_sum_and_failure_not_counted(self):
        r = {"turns": [{"success": True, "server_metrics": {"pd_bootstrap_room": 7}}, {"success": False}]}
        counts = {("7", "prefill"): {0: {"pf": 8923, "decode": 0}, 1: {"pf": 8923, "decode": 0}},
                  ("7", "decode"): {0: {"pf": 0, "decode": 63}}}
        joined = join_counts(r, counts, 2)
        self.assertEqual(joined["turns"][0]["server_metrics"]["prefill_compute_tokens"], 8923)
        self.assertEqual(joined["turns"][0]["server_metrics"]["decode_compute_tokens"], 63)
        self.assertNotIn("prefill_compute_tokens", r["turns"][0]["server_metrics"])
        counts[("7", "prefill")][1]["pf"] = 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            join_counts(r, counts, 2)

    def test_missing_data_must_not_become_zero(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            join_counts({"turns": [{"success": True, "server_metrics": {"pd_bootstrap_room": 7}}]}, {}, 2)


if __name__ == "__main__":
    unittest.main()
