"""CPU checks for the physical counting boundary; no model imports."""

from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
import unittest

from launch_pd_counted import batch_work
from summarize_pd_matrix import join_counts, read_counts, rebuild_summaries
from run_pd_matrix import overlap_command, predecessor_ready
from test_serving import parser as client_parser


def batch(mode, lengths):
    return NS(forward_mode=NS(**{f"is_{m}": (lambda m=m: m == mode) for m in ("prebuilt", "idle", "decode", "extend")}),
              reqs=[NS(time_stats=NS()) for _ in lengths], extend_lens=lengths)


class Counters(unittest.TestCase):
    def test_client_accepts_all_approved_matrix_cases(self):
        common = ["--mini-root", "/mini", "--requests-path", "/tasks.jsonl",
                  "--output-dir", "/results", "--model", "/model", "--tokenizer", "/model"]
        for c in (1, 2, 4, 8, 32):
            for drop in (False, True):
                with self.subTest(concurrency=c, drop=drop):
                    args = client_parser().parse_args(common + [
                        "--concurrency", str(c), "--num-tasks", str(3*c),
                        "--prefill-url", "http://127.0.0.1:32041/v1/chat/completions",
                        "--url", "http://127.0.0.1:32042/v1/chat/completions",
                    ] + (["--drop"] if drop else []))
                    self.assertEqual((args.concurrency, args.num_tasks, args.drop), (c, 3*c, drop))

    def test_overlap_plan_removes_capacity_cap_and_allows_c8(self):
        original = ["python", "launch.py", "--max-total-tokens", "262144",
                    "--mem-fraction-static", "0.84", "--max-running-requests", "4",
                    "--disable-overlap-schedule", "--page-size", "1",
                    "--cuda-graph-config", "old"]
        command = overlap_command(original)
        self.assertNotIn("--max-total-tokens", command)
        self.assertNotIn("--disable-overlap-schedule", command)
        self.assertEqual(command[command.index("--mem-fraction-static") + 1], "0.9")
        self.assertEqual(command[command.index("--max-running-requests") + 1], "8")
        self.assertEqual(command, overlap_command(command))
        self.assertIn("262144", original)

    def test_c32_admits_and_captures_full_batch(self):
        command = overlap_command(["python", "launch.py"], 32)
        self.assertEqual(command[command.index("--max-running-requests") + 1], "32")
        graph = json.loads(command[command.index("--cuda-graph-config") + 1])
        self.assertEqual(graph["decode"], {"bs": [1, 2, 4, 8, 16, 32], "max_bs": 32})

    def test_handoff_requires_valid_result_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("run_pd_matrix.os.kill"):
                self.assertFalse(predecessor_ready(root, 123))
            with patch("run_pd_matrix.os.kill", side_effect=ProcessLookupError):
                with self.assertRaisesRegex(RuntimeError, "without a valid outcome"):
                    predecessor_ready(root, 123)
                (root / "outcome.json").write_text('{"valid":true}')
                with patch("run_pd_matrix.gpu_free", return_value=False):
                    self.assertFalse(predecessor_ready(root, 123))
                with patch("run_pd_matrix.gpu_free", return_value=True):
                    self.assertTrue(predecessor_ready(root, 123))
            (root / "failure.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "Predecessor failed"):
                predecessor_ready(root, 123)

    def test_rounds_use_joined_counts_and_user_latency_excludes_filler(self):
        calls = []
        def summary(rows, start, end):
            tokens = sum(r["server_metrics"]["prefill_compute_tokens"] for r in rows
                         if start < r["end_time"] <= end)
            calls.append((start, end, tokens))
            return {"tokens": tokens}
        tasks = [dict(instance=i, case_id=i, start_time=start, end_time=end,
                      filler=filler, status="all_turns_completed")
                 for i, start, end, filler in [(0, 0, 5, False), (1, 4, 9, False), (2, 5, 8, True)]]
        rows = [dict(instance=i, end_time=end, server_metrics={"prefill_compute_tokens": n})
                for i, end, n in [(0, 5, 10), (1, 9, 20), (2, 8, 30)]]
        result = dict(start=0, cutoff=9, turns=rows, tasks=tasks,
                      round_ends=[5, 9])
        rebuild_summaries(result, NS(summary=summary, stats=lambda x: x))
        self.assertEqual(result["overall"]["tokens"], 60)
        self.assertEqual(result["rounds"][1]["window"]["tokens"], 50)
        self.assertEqual(result["rounds"][1]["cohort"]["tokens"], 20)
        self.assertEqual(result["user_latency"]["seconds"], [5, 5])
        self.assertEqual(len(result["user_latency"]["tasks"]), 2)

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

    def test_reused_health_room_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prefill.log").write_text("\n".join(
                '[TP0] ReqTimeStats(rid=HEALTH_CHECK_x, bootstrap_room=0): '
                'pd_matrix_compute={"pf": ' + str(n) + ', "decode": 0}, r2_pd_timing={}'
                for n in (1, 2)))
            (root / "decode.log").write_text("")
            self.assertEqual(read_counts(root, 2), {})


if __name__ == "__main__":
    unittest.main()
