import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serving_cohort import Journal, UniqueCohort, context_stop
from summarize_pd_matrix import detailed_accounting
from run_swe_top80 import CASES, command
from run_pd_matrix import continuous_warmup


class Cohort(unittest.IsolatedAsyncioTestCase):
    async def test_unique_filler_keeps_slot_busy_until_fixed_primary_finishes(self):
        release = asyncio.Event()
        events = []
        async def execute(case, instance):
            if case["case_id"] == "0":
                await release.wait()
            if case["case_id"] == "5":
                release.set()
                await asyncio.Event().wait()
            return "all_turns_completed"
        cohort = await UniqueCohort([dict(case_id=str(i)) for i in range(8)], 2, 4, execute, events.append).run()
        self.assertEqual({c["case"]["case_id"] for c in cohort.completed}, {"0", "1", "2", "3"})
        self.assertEqual(len(cohort.instances), 6)
        self.assertEqual(cohort.instances[-1]["status"], "cutoff_cancelled")
        self.assertEqual([i["filler"] for i in cohort.instances], [False]*4 + [True]*2)
        active = set()
        for event in events:
            if event["kind"] == "task_start":
                self.assertNotIn(event["case_id"], active)
                active.add(event["case_id"])
                self.assertLessEqual(len(active), 2)
            elif event["kind"] == "task_end":
                active.remove(event["case_id"])
        self.assertEqual(len(cohort.round_ends), 2)

    async def test_failure_does_not_count_as_task_completion(self):
        async def fail(case, instance):
            return "timeout"
        c = await UniqueCohort([dict(case_id=str(i)) for i in range(4)], 2, 4, fail, lambda e: None).run()
        self.assertEqual(c.completed, [])
        self.assertEqual(c.failure["status"], "timeout")


class Boundaries(unittest.TestCase):
    def test_early_warmup_completion_starts_new_pass_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def warmup(args, target):
                self.assertTrue(target.is_dir())
                (target / "usage.json").write_text("{}")
                if target.name.endswith("0002"):
                    raise RuntimeError("parent budget expired")
            with patch("run_pd_matrix.warmup", side_effect=warmup):
                with self.assertRaisesRegex(RuntimeError, "parent budget expired"):
                    continuous_warmup(None, root)
            self.assertEqual(len(list(root.glob("warmup-pass-*/usage.json"))), 3)

    def test_context_uses_active_and_absolute_position_not_full_history(self):
        self.assertIsNone(context_stop(180000, 100, dict(active_tokens=80000, position_tokens=90000), 131072))
        self.assertEqual(context_stop(180000, 100, dict(active_tokens=131073, position_tokens=100000), 131072), "active_context_limit_reached")
        self.assertEqual(context_stop(131000, 100, {}, 131072), "model_context_budget_reached")
        self.assertIsNone(context_stop(130972, 100, {}, 131072))

    def test_journal_snapshots_every_raw_sse_and_preserves_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = dict(kind="raw_sse", sequence=1, data='{"usage":{"completion_tokens":4}}')
            with Journal(tmp) as journal:
                journal.emit(event)
                event["data"] = "changed"
                journal.emit(dict(kind="raw_sse", sequence=2, data="[DONE]"))
                journal.emit(dict(kind="turn_end", usage=dict(completion_tokens=4)))
            rows = [json.loads(x) for x in (Path(tmp)/"sse.jsonl").read_text().splitlines()]
            self.assertEqual(json.loads(rows[0]["data"])["usage"]["completion_tokens"], 4)
            self.assertEqual(rows[1]["data"], "[DONE]")
            self.assertEqual(json.loads((Path(tmp)/"events.jsonl").read_text())["usage"], dict(completion_tokens=4))

    def test_partial_usage_and_forward_pairs_are_counted_once(self):
        def row(output, success, done):
            return dict(start_time=1, raw_done_time=done, last_received_usage=dict(prompt_tokens=100, completion_tokens=output),
                        usage_received_perf=3, filler=not success, success=success, latency=2 if done else None,
                        output_len=output, usage=dict(completion_tokens=output), cleanup_time_s=1, ttft=1,
                        tpot_s=1 if success else None)
        result = dict(start=0, cutoff=10, turns=[row(4, True, 3), row(2, False, None)])
        method = SimpleNamespace(stats=lambda values: dict(count=len(values), values=values))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in ("prefill", "decode"):
                step = dict(pid=1, start_perf=2, cpu_ms=1, prefill_tokens=10, decode_tokens=1)
                (root/f"{stage}-forward.jsonl").write_text(json.dumps(step)+"\n"+json.dumps(dict(step, gpu_ms=2))+"\n")
            value = detailed_accounting(result, method, root)
        self.assertEqual(value["all_known"]["output_tokens"], 6)
        self.assertEqual(value["abnormal_known"]["output_tokens"], 2)
        self.assertEqual(value["user_latency_s"]["count"], 1)
        self.assertEqual(value["physical_forward_dispatch"]["prefill_tokens"], 20)

    def test_all_ten_commands_use_original_seed_and_common_settings(self):
        args = SimpleNamespace(server_repo="/repo", mini_root="/mini", source_launch="/launch", requests_path="/data", seed=42, port=41101)
        self.assertEqual(len(CASES), 10)
        for c, drop in CASES:
            cmd = command(args, "head", "/out", c, drop)
            self.assertEqual(cmd[cmd.index("--seed")+1], "42")
            self.assertEqual(cmd[cmd.index("--rounds")+1], "3")
            self.assertEqual(cmd[cmd.index("--max-running-requests")+1], "16")
            self.assertEqual("--drop" in cmd, drop)
            self.assertIn("--unique-cohort", cmd)


if __name__ == "__main__":
    unittest.main()
