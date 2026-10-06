"""CPU checks for pairing, admission and overlapping TP timeline accounting."""

import copy
import json
import sqlite3
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyze_decode_breakdown import exclusive_intervals, read_nsys, summarize_steps
from build_decode_breakdown_fixture import PLAN, digest, paired_payloads
from profile_decode_breakdown import cohort_ready, validate_completed_case


class AccountingTests(unittest.TestCase):
    def test_graph_clone_uses_creation_component_and_replay_step_per_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.sqlite"
            db = sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1,'Graph Node Creation'),(2,'gemm');
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
                CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER,globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER,nameId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,graphNodeId INTEGER,streamId INTEGER,deviceId INTEGER);
            """)
            for pid, category in ((1, "full_attention"), (2, "moe_experts")):
                tid = (pid << 24) | 1
                db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)", [
                    (0, 100, tid, "component:outer"),
                    (10, 30, tid, "component:" + category),
                    (200, 300, tid, "decode_step:128")])
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(210,220,?,7)", (tid,))
                db.executemany("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?,?,?,?,1)", [
                    (20, tid, 10, None), (40, tid, 11, 10), (50, tid, 12, 11)])
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(230,250,?,7,2,12,1,0)", (pid << 24,))
            db.commit()
            db.close()
            kernels = read_nsys(path)
            self.assertEqual([v["category"] for v in kernels], ["full_attention", "moe_experts"])
            self.assertTrue(all(v["step"] == 128 and v["attribution"] == "graph_node" for v in kernels))

    def test_resume_rejects_failed_or_different_arm(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "profile_decode_breakdown.validate_fixture", side_effect=lambda value: value):
            root = Path(tmp)
            fixture = dict(batch_size=8)
            (root / "fixture.json").write_text(json.dumps(fixture))
            case = dict(fixture=str(root / "fixture.json"), strategy="drop", profile=False)
            (root / "outcome.json").write_text('{"success": true}')
            (root / "launch.json").write_text(json.dumps(dict(
                runtime_head="cfe9a570751218eab8ae8890777998f489ab5e21", fixture_sha256=digest(fixture))))
            cfg = dict(strategy="drop", profile=False, warm_steps=128, measure_steps=512)
            for role in ("prefill", "decode"):
                (root / f"{role}-config.json").write_text(json.dumps(cfg))
            (root / "analysis").mkdir()
            (root / "analysis/summary.json").write_text(json.dumps(dict(ranks={
                str(r): dict(steps=512, batch_sizes=[8]) for r in range(4)})))
            self.assertEqual(validate_completed_case(root, case), str(root.resolve()))
            with self.assertRaises(ValueError):
                validate_completed_case(root, dict(case, strategy="no_drop"))
            (root / "outcome.json").write_text('{"success": false}')
            with self.assertRaises(ValueError):
                validate_completed_case(root, case)

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
