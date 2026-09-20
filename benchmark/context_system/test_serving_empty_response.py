"""CPU replay of full-budget native SSE with no visible assistant delta."""
import copy
import json
import os
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from test_serving import Transport, load_method


class EmptyResponse(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("MINI_BENCHMARK_ROOT") or Path(__file__).resolve().parents[3] / "mini-sglang"
        cls.method = load_method(root)

    async def replay(self, *, done=True, finish="length", tokens=40, usage=True, text="", error=None):
        events = [dict(choices=[dict(delta=dict(role="assistant", content=text), finish_reason=None)]),
                  dict(choices=[dict(delta=dict(reasoning_content=None), finish_reason=finish)])]
        if error:
            events.append(dict(error=error))
        if usage:
            events.append(dict(choices=[], usage=dict(prompt_tokens=7708, completion_tokens=tokens,
                                                      total_tokens=7708 + tokens, reasoning_tokens=36)))
        frames = [json.dumps(e) for e in events] + (["[DONE]"] if done else [])
        async def content():
            for frame in frames:
                yield ("data: " + frame + "\n").encode()
                yield b"\n"
        @asynccontextmanager
        async def post(url, **kwargs):
            yield SimpleNamespace(status=200, content=content(), text=None)
        return await Transport(SimpleNamespace(post=post), self.method).request("/test", dict(max_tokens=40))

    async def test_complete_empty_response_preserves_output_and_missing_timing(self):
        row = await self.replay()
        self.assertTrue(row["success"])
        self.assertTrue(row["completion_without_visible_delta"])
        self.assertEqual(row["assistant"], dict(role="assistant", content=""))
        self.assertEqual(row["usage"]["completion_tokens"], 40)
        self.assertIsNone(row["ttft"])
        self.assertIsNone(row["tpot_s"])
        self.assertEqual(row["latency"], row["raw_done_time"] - row["start_time"])
        self.assertGreaterEqual(row["cleanup_time_s"], 0)

    async def test_missing_completion_evidence_and_server_error_still_fail(self):
        for args in (dict(done=False), dict(finish=None), dict(tokens=39), dict(usage=False),
                     dict(error="engine failed")):
            with self.subTest(args=args):
                self.assertFalse((await self.replay(**args))["success"])

    async def test_summary_counts_empty_work_but_excludes_unknown_ttft(self):
        empty, visible = await self.replay(), await self.replay(text="actual answer")
        for row in (empty, visible):
            row.update(filler=False)
        before = copy.deepcopy([empty, visible])
        summary = self.method.summary([empty, visible], empty["start_time"] - 1, visible["end_time"] + 1)
        logical = summary["sglang_logical"]
        self.assertEqual(logical["completed"], 2)
        self.assertEqual(logical["total_output"], 80)
        self.assertEqual(logical["total_input"], 15416)
        self.assertEqual(logical["missing_ttft_requests"], 1)
        self.assertEqual(logical["mean_ttft_ms"], visible["ttft"] * 1000)
        self.assertAlmostEqual(logical["mean_e2e_latency_ms"], (empty["latency"] + visible["latency"]) * 500)
        self.assertIsNone(logical["max_output_tokens_per_s"])
        self.assertEqual([empty, visible], before)
        only_empty = self.method.summary([empty], empty["start_time"] - 1, empty["end_time"] + 1)
        self.assertIsNone(only_empty["sglang_logical"]["mean_ttft_ms"])
        self.assertIsNone(only_empty["sglang_logical"]["mean_tpot_ms"])


if __name__ == "__main__":
    unittest.main()
