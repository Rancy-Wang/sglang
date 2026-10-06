"""CPU-only checks of the client DONE boundary, with a delayed P response."""

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_serving import Transport


class ClientTiming(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, slow_prefill, fail_prefill=False, events=None):
        clock = NS(now=10.0)
        done_received = asyncio.Event()

        async def sse_events(content):
            async for item in content:
                if isinstance(item, bytes):
                    item = item.decode().strip()
                    if not item:
                        continue
                    item = item.removeprefix("data: ")
                yield item

        async def request(session, url, payload):
            start = clock.now
            success = False
            try:
                async with session.post(url, json=payload) as response:
                    async for item in sse_events(response.content):
                        success |= item == "[DONE]"
            except (RuntimeError, asyncio.CancelledError):
                success = False
            return dict(start_time=start, end_time=clock.now,
                        latency=clock.now-start, ttft=.5, output_len=4,
                        success=success)

        @asynccontextmanager
        async def post(url, **kwargs):
            async def response_json():
                if slow_prefill:
                    await done_received.wait()
                    clock.now = 20.0
                return {"error": "P failed"} if fail_prefill else {}

            async def response_content():
                # Ensure the P task runs before D emits its final event.
                await asyncio.sleep(0)
                clock.now = 10.5
                yield json.dumps({"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 10}})
                clock.now = 12.0
                done_received.set()
                yield "[DONE]"

            yield NS(status=500 if url == "P" and fail_prefill else 200,
                     json=response_json, content=response_content(), text=None)

        transport = Transport(NS(post=post), NS(request=request, sse_events=sse_events), "P", 123,
                              emit=events.append if events is not None else None)
        with patch("test_serving.time", NS(perf_counter=lambda: clock.now, time=lambda: clock.now)):
            return await transport.request("D", {"max_tokens": 4})

    async def test_raw_usage_and_done_arrival_recorded_before_p_cleanup(self):
        events = []
        row = await self.run_request(True, events=events)
        sse = [r for r in events if r["kind"] == "raw_sse"]
        self.assertEqual([r["sequence"] for r in sse], [1, 2])
        self.assertEqual(sse[-1]["received_perf"], 12)
        self.assertEqual(json.loads(sse[0]["data"])["usage"], row["last_received_usage"])
        self.assertEqual([r for r in events if r["kind"] == "prefill_response"][0]["received_perf"], 20)

    async def test_slow_prefill_is_cleanup_not_user_latency_or_tpot(self):
        row = await self.run_request(True)
        self.assertTrue(row["success"])
        self.assertEqual(row["raw_done_time"], 12.0)
        self.assertEqual(row["latency"], 2.0)
        self.assertEqual(row["tpot_s"], .5)
        self.assertEqual(row["cleanup_time_s"], 8.0)
        self.assertEqual(row["lifecycle_latency_s"], 10.0)
        self.assertEqual(row["end_time"], 20.0)

    async def test_prefill_already_complete_has_no_cleanup_wait(self):
        row = await self.run_request(False)
        self.assertTrue(row["success"])
        self.assertEqual(row["latency"], 2.0)
        self.assertEqual(row["cleanup_time_s"], 0.0)

    async def test_done_does_not_hide_prefill_failure(self):
        row = await self.run_request(True, fail_prefill=True)
        self.assertFalse(row["success"])
        self.assertEqual(row["raw_done_time"], 12.0)
        self.assertEqual(row["cleanup_time_s"], 8.0)
        self.assertIsNone(row["tpot_s"])


if __name__ == "__main__":
    unittest.main()
