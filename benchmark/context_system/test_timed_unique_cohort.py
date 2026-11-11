"""Deadline semantics and accounting for bounded, unique replay cohorts."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from serving_cohort import UniqueCohort
from analyze_system_ablation import account

class TimedCohort(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_cancels_without_completing_tasks(self):
        async def execute(case, instance):
            await asyncio.Event().wait()
        c = await UniqueCohort([{'case_id': str(i)} for i in range(4)], 2, 4,
                              execute, lambda e: None, .01).run()
        self.assertEqual(c.stop_reason, 'measurement_time_limit')
        self.assertEqual(c.completed, [])
        self.assertIsNone(c.failure)
        self.assertEqual([x['status'] for x in c.instances], ['cutoff_cancelled'] * 2)
        self.assertTrue(all(t.done() for t in c.workers))

    async def test_completion_and_failure_keep_their_reason(self):
        for status, reason in [('all_turns_completed', 'primary_cohort_completed'),
                               ('context_stop', 'request_failure')]:
            async def execute(case, instance):
                return status
            c = await UniqueCohort([{'case_id': 'a'}], 1, 1, execute,
                                  lambda e: None, 60).run()
            self.assertEqual(c.stop_reason, reason)
            self.assertEqual(len(c.completed), int(status == 'all_turns_completed'))

    async def test_invalid_duration_rejected(self):
        for duration in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                UniqueCohort([{'case_id': 'a'}], 1, 1, None, None, duration)

class TimedAccounting(unittest.TestCase):
    def fixture(self, root):
        (root/'workload').mkdir()
        data = dict(valid=True, start=1, cutoff=11, stop_reason='measurement_time_limit',
                    tasks=[dict(instance=0, filler=False, status='all_turns_completed', end_time=5),
                           dict(instance=1, filler=False, status='cutoff_cancelled', end_time=12)],
                    turns=[dict(instance=0, turn=0, start_time=2, success=True, filler=False,
                                ttft=1, output_len=3, requested_max_tokens=3, usage={'completion_tokens':3}),
                           dict(instance=1, turn=0, start_time=3, success=False, filler=False)])
        events = [dict(instance=0, turn=0, received_perf=4,
                       data=json.dumps({'usage':dict(prompt_tokens=100,completion_tokens=3)})),
                  dict(instance=0, turn=0, received_perf=5, data='[DONE]'),
                  dict(instance=1, turn=0, received_perf=12,
                       data=json.dumps({'usage':dict(prompt_tokens=100,completion_tokens=9)}))]
        (root/'workload/result.json').write_text(json.dumps(data))
        (root/'workload/sse.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
        return data

    def test_partial_window_preserves_unknown_and_does_not_fake_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            with self.assertRaises(AssertionError): account(root, 2)
            result=account(root, 2, allow_timed=True)
            self.assertFalse(result['designated_tasks_completed'])
            self.assertEqual(result['completed_primary_tasks'], 1)
            self.assertEqual(result['unknown_usage_calls'], 1)
            self.assertEqual(result['output_tokens'], 3)
            self.assertEqual(result['duration_s'], 10)
            self.assertEqual(result['latency_s']['n'], 1)
            first=account(root, 1, allow_timed=True)
            self.assertEqual(first['cutoff'], 5)
            self.assertTrue(first['designated_tasks_completed'])

    def test_failure_cannot_be_relabelled_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); d=self.fixture(root)
            d['cohort_failure']={'status':'timeout'}
            (root/'workload/result.json').write_text(json.dumps(d))
            with self.assertRaises(AssertionError): account(root, 2, allow_timed=True)

if __name__ == '__main__': unittest.main()
