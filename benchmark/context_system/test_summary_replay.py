"""CPU checks for source reconstruction, generated compaction, and cutoff accounting."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serving_cohort import UniqueCohort
from summary_replay import Replay, load_cases
from summarize_summary_serving import summarize


class WindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_summary_latency_matches_original_http_boundary(self):
        from types import SimpleNamespace
        from concurrent.futures import ThreadPoolExecutor
        from summary_replay import execute_case
        class FakeReplay:
            original = []
            def __init__(self, *unused): pass
            def prepare(self, group): return dict(requests=[], event=None)
            def active(self, prepared): return [dict(role="user", content="input")]
            def commit_summary(self, *unused): pass
            def commit_agent(self, *unused): pass
        class Transport:
            async def request(self, url, payload, identity):
                assert payload["stream_options"]["continuous_usage_stats"] is True
                return dict(success=True, prompt_len=10, start_time=100, raw_done_time=102,
                            output_len=2, ttft=.5, tpot_s=1.5, cleanup_time_s=.1,
                            assistant=dict(role="assistant",content="generated"))
        args=SimpleNamespace(summary_policy_dir="unused", summary_smoke=False,
                             model_context_limit=100,model="model",template_kwargs="{}",url="unused")
        case=dict(case_id="task",trial=0,tools=[],groups=[dict(agent=dict(logical_call_index=0,
                  replay_output_budget_known=True,observed_output_tokens=2,source_usage={}),summaries=[])])
        parents=[]
        with ThreadPoolExecutor(max_workers=1) as executor, patch('summary_replay.Replay',FakeReplay), patch('time.perf_counter',return_value=97):
            status=await execute_case(case,dict(instance=0,filler=False),args=args,
                renderer=SimpleNamespace(render=lambda *a:(10,None)),rendering=executor,
                transport=Transport(),emit=lambda e:None,rows=[],user_turns=parents)
        self.assertEqual(status,"all_turns_completed")
        self.assertEqual(parents[0]['latency'],2)
        self.assertEqual(parents[0]['ttft'],.5)
        self.assertEqual(parents[0]['prepare_time_s'],3)

    async def test_filler_does_not_finish_primary_and_is_cancelled(self):
        events = []
        async def execute(case, inst):
            await asyncio.sleep({0: .025, 1: .002, 2: 1}[case['case_id']])
            return 'all_turns_completed'
        scheduler = UniqueCohort([dict(case_id=i) for i in range(3)], 2, 2, execute, events.append)
        await scheduler.run()
        self.assertEqual(scheduler.stop_reason, 'primary_cohort_completed')
        self.assertEqual(len(scheduler.completed), 2)
        self.assertTrue(scheduler.instances[2]['filler'])
        self.assertEqual(scheduler.instances[2]['status'], 'cutoff_cancelled')

    async def test_time_limit_preserves_cancelled_tasks(self):
        async def execute(*unused):
            await asyncio.sleep(10)
        scheduler = UniqueCohort([dict(case_id=1)], 1, 1, execute, lambda e: None, max_seconds=.01)
        await scheduler.run()
        self.assertEqual(scheduler.stop_reason, 'measurement_time_limit')
        self.assertEqual(scheduler.instances[0]['status'], 'cutoff_cancelled')
        self.assertIsNone(scheduler.failure)

    async def test_source_terminal_is_not_transport_failure(self):
        async def execute(*unused):
            return 'source_terminal_replayed'
        scheduler = UniqueCohort([dict(case_id=1)], 1, 1, execute, lambda e: None, source_terminal_ok=True)
        await scheduler.run()
        self.assertEqual(len(scheduler.completed), 1)

    async def test_pressure_stop_requires_hour_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'STOP.json'
            path.write_text(json.dumps(dict(reason='cache_pressure_after_1h', evidence='KV allocation queue')))
            async def execute(*unused):
                await asyncio.sleep(.03)
                return 'all_turns_completed'
            scheduler = UniqueCohort([dict(case_id=1)], 1, 1, execute, lambda e: None, stop_file=path)
            await scheduler.run()
            self.assertEqual(scheduler.stop_reason,'primary_cohort_completed')
            scheduler = UniqueCohort([dict(case_id=1)], 1, 1, execute, lambda e: None, stop_file=path)
            scheduler.start = 0
            with patch('serving_cohort.time.perf_counter',return_value=3601):
                await scheduler.monitor()
            self.assertEqual(scheduler.stop_reason,'cache_pressure_after_1h')


class AccountingTests(unittest.TestCase):
    def test_partial_summary_counted_once_and_post_cutoff_usage_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = dict(start=1, cutoff=11, valid=True, stop_reason='measurement_time_limit',
                          logical_user_turns=[], turns=[], tasks=[], source_terminal_tasks=0)
            (root/'result.json').write_text(json.dumps(result))
            (root/'events.jsonl').write_text(json.dumps(dict(kind='turn_start',time=2,instance=0,turn=0,purpose='summary_history'))+'\n')
            events = []
            for stamp,n in [(3,2),(5,4),(12,1000)]:
                events.append(dict(instance=0,turn=0,received_perf=stamp,data=json.dumps(dict(usage=dict(prompt_tokens=100,completion_tokens=n,prompt_tokens_details={'cached_tokens':20})))))
            (root/'sse.jsonl').write_text('\n'.join(map(json.dumps,events)))
            actual=summarize(root)
            self.assertEqual(actual['output_throughput'],.4)
            self.assertEqual(actual['logical_input_throughput'],10)
            self.assertEqual(actual['counters']['summary']['calls_with_usage'],1)
            self.assertAlmostEqual(actual['counters']['summary']['price_usd'],(20*.075+80*.15+4*.6)/1e6)


@unittest.skipUnless(os.environ.get('SUMMARY_FIXTURE_ROOT'), 'Full source audit requires explicit archived fixtures')
class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = os.environ['SUMMARY_POLICY_DIR']
        cls.datasets = []
        for root in sorted(Path(os.environ['SUMMARY_FIXTURE_ROOT']).glob('throughput*')):
            cases,_ = load_cases(root/'tasks_unique.jsonl',80,42,None,cls.policy)
            cls.datasets.append(cases)

    def test_all_source_prompts_and_final_requests(self):
        self.assertEqual([sum(c['validated_summary_prompts'] for c in cases) for cases in self.datasets],[152,32])
        self.assertEqual([c['case_id'] for c in self.datasets[0]],[c['case_id'] for c in self.datasets[1]])

    def test_generated_history_and_summary_replace_source(self):
        case = next(c for c in self.datasets[0] if any(g['summaries'] for g in c['groups']))
        replay=Replay(case,self.policy)
        first = case['groups'][0]['agent']
        replay.commit_agent(first,dict(role='assistant',content='GENERATED_UNIQUE_SENTINEL'))
        group=next(g for g in case['groups'] if g['summaries'])
        prepared=replay.prepare(group)
        self.assertTrue(any('GENERATED_UNIQUE_SENTINEL' in r['prompt'] for r in prepared['requests']))
        replay.commit_summary(prepared,{r['operation']['summary']['kind']:'ACTUAL_SUMMARY' for r in prepared['requests']})
        active=replay.active(prepared)
        self.assertIn('ACTUAL_SUMMARY',active[1]['content'])
        self.assertNotIn('GENERATED_UNIQUE_SENTINEL',str(active))
        self.assertEqual(active[2:],prepared['canonical'][replay.first_kept:])

    def test_format_error_generation_discarded_but_feedback_kept(self):
        case=next(c for c in self.datasets[0] if any(g['agent']['status']=='format_error' for g in c['groups']))
        group=next(g for g in case['groups'] if g['agent']['status']=='format_error')
        replay=Replay(case,self.policy); op=group['agent']; index=op['source_response_message_index']
        replay.commit_agent(op,dict(role='assistant',content='MUST_NOT_COMMIT'))
        self.assertNotIn(index,replay.replacements)
        self.assertEqual(replay.canonical(index+1)[-1]['role'],'user')


if __name__=='__main__':
    unittest.main()
