import asyncio
import unittest
from test_serving_live import LiveTaskScheduler, tool_wait, make_payload, validate_budget, positive
from analyze_serving_live import summarize


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sleep_keeps_slot_other_task_can_finish_and_refill(self):
        events, entered = [], []
        sleeping, release = asyncio.Event(), asyncio.Event()
        async def execute(case, identity):
            entered.append(case['case_id'])
            if case['case_id'] == 'a':
                sleeping.set()
                await release.wait()
            else:
                await sleeping.wait()
            return 'all_recorded_steps_replayed'
        cases = [dict(case_id=x, rank=i) for i,x in enumerate('abcd')]
        s = LiveTaskScheduler(cases, 2, execute, events.append)
        task = asyncio.create_task(s.run())
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertEqual(entered, list('abcd'))
        self.assertEqual(len(s.active), 1)
        self.assertFalse(task.done())
        # All replacements followed a task end, never just the a task's wait.
        balance = 0
        for e in events:
            if e['kind'] == 'task_start': balance += 1
            if e['kind'] == 'task_end': balance -= 1
            self.assertLessEqual(balance, 2)
        release.set()
        await task
        self.assertEqual(len(s.completed), 4)

    async def test_no_refill_while_all_tasks_sleep(self):
        gate, entered = asyncio.Event(), []
        async def execute(case, identity):
            entered.append(case['case_id'])
            await gate.wait()
            return 'all_recorded_steps_replayed'
        s = LiveTaskScheduler([dict(case_id=str(i),rank=i) for i in range(5)], 2, execute, lambda _:None)
        run = asyncio.create_task(s.run())
        for _ in range(5): await asyncio.sleep(0)
        self.assertEqual(entered, ['0','1'])
        gate.set()
        await run
        self.assertEqual(entered, ['0','1','2','3','4'])

    async def test_final_wait_and_long_delay_are_not_capped(self):
        now, events = [0.], []
        async def sleep(sec): now[0] += sec
        await tool_wait(2400.125, {}, events.append, sleep=sleep, clock=lambda:now[0])
        self.assertEqual(events[-1]['actual_seconds'],2400.125)
        self.assertEqual(events[-1]['wake_delay_seconds'],0)

    async def test_unknown_wait_not_zero(self):
        events=[]
        await tool_wait(None, {}, events.append)
        self.assertEqual(events,[])

    async def test_failure_cancels_sleepers_no_false_complete(self):
        ready=asyncio.Event()
        async def execute(case, identity):
            if case['case_id']=='a':
                ready.set(); await asyncio.sleep(1000)
            await ready.wait()
            raise RuntimeError('network failure')
        s=LiveTaskScheduler([dict(case_id=x,rank=i) for i,x in enumerate('abc')],2,execute,lambda _:None)
        with self.assertRaisesRegex(RuntimeError,'network failure'): await s.run()
        self.assertEqual(set(x['status'] for x in s.completed),{'runtime_failed','cancelled'})
        self.assertEqual(len(s.pending),1)

    def test_payload_exact_budget_eos_and_ops(self):
        state=dict(drop_message={'27':[2]},reposition=[27],active_tokens=10,position_tokens=10)
        p=make_payload('m',[dict(role='user',content='x')],state,173)
        self.assertEqual(p['max_tokens'],173);self.assertTrue(p['ignore_eos'])
        self.assertEqual(p['drop_message'],state['drop_message'])
        with self.assertRaises(ValueError):validate_budget(state,100,100)
        validate_budget(state,90,100)
        self.assertEqual(positive('37'),37)
        with self.assertRaises(Exception):positive('0')

    def test_restoration_uses_previous_input_and_physical_prefill(self):
        import tempfile
        from pathlib import Path
        from analyze_serving_live import restoration
        rows=[dict(success=True,instance=1,turn=i,pd_bootstrap_room=i+10,prompt_len=n) for i,n in enumerate([100,130,150])]
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'p.log'
            log.write_text('[TP0] ReqTimeStats(bootstrap_room=10, input_len=100), pd_matrix_compute={"pf":100}\n'
                           '[TP0] ReqTimeStats(bootstrap_room=11, input_len=130), pd_matrix_compute={"pf":50}\n')
            report=restoration(dict(turns=rows),log)
        self.assertEqual(report['positive_requests'],1)
        self.assertEqual(report['unknown_requests'],1)
        self.assertEqual(report['rows'][0]['restore'],0)
        self.assertEqual(report['rows'][1]['restore'],20)
        self.assertEqual(report['rows'][1]['new_prefill'],30)
        self.assertEqual(report['positive_fraction'],1/3)
        self.assertFalse(report['exact_zero'])

    def test_wall_metrics_include_tool_wait(self):
        row=dict(success=True,prompt_len=1000,output_len=100,latency=2,ttft=.1,tpot_s=.02)
        r=summarize(dict(turns=[row],duration_s=10,tasks=[dict(status='all_recorded_steps_replayed',latency_s=10)]))
        self.assertEqual(r['output_throughput'],10)
        self.assertEqual(r['latency_s']['mean'],2)

if __name__=='__main__':unittest.main()
