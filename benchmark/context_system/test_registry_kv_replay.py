import asyncio, copy, json, time, unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from registry_kv_replay import History, execute_case
from serving_cohort import UniqueCohort


def case():
    return dict(case_id='one', trial=0, tools=[], trajectory=[
        {'role':'user','content':'start'}, {'role':'assistant','content':'source one'},
        {'role':'tool','content':'observation','name':'bash'},
        {'role':'user','content':'format error: retry'}, {'role':'assistant','content':'source last'}],
        operations=[dict(logical_call_index=i, source_response_message_index=pos,
                         source_response_role=role, observed_output_tokens=budget,
                         source_usage={'prompt_tokens':100,'completion_tokens':budget})
                    for i,(pos,role,budget) in enumerate([(1,'assistant',7),(3,'user',11),(4,'assistant',5)])])

class Contracts(unittest.TestCase):
    def test_generated_history_kept_but_format_error_output_discarded(self):
        c=case();h=History(c)
        self.assertTrue(h.commit(c['operations'][0],{'role':'assistant','content':'actual one'}))
        self.assertFalse(h.commit(c['operations'][1],{'role':'assistant','content':'discard this'}))
        messages,_=h.prepare(c['operations'][2])
        self.assertEqual([m['content'] for m in messages],['start','actual one','observation','format error: retry'])
        self.assertEqual(c['trajectory'][1]['content'],'source one')

    def test_invalid_deadline_rejected(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):UniqueCohort([{'case_id':'a'}],1,1,None,None,max_seconds=value)

    def test_client_and_runner_accept_500_source_c30_without_changing_primary(self):
        from test_serving import parser
        args=parser().parse_args(['--mini-root','/mini','--requests-path','/data','--output-dir','/out','--model','m','--tokenizer','m',
            '--concurrency','30','--num-tasks','30','--source-tasks','500','--measurement-seconds','21600','--registry-kv','--drop','--unique-cohort'])
        self.assertEqual((args.concurrency,args.num_tasks,args.source_tasks,args.measurement_seconds),(30,30,500,21600))
        import run_pd_matrix
        args=run_pd_matrix.parser().parse_args(['--output-dir','/out','--server-repo','/server','--source-launch','/launch','--mini-root','/mini','--requests-path','/data','--one','--source-tasks','500','--registry-kv','--concurrency','30','--primary-tasks','30'])
        self.assertEqual((args.concurrency,args.primary_tasks,args.source_tasks),(30,30,500))

class Execution(unittest.IsolatedAsyncioTestCase):
    async def replay(self, *, bad_budget=False, limit=1000):
        payloads=[];events=[];rows=[];parents=[]
        class Rolling:
            def __init__(self,**kwargs):self.previous=[]
            def extend(self,messages,owners,full):
                assert messages[:len(self.previous)]==self.previous
                self.previous=copy.deepcopy(messages)
                return dict(active_tokens=100,position_tokens=100,drop_message={},reposition=[])
        class Transport:
            async def request(self,url,payload,identity):
                payloads.append(copy.deepcopy(payload));now=time.perf_counter()
                return dict(success=True, status='success', prompt_len=100,
                    output_len=payload['max_tokens']-(1 if bad_budget else 0),
                    assistant={'role':'assistant','content':f'generated {len(payloads)}'},
                    start_time=now,raw_done_time=now+.01,latency=.01,ttft=.001,tpot_s=.001,cleanup_time_s=0)
        with ThreadPoolExecutor(1) as pool:
            status=await execute_case(case(),{'instance':0,'filler':False},args=SimpleNamespace(model='m',url='fake',model_context_limit=limit,template_kwargs='{}'),
                renderer=SimpleNamespace(render=lambda m,t:(100,[])),rendering=pool,transport=Transport(),emit=events.append,
                rows=rows,user_turns=parents,method=SimpleNamespace(RollingState=Rolling))
        return status,payloads,events,rows,parents

    async def test_full_budget_all_calls_and_feedback_not_fake_history(self):
        status,payloads,events,rows,parents=await self.replay()
        self.assertEqual(status,'all_turns_completed');self.assertEqual([p['max_tokens'] for p in payloads],[7,11,5])
        self.assertEqual([m['content'] for m in payloads[-1]['messages']],['start','generated 1','observation','format error: retry'])
        self.assertEqual(len(rows),3);self.assertEqual(len(parents),3)
        self.assertEqual([e['generated_response_retained'] for e in events if e['kind']=='registry_response_commit'],[True,False,True])
        self.assertTrue(all(p['ignore_eos'] and p['temperature']==0 for p in payloads))

    async def test_budget_or_context_failure_cannot_complete(self):
        status,payloads,events,rows,parents=await self.replay(bad_budget=True)
        self.assertEqual(status,'registry_budget_or_template_mismatch');self.assertEqual(len(payloads),1);self.assertEqual(parents,[])
        status,payloads,*_=await self.replay(limit=105)
        self.assertEqual(status,'registry_context_limit');self.assertEqual(payloads,[])

    async def test_c30_cohort_and_timed_partial(self):
        cases=[dict(case_id=str(i)) for i in range(500)]
        async def done(c,i):await asyncio.sleep(0);return 'all_turns_completed'
        c=await UniqueCohort(cases,30,30,done,lambda e:None,max_seconds=6).run()
        self.assertEqual(len(c.completed),30);self.assertEqual(c.stop_reason,'primary_cohort_completed')
        self.assertEqual(len({x['case_id'] for x in c.instances}),len(c.instances))
        async def wait(c,i):await asyncio.Event().wait()
        p=await UniqueCohort(cases,30,30,wait,lambda e:None,max_seconds=.02).run()
        self.assertEqual(p.stop_reason,'measurement_time_limit');self.assertEqual(len(p.completed),0)
        self.assertEqual({x['status'] for x in p.instances},{'cutoff_cancelled'});self.assertIsNone(p.failure)

if __name__=='__main__':unittest.main()
