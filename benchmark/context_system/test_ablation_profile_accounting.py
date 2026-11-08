import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from analyze_system_ablation import stats, profile
import observe_cache_ablation as obs

class Profile(unittest.TestCase):
    def test_nested_cpu_exclusive_and_one_call(self):
        class Cache:
            def inner(self):return 3
            def outer(self):return self.inner()
        rows=[]
        obs.observe(Cache,'inner','inner');obs.observe(Cache,'outer','outer')
        with patch.object(obs,'rank',return_value=0),patch.object(obs,'emit',side_effect=rows.append):
            self.assertEqual(Cache().outer(),3)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['parent_id'],rows[1]['span_id'])
        self.assertAlmostEqual(rows[1]['exclusive_ms']+rows[0]['cpu_ms'],rows[1]['cpu_ms'])
    def test_nonzero_rank_not_duplicated(self):
        class C:
            def run(self):return 1
        obs.observe(C,'run','run')
        with patch.object(obs,'rank',return_value=1),patch.object(obs,'emit') as sink:
            self.assertEqual(C().run(),1);sink.assert_not_called()
    def test_late_gpu_record_and_unknown_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=dict(rank=0,stage='decode',pid=1,start_perf=2,end_perf=3,cpu_ms=1,decode_tokens=8,batch={'step_id':1})
            p=Path(tmp)/'decode-profile.1.jsonl'
            p.write_text('\n'.join(json.dumps(dict(row,**v)) for v in [dict(record='dispatched'),dict(record='gpu_completed',gpu_ms=2,observed_perf=100)]))
            r=profile(tmp,0,10)
            self.assertEqual(r['missing_gpu_records'],0)
            self.assertEqual(next(x for x in r['rows'] if x['stage']=='decode' and x['category']=='decode')['count'],1)
    def test_empty_is_not_fabricated_mean(self):
        self.assertEqual(stats([]),dict(n=0,total=0,mean=None,p90=None))

class OperationCounts(unittest.TestCase):
    def test_skip_same_position_and_repeated_sources(self):
        result=obs.operation_counts([0,5,5,6], [9,10,11,12],
                                    [[1,2],[3,3],[4,9],[5,8]], True)
        self.assertEqual(result, dict(token_events=4,valid_token_events=3,
            skipped_token_events=1,moved_token_events=2,same_position_events=1,
            unique_source_slots_in_launch=2))
    def test_full_pool_zero_slot_not_assumed_unmapped(self):
        result=obs.operation_counts([0],[2],[[10,20]],False)
        self.assertEqual(result['moved_token_events'],1)
    def test_mismatched_snapshot_rejected(self):
        with self.assertRaises(ValueError):obs.operation_counts([1],[],[],True)

if __name__=='__main__':unittest.main()
