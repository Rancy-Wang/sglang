import copy
import math
import unittest
from mlgym_replay import ObservationRollingState, source_calls, tool_delay


class ReplayTests(unittest.TestCase):
    def test_drop_13th_observation_only_and_reposition(self):
        h=[dict(role='system',content='s'),dict(role='user',content='task')]
        obs=set()
        for i in range(13):
            h.append(dict(role='assistant',content=str(i)))
            obs.add(len(h));h.append(dict(role='user',content='feedback'))
        owners=[i for i in range(len(h)) for _ in range(10)]
        r=ObservationRollingState(threshold=100)
        state=r.extend(h,owners,len(owners),obs)
        self.assertEqual(state['drop_message'],{'27':[3]})
        self.assertEqual(state['active_tokens'],len(owners)-10)
        self.assertEqual(state['reposition'],[27])
        self.assertNotIn(0, state['drop_message']['27'])
        self.assertEqual(r.extend(h,owners,len(owners),obs),state)
        changed=copy.deepcopy(h);changed[0]['content']='changed'
        with self.assertRaisesRegex(ValueError,'changed'):r.extend(changed,owners,len(owners),obs)

    def test_append_actual_assistant_preserves_bounds(self):
        h=[dict(role='system',content='s'),dict(role='user',content='t')]
        r=ObservationRollingState();r.extend(h,[0]*5+[1]*6,11,set())
        h += [dict(role='assistant',content='actual'),dict(role='user',content='obs')]
        result=r.extend(h,[0]*5+[1]*6+[2]*9+[3]*20,40,{3})
        self.assertEqual(result['active_tokens'],40)
        with self.assertRaisesRegex(ValueError,'boundary'):r.extend(h,[0]*4+[1]*7+[2]*9+[3]*20,40,{3})

    def test_invalid_tool_times_rejected(self):
        for v in [None,-1,math.inf,math.nan,True,'10']:
            with self.subTest(v=v),self.assertRaises(ValueError):tool_delay(dict(action='ls',execution_time=v))
        self.assertEqual(tool_delay(dict(action='submit',execution_time=2401.5)),2401.5)
        self.assertIsNone(tool_delay(dict(action='exit_context',execution_time=.2)))
        self.assertIsNone(tool_delay(None))

    def test_mapping_sentinel_orphan_and_observation_suffix(self):
        h=[dict(role='system',content='s'),dict(role='user',content='task'),
           dict(role='assistant',content='ls'),dict(role='user',content='files\nstate=x'),
           dict(role='assistant',content='orphan')]
        data=dict(history=h,trajectory=[dict(response='ls',action='ls',observation='files',execution_time=5.)])
        calls=source_calls(data)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0]['tool_seconds'],5.)
        self.assertEqual(calls[1]['new_messages'][0]['content'],'files\nstate=x')
        self.assertTrue(calls[1]['orphan_tool_step']);self.assertIsNone(calls[1]['tool_seconds'])
        data['history'][-1]=dict(role='assistant',content='Exit due to context window',action='exit_context')
        self.assertEqual(len(source_calls(data)),1)
        data['trajectory'][0]['response']='mismatch'
        with self.assertRaisesRegex(ValueError,'mismatch'):source_calls(data)

if __name__=='__main__':unittest.main()
