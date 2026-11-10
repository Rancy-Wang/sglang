import copy
import unittest
from ablation_replay import text_drop_projection, require_complete_budget, context_budget_audit

class Replay(unittest.TestCase):
    def test_cumulative_canonical_projection_without_mutation(self):
        h=[dict(role='user',content='x'),dict(role='tool',content='old'),dict(role='assistant',content='a',reasoning_content='keep'),dict(role='tool',content='new')]
        old=copy.deepcopy(h); wire,kept,removed=text_drop_projection(h,{'drop_message':{'3':[1]}})
        self.assertEqual(kept,[0,2,3]);self.assertEqual(removed,[1]);self.assertEqual(wire[1]['reasoning_content'],'keep')
        wire[0]['content']='mutated';self.assertEqual(h,old)
    def test_reject_bad_target_and_partial_rule(self):
        h=[dict(role='user'),dict(role='tool')]
        for rule in ({'1':[0]},{'1':[1]},{'2':[1]},{'1':{'range':[0,1]}}):
            with self.assertRaises(ValueError):text_drop_projection(h,{'drop_message':rule})
    def test_strict_length_cannot_be_successful_terminal(self):
        with self.assertRaises(ValueError):require_complete_budget(200000,100,{},131072)
        require_complete_budget(200000,100,{},262144)
        require_complete_budget(200000,100,{'active_tokens':40000,'position_tokens':80000},131072)
    def test_audit_never_caps_pair_at128k(self):
        class Tok:
            def encode(self,text,**kwargs):return list(text)
        a=context_budget_audit([dict(case_id='a',turns=[dict(new_messages=[],max_new_tokens=200000)])],Tok())
        self.assertGreater(a['context_length'],400000)
        self.assertEqual(a['cases'][0]['output_budget'],200000)

if __name__=='__main__':unittest.main()
