"""Exercise score/evidence validation with disposable synthetic HTTP cells."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from report import checked
LOOPVQ=HERE.parents[4]/'LoopVQ'
sys.path.insert(0,str(LOOPVQ/'experiments/loopvq_reproduction'))
from run_evaluation import collect_cell, summarize


class EvidenceTests(unittest.TestCase):
    def test_selected_report_excludes_cancelled_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'done').mkdir()
            (root/'done/residual__aime24__0.json').write_text('cancelled evidence, excluded')
            cmd=[sys.executable,str(HERE/'report.py'),'--root',str(root),'--loopvq',str(LOOPVQ)]
            subprocess.run(cmd,check=True)
            result=json.loads((root/'accuracy.json').read_text())
            self.assertEqual(result['selected_arms'],['bf16','loopvq','residual-ft'])
            self.assertIn('0/2400',(root/'accuracy.md').read_text())
            self.assertNotIn('原码本＋INT4',(root/'accuracy.md').read_text())
            (root/'done/bf16__aime24__0.json').write_text('invalid selected evidence')
            self.assertNotEqual(subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)

    def test_greedy_complete_score_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cell=root/'full/aime24/0';cell.mkdir(parents=True)
            stem='bf16_0_aime24'
            def save(p,data):p.write_text(json.dumps(data))
            protocol={'api_seed':0,'temperature':0.,'top_p':1.,'max_gen_toks':32768,'limit':None,'dataset_documents_sha256':'fixture'}
            save(cell/(stem+'_protocol.json'),protocol)
            save(cell/(stem+'.json'),{'aime24':{'exact_match,none':.5}})
            (cell/(stem+'_samples.jsonl')).write_text('\n'.join(json.dumps({'doc_id':i,'exact_match':int(i<15)}) for i in range(30)))
            responses=cell/'http/responses';responses.mkdir(parents=True)
            for i in range(30):
                save(responses/f'{i}.json',{'payload':{'temperature':0.,'top_p':1.,'max_tokens':32768,'seed':0,'chat_template_kwargs':{'enable_thinking':True}},
                    'input_token_ids':[i], 'response':{'usage':{'prompt_tokens':1,'completion_tokens':32768},'choices':[{'finish_reason':'length' if i==0 else 'stop'}]}})
            scores=collect_cell(cell,stem,'aime24',30)
            save(cell/'complete.json',{'audit':False,'exit_code':0,'method':'bf16','benchmark':'aime24','seed':0,'scores':scores})
            save(root/'recipe.json',{'temperature':0.,'top_p':1.,'max_tokens':32768,'tokenizer_files':{},'sources':{},'packages':{},
                'server':{'tp':2,'total_ut_steps':4,'context_length':40960,'implementation':'sglang','decoding':'greedy','model_sha256':'fixture','sglang_git_sha':'fixture'}})
            done=root/'done.json';save(done,{'arm':'bf16','benchmark':'aime24','seed':0,'cell':str(cell),'recipe':str(root/'recipe.json'),'scores':scores})
            score,_,_=checked(done,'bf16','aime24',collect_cell)
            self.assertEqual(score['accuracy_percent'],50.)
            self.assertEqual(score['length_finishes'],1)
            summarize(root,'bf16',None,greedy=True)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['benchmarks']['aime24']['accuracy_percent'],50.)
            self.assertFalse(summary['complete'])
            bad=json.loads((responses/'0.json').read_text());bad['payload']['temperature']=1.;save(responses/'0.json',bad)
            with self.assertRaises(AssertionError):checked(done,'bf16','aime24',collect_cell)


if __name__=='__main__':unittest.main()
