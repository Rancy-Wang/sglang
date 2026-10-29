"""Thinking IDs, heterogeneous decisions, cancellation and repeated KV reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
import urllib.error

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'2026-10-29-cdb-v2'))
from profile_workload import http


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    prompts=json.loads(Path('/all-results/2026-10-29-thinking-batch-profile/native-reference-r4/prompts.json').read_text())['prompts']
    def generate(i,rid,params):
        with http(a.url,'/generate',{'rid':rid,'input_ids':prompts[i]['input_ids'],
            'sampling_params':params,'return_logprob':True,'logprob_start_len':-1,'top_logprobs_num':2}) as r:
            return json.load(r)
    def flush():
        with http(a.url,'/flush_cache?timeout=15',{},timeout=30) as r:assert r.status==200
    cases=[]
    for repeat in range(2):
        with ThreadPoolExecutor(max_workers=16) as pool:
            jobs=[]
            for i in range(16):
                params={'temperature':0,'max_new_tokens':1+(i*7)%24}
                if i%2:
                    params.update(temperature=.8,top_k=40,top_p=.9,min_p=.02,
                        frequency_penalty=.1,presence_penalty=.05,repetition_penalty=1.05,
                        sampling_seed=102+i,min_new_tokens=8,max_new_tokens=24)
                jobs.append((params,pool.submit(generate,i,f'mixed-{repeat}-{i}',params)))
                time.sleep(.013)
            for params,job in jobs:
                out=job.result();meta=out['meta_info'];n=len(out['output_ids'])
                assert n==meta['completion_tokens'] and params.get('min_new_tokens',0)<=n<=params['max_new_tokens']
                assert meta['finish_reason']['type'] in ('length','stop')
                assert len(meta['output_token_logprobs'])==n
                assert all(math.isfinite(v[0]) for v in meta['output_token_logprobs'])
                cases.append(out)
        flush()
    for delay in (.03,.4):
        rid=f'cancel-{delay}'
        with ThreadPoolExecutor(max_workers=2) as pool:
            doomed=pool.submit(generate,10,rid,{'temperature':0,'max_new_tokens':512})
            time.sleep(delay)
            with http(a.url,'/abort_request',{'rid':rid}):pass
            survivor=pool.submit(generate,1,f'survive-{delay}',{'temperature':0,'max_new_tokens':16})
            try:
                out=doomed.result();assert out['meta_info']['finish_reason']['type']=='abort';cases.append(out)
            except urllib.error.HTTPError as e:
                message=e.read().decode();assert e.code in (400,500) and 'abort' in message.lower()
                cases.append({'abort_http':e.code,'body':message})
            out=survivor.result();assert out['meta_info']['completion_tokens']==16;cases.append(out)
        flush()
    out=generate(0,'eos',{'temperature':0,'max_new_tokens':16,'min_new_tokens':3,'logit_bias':{'2':1000}})
    assert out['meta_info']['finish_reason']['type']=='stop' and 3<=out['meta_info']['completion_tokens']<16
    cases.append(out);flush()
    out=generate(0,'zero',{'temperature':0,'max_new_tokens':0})
    assert out['meta_info']['completion_tokens']==0;cases.append(out);flush()
    a.output.write_text(json.dumps({'passed':True,'cases':cases},indent=2)+'\n')
    print(json.dumps({'lifecycle_passed':True,'cases':len(cases)}),flush=True)


if __name__=='__main__':main()
