"""Native V2 smoke/lifecycle and paired greedy comparison (not a speed claim)."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def http(url, path, payload=None):
    req=urllib.request.Request(url+path, data=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithms',nargs='+',default=['ar','DAS_WAVE'])
    p.add_argument('--batches',nargs='+',type=int,default=[1])
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--port',type=int,default=30215)
    p.add_argument('--graph',action='store_true')
    p.add_argument('--overlap',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model='/home/ruoxi/models/Ouro-2.6B-Thinking'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    problems=['Compute 7 times 8.','What is 13 plus 21?','Solve x + 9 = 17.','What is the capital of France?']
    prompts=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=True,
             enable_thinking=True,add_generation_prompt=True,return_dict=False) for s in problems]
    manifest={'sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'prompt_ids':prompts,'args':vars(a)|{'output':str(a.output)}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result={};url=f'http://127.0.0.1:{a.port}'
    for algorithm in a.algorithms:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError('Port occupied')
        command=[sys.executable,'-m','sglang.launch_server','--model-path',model,'--host','127.0.0.1',
            '--port',str(a.port),'--dtype','bfloat16','--attention-backend','triton','--page-size','1',
            '--disable-radix-cache','--context-length','2048','--max-running-requests',str(max(a.batches)),
            '--return-output-ids','--max-total-tokens','6000','--mem-fraction-static','.70','--stream-interval','1']
        if not a.graph:command+=['--disable-cuda-graph']
        else:command+=['--cuda-graph-max-bs-decode',str(max(a.batches))]
        if not a.overlap:command+=['--disable-overlap-schedule']
        if algorithm!='ar':command+=['--speculative-algorithm',algorithm]
        env=dict(os.environ,SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY='1',SGLANG_RUST_BUILD_MODE='auto')
        log=(a.output/f'{algorithm}.log').open('w')
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        (a.output/f'{algorithm}-process.json').write_text(json.dumps({'pid':proc.pid,'command':command}))
        try:
            for _ in range(600):
                if proc.poll() is not None:raise RuntimeError(f'{algorithm} exited {proc.returncode}')
                try:http(url,'/get_server_info');break
                except Exception:time.sleep(1)
            else:raise TimeoutError('startup')
            cases=[]
            for bs in a.batches:
                for turn in range(2):
                    def generate(i):
                        payload={'input_ids':prompts[i%len(prompts)],'sampling_params':
                            {'temperature':0,'max_new_tokens':max(1,a.tokens-i%4),'ignore_eos':False},
                            'return_logprob':False}
                        started=time.perf_counter()
                        response=http(url,'/generate',payload)
                        return {'bs':bs,'turn':turn,'i':i,'payload':payload,'response':response,
                                'wall_ms':1000*(time.perf_counter()-started)}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                        cases+=list(pool.map(generate,range(bs)))
                    (a.output/f'{algorithm}-partial.json').write_text(json.dumps(cases,indent=2))
                    http(url,'/flush_cache?timeout=15',{})
            result[algorithm]=cases
            (a.output/f'{algorithm}.json').write_text(json.dumps(cases,indent=2))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
    comparisons=[]
    if 'ar' in result:
        for algorithm,cases in result.items():
            if algorithm=='ar':continue
            for baseline,candidate in zip(result['ar'],cases,strict=True):
                l,r=baseline['response'],candidate['response']
                comparisons.append({'algorithm':algorithm,'bs':candidate['bs'],'turn':candidate['turn'],'i':candidate['i'],
                    'text_equal':l['text']==r['text'], 'tokens_equal':l.get('output_ids')==r.get('output_ids'),
                    'token_ids_available':'output_ids' in l and 'output_ids' in r,
                    'baseline_tokens':l['meta_info']['completion_tokens'],
                    'candidate_tokens':r['meta_info']['completion_tokens']})
    (a.output/'comparison.json').write_text(json.dumps(comparisons,indent=2))
    print(json.dumps({'completed':True,'comparisons':comparisons}),flush=True)

if __name__=='__main__':main()
