"""Compare released and ported VQ on the same A10 GPU pair, under tmux."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import urllib.request

import benchmark_throughput as bench
import decode_sweep as sweep

TASK = sweep.safe.TASK
MODEL = sweep.safe.MODEL
BUNDLE = TASK / 'artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e'
HERE = Path(__file__).resolve().parent
OFFICIAL = Path.home() / 'repo/LoopVQ'


def request(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data, {'Content-Type': 'application/json'}), timeout=timeout)


def batch(root, name, url, inputs, length, capture):
    payload = {'input_ids': inputs, 'sampling_params': {'temperature': 0, 'ignore_eos': True,
               'max_new_tokens': length, 'stream_interval': 1}, 'stream': True}
    stats = bench.StreamStats(4, length, 2048)
    events = root / (name + '.jsonl')
    start = time.perf_counter()
    done = False
    trigger = None
    with request(url+'/generate', payload, 600) as response, events.open('x') as out:
        for line in response:
            if not line.startswith(b'data:'): continue
            item = line[5:].strip()
            if item == b'[DONE]':
                done = True
                break
            obs = stats.feed(json.loads(item), time.perf_counter()-start)
            out.write(json.dumps(obs)+'\n')
            if capture and trigger is None and min(r['completion_tokens'] for r in stats.requests) >= 928:
                with request(url+'/start_profile', {'activities':['CUDA_PROFILER'], 'num_steps':16,'detailed_annotations':True}, 120) as r:
                    trigger = r.read().decode()
    assert done and (not capture or trigger is not None)
    stats.finalize(time.perf_counter()-start)
    result = sweep.decode_interval(events, 4, length)
    result['profile_response'] = trigger
    return result


def main(args):
    assert os.environ.get('TMUX')
    root = TASK / 'results' / args.run / args.mode / args.stack
    root.mkdir(parents=True, exist_ok=False)
    def gpu(field,kind='gpu'):
        return subprocess.check_output(['nvidia-smi','-i',args.cards,'--query-'+kind+'='+field,'--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(args.cards.split(','))==2 and not gpu('pid','compute-apps')
    with socket.socket() as sock: sock.bind(('127.0.0.1',args.port))
    url = 'http://127.0.0.1:'+str(args.port)
    official = args.stack == 'official'
    repo = OFFICIAL if official else HERE.parents[3]
    python = str(Path.home()/'anaconda3/envs/loopvq-reference-serve/bin/python') if official else os.environ['LOOPVQ_PYTHON']
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.cards,PYTHONPATH=str(repo/'python'),TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='3')
    common = ['--port',str(args.port),'--dtype','bfloat16','--page-size','8','--context-length','40960','--max-prefill-tokens','8192','--chunked-prefill-size','-1','--mem-fraction-static','.90','--enable-metrics','--decode-log-interval','1','--incremental-streaming-output']
    if official:
        for key in ('LD_LIBRARY_PATH','CUDA_HOME','VIRTUAL_ENV','SGLANG_CACHE_DIR','SGLANG_JIT_CACHE_DIR','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','PYTORCH_CUDA_ALLOC_CONF'):
            env.pop(key,None)
        env.update(PATH=str(Path(python).parent)+':'+os.defpath+':/usr/local/bin',TP='2',PREFIX_TOKENS='8',RECENT_TOKENS='64',MAX_TOKENS='172032',PREFILL_BACKEND='triton',SERVE_PYTHON=python)
        command = ['bash',str(repo/'scripts/serve_method.sh'),str(MODEL),'vq',str(BUNDLE),*common,'--cuda-graph-bs','1','2','4','8','16','--disable-piecewise-cuda-graph']
    else:
        command = [python,'-m','sglang.launch_server','--tp-size','2','--model-path',str(MODEL),'--trust-remote-code','--host','127.0.0.1','--attention-backend','triton','--sampling-backend','pytorch','--max-running-requests','16','--disable-radix-cache','--cuda-graph-backend-decode','full','--cuda-graph-bs-decode','1','2','4','8','16','--cuda-graph-backend-prefill','disabled','--ouro-decode-profile','optimized','--triton-attention-num-kv-splits','48','--loopvq-codebook-dir',str(BUNDLE),'--loopvq-prefix-tokens','8','--loopvq-recent-tokens','64','--max-total-tokens','172032',*common]
    profile = args.mode != 'timing'
    if profile:
        temp = Path('/tmp')/f'loopvq-nsys-{os.getuid()}'
        temp.mkdir(mode=0o700,exist_ok=True)
        env.update(TMPDIR=str(temp),NSYS_TMPDIR=str(temp))
        options = ['--cuda-graph-trace=node']
        if args.mode == 'light':
            nvtx = TASK/'envs'/('nvtx-step20-'+args.stack)
            assert (nvtx/'nvtx').is_dir()
            env['PYTHONPATH'] += os.pathsep + str(nvtx)
            methods = {'sglang.srt.managers.scheduler': [
                'Scheduler.get_next_batch_to_run', 'Scheduler.run_batch',
                'Scheduler.process_batch_result']}
            if official:
                methods.update({
                    'sglang.srt.mem_cache.common': ['_alloc_for_decode_mixed'],
                    'sglang.srt.mem_cache.unified_kv_allocator': ['UnifiedInt2HPKVAllocator.alloc_hp_recent'],
                    'sglang.srt.mem_cache.unified_kv_pool': ['UnifiedInt2HPKVPool.wait_pending_forward'],
                    'sglang.srt.model_executor.cuda_graph_runner': ['CudaGraphRunner.replay', 'CudaGraphRunner.replay_prepare']})
            else:
                methods.update({
                    'sglang.srt.mem_cache.loopvq_allocator': ['alloc_for_decode', 'LoopVQKVAllocator.alloc_hp_recent', 'LoopVQKVAllocator.alloc_quant'],
                    'sglang.srt.mem_cache.loopvq_pool': ['LoopVQKVPool.wait_pending_forward', 'LoopVQKVPool.vq_flush_k', 'LoopVQKVPool.vq_flush_v']})
            annotations = root/'annotations.json'
            bench.atomic_json(annotations, [dict(module=m,functions=funcs,domain='vq-step') for m,funcs in methods.items()])
            options = ['--cuda-graph-trace=graph', '--python-functions-trace='+str(annotations)]
        command = ['/usr/local/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',*options,'--capture-range=cudaProfilerApi','--capture-range-end=repeat:1:async','--wait=primary','--kill=none','--output='+str(root/'ranks'),*command]
    inputs=json.loads((sweep.ROOT/'input-2048.json').read_text())['input_ids'][:4]
    assert len(inputs)==4 and all(len(i)==2048 for i in inputs) and json.loads((MODEL/'config.json').read_text())['total_ut_steps']==4
    manifest=dict(command=command,host=socket.gethostname(),cards=gpu('uuid'),bundle={f.name:sweep.safe.sha256(f) for f in BUNDLE.glob('*.pt')},input_hash=bench.digest(inputs),repo=str(repo),sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),python=python)
    bench.atomic_json(root/'manifest.json',manifest)
    server_log=root/'server.log'
    with server_log.open('x') as log:
        process=subprocess.Popen(command,env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError(f'launch exited {process.returncode}')
            try:
                with request(url+'/health',timeout=3): break
            except (OSError,TimeoutError): time.sleep(2)
        else: raise TimeoutError('startup')
        with request(url+'/get_server_info') as response: info=json.load(response)
        bench.atomic_json(root/'server-info.json',info)
        info=info.get('server_args',info)
        assert info['tp_size']==2 and info['dp_size']==1
        ranks=gpu('pid','compute-apps'); assert len(ranks)==2 and all(int(p) in sweep.safe.process_tree(process.pid) for p in ranks)
        bench.atomic_json(root/'rank-pids.json',ranks)
        for i in range(2): batch(root,f'warmup-{i}',url,inputs,32,False)
        rates=[]
        for i in range(1 if profile else 3):
            start=server_log.stat().st_size
            row=batch(root,f'measure-{i}',url,inputs,1024,profile)
            time.sleep(.3)
            with server_log.open('rb') as f:
                f.seek(start); lines=f.read().decode(errors='replace').splitlines()
            decoded=[l for l in lines if bench.DECODE_PATTERN.search(l) and int(bench.DECODE_PATTERN.search(l)[1])>0]
            row['logged_steps']=len(decoded)
            assert len(decoded)>=1000 and all(int(bench.DECODE_PATTERN.search(l)[1])==4 and bench.GRAPH_PATTERN.search(l) and bench.GRAPH_PATTERN.search(l)[1].lower()=='true' for l in decoded), (len(decoded),decoded[:1])
            bench.atomic_json(root/f'measure-{i}.json',row); rates.append(row['decode_tokens_s'])
        if profile:
            for _ in range(120):
                if any(p.stat().st_size>10000 for p in root.glob('*.nsys-rep')): break
                time.sleep(1)
        bench.atomic_json(root/'summary.json',dict(rates=rates,median=statistics.median(rates),profile=profile))
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run');p.add_argument('mode',choices=['timing','trace','light']);p.add_argument('stack',choices=['official','port']);p.add_argument('cards');p.add_argument('port',type=int);main(p.parse_args())
