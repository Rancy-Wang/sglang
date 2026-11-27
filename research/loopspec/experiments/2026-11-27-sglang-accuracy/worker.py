"""Own one TP2 server at a time and run immutable official scoring cells."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ARMS = {'bf16':'bf16','loopvq':'loopvq_0.5bit','residual':'loopvq_residual4',
        'plain-continued':'loopvq_plain_continued','residual-ft':'loopvq_residual_ft'}
BENCHES = ['aime24','aime25','gpqa_diamond','evalplus_humaneval','evalplus_mbpp']


def output(cmd): return subprocess.check_output(cmd, text=True).strip()


def stop(p):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=30)


def server_command(model, book, residual, port, context=40960, log_interval=100):
    cmd=[sys.executable,'-m','sglang.launch_server','--model-path',str(model),'--trust-remote-code',
        '--host','127.0.0.1','--port',str(port),'--tp-size','2','--dtype','bfloat16',
        '--attention-backend','triton','--sampling-backend','pytorch','--page-size','8',
        '--context-length',str(context),'--max-total-tokens',str(81920 if book is None else 262144),'--max-running-requests','16',
        '--max-prefill-tokens','16384','--mem-fraction-static','.85','--disable-radix-cache',
        '--chunked-prefill-size','-1','--ouro-decode-profile','optimized',
        '--triton-attention-num-kv-splits','48' if book else '8',
        '--cuda-graph-backend-decode','full','--cuda-graph-bs-decode','1','2','4','8','16',
        '--cuda-graph-backend-prefill','disabled','--decode-log-interval',str(log_interval),
        '--random-seed','605366612']
    if book:cmd+=['--loopvq-codebook-dir',str(book),'--loopvq-prefix-tokens','8','--loopvq-recent-tokens','64']
    if residual:cmd+=['--loopvq-residual-bits','4','--loopvq-residual-codebook','global','--loopvq-residual-layout','page-head']
    return cmd

def serving_env(gpus):
    senv=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,PYTHONPATH=str(REPO/'python'),OMP_NUM_THREADS='4',MAX_JOBS='2')
    senv['CC']=shutil.which('gcc-13') or shutil.which('gcc');senv['CXX']=shutil.which('g++-13') or shutil.which('g++')
    return senv


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--loopvq',type=Path,required=True)
    ap.add_argument('--original-root',type=Path,required=True)
    ap.add_argument('--training-root',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--eval-python',required=True)
    ap.add_argument('--worker',required=True)
    ap.add_argument('--gpus',required=True)
    ap.add_argument('--port',type=int,required=True)
    ap.add_argument('--arms',nargs='+',choices=ARMS,required=True)
    ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(args.loopvq/'experiments/loopvq_reproduction'))
    from evaluation_io import atomic_json, file_hash, digest
    from run_evaluation import collect_cell, COUNTS
    assert os.environ.get('TMUX')
    assert len(set(args.gpus.split(',')))==2
    assert output(['git','-C',str(REPO),'branch','--show-current'])=='loopspec-test'
    assert not output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'])
    assert output(['git','-C',str(args.loopvq),'rev-parse','HEAD'])=='7397aef8072ff3e61044c1bead66849e07963252'
    assert not output(['git','-C',str(args.loopvq),'status','--porcelain','--untracked-files=no'])
    assert json.loads((args.model/'config.json').read_text())['total_ut_steps']==4
    root=args.root; job=root/'workers'/args.worker; job.mkdir(parents=True,exist_ok=True)
    lock=(job/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.audit_only:
        assert json.loads((root/'context-r1-gcc13.json').read_text())['passed']
        assert (root/'context-memcheck.exit').read_text().strip()=='0'
        assert json.loads((root/'acceptance.json').read_text())['passed']
    sha=output(['git','-C',str(REPO),'rev-parse','HEAD'])
    model_files=sorted(p for p in args.model.iterdir() if p.is_file() and p.suffix in ('.json','.py','.safetensors','.model'))
    with (root/'hashes.lock').open('a') as hlock:
        fcntl.flock(hlock,fcntl.LOCK_EX)
        stat={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in model_files}
        hp=root/'model-hashes.json'
        if hp.exists():
            hashes=json.loads(hp.read_text()); assert hashes['stat']==stat
        else:
            hashes={'stat':stat,'files':{p.name:file_hash(p) for p in model_files}}; atomic_json(hp,hashes)
    state={'worker':args.worker,'host':platform.node(),'gpus':args.gpus,'source':sha,'status':'starting'}
    server=client=None; current=None; common=None; env=None
    def persist(**kw):
        state.update(kw,updated_unix=time.time()); atomic_json(job/'state.json',state)
    def run_client(tail, label):
        nonlocal client
        command=common+tail
        persist(status=label,client_command=command)
        with (cellroot/(label+'.log')).open('a',buffering=1) as log:
            client=subprocess.Popen(command,env=env,cwd=args.loopvq,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        persist(client_pid=client.pid)
        while client.poll() is None:
            if server.poll() is not None: raise RuntimeError('Server died during evaluation')
            time.sleep(5)
        if client.returncode: raise RuntimeError(f'{label} client failed exit{client.returncode}')
    try:
        jobs=[(b,s,a) for b in BENCHES for s in (0,) for a in args.arms]
        if args.audit_only: jobs=[('aime24',0,a) for a in args.arms]
        for bench,seed,arm in jobs:
            name=f'{arm}__{bench}__{seed}'; done=root/'done'/(name+'.json')
            if done.exists() and not args.audit_only:
                from report import checked
                checked(done, arm, bench, collect_cell)
                continue
            cellroot=job/arm; cellroot.mkdir(exist_ok=True)
            persist(arm=arm,benchmark=bench,seed=seed)
            if current!=arm:
                stop(server);server=None
                deadline=time.time()+120
                while output(['nvidia-smi','-i',args.gpus,'--query-compute-apps=pid','--format=csv,noheader']):
                    if current is None or time.time()>deadline: raise RuntimeError('GPU pair is occupied')
                    time.sleep(2)
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); sock.bind(('127.0.0.1',args.port))
                book=None
                if arm in ('loopvq','residual'): book=args.original_root/'artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e'
                elif arm!='bf16':book=args.training_root/'artifacts'/arm
                residual=arm in ('residual','residual-ft')
                concurrency=2 if arm=='bf16' else 6
                tokens=81920 if arm=='bf16' else 262144
                cmd=server_command(args.model, book, residual, args.port)
                senv=serving_env(args.gpus)
                logpath=cellroot/f'server-{time.time_ns()}.log'
                with logpath.open('w') as log:
                    server=subprocess.Popen(cmd,env=senv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                persist(status='server-starting',server_pid=server.pid,server_log=str(logpath),server_command=cmd)
                url=f'http://127.0.0.1:{args.port}'
                deadline=time.time()+1800
                while time.time()<deadline:
                    if server.poll() is not None:raise RuntimeError(f'Server exited{server.returncode}')
                    try:
                        with urlopen(url+'/health',timeout=3):break
                    except OSError:time.sleep(4)
                else:raise TimeoutError('server startup')
                with urlopen(url+'/get_server_info',timeout=30) as response:info=json.load(response)
                cfg=info.get('server_args',info)
                for k,v in {'tp_size':2,'dp_size':1,'context_length':40960,'page_size':8,'disable_radix_cache':True,'chunked_prefill_size':-1,'speculative_algorithm':None,'ouro_decode_profile':'optimized','loopvq_residual_bits':4 if residual else 0}.items():assert cfg[k]==v,(k,cfg[k],v)
                assert cfg['cuda_graph_config']['decode']['backend']=='full'
                assert not cfg.get('allow_auto_truncate',False)
                manifest={'method':ARMS[arm],'arm':arm,'implementation':'sglang','sglang_git_sha':sha,
                    'model_sha256':digest(hashes['files']),'model_files':hashes['files'],'total_ut_steps':4,
                    'bundle_sha256':{p.name:file_hash(p) for p in sorted(book.iterdir()) if p.is_file()} if book else None,
                    'tp':2,'context_length':40960,'output_budget':32768,'prefix_tokens':8,'recent_tokens':64,
                    'decoding':'greedy','residual':residual,'cuda_graph':True,'attention_backend':'triton','sampling_backend':'pytorch',
                    'layout':'page-head' if residual else 'token','adaptive':False,'command':cmd,
                    'gpu':output(['nvidia-smi','-i',args.gpus,'--query-gpu=uuid,name','--format=csv,noheader'])}
                mp=cellroot/'server-manifest.json'
                if mp.exists():assert json.loads(mp.read_text())==manifest,'Immutable manifest changed'
                else:atomic_json(mp,manifest)
                atomic_json(cellroot/'server-observed.json',cfg)
                env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_DATASETS_CACHE=str(args.original_root/'cache/huggingface/datasets'))
                for k in ('PYTHONPATH','LD_LIBRARY_PATH','VIRTUAL_ENV','HUMANEVAL_OVERRIDE_PATH','MBPP_OVERRIDE_PATH','EVALPLUS_TIMEOUT_PER_TASK','WANDB_PROJECT','WANDB_ENTITY'):env.pop(k,None)
                common=[args.eval_python,str(args.loopvq/'experiments/loopvq_reproduction/run_evaluation.py'),
                    '--model',str(args.model),'--base-url',url,'--method',ARMS[arm],
                    '--server-manifest',str(mp),'--root',str(cellroot/'evaluation'),'--num-concurrent',str(concurrency),
                    '--api-seed-mode','explicit','--greedy']
                run_client(['--audit','--benchmarks','aime24','evalplus_humaneval','--seeds','0'],'audit')
                current=arm
            if args.audit_only:continue
            run_client(['--benchmarks',bench,'--seeds',str(seed),'--wandb-entity','529513043-tsinghua-university',
                '--wandb-project','loopvq-ouro-2p6b-thinking','--wandb-group','2026-11-27-sglang-accuracy'],'full')
            candidates=list((cellroot/'evaluation'/ARMS[arm]).glob(f'*/full/{bench}/{seed}/complete.json'))
            assert len(candidates)==1
            cell=candidates[0].parent; marker=json.loads(candidates[0].read_text())
            assert marker['scores']==collect_cell(cell,f'{ARMS[arm]}_{seed}_{bench}',bench,COUNTS[bench])
            atomic_json(done,{'arm':arm,'benchmark':bench,'seed':seed,'worker':args.worker,'cell':str(cell),
                'recipe':str(cell.parents[2]/'recipe.json'),'scores':marker['scores']})
            subprocess.run([args.eval_python,str(HERE/'report.py'),'--root',str(root),'--loopvq',str(args.loopvq)],env=env,check=True)
        persist(status='audit-complete' if args.audit_only else 'complete',exit_code=0)
    except BaseException as exc:
        persist(status='failed',error=f'{type(exc).__name__}: {exc}',exit_code=1);raise
    finally:
        stop(client);stop(server)
        persist(stopped_unix=time.time())


if __name__=='__main__':main()
