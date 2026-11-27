"""Keep selected live work, replace excluded arm, and drain legacy method queue."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from worker import HERE, REPO, output

ROOT=Path('/homes/ruoxi07/experiments/loopvq/2026-11-27-sglang-accuracy')
LOOPVQ=Path('/homes/ruoxi07/repo/LoopVQ')
TRAIN=ROOT.parent/'2026-11-26-residual'
ORIGINAL=ROOT.parent/'2026-11-23-reproduction'
MODEL=Path('/homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking')
EVAL='/homes/ruoxi07/anaconda3/envs/loopvq-eval/bin/python'
sys.path.insert(0,str(LOOPVQ/'experiments/loopvq_reproduction'))
from evaluation_io import atomic_json, file_hash
from run_evaluation import collect_cell
from report import checked


def cmdline(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    except (FileNotFoundError,PermissionError,ProcessLookupError):return []


def legacy_worker():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid != os.getuid():continue
        except FileNotFoundError:continue
        args=cmdline(int(p.name))
        if str(HERE/'worker.py').encode() not in args or b'--worker' not in args:continue
        if args[args.index(b'--worker')+1] == b'plain':found.append(int(p.name))
    assert len(found)==1, found
    return found[0],cmdline(found[0])


def launch(name,gpus,port,arm,revision):
    cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--root',str(ROOT),'--loopvq',str(LOOPVQ),
         '--original-root',str(ORIGINAL),'--training-root',str(TRAIN),'--model',str(MODEL),
         '--eval-python',EVAL,'--worker',name,'--gpus',gpus,'--port',str(port),'--arms',arm,
         '--serving-revision',revision]
    keys=('PATH','PYTHONPATH','LD_LIBRARY_PATH','VIRTUAL_ENV','CUDA_HOME','CC','CXX','TORCH_EXTENSIONS_DIR',
          'TORCH_CUDA_ARCH_LIST','MAX_JOBS','OMP_NUM_THREADS','TMPDIR','HF_HOME','XDG_CACHE_HOME',
          'SGLANG_CACHE_DIR','SGLANG_JIT_CACHE_DIR','TRITON_CACHE_DIR','WANDB_DIR','WANDB_BASE_URL')
    env=[f'{k}={os.environ[k]}' for k in keys if k in os.environ]
    command=shlex.join(['env']+env+cmd)+' > '+shlex.quote(str(ROOT/(name+'-three-arm.log')))+' 2>&1; echo $? > '+shlex.quote(str(ROOT/(name+'-three-arm.exit')))
    (ROOT/(name+'-three-arm-command.txt')).write_text(command+'\n')
    subprocess.run(['tmux','new-window','-t','vq-accuracy','-n',name+'-selected',command],check=True)


def main():
    import fcntl
    assert os.environ.get('TMUX')
    lock=(ROOT/'rearrange.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'rearrange-start.json').exists(), 'Inspect existing rearrangement before retry'
    plan=json.loads((HERE/'plan.json').read_text());revision=plan['serving_revision']
    subprocess.run(['git','-C',str(REPO),'diff','--exit-code',revision,'HEAD','--','python'],check=True,stdout=subprocess.DEVNULL)
    assert not output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'])
    evidence={}
    for arm in ('plain-continued','residual-ft'):
        b=TRAIN/'artifacts'/arm/'build_info.json';t=TRAIN/'results'/(arm+'-trace.json')
        info=json.loads(b.read_text());trace=json.loads(t.read_text())
        assert info['e2e_finetune']['held_out_kl']==trace['best']==plan['held_out_best_kl'][arm]
        evidence[arm]={'build_info_sha256':file_hash(b),'training_trace_sha256':file_hash(t),'best_kl':trace['best']}
    assert evidence['residual-ft']['best_kl']<evidence['plain-continued']['best_kl']
    assert not output(['nvidia-smi','-i','6,7','--query-compute-apps=pid','--format=csv,noheader'])
    old_pid,old_args=legacy_worker()
    state={'selected':plan,'selection_evidence':evidence,'old_plain_worker_pid':old_pid,
           'started_unix':time.time(),'status':'waiting-for-selected-current-cell'}
    atomic_json(ROOT/'rearrange-start.json',state)
    launch('best-kl','6,7',30946,'residual-ft',revision)
    done=ROOT/'done/loopvq__aime24__0.json'
    while not done.exists():
        if cmdline(old_pid)!=old_args:raise RuntimeError('Old LoopVQ worker exited before completed cell; preserve cache and inspect')
        state['updated_unix']=time.time();atomic_json(ROOT/'rearrange-state.json',state)
        time.sleep(1)
    scores,_,_=checked(done,'loopvq','aime24',collect_cell)
    if cmdline(old_pid)==old_args:os.kill(old_pid,signal.SIGINT)
    deadline=time.time()+180
    while cmdline(old_pid):
        if time.time()>deadline:raise TimeoutError('Old wrapper cleanup')
        time.sleep(1)
    while output(['nvidia-smi','-i','4,5','--query-compute-apps=pid','--format=csv,noheader']):
        if time.time()>deadline:raise TimeoutError('Old server cleanup')
        time.sleep(1)
    launch('plain','4,5',30944,'loopvq',revision)
    state.update(status='three-dedicated-workers',preserved_aime24=scores,finished_unix=time.time())
    atomic_json(ROOT/'rearrange-state.json',state)
    print('Switched original LoopVQ queue after complete cell; preserved every answer',flush=True)


if __name__=='__main__':main()
