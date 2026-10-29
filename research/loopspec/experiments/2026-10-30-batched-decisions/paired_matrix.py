"""Matched workload, serial servers; rotate variant order at each batch size.

Each cell retains two warmups and five consecutive measurements. This is
paired algorithm blocks, not per-repetition ABBA. Never modifies either source.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORKLOAD = HERE.parent/'2026-10-29-cdb-v2/profile_workload.py'
sys.path.insert(0, str(WORKLOAD.parent))
from profile_workload import http, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke','lifecycle','performance','profile'], required=True)
    p.add_argument('--baseline', type=Path, default=Path('/home/ruoxi/repo/test/baselines/loopspec-7df281a'))
    p.add_argument('--candidate', type=Path, default=Path('/workspace/sglang'))
    p.add_argument('--variants', nargs='+', default=['ar','old','new'])
    p.add_argument('--batches', nargs='+', type=int, default=[1,2,4,8,16])
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--port', type=int, default=30127)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sha={name: subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
         for name,root in [('baseline',a.baseline),('candidate',a.candidate)]}
    write_json(a.output/'versions.json',sha)
    url=f'http://127.0.0.1:{a.port}'
    for turn,bs in enumerate(a.batches):
        order=a.variants[turn%len(a.variants):]+a.variants[:turn%len(a.variants)]
        for variant in order:
            source=a.baseline if variant=='old' else a.candidate
            algorithm='ar' if variant=='ar' else 'loopspec'
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',a.port))==0:
                    raise RuntimeError('Port occupied; refusing to stop another server')
            label=f'{variant}-b{bs}'
            (a.output/f'{label}-gpu-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
            command=[sys.executable,'-m','sglang.launch_server','--model-path','/home/ruoxi/models/Ouro-2.6B-Thinking',
                '--host','127.0.0.1','--port',str(a.port),'--dtype','bfloat16','--attention-backend','triton',
                '--bf16-gemm-backend','auto','--max-running-requests','16','--page-size','1','--disable-radix-cache',
                '--context-length','2048','--mem-fraction-static','.78','--max-total-tokens','12000',
                '--cuda-graph-max-bs-decode','16','--enable-metrics','--stream-interval','1']
            if algorithm=='loopspec':command += ['--speculative-algorithm','loopspec','--speculative-draft-depth','1']
            env=dict(os.environ,PYTHONPATH=str(source/'python'),SGLANG_RUST_BUILD_MODE='auto')
            env.pop('SGLANG_RECURRENT_TRACE',None)
            if a.phase=='profile':env.update(SGLANG_RECURRENT_TRACE='1',SGLANG_PROFILE_V2='0')
            if a.phase in ('smoke','lifecycle'):env['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY']='1'
            with (a.output/f'{label}-server.log').open('w') as log:
                process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(a.output/f'{label}-process.json',{'pid':process.pid,'command':command,'source':str(source),
                'sha':sha['baseline' if variant=='old' else 'candidate'],'started':time.time()})
            try:
                for _ in range(600):
                    if process.poll() is not None:raise RuntimeError(f'{label} server exited {process.returncode}')
                    try:
                        with http(url,'/health',timeout=2):break
                    except Exception:time.sleep(1)
                else:raise TimeoutError('Server readiness')
                if a.phase=='lifecycle':
                    subprocess.run([sys.executable,str(HERE/'lifecycle.py'),'--url',url,
                        '--output',str(a.output/f'{label}-lifecycle.json')],cwd=source,env=env,check=True)
                    continue
                for pattern in ['simultaneous','staggered']:
                    folder=a.output/f'{variant}-b{bs}-{pattern}'
                    cmd=[sys.executable,str(WORKLOAD),'--output',str(folder),'--external-server','--thinking',
                        '--model','/home/ruoxi/models/Ouro-2.6B-Thinking',
                        '--first',str(bs),'--arrival-ms',*[str(0 if pattern=='simultaneous' else 100*i) for i in range(bs)],
                        '--max-new-tokens',str(a.max_new_tokens),'--max-running-requests','16','--graph-max-bs','16',
                        '--mem-fraction','.78','--max-total-tokens','12000','--algorithm',algorithm,
                        '--base-url',url,'--port',str(a.port),'--warmup','0' if a.phase=='smoke' else '2',
                        '--repeat','5' if a.phase=='performance' else '1','--report-mismatches']
                    if a.phase=='profile':cmd.append('--profile')
                    for suffix in ['before','after']:
                        if suffix=='after':
                            print(json.dumps({'start':folder.name,'phase':a.phase,'time':time.time()}),flush=True)
                            subprocess.run(cmd,cwd=source,env=env,check=True)
                        with http(url,'/metrics') as response:
                            (a.output/f'{folder.name}-metrics-{suffix}.txt').write_bytes(response.read())
                    manifest=json.loads((folder/'manifest.json').read_text())
                    assert manifest['sha']==sha['baseline' if variant=='old' else 'candidate']
                    assert Path(manifest['sglang_file']).is_relative_to(source)
                    write_json(folder/'variant.json',{'variant':variant,'source':str(source),'sha':manifest['sha']})
                    print(json.dumps({'done':folder.name,'phase':a.phase,'time':time.time()}),flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    write_json(a.output/'status.json',{'execution_complete':True,'versions':sha})


if __name__=='__main__':main()
