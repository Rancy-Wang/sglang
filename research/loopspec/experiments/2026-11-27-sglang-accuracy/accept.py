"""Verify TP2 old-range greedy identity, cache reuse, and >32K production graph."""
import argparse
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from worker import REPO, HERE, output, stop, serving_env, server_command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--book',type=Path,required=True)
    ap.add_argument('--gpus',required=True);ap.add_argument('--port',type=int,default=30946)
    a=ap.parse_args();assert os.environ.get('TMUX')
    assert json.loads((a.root/'context-r1-gcc13.json').read_text())['passed']
    assert (a.root/'context-memcheck.exit').read_text().strip()=='0'
    work=a.root/'acceptance';work.mkdir(exist_ok=False)
    env=serving_env(a.gpus);url=f'http://127.0.0.1:{a.port}'
    checker=HERE.parent/'2026-11-26-vqplus-residual/check_serving.py'
    server=None
    try:
        for context in (32768,40960):
            assert not output(['nvidia-smi','-i',a.gpus,'--query-compute-apps=pid','--format=csv,noheader'])
            with socket.socket() as sock: sock.bind(('127.0.0.1',a.port))
            cmd=server_command(a.model,a.book,True,a.port,context,1)
            (work/f'command-{context}.json').write_text(json.dumps(cmd))
            logpath=work/f'server-{context}.log'
            with logpath.open('w') as log:server=subprocess.Popen(cmd,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.time()+1800
            while time.time()<deadline:
                assert server.poll() is None, f'server exited {server.returncode}'
                try:
                    with urlopen(url+'/health',timeout=3): break
                except OSError: time.sleep(3)
            else:raise TimeoutError('startup')
            base=[sys.executable,str(checker),'--url',url,'--model',str(a.model),'--server-log',str(logpath),
                  '--method','vqplus','--expected-layout','page-head','--expected-page-kernel','cuda']
            extra=['--reference',str(work/'lifecycle-32768')] if context==40960 else []
            subprocess.run(base+['--suite','lifecycle','--out',str(work/f'lifecycle-{context}')]+extra,env=env,check=True)
            if context==40960:
                subprocess.run(base+['--suite','profile','--out',str(work/'boundary-profile'),
                    '--input-length','32760','--output-length','128','--batches','4'],env=env,check=True)
                subprocess.run(base+['--suite','timing','--out',str(work/'near40k'),
                    '--input-length','40928','--output-length','32','--batches','1','--repeats','1'],env=env,check=True)
            stop(server);server=None
            deadline=time.time()+120
            while output(['nvidia-smi','-i',a.gpus,'--query-compute-apps=pid','--format=csv,noheader']):
                assert time.time()<deadline;time.sleep(2)
        kernels=[]
        for p in (work/'boundary-profile').rglob('*.json.gz'):
            data=json.loads(gzip.decompress(p.read_bytes()))
            kernels.extend(e['name'] for e in data.get('traceEvents',[]) if e.get('cat')=='kernel')
        for p in (work/'boundary-profile').rglob('*.trace.json'):
            data=json.loads(p.read_text());kernels.extend(e['name'] for e in data.get('traceEvents',[]) if e.get('cat')=='kernel')
        matched=sorted({name for name in kernels if 'payload' in name.lower()})
        assert matched, 'No optimized payload kernel observed in production graph trace'
        result={'passed':True,'sglang_sha':output(['git','-C',str(REPO),'rev-parse','HEAD']),
                'context_range':[32768,40960],'greedy_lifecycle_exact':True,'production_payload_kernels':matched,
                'new_context_boundary':True,'gpu':output(['nvidia-smi','-i',a.gpus,'--query-gpu=uuid,name','--format=csv,noheader'])}
        (a.root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        print('TP2 40K acceptance passed',flush=True)
    finally:stop(server)


if __name__=='__main__':main()
