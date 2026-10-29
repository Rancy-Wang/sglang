"""Serial GPU benchmark matrix. Server lifetime spans cells, never measurements."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

OLD = Path(__file__).resolve().parent.parent / '2026-10-29-cdb-v2'
sys.path.insert(0, str(OLD))
from profile_workload import http, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['performance', 'profile', 'smoke'], required=True)
    p.add_argument('--algorithms', nargs='+', default=['ar', 'loopspec'])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--port', type=int, default=30126)
    p.add_argument('--model', default='/home/ruoxi/models/Ouro-2.6B-Thinking')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    url = f'http://127.0.0.1:{a.port}'
    for algorithm in a.algorithms:
        # Refuse port collisions; never stop an unrelated server.
        import socket
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', a.port)) == 0:
                raise RuntimeError('Port already occupied')
        resource = subprocess.check_output(['nvidia-smi'], text=True)
        (a.output / f'{algorithm}-gpu-before.txt').write_text(resource)
        common = ['--model-path', a.model, '--host', '127.0.0.1', '--port', str(a.port),
            '--dtype', 'bfloat16', '--attention-backend', 'triton', '--bf16-gemm-backend', 'auto',
            '--max-running-requests', '16', '--page-size', '1', '--disable-radix-cache',
            '--context-length', '2048', '--mem-fraction-static', '0.78', '--max-total-tokens', '12000',
            '--cuda-graph-max-bs-decode', '16', '--enable-metrics', '--stream-interval', '1']
        if algorithm == 'loopspec':
            common += ['--speculative-algorithm', 'loopspec', '--speculative-draft-depth', '1']
        command = [sys.executable, '-m', 'sglang.launch_server', *common]
        env = dict(os.environ, SGLANG_RUST_BUILD_MODE='auto')
        if a.phase == 'profile':
            env.update(SGLANG_RECURRENT_TRACE='1', SGLANG_PROFILE_V2='0')
        else:
            env.pop('SGLANG_RECURRENT_TRACE', None)
        if a.phase == 'smoke':
            env['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY'] = '1'
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        with (a.output / f'{algorithm}-server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        write_json(a.output / f'{algorithm}-process.json', {'command': command, 'pid': process.pid,
            'sha': sha})
        try:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited {process.returncode}')
                try:
                    with http(url, '/health', timeout=2): break
                except Exception: time.sleep(1)
            else: raise TimeoutError('Server startup')
            for bs in a.batches:
                for pattern in ['simultaneous', 'staggered']:
                    name = f'{algorithm}-b{bs}-{pattern}'
                    folder = a.output / name
                    arrivals = [0 if pattern == 'simultaneous' else 100*i for i in range(bs)]
                    cmd = [sys.executable, str(OLD/'profile_workload.py'), '--output', str(folder),
                        '--external-server', '--thinking', '--model', a.model, '--first', str(bs),
                        '--arrival-ms', *map(str, arrivals), '--max-new-tokens', str(a.max_new_tokens),
                        '--max-running-requests', '16', '--graph-max-bs', '16', '--mem-fraction', '.78', '--max-total-tokens', '12000',
                        '--algorithm', algorithm, '--base-url', url, '--port', str(a.port),
                        '--warmup', str(a.warmup), '--repeat', '1' if a.phase == 'profile' else str(a.repeat),
                        '--report-mismatches']
                    if a.phase == 'profile': cmd += ['--profile']
                    with http(url, '/metrics') as response:
                        (a.output/f'{name}-metrics-before.txt').write_bytes(response.read())
                    print(json.dumps({'start': name, 'phase': a.phase, 'time': time.time()}), flush=True)
                    completed = subprocess.run(cmd)
                    if completed.returncode:
                        raise RuntimeError(f'{name} client exited {completed.returncode}')
                    with http(url, '/metrics') as response:
                        (a.output/f'{name}-metrics-after.txt').write_bytes(response.read())
                    (folder/'gpu-after.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
                    print(json.dumps({'done': name, 'time': time.time()}), flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
    write_json(a.output/'status.json', {'execution_complete': True})

if __name__ == '__main__': main()
