"""Serial full AIME25 benchmark and separately bounded profile windows.

Immutable output directories; completed requests are persisted immediately.
Profile prefixes come from the corresponding unprofiled trajectory, never from
an answer key. Profile windows are diagnostics, not end-to-end benchmark times.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

METHODS = ('ar', 'dflash', 'loopspec1', 'loopspec13', 'das_wave')
MODEL = '/home/ruoxi/models/Ouro-2.6B-Thinking'
DRAFT = MODEL + '-DFlash'
DATA_HASH = 'b1cb6f70eb12b5a0e68f9ea88be9a35c6a3c66f0d00112b5b26160b042326b19'


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def http(url, route, payload=None, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(url + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}), timeout=timeout)


def generate(url, ids, rid, limit):
    start = time.perf_counter_ns()
    events, final, first, last_count = [], None, None, 0
    output, text_parts = [], []
    with http(url, '/generate', {'rid': rid, 'input_ids': ids, 'stream': True,
            'sampling_params': {'temperature': 0, 'max_new_tokens': limit, 'ignore_eos': False}}) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            body = line[5:].strip()
            if body == b'[DONE]':
                break
            final = json.loads(body)
            if 'error' in final:
                raise RuntimeError(final)
            now = time.perf_counter_ns()
            count = final['meta_info']['completion_tokens']
            output.extend(final.get('output_ids', []))
            text_parts.append(final.get('text', ''))
            assert len(output) == count, (rid, len(output), count)
            assert count >= last_count
            last_count = count
            if count and first is None:
                first = now
            events.append({'ns': now, 'tokens': count})
    assert final is not None and final['meta_info'].get('finish_reason'), rid
    assert len(output) == final['meta_info']['completion_tokens'], (rid, len(output), final['meta_info'])
    assert first is not None
    end = events[-1]['ns']
    return {'rid': rid, 'output_ids': output, 'text': ''.join(text_parts),
        'meta_info': final['meta_info'], 'events': events, 'sent_ns': start,
        'ttft_ms': (first-start)/1e6, 'decode_ms': (end-first)/1e6,
        'e2e_ms': (end-start)/1e6, 'decode_tokens': max(0, len(output)-events[next(i for i,e in enumerate(events) if e['tokens'])]['tokens']),
        'first_event_tokens': next(e['tokens'] for e in events if e['tokens'])}


def command(method, a):
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', MODEL,
        '--host', '127.0.0.1', '--port', str(a.port), '--dtype', 'bfloat16',
        '--attention-backend', 'triton', '--page-size', '1', '--disable-radix-cache',
        '--context-length', str(a.context), '--max-running-requests', '1',
        '--max-total-tokens', str(a.capacity), '--mem-fraction-static', '.85',
        '--return-output-ids', '--incremental-streaming-output', '--stream-interval', '1', '--cuda-graph-max-bs-decode', '1',
        '--chunked-prefill-size', '512']
    if method != 'ar':
        cmd += ['--speculative-algorithm', 'LOOPSPEC' if method.startswith('loopspec') else method.upper()]
    if method.startswith('loopspec'):
        cmd += ['--speculative-draft-depth', '1'] + (['3'] if method == 'loopspec13' else [])
    if method == 'dflash':
        cmd += ['--speculative-draft-model-path', DRAFT, '--speculative-dflash-block-size', '16']
    return cmd


def run_method(method, repeat, a, rows, prompts, warmups):
    output = a.output / f'{method}-r{repeat}'
    output.mkdir()
    url = f'http://127.0.0.1:{a.port}'
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', a.port)) == 0:
            raise RuntimeError('Port occupied')
    env = dict(os.environ, SGLANG_RUST_BUILD_MODE='auto')
    env.pop('SGLANG_RECURRENT_TRACE', None)
    if a.profile_reference:
        env.update(SGLANG_RECURRENT_TRACE='1', SGLANG_PROFILE_V2='0')
    cmd = command(method, a)
    save(output/'launch.json', {'command': cmd, 'profile': bool(a.profile_reference)})
    with (output/'server.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(output/'process.json', {'pid': proc.pid})
    active = False
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f'{method} server exit {proc.returncode}')
            try:
                with http(url, '/get_server_info', timeout=2) as response:
                    info = json.load(response)
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError('Server startup')
        save(output/'server-info.json', info)
        for i, ids in enumerate(warmups):
            save(output/f'warmup-{i}.json', generate(url, ids, f'warmup-{method}-{i}', 64))
        count = 0
        for row, prompt in zip(rows, prompts, strict=True):
            qid = str(row['id'])
            if a.profile_reference:
                if qid not in a.profile_ids:
                    continue
                ref = json.loads((a.profile_reference/f'{method}-r0'/f'q{qid}.json').read_text())
                for fraction in (0.0, 0.5, 0.9):
                    offset = int(len(ref['output_ids']) * fraction)
                    ids = prompt['input_ids'] + ref['output_ids'][:offset]
                    label = f'q{qid}-p{int(fraction*100)}'
                    trace = output/label/'trace'
                    trace.mkdir(parents=True)
                    with http(url, '/start_profile', {'output_dir': str(trace), 'activities': ['CPU','GPU'],
                        'profile_id': label, 'with_stack': False, 'record_shapes': False,
                        'detailed_annotations': True}) as response:
                        response.read()
                    active = True
                    result = generate(url, ids, f'{method}-{label}', a.profile_tokens)
                    with http(url, '/stop_profile', {}) as response:
                        response.read()
                    active = False
                    save(output/label/'request.json', result | {'id': qid, 'prefix_tokens': len(ids),
                        'trajectory_offset': offset, 'reference': str(a.profile_reference)})
                    print(json.dumps({'method': method, 'profile': label, 'tokens': len(result['output_ids'])}), flush=True)
            else:
                result = generate(url, prompt['input_ids'], f'{method}-r{repeat}-q{qid}', a.tokens)
                save(output/f'q{qid}.json', result | {'id': qid, 'answer': row['answer']})
                print(json.dumps({'method': method, 'repeat': repeat, 'id': qid,
                    'tokens': len(result['output_ids']), 'seconds': result['e2e_ms']/1000,
                    'finish': result['meta_info']['finish_reason']}), flush=True)
            count += 1
            with http(url, '/flush_cache?timeout=30', {}) as response:
                response.read()
        expected = len(a.profile_ids) if a.profile_reference else len(rows)
        assert count == expected
        save(output/'status.json', {'execution_complete': True, 'questions': count,
            'full30': count == 30 and not a.profile_reference, 'profile': bool(a.profile_reference)})
    except BaseException as error:
        save(output/'status.json', {'execution_complete': False, 'error': repr(error)})
        raise
    finally:
        if active:
            try:
                with http(url, '/stop_profile', {}, timeout=120) as response:
                    response.read()
            except Exception:
                pass
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('/results/aime25-matharena.jsonl'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--context', type=int, default=12288)
    p.add_argument('--capacity', type=int, default=11264)
    p.add_argument('--port', type=int, default=30215)
    p.add_argument('--smoke', action='store_true', help='First two questions only; never counts as full30')
    p.add_argument('--profile-reference', type=Path)
    p.add_argument('--profile-ids', nargs='+', default=['0','14','29'])
    p.add_argument('--profile-tokens', type=int, default=32)
    a = p.parse_args()
    assert a.repeat > 0 and (not a.profile_reference or a.repeat == 1)
    a.output.mkdir(parents=True, exist_ok=False)
    raw = a.dataset.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_HASH
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == len({str(r['id']) for r in rows}) == 30
    from transformers import AutoTokenizer
    import sglang
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    def encode(text):
        rendered = tokenizer.apply_chat_template([{'role':'user','content':text}],
            tokenize=False, enable_thinking=True, add_generation_prompt=True)
        assert '<think>' in rendered.split('<|im_start|>assistant')[-1]
        return {'rendered': rendered, 'input_ids': tokenizer.encode(rendered, add_special_tokens=False)}
    prompts = [encode(r['problem']) | {'id': str(r['id'])} for r in rows]
    assert max(len(r['input_ids']) for r in prompts) + a.tokens + 32 < min(a.context, a.capacity)
    save(a.output/'prompts.json', {'chat_template': tokenizer.chat_template,
        'eos_token_id': tokenizer.eos_token_id, 'prompts': prompts})
    save(a.output/'manifest.json', {'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sglang_file': sglang.__file__, 'python': sys.executable, 'dataset_sha256': DATA_HASH,
        'versions': {n:importlib.metadata.version(n) for n in ('torch','transformers','triton')},
        'nvidia_smi': subprocess.check_output(['nvidia-smi'],text=True),
        'target_config': json.loads(Path(MODEL,'config.json').read_text()),
        'draft_manifest': json.loads(Path(DRAFT,'export-manifest.json').read_text())})
    if a.smoke:
        rows, prompts = rows[:2], prompts[:2]
    warmups = [encode(s)['input_ids'] for s in ('Compute 7 times 8.', 'What is 13 plus 21?')]
    for repeat in range(a.repeat):
        order = a.methods[repeat % len(a.methods):] + a.methods[:repeat % len(a.methods)]
        for method in order:
            run_method(method, repeat, a, rows, prompts, warmups)
    save(a.output/'status.json', {'execution_complete': True, 'full30': not a.smoke and not a.profile_reference})


if __name__ == '__main__':
    main()
