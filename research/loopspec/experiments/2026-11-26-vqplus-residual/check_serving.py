"""TP2 client validation and pure-decode timing of an explicitly launched server.

No accuracy benchmarks and no automatic process termination. Raw input IDs,
SSE counts, log slices and identity allow independent verification.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / '2026-11-23-loopvq-reproduction'))
import benchmark_throughput as bench
from decode_sweep import decode_interval
from serving_probe import exact_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:30664')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--server-log', type=Path, required=True)
    p.add_argument('--method', choices=['bf16', 'loopvq', 'vqplus'], required=True)
    p.add_argument('--suite', choices=['timing', 'lifecycle', 'profile'], required=True)
    p.add_argument('--input-length', type=int, default=2048)
    p.add_argument('--output-length', type=int, default=128)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--reference', type=Path)
    p.add_argument('--expected-layout', choices=['token','page','page-head'])
    p.add_argument('--expected-page-kernel', choices=['cuda','triton'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    def request(path, payload=None):
        return urlopen(Request(a.url + path, None if payload is None else json.dumps(payload).encode(),
                               {'Content-Type': 'application/json'}), timeout=1800)

    with request('/get_server_info') as response:
        info = json.load(response)
    cfg = info.get('server_args', info)
    assert cfg['tp_size'] == 2 and cfg['dp_size'] == 1
    assert Path(cfg['model_path']).resolve() == a.model.resolve()
    assert cfg['ouro_decode_profile'] == 'optimized' and cfg['speculative_algorithm'] is None
    assert json.loads((a.model / 'config.json').read_text())['total_ut_steps'] == 4
    assert bool(cfg['loopvq_codebook_dir']) == (a.method != 'bf16')
    assert cfg.get('loopvq_residual_bits', 0) == (4 if a.method == 'vqplus' else 0)
    if a.expected_layout is not None:
        assert cfg.get('loopvq_residual_layout','token') == a.expected_layout
    if a.expected_page_kernel is not None:
        assert cfg.get('loopvq_residual_page_kernel','cuda') == a.expected_page_kernel
    if a.method != 'bf16':
        assert (cfg['loopvq_prefix_tokens'], cfg['loopvq_recent_tokens']) == (8, 64)
    assert cfg['disable_radix_cache'] and cfg['chunked_prefill_size'] == -1
    assert cfg['decode_log_interval'] == 1
    capture = bench.capture_from_log(a.server_log)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    fixture = exact_prompt(tokenizer, a.input_length)
    (a.out / 'input.json').write_text(json.dumps(fixture))
    allowed = ['model_path', 'tp_size', 'dp_size', 'dtype', 'context_length', 'page_size',
               'max_total_tokens', 'max_running_requests', 'cuda_graph_config',
               'loopvq_codebook_dir', 'loopvq_residual_bits', 'ouro_decode_profile',
               'loopvq_prefix_tokens', 'loopvq_recent_tokens', 'sampling_backend',
               'loopvq_residual_layout','loopvq_residual_page_kernel','triton_attention_num_kv_splits']
    identity = {'client_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'config': {k: cfg.get(k) for k in allowed}, 'capture': capture, 'total_ut_steps': 4}
    if a.method != 'bf16':
        identity['book_sha256'] = {}
        for name in ('codebook.pt', 'vq_v_codebook.pt'):
            digest = hashlib.sha256()
            with (Path(cfg['loopvq_codebook_dir']) / name).open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''): digest.update(chunk)
            identity['book_sha256'][name] = digest.hexdigest()
    (a.out / 'identity.json').write_text(json.dumps(identity, indent=2))

    def generate(label, b, n, profile=False):
        payload = {'input_ids': [fixture['input_ids']] * b, 'stream': True,
                   'sampling_params': {'temperature': 0, 'ignore_eos': True, 'max_new_tokens': n, 'stream_interval': 1}}
        stats = bench.StreamStats(b, n, a.input_length)
        log_start = a.server_log.stat().st_size
        events = a.out / (label + '.jsonl')
        start, done, profiled = time.perf_counter(), False, False
        with request('/generate', payload) as response, events.open('x') as stream:
            for line in response:
                if not line.startswith(b'data:'): continue
                item = line[5:].strip()
                if item == b'[DONE]': done = True; break
                obs = stats.feed(json.loads(item), time.perf_counter() - start)
                stream.write(json.dumps(obs) + '\n')
                if profile and not profiled and min(r['completion_tokens'] for r in stats.requests) >= 32:
                    with request('/start_profile', {'activities': ['CPU', 'GPU'], 'num_steps': 10,
                         'output_dir': str(a.out / label), 'with_stack': False, 'record_shapes': False}) as r:
                        (a.out / (label + '-profile-response.txt')).write_bytes(r.read())
                    profiled = True
        assert done
        stats.finalize(time.perf_counter() - start)
        # Scheduler output can trail the final HTTP chunk by one event-loop pass.
        time.sleep(.1)
        with a.server_log.open('rb') as f:
            f.seek(log_start); logs = f.read().decode(errors='replace')
        (a.out / (label + '-server.log')).write_text(logs)
        evidence = bench.decode_evidence(logs, b, n, cfg, capture)
        result = decode_interval(events, b, n)
        result.update(batch=b, input_length=a.input_length, output_length=n,
                      decode_evidence=evidence, profile=profile, label=label)
        (a.out / (label + '.json')).write_text(json.dumps(result, indent=2))
        assert evidence['verified'], evidence
        if profile: assert profiled
        return result

    results = []
    if a.suite in ('timing', 'profile'):
        for b in a.batches:
            generate(f'warm-b{b}', b, 32)
            for r in range(a.repeats if a.suite == 'timing' else 1):
                result = generate(f'b{b}-r{r}', b, a.output_length, a.suite == 'profile')
                results.append(result)
                (a.out / 'summary.json').write_text(json.dumps(results, indent=2))
                print(b, r, result['decode_tokens_s'], flush=True)
    else:
        def greedy(label, b, n):
            with request('/generate', {'input_ids': [fixture['input_ids']] * b,
                         'sampling_params': {'temperature': 0, 'max_new_tokens': n, 'ignore_eos': True}}) as r:
                rows = json.load(r)
            assert len(rows) == b and all(len(x['output_ids']) == n for x in rows)
            (a.out / (label + '.json')).write_text(json.dumps(rows))
            if a.reference:
                ref = json.loads((a.reference / (label + '.json')).read_text())
                assert [r['output_ids'] for r in ref] == [r['output_ids'] for r in rows], label
            return rows
        before = greedy('before', 1, 128)
        for b in (3, 5, 15, 16): greedy(f'pad{b}', b, 48)
        rid = uuid.uuid4().hex
        with request('/generate', {'rid': rid, 'input_ids': fixture['input_ids'], 'stream': True,
                     'sampling_params': {'temperature': 0, 'max_new_tokens': 512, 'ignore_eos': True}}) as stream:
            for line in stream:
                if line.startswith(b'data:') and b'completion_tokens' in line:
                    with request('/abort_request', {'rid': rid}) as r: r.read()
                    break
            else: raise AssertionError('No abort opportunity')
        time.sleep(2)
        after = greedy('after', 1, 128)
        assert before[0]['output_ids'] == after[0]['output_ids']
        (a.out / 'summary.json').write_text(json.dumps({'passed': True, 'abort_and_reuse': True}))


if __name__ == '__main__':
    main()
