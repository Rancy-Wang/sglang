"""Validate complete greedy cells and publish a Markdown accuracy table."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys

ARMS = {'bf16': 'bf16', 'loopvq': 'loopvq_0.5bit', 'residual': 'loopvq_residual4',
        'plain-continued': 'loopvq_plain_continued', 'residual-ft': 'loopvq_residual_ft'}
LABELS = ['BF16', 'LoopVQ 0.5bit', '原码本＋INT4 残差', '普通 KL 续训', '残差感知续训＋INT4']
BENCHES = {'aime24': ('AIME24', 30), 'aime25': ('AIME25', 30), 'gpqa_diamond': ('GPQA-D', 198),
           'evalplus_humaneval': ('HumanEval+', 164), 'evalplus_mbpp': ('MBPP+', 378)}


def checked(path, arm, bench, collect_cell):
    item = json.loads(path.read_text()); cell = Path(item['cell']); method = ARMS[arm]
    assert (item['arm'], item['benchmark'], item['seed']) == (arm, bench, 0)
    marker = json.loads((cell/'complete.json').read_text())
    assert not marker['audit'] and marker['exit_code'] == 0
    assert (marker['method'], marker['benchmark'], marker['seed']) == (method, bench, 0)
    stem = f'{method}_0_{bench}'
    scores = collect_cell(cell, stem, bench, BENCHES[bench][1])
    assert scores == marker['scores'] == item['scores']
    recipe = json.loads(Path(item['recipe']).read_text())
    protocol = json.loads((cell/(stem+'_protocol.json')).read_text())
    for k, v in {'temperature': 0.0, 'top_p': 1.0, 'max_gen_toks': 32768, 'limit': None, 'api_seed': 0}.items():
        assert protocol[k] == v, (cell, k)
    assert recipe['temperature'] == 0 and recipe['top_p'] == 1 and recipe['max_tokens'] == 32768
    server = recipe['server']
    assert server['tp'] == 2 and server['total_ut_steps'] == 4 and server['context_length'] == 40960
    assert server['implementation'] == 'sglang' and server['decoding'] == 'greedy'
    rows = list(map(json.loads, (cell/(stem+'_samples.jsonl')).read_text().splitlines()))
    correct = sum(float(r['plus'] == 'pass') if bench.startswith('evalplus_') else float(r['exact_match']) for r in rows)
    assert math.isclose(100*correct/len(rows), scores['accuracy_percent'], abs_tol=1e-8)
    responses = list((cell/'http/responses').glob('*.json'))
    assert len(responses) == len(rows)
    identities = []
    truncated = 0
    for p in responses:
        r = json.loads(p.read_text()); payload = r['payload']; body = r['response']
        assert payload['temperature'] == 0 and payload['top_p'] == 1 and payload['max_tokens'] == 32768
        assert payload['seed'] == 0 and payload['chat_template_kwargs']['enable_thinking'] is True
        assert body['usage']['prompt_tokens'] == len(r['input_token_ids'])
        assert len(r['input_token_ids']) + 32768 <= 40958
        if body['choices'][0]['finish_reason'] == 'length':
            assert body['usage']['completion_tokens'] == 32768, 'Output budget was shortened'
            truncated += 1
        identities.append((r['input_token_ids'], payload.get('stop')))
    identities.sort(key=lambda x: json.dumps(x))
    shared = {k: recipe[k] for k in ('tokenizer_files', 'sources', 'packages')}
    shared['model_sha256'] = server['model_sha256']
    shared['data'] = {k: protocol[k] for k in ('dataset_documents_sha256', 'dataset_hash', 'task_ids') if k in protocol}
    assert shared['data']
    return {**scores, 'correct': int(correct), 'length_finishes': truncated}, shared, identities


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--loopvq',type=Path,required=True)
    args=ap.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0, str(args.loopvq/'experiments/loopvq_reproduction'))
    from run_evaluation import collect_cell
    from evaluation_io import atomic_json
    with (args.root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        results={}; baselines={}; inputs={}
        for bench in BENCHES:
            for arm in ARMS:
                p=args.root/'done'/f'{arm}__{bench}__0.json'
                if not p.exists(): continue
                score, shared, ids = checked(p,arm,bench,collect_cell)
                if bench in baselines:
                    assert baselines[bench] == shared, f'Incompatible evidence {arm}/{bench}'
                    assert inputs[bench] == ids, f'Prompts/stops differ {arm}/{bench}'
                else: baselines[bench],inputs[bench] = shared,ids
                results[f'{arm}/{bench}']=score
        lines=['# Greedy accuracy — Ouro-2.6B-Thinking', '',
               'TP=2，完整 4 loops；temperature=0、top_p=1；新增输出上限 32768，总上下文 40960。每题只生成一次，保留 EOS/原评分器停止条件。',
               '数值仅包含完整 benchmark；未完成不以部分样本估计。HumanEval+/MBPP+ 使用 plus pass@1。', '',
               '| Benchmark | '+ ' | '.join(LABELS)+' |', '| --- | '+' | '.join(['---']*5)+' |']
        for b,(label,n) in BENCHES.items():
            cells=[]
            for a in ARMS:
                r=results.get(f'{a}/{b}'); cells.append(f"{r['accuracy_percent']:.2f}% ({r['correct']}/{n})" if r else '待完成')
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines+=['', f'已验收 {len(results)}/25 个单元，{sum(r["n"] for r in results.values())}/4000 个评分样本。', '',
                '| Benchmark | 方法 | 达到输出长度上限 |', '| --- | --- | --- |']
        for key,r in results.items():
            a,b=key.split('/'); lines.append(f'| {BENCHES[b][0]} | {LABELS[list(ARMS).index(a)]} | {r["length_finishes"]}/{r["n"]} |')
        atomic_json(args.root/'accuracy.json',{'complete':len(results)==25,'decoding':'greedy','results':results})
        temp=args.root/'accuracy.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(args.root/'accuracy.md')


if __name__=='__main__': main()
