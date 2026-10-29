"""Merge performance, token differences and shared-stage profile into auditable tables."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
med=lambda xs:st.median(xs) if xs else None
rows=[];requests=[];differences=[];depths=[]
for bs in [1,2,4,8,16]:
    for pattern in ['simultaneous','staggered']:
        paired={}
        for algo in ['ar','loopspec']:
            name=f'{algo}-b{bs}-{pattern}'; folder=a.root/'performance'/name
            runs=[json.loads(f.read_text()) for f in sorted(folder.glob('cohort-*.json'))]
            if not runs:continue
            paired[algo]=runs
            reqs=[q for r in runs for q in r['requests']]
            row={'algorithm':algo,'bs':bs,'pattern':pattern,'repeats':len(runs),
                'makespan_ms':med([r['makespan_ms'] for r in runs]),
                'makespan_min_ms':min(r['makespan_ms'] for r in runs),
                'makespan_max_ms':max(r['makespan_ms'] for r in runs),
                'request_e2e_median_ms':med([q['e2e_ms'] for q in reqs]),
                'ttft_median_ms':med([q['ttft_ms'] for q in reqs]),
                'decode_median_ms':med([q['e2e_ms']-q['ttft_ms'] for q in reqs]),
                'output_tokens_per_cohort':med([sum(len(q['output_ids']) for q in r['requests']) for r in runs]),
                'throughput_tokens_s':med([sum(len(q['output_ids']) for q in r['requests'])/(r['makespan_ms']/1000) for r in runs]),
                'length_stops':sum(q['finish_reason']['type']=='length' for q in reqs),
                'requests_measured':len(reqs),'max_send_lateness_ms':max(q['send_lateness_ms'] for q in reqs)}
            rows.append(row)
            for q in range(bs):
                rr=[r['requests'][q] for r in runs]
                requests.append({'algorithm':algo,'bs':bs,'pattern':pattern,'question':q+1,
                    'e2e_ms':med([r['e2e_ms'] for r in rr]),'ttft_ms':med([r['ttft_ms'] for r in rr]),
                    'decode_ms':med([r['e2e_ms']-r['ttft_ms'] for r in rr]),
                    'tokens':[len(r['output_ids']) for r in rr],
                    'finish_reasons':[r['finish_reason'] for r in rr]})
            prof=a.root/'profile'/name/'summary.json'
            if prof.exists():
                summary=json.loads(prof.read_text())
                for d,stages in summary['stage_time_per_output'].items():
                    count=summary['entry_depth'][d]['cpu_ms']['n']
                    if not count:continue
                    for stage,metrics in stages.items():
                        depths.append({'algorithm':algo,'bs':bs,'pattern':pattern,'entry_depth':int(d),
                            'outputs':count,'stage':stage,'cpu_median_ms':metrics['cpu_ms']['median'],
                            'gpu_stream_median_ms':metrics['gpu_stream_ms']['median'],
                            'calls_median':metrics['calls']['median']})
        if len(paired)==2:
            for repeat,(ar,ls) in enumerate(zip(paired['ar'],paired['loopspec'],strict=True)):
                for q,(x,y) in enumerate(zip(ar['requests'],ls['requests'],strict=True)):
                    first=next((i for i,(u,v) in enumerate(zip(x['output_ids'],y['output_ids'])) if u!=v),None)
                    equal=x['output_ids']==y['output_ids'] and x['finish_reason']==y['finish_reason']
                    differences.append({'bs':bs,'pattern':pattern,'repeat':repeat,'question':q+1,
                        'exact':equal,'first_difference':first,'ar_tokens':len(x['output_ids']),
                        'loopspec_tokens':len(y['output_ids'])})
for name,data in [('performance',rows),('requests',requests),('depth-stages',depths),('token-comparison',differences)]:
    (a.root/f'{name}.json').write_text(json.dumps(data,indent=2))
    if data:
        with (a.root/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
text=['# Ouro-2.6B-Thinking：AR / LoopSpec AIME profile',
    '', 'BF16，完整 R=4，LoopSpec draft depth=1。每题最多 512 新 token，保留自然 EOS；达到 length 上限为截断，不能解释成完整解题耗时。',
    '', '同时发送延迟全 0；交错发送每题间隔 100ms。每格 2 次预热、5 次测量。E2E 为客户端发起到最后 token；整批耗时从最早发送到最后 token。Profile 单独运行，不能代替下表延迟。',
    '', '|发送|bs|AR 整批 s|LoopSpec 整批 s|AR/LS|AR 每请求 E2E 中位 s|LS 每请求 E2E 中位 s|AR/LS 输出 token|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for pattern in ['simultaneous','staggered']:
    for bs in [1,2,4,8,16]:
        pair={r['algorithm']:r for r in rows if r['bs']==bs and r['pattern']==pattern}
        if len(pair)!=2:continue
        ar,ls=pair['ar'],pair['loopspec']
        text.append(f"|{pattern}|{bs}|{ar['makespan_ms']/1000:.3f}|{ls['makespan_ms']/1000:.3f}|{ar['makespan_ms']/ls['makespan_ms']:.3f}×|{ar['request_e2e_median_ms']/1000:.3f}|{ls['request_e2e_median_ms']/1000:.3f}|{ar['output_tokens_per_cohort']}/{ls['output_tokens_per_cohort']}|")
text+=['', f"逐请求、逐重复 token 完全一致：{sum(d['exact'] for d in differences)}/{len(differences)}。首个差异保留于 token-comparison.csv；不能仅凭此速度表宣称逐 token 无损。",'',
    '## 精细 Profile（每输出 token 路径所经历的共享阶段）','',
    'depth 从 0 计数：0 表示尚需 4 个 core loop，3 表示已经提前算完 3 个 loop，仍需 1 个。AR decode 均执行完整 4 loop；prefill 首 token 独立记录。',
    'CPU 为调用墙钟区间；GPU 为当前计算流上的 CUDA event 区间，含流等待。嵌套阶段不可相加，多请求共同经历的阶段不是每请求独占成本，亦不可跨请求累加。原始共享 step IDs 见每格 output-paths.json；全局不重复统计见 summary.json。', '',
    '|发送|bs|算法|入口 depth|输出数|准备 CPU/GPU ms|core CPU/GPU ms|readout CPU/GPU ms|决策 CPU/GPU ms|提交 CPU/GPU ms|回收 CPU/GPU ms|',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for pattern in ['simultaneous','staggered']:
    for bs in [1,2,4,8,16]:
        for algo in ['ar','loopspec']:
            for d in range(4):
                group={r['stage']:r for r in depths if (r['pattern'],r['bs'],r['algorithm'],r['entry_depth'])==(pattern,bs,algo,d)}
                if not group:continue
                def value(stage):
                    if stage not in group:return '—'
                    r=group[stage];return f"{r['cpu_median_ms']:.3f}/{r['gpu_stream_median_ms']:.3f}"
                stages=['loopspec.prepare_rows' if algo=='loopspec' else 'recurrent.start','recurrent.execute_core',
                        'recurrent.finish','loopspec.decision','loopspec.commit','loopspec.retain']
                text.append(f"|{pattern}|{bs}|{algo}|{d}|{next(iter(group.values()))['outputs']}|"+'|'.join(value(s) for s in stages)+'|')
(a.root/'report.md').write_text('\n'.join(text)+'\n')
print(json.dumps({'cells':len(rows),'profiles':len({(r['algorithm'],r['bs'],r['pattern']) for r in depths}),
                  'token_pairs':len(differences),'exact_pairs':sum(r['exact'] for r in differences)}))
