"""Generate paired tables, per-depth paths and signed speedup-loss accounting."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

from partition_profile import CATEGORIES

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'2026-10-29-thinking-batch-profile'
BATCHES = (1,2,4,8,16)
PATTERNS = ('simultaneous','staggered')
VARIANTS = ('ar','old','new')


def read(path):
    return json.loads(path.read_text())


def write(root, name, rows):
    (root/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows and isinstance(rows,list):
        with (root/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


def stats(values):
    v=sorted(values)
    return dict(median=st.median(v),p95=v[math.ceil(.95*len(v))-1],min=v[0],max=v[-1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--analyze',action='store_true');a=parser.parse_args()
    out=a.root/'analysis';out.mkdir(exist_ok=True)
    timings=[];profiles=[];depth_rows=[];token_pairs=[];request_rows=[];versions={}
    cells={};profile_map={}
    for bs,pattern in itertools.product(BATCHES,PATTERNS):
        runs_by_variant={}
        for variant in VARIANTS:
            name=f'{variant}-b{bs}-{pattern}'
            perf=a.root/'performance'/name;prof=a.root/'profile'/name
            if a.analyze:
                for script,product in [(OLD/'analyze_profile.py','summary.json'),
                                       (OLD/'enrich_profile.py','token-intervals.json'),
                                       (HERE/'partition_profile.py','partition.json')]:
                    if not (prof/product).exists():
                        subprocess.run([sys.executable,str(script),str(prof)],check=True)
            runs=[read(p) for p in sorted(perf.glob('cohort-*.json'))]
            assert len(runs)==5, (name,len(runs))
            manifest=read(perf/'manifest.json');versions[name]=manifest['sha']
            runs_by_variant[variant]=runs
            requests=[r for run in runs for r in run['requests']]
            span=stats([r['makespan_ms'] for r in runs])
            latency=stats([r['e2e_ms'] for r in requests])
            row=dict(variant=variant,bs=bs,pattern=pattern,repeats=len(runs),
                makespan_ms=span['median'],makespan_min_ms=span['min'],makespan_max_ms=span['max'],
                request_e2e_median_ms=latency['median'],request_e2e_p95_ms=latency['p95'],
                ttft_ms=st.median(r['ttft_ms'] for r in requests),
                decode_ms=st.median(r['e2e_ms']-r['ttft_ms'] for r in requests),
                tokens_per_cohort=st.median(sum(len(r['output_ids']) for r in run['requests']) for run in runs),
                tokens_per_second=st.median(sum(len(r['output_ids']) for r in run['requests'])/run['makespan_ms']*1000 for run in runs),
                length_stops=sum(r['finish_reason']['type']=='length' for r in requests),
                measured_requests=len(requests),max_send_lateness_ms=max(r['send_lateness_ms'] for r in requests))
            timings.append(row);cells[(variant,bs,pattern)]=row
            for q in range(bs):
                rs=[run['requests'][q] for run in runs]
                request_rows.append(dict(variant=variant,bs=bs,pattern=pattern,question=q+1,
                    e2e_ms=st.median(r['e2e_ms'] for r in rs),ttft_ms=st.median(r['ttft_ms'] for r in rs),
                    tokens=[len(r['output_ids']) for r in rs],stops=[r['finish_reason'] for r in rs]))
            summary=read(prof/'summary.json');detail=read(prof/'token-intervals.json');partition=read(prof/'partition.json')
            assert not detail['failures']
            p=dict(name=name,variant=variant,bs=bs,pattern=pattern,
                profile_ms=summary['makespan_ms']['median'],
                profiler_slowdown=summary['makespan_ms']['median']/row['makespan_ms'],
                stages=summary['stages'],depths=detail['depths'],prefill=detail['prefill'],
                kernel_busy_union_ms=summary['gpu_kernel_busy_union_ms'],partition=partition)
            profiles.append(p);profile_map[(variant,bs,pattern)]=p
            for depth,stages in summary['stage_time_per_output'].items():
                n=summary['entry_depth'][depth]['cpu_ms']['n']
                if not n:continue
                for stage,metrics in stages.items():
                    depth_rows.append(dict(variant=variant,bs=bs,pattern=pattern,depth=int(depth),outputs=n,stage=stage,
                        cpu_ms=metrics['cpu_ms'].get('median',0),gpu_stream_ms=metrics['gpu_stream_ms'].get('median',0),
                        calls=metrics['calls'].get('median',0)))
        # A prompt mismatch invalidates paired results, even if token counts agree.
        prompt_lists=[read(a.root/'performance'/f'{v}-b{bs}-{pattern}'/'prompts.json') for v in VARIANTS]
        assert prompt_lists[0]==prompt_lists[1]==prompt_lists[2]
        for first,second in [('ar','old'),('ar','new'),('old','new')]:
            for rep,(x,y) in enumerate(zip(runs_by_variant[first],runs_by_variant[second],strict=True)):
                for q,(u,v) in enumerate(zip(x['requests'],y['requests'],strict=True)):
                    diff=next((i for i,(l,r) in enumerate(zip(u['output_ids'],v['output_ids'])) if l!=r),None)
                    if diff is None and len(u['output_ids'])!=len(v['output_ids']):diff=min(len(u['output_ids']),len(v['output_ids']))
                    token_pairs.append(dict(pair=f'{first}/{second}',bs=bs,pattern=pattern,repeat=rep,question=q+1,
                        exact=u['output_ids']==v['output_ids'] and u['finish_reason']==v['finish_reason'],
                        first_difference=diff))
    comparison=[];losses=[];contributions=[];projection_limits=[]
    def elapsed(v,b,p):return cells[(v,b,p)]['makespan_ms']
    def projected(v,b,p,category,model):
        profile=profile_map[(v,b,p)]
        if model=='whole_wall':
            return elapsed(v,b,p)*profile['partition']['partition_ms'][category]/profile['profile_ms']
        parts=profile['partition']['resource_partition_ms']
        busy=sum(parts['gpu'].values());host=profile['profile_ms']-busy
        if elapsed(v,b,p)<=busy or host<=0:
            return None
        return parts['gpu'][category]+(parts['cpu'][category]+parts['uncovered'][category])*(elapsed(v,b,p)-busy)/host
    for pattern in PATTERNS:
        for bs in BATCHES:
            ar,old,new=(elapsed(v,bs,pattern) for v in VARIANTS)
            comparison.append(dict(bs=bs,pattern=pattern,ar_ms=ar,old_ms=old,new_ms=new,
                old_speedup=ar/old,new_speedup=ar/new,new_vs_old_reduction_percent=(1-new/old)*100,
                performance_gate=new<=1.02*old))
            if bs==1:continue
            e={}
            for v in ('old','new'):
                ratio=elapsed(v,1,pattern)/elapsed('ar',1,pattern)
                e[v]=elapsed(v,bs,pattern)-ratio*ar
                for model in ('whole_wall','gpu_fixed'):
                    endpoints=[(v,bs),(v,1),('ar',bs),('ar',1)]
                    values={key:{k:projected(*key,pattern,k,model) for k in CATEGORIES} for key in endpoints}
                    if any(x is None for group in values.values() for x in group.values()):
                        projection_limits.append(dict(variant=v,bs=bs,pattern=pattern,projection=model,
                            reason='Measured profile GPU occupancy exceeds unprofiled wall; GPU-fixed projection invalid.'))
                        continue
                    parts={k:(values[(v,bs)][k]-values[(v,1)][k])-
                        ratio*(values[('ar',bs)][k]-values[('ar',1)][k]) for k in CATEGORIES}
                    assert abs(sum(parts.values())-e[v]) < .01
                    for k,ms in parts.items():
                        contributions.append(dict(variant=v,bs=bs,pattern=pattern,projection=model,category=k,
                            projected_contribution_ms=ms,percent_of_loss=100*ms/e[v] if e[v]>0 else None,
                            total_loss_ms=e[v]))
            gaps={v:elapsed('ar',1,pattern)/elapsed(v,1,pattern)-ar/elapsed(v,bs,pattern) for v in ('old','new')}
            losses.append(dict(bs=bs,pattern=pattern,old_loss_ms=e['old'],new_loss_ms=e['new'],
                measured_loss_recovery_ms=e['old']-e['new'],
                measured_loss_recovery_percent=100*(e['old']-e['new'])/e['old'] if e['old']>0 else None,
                old_speedup_drop=gaps['old'],new_speedup_drop=gaps['new'],
                speedup_drop_recovery_percent=100*(gaps['old']-gaps['new'])/gaps['old'] if gaps['old']>0 else None))
    for name,data in [('performance',timings),('comparison',comparison),('requests',request_rows),
                      ('depth-stages',depth_rows),('profile-summary',profiles),('token-comparison',token_pairs),
                      ('speedup-loss',losses),('attribution-estimate',contributions),
                      ('projection-limits',projection_limits),('versions',versions)]:
        write(out,name,data)
    lines=['# 批量 endpoint 决策：AR / 旧 LoopSpec / 新 LoopSpec', '',
        'Ouro-2.6B-Thinking，BF16，R=4，draft depth=1，原生 overlap / LoopSpec V2、core/readout CUDA graph。新决策为批量 eager，未新增决策 CUDA graph。', '',
        'AIME 前 bs 题，各最多512新token，保留自然EOS；不是完整解题耗时。每格2次预热、5次无Profiler测量；算法块按bs轮换顺序，同GPU串行。交错每题间隔100ms。E2E指整批最早发送到最晚token。', '',
        '|发送|bs|AR s|旧 LS s|新 LS s|旧加速比|新加速比|新比旧缩时|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"|{r['pattern']}|{r['bs']}|{r['ar_ms']/1000:.3f}|{r['old_ms']/1000:.3f}|{r['new_ms']/1000:.3f}|{r['old_speedup']:.3f}×|{r['new_speedup']:.3f}×|{r['new_vs_old_reduction_percent']:.1f}%|")
    lines+=['','全部单请求E2E/P95、TTFT、decode、输出量、发送偏差与重复范围见 performance.csv / requests.csv。', '',
        '## 正确性和比较边界','']
    for pair in ('ar/old','ar/new','old/new'):
        group=[r for r in token_pairs if r['pair']==pair]
        lines.append(f"- {pair} 完整token序列及停止原因一致：{sum(r['exact'] for r in group)}/{len(group)}；首差异见 token-comparison.csv。")
    lines+=['','自由生成不一致不等于这次决策合批错误，也不能被当成无损通过。必须结合固定logits严格决策测试、独立同前缀数值和冻结旧版对照。测试记录另见 record.md。','',
        '## 每输出token的精细Profile','',
        'depth0尚需4轮、depth3已有3轮状态可复用。下表是该token路径经历的共享阶段中位数CPU/GPU stream ms；各阶段嵌套且CPU/GPU重叠，不能相加为E2E，也不能跨请求累计为独占成本。GPU stream包括等待。首token的prefill见profile-summary.json，原始共享step IDs仍在X5 output-paths.json。', '',
        '|发送|bs|入口depth|输出数|准备CPU/GPU|core CPU/GPU|readout CPU/GPU|decision CPU/GPU|commit CPU/GPU|retain CPU/GPU|回读CPU|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pattern,bs,depth in itertools.product(PATTERNS,BATCHES,(0,3)):
        group={r['stage']:r for r in depth_rows if (r['variant'],r['bs'],r['pattern'],r['depth'])==('new',bs,pattern,depth)}
        if not group:continue
        def fmt(stage):
            r=group.get(stage,{})
            return f"{r.get('cpu_ms',0):.3f}/{r.get('gpu_stream_ms',0):.3f}"
        values=[fmt(s) for s in ('loopspec.prepare_rows','recurrent.execute_core','recurrent.finish','loopspec.decision','loopspec.commit','loopspec.retain')]
        lines.append(f"|{pattern}|{bs}|{depth}|{next(iter(group.values()))['outputs']}|"+'|'.join(values)+f"|{group.get('loopspec.decision_readback',{}).get('cpu_ms',0):.3f}|")
    lines+=['', '新版本的独立readback span在旧版本属于scatter余项；比较余项时须加回新readback，不应将插桩边界变化当成工作消失。计划/collect、native result handler和客户端ITL见profile-summary.json中的depths；它们是包含GPU等待的主机区间，不能再与上述GPU时间相加。','',
        '## 实际分支计算量和readout graph覆盖','',
        'LoopSpec并行推进未来token的depth；全接受时主要减少串行关键路径，不按同样倍数减少core行计算量。下表按共享step只计一次。graph覆盖按真实endpoint数、捕获集合和当前无hidden-state请求的精确行数路由核对。', '',
        '|发送|版本|bs|平均core行数|每提交token的core行推进数|含padding行推进数|readout graph覆盖|',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in profiles:
        shape=p['partition']['execution_shapes']
        if shape:
            lines.append(f"|{p['pattern']}|{p['variant']}|{p['bs']}|{shape['mean_live_core_rows']:.2f}|{shape['core_row_updates_per_token']:.3f}|{shape['padded_core_row_updates_per_token']:.3f}|{shape['finish_graph_eligible_percent']:.1f}%|")
    lines+=['',
        '## 为什么batch增大时加速比下降：实测修复收益', '',
        '定义 r=L1/A1，E_B=L_B−r·A_B。E_B表示在bs=B保持bs1加速比还需省下的LoopSpec时间。E_B≤0表示加速比没有下降；保留负值表示改善，不定义损失贡献或恢复百分比。旧、新各用自己的r；因此“恢复比例”扣除了bs1也获得的优化，不等于新旧E2E降幅。该干预包含greedy特化、合批和减少无用操作，不能单独归因于某一个算子。', '',
        '|发送|bs|旧损失s|新损失s|修复消除损失s|消除比例|',
        '|---|---:|---:|---:|---:|---:|']
    for r in losses:
        percent=r['measured_loss_recovery_percent']
        recovery='无下降' if percent is None else f'{percent:.1f}%'
        lines.append(f"|{r['pattern']}|{r['bs']}|{r['old_loss_ms']/1000:.3f}|{r['new_loss_ms']/1000:.3f}|{r['measured_loss_recovery_ms']/1000:.3f}|{recovery}|")
    lines+=['','上表是额外耗时损失的恢复比例。直接比较加速比跌幅 D_B=S_1−S_B，数值不同：', '',
        '|发送|bs|旧加速比跌幅|新加速比跌幅|跌幅收窄比例|',
        '|---|---:|---:|---:|---:|']
    for r in losses:
        percent=r['speedup_drop_recovery_percent']
        recovery='无下降' if percent is None else f'{percent:.1f}%'
        lines.append(f"|{r['pattern']}|{r['bs']}|{r['old_speedup_drop']:.3f}|{r['new_speedup_drop']:.3f}|{recovery}|")
    lines+=['', '## 各步骤的有符号时间分解估算（不是精确因果比例）','',
        '先将Profile时间轴分成互斥区间：GPU执行优先，CPU仅计算未被GPU覆盖的阶段，未知空隙保留未归因。whole_wall以各区间占比统一缩放到无Profiler E2E；gpu_fixed保留实测GPU忙时，仅缩放CPU/空隙到剩余E2E。后者假设Profile kernel时长和工作形状能代表无Profiler，若GPU忙时已超过无Profiler E2E则拒绝估算。两种口径差异是方法敏感性，不是置信区间。', '',
        '贡献 Δj=(L̂Bj−L̂1j)−r(ÂBj−Â1j)，各项之和为E_B；Δj/(r·L_B)之和等于加速比跌幅，因此同一版本/bs内的贡献百分比也对应其加速比跌幅。负数表示抵消下降，正数可能超过100%。阶段含其CPU提交空隙；运行时余项可包含由外层runtime提交、未细分的GPU算子。不能把占用分区称为精确因果贡献或可全部回收的优化收益。', '',
        '|发送|版本|bs|投影口径|attention|模型其他计算|decision|分支/KV|prepare|运行时余项|传输|未归因|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pattern,v,bs,model in itertools.product(PATTERNS,('old','new'),BATCHES[1:],('whole_wall','gpu_fixed')):
        group={r['category']:r for r in contributions if (r['pattern'],r['variant'],r['bs'],r['projection'])==(pattern,v,bs,model)}
        values=['不可估' if k not in group else ('无下降' if group[k]['percent_of_loss'] is None else f"{group[k]['percent_of_loss']:.1f}%") for k in CATEGORIES]
        lines.append(f'|{pattern}|{v}|{bs}|{model}|'+ '|'.join(values)+'|')
    lines+=['', 'bs16新增的题目有648/861 token长输入，而bs8最长182；这不是固定长度的纯batch扩展。attention增长包含请求数、分支行数、上下文长度、接受率和计算形状变化。没有独立attention干预实验，不能将上表称为attention优化必然可回收的收益。', '',
        '|发送|bs|AR Profile/实测|旧LS Profile/实测|新LS Profile/实测|',
        '|---|---:|---:|---:|---:|']
    for pattern,bs in itertools.product(PATTERNS,BATCHES):
        lines.append(f'|{pattern}|{bs}|'+ '|'.join(f"{profile_map[(v,bs,pattern)]['profiler_slowdown']:.3f}×" for v in VARIANTS)+'|')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out,'status',dict(cells=len(comparison),all_performance_gates=all(r['performance_gate'] for r in comparison),
        profile_cells=len(profiles),note='Performance gates do not imply token-losslessness; see numerical and sequence evidence.'))
    print(json.dumps(read(out/'status.json')),flush=True)


if __name__=='__main__':main()
