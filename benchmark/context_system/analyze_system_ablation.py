"""First designated task-window accounting; no extrapolation of missing usage."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def stats(values):
    xs = sorted(x for x in values if x is not None and math.isfinite(x))
    if not xs: return dict(n=0,total=0,mean=None,p90=None)
    q=(len(xs)-1)*.9; i=int(q)
    return dict(n=len(xs), total=sum(xs), mean=statistics.mean(xs),
                p90=xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(q-i))


def lines(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def profile(root, start, cutoff):
    groups=defaultdict(list); dispatched=set(); completed=set(); missing_batch=0
    for path in sorted(Path(root).glob('*-profile.*.jsonl')):
        for r in lines(path):
            if r.get('rank') != 0: continue
            a,b=r.get('start_perf'),r.get('end_perf')
            if a is None or b is None or not start <= a <= b <= cutoff: continue
            stage=r['stage']; rec=r['record']
            if rec=='cpu_span':
                g=groups[stage,r['category']]; g.append(r)
            elif rec in ('dispatched','gpu_completed'):
                key=(stage,r['pid'],a)
                if not r.get('batch'): missing_batch+=rec=='dispatched'
                if rec=='dispatched':
                    dispatched.add(key)
                else:
                    if key in completed: raise ValueError('Duplicate GPU step')
                    completed.add(key);groups[stage,'decode' if r.get('decode_tokens',0)>0 else 'extend'].append(r)
    rows=[]
    categories=('cache_lookup_inclusive','cache_tree_lookup','eviction_pressure','eviction_alloc_inclusive',
                'ordinary_leaf_eviction','dropaware_page_eviction','dropaware_lock_configuration',
                'cache_back_finished','cache_back_unfinished',
                'dropaware_eligibility_update','extend','decode')
    for stage in ('prefill','decode'):
        for cat in categories:
            rs=groups[stage,cat]
            values=stats([r['cpu_ms'] for r in rs]);gpu=stats([r.get('gpu_ms') for r in rs])
            freed=[]
            for r in rs:
                before,after=r.get('before',{}),r.get('after',{})
                if cat in ('ordinary_leaf_eviction','dropaware_page_eviction'):
                    freed.append(sum(max(0,v-before.get(k,0)) for k,v in after.items()))
                elif cat in ('eviction_pressure','eviction_alloc_inclusive'):
                    freed.append(after.get('num_tokens_evicted',0)+after.get('swa_num_tokens_evicted',0))
            rows.append(dict(stage=stage,category=cat,count=len(rs),cpu_total_ms=values['total'],
                cpu_mean_ms=values['mean'],cpu_p90_ms=values['p90'],
                exclusive_total_ms=sum(r.get('exclusive_ms',0) for r in rs) if cat not in ('extend','decode') else None,
                gpu_span_total_ms=gpu['total'] if gpu['n'] else None,gpu_span_mean_ms=gpu['mean'],gpu_span_p90_ms=gpu['p90'],
                effective_free_count=sum(x>0 for x in freed) if freed else (0 if not rs else None),
                freed_component_tokens=sum(freed) if freed else (0 if not rs else None),
                hit_tokens_unknown_calls=sum(r.get('after',{}).get('hit_tokens') is None for r in rs) if cat=='cache_lookup_inclusive' else None,
                hit_tokens=sum(r.get('after',{}).get('hit_tokens') or 0 for r in rs) if cat=='cache_lookup_inclusive' else None,
                compute_tokens=sum(r.get('prefill_tokens',0)+r.get('decode_tokens',0) for r in rs) if cat in ('extend','decode') else None))
    out=dict(rows=rows,dispatched=len(dispatched),completed=len(completed),missing_gpu_records=len(dispatched-completed),
             missing_batch_identity=missing_batch,
             notes=['TP0 per stage only; never sum parallel stage/rank times as latency.',
                    'GPU event spans include gaps; not pure kernel union.',
                    'Only wholly contained spans; boundary-crossing spans excluded, raw retained.',
                    'Nested inclusive rows must not be added together; freed_component_tokens may include Full and SWA.'])
    write(Path(root)/'profile-summary.json',out)
    with (Path(root)/'profile.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return out


def account(root, report_tasks, allow_timed=False):
    assert report_tasks > 0
    root=Path(root);d=json.loads((root/'workload/result.json').read_text())
    primary=sorted((t for t in d['tasks'] if not t['filler']),key=lambda t:t['instance'])
    assert d['valid'] and not d.get('cohort_failure')
    completed = [t for t in primary if t['status']=='all_turns_completed']
    designated_complete = len(primary)>=report_tasks and all(t['status']=='all_turns_completed' for t in primary[:report_tasks])
    if not allow_timed:
        assert len(completed)==len(primary) and designated_complete
    else:
        assert all(t['status'] in ('all_turns_completed','cutoff_cancelled') for t in primary)
        if not designated_complete:
            assert d.get('stop_reason')=='measurement_time_limit'
    start=d['start'];end=max(t['end_time'] for t in primary[:report_tasks]) if designated_complete else d['cutoff'];seconds=end-start
    assert seconds > 0
    reqs={(r['instance'],r['turn']):r for r in d['turns'] if start<=r['start_time']<=end}
    usage={};done={}
    for e in lines(root/'workload/sse.jsonl'):
        t=e['received_perf']
        if not start<=t<=end: continue
        key=e['instance'],e['turn'];assert key in reqs
        if e['data']=='[DONE]':done[key]=t;continue
        value=json.loads(e['data'])
        if value.get('usage'):usage[key]=dict(usage=value['usage'],received_perf=t)
    samples=[dict(instance=k[0],turn=k[1],latency=t-reqs[k]['start_time'],ttft=reqs[k]['ttft'],
                       tpot=(t-reqs[k]['start_time']-reqs[k]['ttft'])/(reqs[k]['output_len']-1)
                       if reqs[k].get('ttft') is not None and reqs[k].get('output_len',0)>1 else None) for k,t in done.items() if reqs[k]['success']]
    mismatches=[(r['instance'],r['turn']) for r in d['turns'] if r['success'] and (r.get('usage') or {}).get('completion_tokens')!=r['requested_max_tokens']]
    inp=sum(v['usage']['prompt_tokens'] for v in usage.values());out=sum(v['usage']['completion_tokens'] for v in usage.values())
    result=dict(case=root.name,valid=not mismatches,report_tasks=report_tasks,start=start,cutoff=end,duration_s=seconds,
                input_tokens=inp,output_tokens=out,input_throughput=inp/seconds,output_throughput=out/seconds,
                throughput=(inp+out)/seconds,latency_s=stats([r['latency'] for r in samples]),ttft_s=stats([r['ttft'] for r in samples]),tpot_s=stats([r['tpot'] for r in samples]),
                known_usage_calls=len(usage),unknown_usage_calls=len(reqs)-len(usage),output_budget_mismatches=mismatches,
                primary_tasks=len(primary),primary_statuses=[t['status'] for t in primary],cohort_duration_s=d['cutoff']-start,
                designated_tasks_completed=designated_complete,completed_primary_tasks=len(completed),
                window_kind='designated_task_completion' if designated_complete else 'measurement_time_limit',
                stop_reason=d.get('stop_reason'))
    write(root/'window-metrics.json',result)
    with (root/'window-usage.jsonl').open('w') as f:
        for k,v in sorted(usage.items()):f.write(json.dumps(dict(instance=k[0],turn=k[1],filler=reqs[k]['filler'],**v))+'\n')
    assert not mismatches
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--report-tasks',type=int,required=True)
    p.add_argument('--allow-timed', action='store_true');args=p.parse_args()
    result=account(args.root,args.report_tasks,allow_timed=args.allow_timed);print(json.dumps(result))
    profile(args.root,result['start'],result['cutoff'])
