"""Exclusive resource-occupancy accounting; NOT a causal critical-path model.

GPU work takes precedence over overlapping CPU spans. CPU span remainders are
classified by the innermost relevant phase. Uninstrumented time stays unknown.
Chrome and perf_counter clocks are aligned with matched forward annotations.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import heapq
import json
from pathlib import Path
import statistics as st


CATEGORIES = ('attention', 'model_compute', 'decision', 'branch_kv',
              'prepare', 'host_runtime', 'transfer', 'unassigned')


def phase(name):
    if name in ('loopspec.decision', 'loopspec.decision_readback'):
        return 'decision', 50
    if name in ('loopspec.commit', 'loopspec.retain'):
        return 'branch_kv', 40
    if name in ('loopspec.prepare_rows', 'recurrent.prepare', 'recurrent.start'):
        return 'prepare', 35
    if name in ('recurrent.execute_core', 'recurrent.finish'):
        return 'model_compute', 30
    if name.startswith(('loopspec.', 'scheduler.', 'recurrent.')):
        return 'host_runtime', 10
    return 'unassigned', 0


def merged(intervals):
    result = []
    for left, right in sorted(intervals):
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def exclusive(intervals, left, right, *, resources=False):
    """Priority partition with union within each category; covers the full wall."""
    streams = []
    for (category, priority), values in intervals.items():
        events = []
        for begin, end in merged(values):
            begin, end = max(left, begin), min(right, end)
            if begin < end:
                events.extend(((begin, 1, category, priority), (end, -1, category, priority)))
        streams.append(iter(events))
    active, totals, previous = Counter(), Counter(), left
    for timestamp, delta, category, priority in heapq.merge(*streams):
        winner = max((p, c) for (c, p), count in active.items() if count > 0) if any(active.values()) else (0, 'unassigned')
        resource = 'gpu' if winner[0]>=90 else ('cpu' if winner[0] else 'uncovered')
        totals[(resource,winner[1])] += timestamp - previous
        active[(category, priority)] += delta
        previous = timestamp
    totals[('uncovered','unassigned')] += right - previous
    assert abs(sum(totals.values()) - (right-left)) < 1e-3
    if resources:
        return {resource:{key:totals[(resource,key)]/1000 for key in CATEGORIES}
                for resource in ('gpu','cpu','uncovered')}
    return {key:sum(totals[(resource,key)] for resource in ('gpu','cpu','uncovered'))/1000
            for key in CATEGORIES}


def partition(folder):
    runs = json.loads((folder/'cohort-0.json').read_text())
    records = [json.loads(line) for file in (folder/'trace').glob('*-recurrent-*.jsonl')
               for line in file.read_text().splitlines()]
    files = list((folder/'trace').glob('*.trace.json.gz'))
    assert len(files) == 1, 'This experiment uses one GPU rank'
    with gzip.open(files[0], 'rt') as stream:
        events = json.load(stream)['traceEvents']
    forwards = sorted((e for e in events if e.get('cat') == 'user_annotation'
                       and e.get('name') == 'recurrent::scheduler.forward'), key=lambda e:e['ts'])
    native = sorted((r for r in records if r['stage'] == 'scheduler.forward'), key=lambda r:r['start_ns'])
    assert len(forwards) == len(native) > 0
    offsets = [e['ts']-r['start_ns']/1000 for e,r in zip(forwards,native,strict=True)]
    offset = st.median(offsets)
    errors = sorted(abs(x-offset) for x in offsets)
    assert errors[int(.95*(len(errors)-1))] < 2000, 'Unreliable clock alignment'
    left = min(r['sent_ns'] for r in runs['requests'])/1000 + offset
    right = max(r['last_token_ns'] for r in runs['requests'])/1000 + offset
    spans = defaultdict(list)
    intervals = defaultdict(list)
    for e in events:
        if e.get('cat') != 'user_annotation' or not e.get('name','').startswith('recurrent::'):
            continue
        category, priority = phase(e['name'].removeprefix('recurrent::'))
        if priority:
            spans[(category, priority)].append((e['ts'], e['ts']+e['dur']))
            intervals[(category, priority)].append((e['ts'], e['ts']+e['dur']))
    indexes = {key:(group, [a for a,b in group]) for key,values in spans.items() if (group:=merged(values))}
    def launch_phase(timestamp):
        matches = []
        for (category, priority),(group,starts) in indexes.items():
            index = bisect_right(starts,timestamp)-1
            if index >= 0 and timestamp <= group[index][1]:
                matches.append((priority,category))
        return max(matches)[1] if matches else 'unassigned'
    launches = {e['args']['correlation']:launch_phase(e['ts']) for e in events
                if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})
                and ('Launch' in e.get('name','') or 'Memcpy' in e.get('name',''))}
    counts = Counter()
    kernel_ms = Counter()
    for e in events:
        if e.get('cat') == 'kernel':
            name = e['name'].lower()
            category = launches.get(e.get('args',{}).get('correlation'), 'unassigned')
            if 'attention' in name or 'attn' in name:
                category = 'attention'
            elif any(k in name for k in ('gemm','gemv','matmul','cutlass')):
                category = 'model_compute'
            counts[category] += 1
            kernel_ms[category] += e['dur']/1000
            intervals[(category, 100+CATEGORIES.index(category))].append((e['ts'],e['ts']+e['dur']))
        elif e.get('cat') in ('gpu_memcpy','gpu_memset'):
            intervals[('transfer',90)].append((e['ts'],e['ts']+e['dur']))
    del events
    resource_parts = exclusive(intervals,left,right,resources=True)
    # Per-request lane metadata is captured immediately before plan(), which
    # advances every live lane once. Count shared work once, never per output.
    steps = [r for r in records if r['stage']=='loopspec.step']
    shapes = {}
    if steps:
        rows = [sum(len(q['lanes']) for q in r['input']['requests']) for r in steps]
        committed = sum(len(r['output']['committed']) for r in steps)
        endpoint_groups = Counter()
        for r in records:
            if r['stage']=='loopspec.decision':
                endpoint_groups[r['parent']] += r.get('input',{}).get('endpoints',1)
        info = json.loads((folder/'server_info.json').read_text())
        assert info['speculative_draft_depth']==[1], 'This report covers d1=1 only'
        sizes = set(info['cuda_graph_config']['decode']['bs'])
        sizes |= {4*n for n in tuple(sizes)}
        sizes = sorted(n for n in sizes if n<=4*info['max_running_requests'])
        endpoints = list(endpoint_groups.values())
        expected = sum(lane['depth'] in (0,3) for r in steps
                       for q in r['input']['requests'] for lane in q['lanes'])
        assert sum(endpoints)==expected
        shapes = dict(shared_steps=len(steps),committed_decode_tokens=committed,
            mean_live_core_rows=st.mean(rows),core_row_updates_per_token=sum(rows)/committed,
            padded_core_row_updates_per_token=sum(next(n for n in sizes if n>=r) for r in rows if r)/committed,
            endpoint_batches=len(endpoints),endpoints=sum(endpoints),
            captured_row_sizes=sizes,finish_graph_eligible_batches=sum(n in sizes for n in endpoints),
            finish_graph_eligible_percent=100*sum(n in sizes for n in endpoints)/len(endpoints),
            note='Finish coverage follows actual endpoint counts and exact-row graph dispatch; these requests do not capture hidden states.')
    result = {'note':__doc__, 'wall_ms':runs['makespan_ms'],
        'clock_offset_us':offset, 'clock_alignment_p95_us':errors[int(.95*(len(errors)-1))],
        'clock_alignment_max_us':max(errors), 'matched_forward_spans':len(forwards),
        'partition_ms':{k:sum(part[k] for part in resource_parts.values()) for k in CATEGORIES},
        'resource_partition_ms':resource_parts, 'execution_shapes':shapes,
        'kernel_counts':dict(counts),
        'kernel_sum_ms':dict(kernel_ms),
        'gpu_partition_ms':resource_parts['gpu']}
    assert abs(sum(result['partition_ms'].values())-runs['makespan_ms']) < .01
    (folder/'partition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'partition':folder.name,'ms':result['partition_ms']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    partition(parser.parse_args().folder)
