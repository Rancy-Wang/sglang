"""Request-filtered TP profiles; ranks remain separate, never summed as latency."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics


def stats(values):
    return {'n': len(values), 'sum_ms': sum(values),
            'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values)} if values else {'n': 0}


def category(name):
    name = name.lower()
    if any(s in name for s in ('nccl', 'allreduce', 'all_reduce', 'allgather', 'all_gather',
                              'broadcast', 'reduce_scatter', 'msccl', 'customar')):
        return 'communication'
    if any(s in name for s in ('attn', 'attention', 'flash_fwd', 'decode_stage')):
        return 'attention'
    if any(s in name for s in ('gemm', 'gemv', 'matmul', 'cutlass', 'cublas', 'xmma')):
        return 'gemm'
    if 'norm' in name:
        return 'normalization'
    if any(s in name for s in ('copy', 'memcpy', 'scatter', 'gather', 'cache', 'kv_')):
        return 'memory_index_cache'
    if any(s in name for s in ('sample', 'argmax', 'topk', 'softmax', 'accept')):
        return 'sampling_acceptance'
    return 'other'


def union_ms(intervals):
    end, total = float('-inf'), 0.0
    for left, right in sorted(intervals):
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    return total / 1000


def selected_rows(rows, rid):
    selected = set()
    for row in rows:
        meta = row.get('input', {})
        own = rid in meta.get('rids', []) and meta.get('mode') == 'DECODE'
        own |= row['stage'] == 'loopspec.step' and any(
            item['rid'] == rid for item in meta.get('requests', []))
        own |= row['stage'] == 'recurrent.core' and any(
            rid in owners and mode == 'DECODE'
            for owners, mode in zip(meta.get('owners', []), meta.get('modes', [])))
        if own or row['parent'] in selected:
            selected.add(row['id'])
    return selected


def summarize_rank(rows, events, rid):
    selected = selected_rows(rows, rid)
    stages = defaultdict(lambda: defaultdict(list))
    children = defaultdict(list)
    for row in rows:
        children[row['parent']].append(row)
    for row in rows:
        if row['id'] not in selected:
            continue
        for field in ('cpu_ms', 'gpu_stream_ms'):
            if field not in row:
                continue
            stages[row['stage']][field].append(row[field])
            stages[row['stage']]['exclusive_' + field].append(
                row[field] - sum(c.get(field, 0) for c in children[row['id']]))
    # Match same-stage CPU events with the same rank's exported span records.
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row['stage']].append(row)
    indices = Counter()
    spans = defaultdict(list)
    for event in sorted(events, key=lambda e: e.get('ts', 0)):
        name = event.get('name', '')
        if not name.startswith('recurrent::') or 'dur' not in event:
            continue
        stage = name.removeprefix('recurrent::')
        index = indices[stage]
        indices[stage] += 1
        assert index < len(by_stage[stage]), (stage, index, len(by_stage[stage]))
        row = by_stage[stage][index]
        if row['id'] in selected:
            spans[(event['pid'], event['tid'])].append(event)
    assert all(indices[stage] == len(group) for stage, group in by_stage.items())
    launches = defaultdict(list)
    for event in events:
        if event.get('cat') == 'cuda_runtime' and 'correlation' in event.get('args', {}):
            launches[(event['pid'], event['tid'])].append(event)
    owners = {}
    for thread, calls in launches.items():
        intervals = spans[thread]
        cursor, active = 0, []
        for call in sorted(calls, key=lambda e: e['ts']):
            t = call['ts']
            while cursor < len(intervals) and intervals[cursor]['ts'] <= t:
                active.append(intervals[cursor])
                cursor += 1
            active = [e for e in active if e['ts'] + e['dur'] >= t + call.get('dur', 0)]
            if active:
                owner = min(active, key=lambda e: e['dur'])
                owners[call['args']['correlation']] = owner['name'].removeprefix('recurrent::')
    kernels = defaultdict(float)
    kernel_names = defaultdict(float)
    stage_kernels = defaultdict(lambda: defaultdict(float))
    intervals = []
    unassigned = 0
    for event in events:
        if event.get('cat') != 'kernel' or 'dur' not in event:
            continue
        owner = owners.get(event.get('args', {}).get('correlation'))
        if owner is None:
            unassigned += 1  # Includes intentionally excluded prefill/probe work.
            continue
        kind, ms = category(event['name']), event['dur'] / 1000
        kernels[kind] += ms
        kernel_names[event['name']] += ms
        stage_kernels[owner][kind] += ms
        intervals.append((event['ts'], event['ts'] + event['dur']))
    output_rows = [r for r in rows if r['stage'] == 'scheduler.result' and r['id'] in selected]
    steps = [r for r in rows if r['stage'] == 'loopspec.step' and r['id'] in selected]
    chains, token_chains = defaultdict(list), []
    for step in steps:
        for item in step['input']['requests']:
            if item['rid'] != rid:
                continue
            chains[item['output_index']].append(step)
            if rid in step.get('output', {}).get('committed', []):
                chain = chains.pop(item['output_index'])
                entry = next(i for i in chain[0]['input']['requests'] if i['rid'] == rid)
                token_chains.append({'loops': len(chain), 'entry_depth': entry['entry_depth'],
                    'cpu_ms': sum(r['cpu_ms'] for r in chain),
                    'gpu_stream_ms': sum(r.get('gpu_stream_ms', 0) for r in chain)})
    return {'stages': {s: {k: stats(v) for k, v in group.items()} for s, group in stages.items()},
        'decode_result_callbacks': len(output_rows), 'loopspec_output_chains': token_chains,
        'kernel_ms': dict(kernels), 'kernel_busy_union_ms': union_ms(intervals),
        'kernel_sum_ms': sum(kernels.values()),
        'kernel_ms_by_launch_stage': {s: dict(v) for s, v in stage_kernels.items()},
        'kernel_ms_by_name': dict(sorted(kernel_names.items(), key=lambda x: -x[1])),
        'excluded_or_unmapped_kernel_count': unassigned,
        'cuda_graph_launches': sum('cudaGraphLaunch' in e.get('name', '') for e in events)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    windows = {}
    for directory in sorted(a.directory.glob('*-r*/q*-p*')):
        request = json.loads((directory/'request.json').read_text())
        trace = directory/'trace'
        ranks = defaultdict(list)
        for path in trace.glob('*-recurrent-*.jsonl'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                ranks[row['tp_rank']].append(row)
        assert set(ranks) == {0, 1}, (directory, set(ranks))
        summary = {}
        for rank, rows in sorted(ranks.items()):
            assert len({r['pid'] for r in rows}) == 1
            rows.sort(key=lambda r: r['id'])
            paths = [p for p in trace.glob('*.trace.json.gz') if re.search(fr'-TP-{rank}\.trace', p.name)]
            assert len(paths) == 1, paths
            with gzip.open(paths[0], 'rt') as stream:
                events = json.load(stream)['traceEvents']
            summary[str(rank)] = summarize_rank(rows, events, request['rid'])
        request = {k: v for k, v in request.items() if k not in ('output_ids', 'text', 'events')}
        windows[str(directory.relative_to(a.directory))] = {'request': request, 'ranks': summary}
    assert windows
    (a.directory/'tp2-profile-summary.json').write_text(json.dumps({'windows': windows,
        'notes': ['Ranks reported separately: do not sum their times as wall latency.',
                  'Inclusive spans overlap; direct-child subtraction is only an interval estimate.',
                  'Kernels are attributed by CPU launch correlation and filtered by request/decode.',
                  'Excluded kernel count includes prefill and unmapped launches; not claimed fully mapped.']}, indent=2))


if __name__ == '__main__':
    main()
