"""Summarize existing serving evidence without changing the benchmark evaluator."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median


def category(name):
    if name == '_fwd_kernel_stage1':
        return 'native_attention_stage1'
    if name == '_fwd_kernel_stage2':
        return 'native_attention_merge'
    if 'payload_stage1<' in name:
        return 'payload_attention_stage1'
    if 'stage1<' in name:
        return 'original_attention_stage1'
    if name == '_official_reduce_kernel':
        return 'attention_merge'
    if name == '_encode_residual':
        return 'residual_encode'
    if name == '_encode':
        return 'index_encode'
    if name == '_write_hp':
        return 'hp_write'
    if 'gemm' in name.lower() or 'splitkreduce' in name.lower():
        return 'gemm_and_splitk'
    if 'all_reduce' in name or 'nccl' in name.lower():
        return 'collectives'
    if 'rmsnorm' in name.lower():
        return 'rmsnorm'
    if 'rope' in name.lower():
        return 'current_token_rope'
    return 'other'


def trace_summary(path, root):
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    groups, kernels = defaultdict(lambda: [0, 0.]), defaultdict(lambda: [0, 0.])
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name, duration = event['name'], event['dur']
        for target, key in ((groups, category(name)), (kernels, name)):
            target[key][0] += 1
            target[key][1] += duration
    total = sum(value[1] for value in groups.values())
    return dict(file=str(path.relative_to(root)), summed_kernel_ms=total / 1000,
                groups={key: dict(calls=value[0], duration_us=value[1],
                                 percent_summed_kernel_time=100 * value[1] / total)
                        for key, value in groups.items()},
                top_kernels=[dict(name=key, calls=value[0], duration_us=value[1])
                             for key, value in sorted(kernels.items(),
                                                      key=lambda item: -item[1][1])])


def summarize(root):
    result = dict(notes=[
        'Unprofiled timing uses the unchanged check_serving.py decode interval.',
        'CUPTI activity durations are not hardware counters or end-to-end walltime.',
        'Profile throughput must not be used as acceptance timing.',
    ], methods={})
    for method in ('baseline', 'candidate', 'bf16'):
        directory = root / ('serving-' + method)
        timing = directory / 'timing' / 'summary.json'
        if not timing.exists():
            continue
        batches = defaultdict(list)
        for row in json.loads(timing.read_text()):
            assert not row['profile'] and row['decode_evidence']['verified']
            batches[row['batch']].append(row['decode_tokens_s'])
        result['methods'][method] = dict(
            timing={batch: dict(samples=values, median=median(values),
                                minimum=min(values), maximum=max(values),
                                range_percent=100 * (max(values)-min(values))/median(values))
                    for batch, values in batches.items()},
            traces=[trace_summary(path, directory)
                    for path in sorted((directory / 'profile').rglob('*.trace.json.gz'))],
        )
    if 'baseline' in result['methods'] and 'candidate' in result['methods']:
        old, new = (result['methods'][key]['timing'] for key in ('baseline', 'candidate'))
        result['candidate_throughput_change_percent'] = {
            batch: 100 * (new[batch]['median'] / row['median'] - 1)
            for batch, row in old.items() if batch in new}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence), indent=2) + '\n')
