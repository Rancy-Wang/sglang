"""Window-bounded physical cost and logical user-turn timing for Summary replay."""
import json
from pathlib import Path


def stats(values):
    import numpy as np
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                **{f'p{q}': float(np.percentile(values, q)) if values else None for q in (50, 90, 95, 99)})


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def summarize(root):
    root = Path(root)
    result = json.loads((root/'result.json').read_text())
    start, end = result['start'], result['cutoff']
    duration = end-start
    usage = {}
    identities = {}
    for event in jsonl(root/'events.jsonl'):
        if event['kind'] == 'turn_start' and start <= event['time'] <= end:
            key = (event['instance'], event['turn'])
            identities[key] = event
    # Last cumulative usage per physical request, before the exact cutoff.
    # Successful AND failed/inflight calls contribute, with no parent double count.
    for event in jsonl(root/'sse.jsonl'):
        if not start <= event['received_perf'] <= end or event['data'] == '[DONE]':
            continue
        value = json.loads(event['data'])
        if value.get('usage'):
            usage[event['instance'], event['turn']] = value['usage']
    counters = {}
    for key, value in usage.items():
        purpose = identities[key]['purpose']
        purpose = 'summary' if purpose.startswith('summary_') else purpose
        group = counters.setdefault(purpose, dict(input_tokens=0, output_tokens=0, cached_tokens=0,
                                                  calls_with_usage=0, unknown_cache_calls=0))
        group['calls_with_usage'] += 1
        for dst, src in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')):
            if type(value.get(src)) is not int:
                raise ValueError('Non-integer usage: '+str(key))
            group[dst] += value[src]
        cached = (value.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached is None:
            group['unknown_cache_calls'] += 1
        else:
            group['cached_tokens'] += cached
    for group in counters.values():
        group['logical_input_throughput'] = group['input_tokens']/duration
        group['output_throughput'] = group['output_tokens']/duration
        group['price_usd'] = None if group['unknown_cache_calls'] else (
            group['cached_tokens']*.075 + (group['input_tokens']-group['cached_tokens'])*.15 + group['output_tokens']*.6)/1e6
    parents = [r for r in result['logical_user_turns'] if start <= r['raw_done_time'] <= end]
    physical = [r for r in result['turns'] if r.get('raw_done_time') and start <= r['raw_done_time'] <= end and r['success']]
    def timing(rows):
        return dict(latency_s=stats([r['latency'] for r in rows]), ttft_s=stats([r['ttft'] for r in rows]),
                    tpot_ms=stats([None if r['tpot_s'] is None else r['tpot_s']*1000 for r in rows]),
                    user_output_tokens_per_s=stats([r['output_len']/r['latency'] for r in rows if r['latency']]))
    output = dict(valid=result['valid'], stop_reason=result['stop_reason'], duration_s=duration,
                  gpu_hours=8*duration/3600, counters=counters,
                  logical_input_throughput=sum(x['input_tokens'] for x in counters.values())/duration,
                  output_throughput=sum(x['output_tokens'] for x in counters.values())/duration,
                  user_turns=timing(parents), primary_user_turns=timing([r for r in parents if not r['filler']]),
                  physical_calls=timing(physical),
                  calls_without_known_usage=[identities[k] for k in identities.keys()-usage.keys()],
                  unfinished_tasks=[t for t in result['tasks'] if t['status']=='cutoff_cancelled'],
                  source_terminal_tasks=result['source_terminal_tasks'],
                  prices_usd_per_million=dict(cached=.075, uncached=.15, output=.6),
                  notes=['All physical summary/agent calls with known in-window usage counted once.',
                         'Unknown partial output is not zero; raw SSE retained for server-counter reconciliation.',
                         'Latency statistics only completed logical turns, excluding post-DONE cleanup.',
                         'Logical latency and TTFT include preceding summary calls.'])
    (root/'summary-accounting.json').write_text(json.dumps(output,indent=2))
    return output


if __name__ == '__main__':
    import sys
    print(json.dumps(summarize(sys.argv[1]),indent=2))
