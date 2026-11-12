"""Live replay metrics: wall throughput includes tools; request latency excludes tools."""
from collections import Counter


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(n=0, mean=None, p95=None, p99=None)
    def q(frac):
        i = (len(values)-1)*frac
        lo = int(i)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)
    return dict(n=len(values), mean=sum(values)/len(values), p95=q(.95), p99=q(.99))


def summarize(result):
    rows = [r for r in result['turns'] if r['success']]
    duration = result['duration_s']
    return dict(duration_s=duration, completed_requests=len(rows),
                failed_requests=sum(not r['success'] for r in result['turns']),
                input_throughput=sum(r['prompt_len'] for r in rows)/duration if duration else None,
                output_throughput=sum(r['output_len'] for r in rows)/duration if duration else None,
                latency_s=stats([r['latency'] for r in rows]), ttft_s=stats([r['ttft'] for r in rows]),
                tpot_ms=stats([r['tpot_s']*1000 if r.get('tpot_s') is not None else None for r in rows]),
                output_tokens=stats([r['output_len'] for r in rows]),
                task_latency_s=stats([r['latency_s'] for r in result['tasks']]),
                task_statuses=dict(Counter(r['status'] for r in result['tasks'])))
