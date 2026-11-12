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


def window_turns(result):
    return [r for r in result['turns'] if result.get('start', float('-inf')) <= r.get('end_time', result.get('cutoff', 0)) <= result.get('cutoff', float('inf'))]


def summarize(result):
    observed = window_turns(result)
    rows = [r for r in observed if r['success']]
    tasks = [t for t in result['tasks'] if t['status'] != 'cancelled' and t.get('end', result.get('cutoff', 0)) <= result.get('cutoff', float('inf'))]
    duration = result['duration_s']
    return dict(duration_s=duration, completed_requests=len(rows),
                failed_requests=sum(not r['success'] for r in observed),
                post_cutoff_requests=len(result['turns'])-len(observed),
                input_throughput=sum(r['prompt_len'] for r in rows)/duration if duration else None,
                output_throughput=sum(r['output_len'] for r in rows)/duration if duration else None,
                latency_s=stats([r['latency'] for r in rows]), ttft_s=stats([r['ttft'] for r in rows]),
                tpot_ms=stats([r['tpot_s']*1000 if r.get('tpot_s') is not None else None for r in rows]),
                output_tokens=stats([r['output_len'] for r in rows]),
                task_latency_s=stats([r['latency_s'] for r in tasks]),
                task_statuses=dict(Counter(r['status'] for r in result['tasks'])))


def restoration(result, prefill_log):
    """User-requested estimate from actual TP0 P prefill, not extend-size bins."""
    import json
    import re
    turns = {(r['instance'],r['turn']):r for r in window_turns(result) if r['success']}
    rooms = {str(r['pd_bootstrap_room']):r for r in turns.values()}
    physical = {}
    with open(prefill_log, errors='replace') as f:
        for line in f:
            if not line.endswith('\n') or 'TP0]' not in line or 'ReqTimeStats(' not in line:
                continue
            room=re.search(r'bootstrap_room=(\d+)',line)
            pf=re.search(r'pd_matrix_compute=(\{[^}]*\})',line)
            n=re.search(r'\binput_len=(\d+)',line)
            if room and pf and n and room[1] in rooms:
                physical[room[1]]=(int(n[1]),json.loads(pf[1])['pf'])
    rows, unknown = [], []
    for room,r in rooms.items():
        p=physical.get(room);prev=turns.get((r['instance'],r['turn']-1))
        if p is None or (r['turn'] and prev is None):
            unknown.append(room);continue
        n,pf=p
        if n!=r['prompt_len'] or not 0<=pf<=n:
            unknown.append(room);continue
        prior_input=prev['prompt_len'] if prev else 0
        if r.get('ablation_policy') == 'text_drop':
            projection=r.get('text_projection') or {}
            dropped=projection.get('removed_since_previous_tokens')
            if not isinstance(dropped,int) or dropped < 0:
                unknown.append(room);continue
            prior_input=max(0,prior_input-dropped)
        expected=min(n,prior_input)
        restored=max(0,expected-(n-pf))
        rows.append(dict(instance=r['instance'],turn=r['turn'],room=room,expected_reuse=expected,
                         input_tokens=n,actual_prefill=pf,effective_hit=n-pf,restore=restored,new_prefill=pf-restored))
    positive=sum(r['restore']>0 for r in rows);total=len(turns)
    bins=Counter(r['restore']//2048 for r in rows if r['restore']>0)
    return dict(formula='max(0,min(current_input,previous_input)-(input-actual_P_prefill)); previous output excluded; text_drop subtracts newly removed observation tokens, native does not double-subtract drop',
                total_requests=total,known_requests=len(rows),unknown_requests=len(unknown),unknown_rooms=unknown,
                positive_requests=positive,positive_fraction=positive/total if total else None,
                exact_zero=bool(total) and not unknown and positive==0,rows=rows,
                mean_restore_tokens=sum(r['restore'] for r in rows)/len(rows) if rows else None,
                mean_new_prefill_tokens=sum(r['new_prefill'] for r in rows)/len(rows) if rows else None,
                positive_2K_frequency={str(k):v/total for k,v in sorted(bins.items())})
