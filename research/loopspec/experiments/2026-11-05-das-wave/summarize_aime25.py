# Retains read-only DAS-Wave parsing for archived pre-removal results.
"""Recompute paired metrics from full outputs and real request counters."""
import argparse
import json
from pathlib import Path
import re
import statistics


def ratio(x, y):
    return x/y if y else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def answer(text):
    # Keep the complete text for manual audit of outputs without a boxed integer.
    matches = re.findall(r'\\boxed\s*\{\s*(\d{1,3})\s*\}', text)
    return int(matches[-1]) if matches else None


def counters(method, row, log):
    rid = re.escape(row['rid'])
    if method == 'ar':
        return None
    if method == 'dflash':
        meta = row['meta_info']
        hist = meta['spec_correct_drafts_histogram']
        accepted = sum(i*n for i,n in enumerate(hist))
        assert accepted == meta['spec_num_correct_drafts']
        assert sum(hist) == meta['spec_verify_ct']
        return {'accepted':[accepted], 'tested':[sum(min(i+1,15)*n for i,n in enumerate(hist))],
            'proposed':[15*sum(hist)], 'verify_cycles':sum(hist), 'scope':'raw verify before final stop truncation'}
    if method.startswith('loopspec'):
        match = re.search(r'LoopSpec rid='+rid+r' cycles=(\d+) reused=(\d+) restarts=(\d+) accepted=(\[[^\]]+\]) rejected=(\d+) proposed=(\[[^\]]+\]) tested=(\[[^\]]+\])', log)
        assert match, row['rid']
        cycles,reused,restarts,accepted,rejected,proposed,tested = match.groups()
        accepted,proposed,tested = map(json.loads,(accepted,proposed,tested))
        return {'accepted':accepted, 'tested':tested, 'proposed':proposed,
            'verify_cycles':sum(accepted)+int(rejected), 'core_steps':int(cycles),
            'reused':int(reused), 'restarts':int(restarts), 'depths':[1,3] if method=='loopspec13' else [1],
            'scope':'completed target lane decisions; proposed includes pruned branches'}
    match = re.search(r'DAS_WAVE rid='+rid+r' cycles=(\d+) proposed=(\d+) tested=(\d+) accepted=(\d+) emitted=(\d+)', log)
    assert match, row['rid']
    cycles,proposed,tested,accepted,emitted = map(int,match.groups())
    return {'accepted':[accepted], 'tested':[tested], 'proposed':[proposed],
        'verify_cycles':cycles, 'emitted':emitted, 'scope':'retained prefix through actual stop'}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    manifest=json.loads((a.directory/'manifest.json').read_text())
    assert not manifest['args']['profile_reference']
    full=not manifest['args']['smoke']
    runs={};details={}
    for directory in sorted(a.directory.glob('*-r*')):
        if not directory.is_dir():continue
        method,repeat=directory.name.rsplit('-r',1)
        status=json.loads((directory/'status.json').read_text())
        assert status['execution_complete']
        rows=[json.loads(f.read_text()) for f in directory.glob('q*.json')]
        assert len(rows)==len({r['id'] for r in rows})==(30 if full else 2)
        log=(directory/'server.log').read_text()
        for row in rows:
            row['counts']=counters(method,row,log)
            row['parsed_answer']=answer(row['text'])
            row['correct']=row['parsed_answer']==int(row['answer'])
            row['truncated']=row['meta_info']['finish_reason'].get('type')=='length'
        runs[(method,int(repeat))]=rows
    for (method,repeat),rows in runs.items():
        totals={key:sum(r[key] for r in rows) for key in ('e2e_ms','decode_ms','ttft_ms','decode_tokens')}
        totals['output_tokens']=sum(len(r['output_ids']) for r in rows)
        totals.update(questions=len(rows), correct=sum(r['correct'] for r in rows),
            unparsed=sum(r['parsed_answer'] is None for r in rows), truncated=sum(r['truncated'] for r in rows),
            e2e_tokens_s=ratio(totals['output_tokens']*1000,totals['e2e_ms']),
            decode_tokens_s=ratio(totals['decode_tokens']*1000,totals['decode_ms']))
        if method!='ar':
            totals['acceptance']={}
            for d in range(2 if method=='loopspec13' else 1):
                counts={key:sum(r['counts'][key][d] for r in rows) for key in ('accepted','tested','proposed')}
                totals['acceptance'][str(d)] = counts | {
                    'tested_acceptance_micro':ratio(counts['accepted'],counts['tested']),
                    'tested_acceptance_macro':mean(ratio(r['counts']['accepted'][d],r['counts']['tested'][d]) for r in rows),
                    'draft_utilization_micro':ratio(counts['accepted'],counts['proposed']),
                    'draft_utilization_macro':mean(ratio(r['counts']['accepted'][d],r['counts']['proposed'][d]) for r in rows),
                    'accepted_per_verify_macro':mean(ratio(r['counts']['accepted'][d],r['counts']['verify_cycles']) for r in rows)}
        baseline=runs.get(('ar',repeat))
        if baseline:
            base={r['id']:r for r in baseline}
            assert set(base)=={r['id'] for r in rows}
            mismatches=[]
            for r in rows:
                b=base[r['id']]
                if (r['output_ids'],r['meta_info']['finish_reason'])!=(b['output_ids'],b['meta_info']['finish_reason']):
                    first=next((i for i,(x,y) in enumerate(zip(b['output_ids'],r['output_ids'])) if x!=y),min(len(b['output_ids']),len(r['output_ids'])))
                    mismatches.append({'id':r['id'],'first_difference':first})
            totals['paired_ar']={'mismatches':mismatches,'same_workload':not mismatches,
                'e2e_elapsed_ratio':sum(r['e2e_ms'] for r in baseline)/totals['e2e_ms'],
                'decode_throughput_speedup':totals['decode_tokens_s']/(sum(r['decode_tokens'] for r in baseline)*1000/sum(r['decode_ms'] for r in baseline)),
                'e2e_throughput_speedup':totals['e2e_tokens_s']/(sum(len(r['output_ids']) for r in baseline)*1000/sum(r['e2e_ms'] for r in baseline)),
                'per_question_decode_speedup_mean':mean((r['decode_tokens']/r['decode_ms'])/(base[r['id']]['decode_tokens']/base[r['id']]['decode_ms']) for r in rows)}
        details[f'{method}-r{repeat}']=totals
    (a.directory/'summary.json').write_text(json.dumps({'full30':full,'runs':details,
        'notes':['Output divergences preclude a lossless same-workload speedup claim.',
                 'Decode timing uses observed first SSE batch; its token count is excluded.',
                 'DFLASH counters describe raw verification; Wave counts are stop-trimmed.',
                 'Acceptance keys 0/1 mean LoopSpec depth 1/3; AR acceptance is N/A.']},indent=2)+'\n')
    print(json.dumps(details,indent=2))


if __name__=='__main__':main()
