"""Read-only attribution from existing traces: CPU remainders and client ITL.

Remainders are parent wall time minus direct nested spans. They group multiple
host functions and must not be presented as individually instrumented kernels.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_profile import describe


def enrich(folder):
    records=[]
    for path in sorted((folder/'trace').glob('*-recurrent-*.jsonl')):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    by_id={r['id']:r for r in records}
    assert len(by_id)==len(records), 'One scheduler rank expected'
    children=collections.defaultdict(list)
    for r in records:
        if r['parent'] is not None: children[r['parent']].append(r)
    remainders={}
    for r in records:
        if r['stage'] not in ['scheduler.forward','loopspec.core','loopspec.step']:
            continue
        rest=r['cpu_ms']-sum(c['cpu_ms'] for c in children[r['id']])
        assert rest>=-1e-5,(r['id'],rest)
        remainders[r['id']]=max(0,rest)
    result_rows=collections.defaultdict(list)
    for r in records:
        # Actual result handlers recurse inside an outer recurrent result relay.
        # Partial-loop outer calls must not be mistaken for emitted tokens.
        if (r['stage']=='scheduler.result' and r['parent'] is not None
                and by_id[r['parent']]['stage']=='scheduler.result'):
            m=r['input']
            for rid,index in zip(m['rids'],m['output_lengths'],strict=True):
                result_rows[(rid,index)].append(r)
    runs=json.loads((folder/'cohort-0.json').read_text())['requests']
    arrivals={}; counts={}; coalesced=0
    for req in runs:
        prior=0
        for event in req['events']:
            count=event['response']['meta_info']['completion_tokens']
            assert count>=prior
            coalesced+=max(0,count-prior-1)
            for index in range(prior,count): arrivals[(req['rid'],index)]=event['received_ns']
            prior=count
        counts[req['rid']]=len(req['output_ids'])
        assert prior==counts[req['rid']]
    paths=json.loads((folder/'output-paths.json').read_text())
    items=[]; failures=[]
    for token in paths:
        rid,index=token['rid'],token['output_index']
        if index==0 or token.get('mode','DECODE')!='DECODE':continue
        key=rid,index
        if key not in arrivals or (rid,index-1) not in arrivals:
            failures.append({'rid':rid,'index':index,'issue':'missing client emission'})
            continue
        step_ids=token['shared_step_ids']
        stages=collections.defaultdict(float)
        for i in step_ids:
            row=by_id[i]
            if row['stage']=='loopspec.step':
                # Its own remainder includes planning/Pipeline.collect/result assembly.
                stages['loopspec.plan_and_collect_host_ms']+=remainders[i]
                for c in children[i]:
                    if c['stage']=='loopspec.core':
                        stages['loopspec.scatter_and_decision_readback_host_ms']+=remainders[c['id']]
            elif row['stage']=='scheduler.forward':
                stages['native.sample_publish_copy_host_ms']+=remainders[i]
        handlers=result_rows.get(key,[])
        if not handlers:failures.append({'rid':rid,'index':index,'issue':'missing actual result handler'})
        stages['native.result_handler_cpu_ms']=sum(r['cpu_ms'] for r in handlers)
        items.append({'rid':rid,'output_index':index,'entry_depth':token['entry_depth'],
            'cycles':token['new_cycles'], 'client_itl_ms':(arrivals[key]-arrivals[(rid,index-1)])/1e6,
            **stages})
    expected=sum(max(0,n-1) for n in counts.values())
    if len(items)!=expected:failures.append({'issue':'decode output coverage','actual':len(items),'expected':expected})
    grouped={}
    keys=sorted({k for x in items for k in x if k.endswith('_ms')})
    for depth in sorted({x['entry_depth'] for x in items}):
        group=[x for x in items if x['entry_depth']==depth]
        grouped[str(depth)]={'outputs':len(group),'new_cycles':describe([x['cycles'] for x in group]),
            **{k:describe([x.get(k,0) for x in group]) for k in keys}}
    # Prefill is the ordinary CDB executor for BOTH algorithms. Preserve its
    # first-token stages separately from decode paths and their depth histogram.
    prefill_calls=collections.defaultdict(list)
    forward_children=collections.defaultdict(list)
    for r in records:
        parent=r
        while parent['stage']!='scheduler.forward' and parent['parent'] is not None:
            parent=by_id[parent['parent']]
        if parent['stage']=='scheduler.forward':forward_children[parent['id']].append(r)
        if r['stage']=='recurrent.core':
            meta=r['input']
            for owners,mode in zip(meta['owners'],meta['modes'],strict=True):
                if mode=='DECODE':continue
                assert parent['stage']=='scheduler.forward'
                for rid in owners:prefill_calls[rid].append(parent['id'])
    prefill=[]
    for req in runs:
        calls=prefill_calls[req['rid']]
        assert len(calls)==4,(req['rid'],len(calls))  # this workload fits one prefill chunk
        groups=collections.defaultdict(list)
        for i in calls:
            for child in forward_children[i]:groups[child['stage']].append(child)
        prefill.append({'rid':req['rid'],'ttft_ms':req['ttft_ms'],'shared_forward_ids':calls,
            'stages':{k:{'calls':len(group),'cpu_ms':sum(x['cpu_ms'] for x in group),
                'gpu_stream_ms':sum(x.get('gpu_stream_ms',0) for x in group)} for k,group in groups.items()}})
    output={'note':'CPU remainders group host functions between direct child spans. Shared steps are not exclusive owner costs. Client ITL includes buffering/network. GPU stages remain inclusive; do not add CPU/GPU intervals.',
        'decode_outputs':len(items),'expected_decode_outputs':expected,'coalesced_tokens':coalesced,
        'failures':failures,'depths':grouped,'tokens':items,'prefill':prefill}
    (folder/'token-intervals.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'directory':str(folder),'outputs':len(items),'failures':failures[:5]}),flush=True)
    if failures:raise RuntimeError(f'{folder}: {len(failures)} attribution failures')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
    for folder in a.folders:enrich(folder)
