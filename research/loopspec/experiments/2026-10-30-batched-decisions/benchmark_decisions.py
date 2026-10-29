"""Replay fixed checkpoint logits; compare whole packets including the host read.

Role mixtures/proposals are controlled synthetic metadata, not an assertion
that these are live draft logits. Full serving tests supply that evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS, MethodType

import torch

from sglang.srt.speculative.loopspec.sampling import decide_endpoints, probabilities


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('/all-results/2026-10-29-thinking-batch-profile/native-reference-r4/tensors.pt'))
    p.add_argument('--baseline',type=Path,default=Path('/home/ruoxi/repo/test/baselines/loopspec-7df281a'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('sglang.srt.speculative.loopspec.baseline_utils',
        a.baseline/'python/sglang/srt/speculative/loopspec/utils.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    refs=torch.load(a.reference,weights_only=True,map_location='cpu')['single']
    raw=[item['logits'][0] for history in refs for item in history]
    rows=[]
    for bs in [1,2,4,8,16]:
        for mode in ['greedy','sampling']:
            logits=torch.stack([raw[(i*17)%len(raw)] for i in range(2*bs)]).to('cuda')
            ends=[]
            for i in range(2*bs):
                params=NS(temperature=0. if mode=='greedy' else .8,top_k=32768,top_p=1.,min_p=0.,
                    frequency_penalty=0.,presence_penalty=0.,repetition_penalty=1.,logit_bias=None,min_new_tokens=0)
                c=NS(params=params,stop_ids={2},total=4,depths=(1,),device=logits.device,
                    generator=torch.Generator(device='cuda').manual_seed(i+900),logits={i:logits[i]})
                c.distribution=MethodType(lambda self,l:probabilities(self.logits[l.uid],self.params,l.history,self.stop_ids),c)
                l=NS(uid=i,depth=1 if i%2==0 else 4,history=tuple(range(64)),proposals={})
                if l.depth==4:
                    q=probabilities(logits[i].roll(i%3),params,l.history,c.stop_ids)
                    l.proposals[1]=(int(q.argmax()),q)
                ends.append((c,l))
            def scalar():
                pairs=[old.RequestContext.queue_decision(c,l)[0] for c,l in ends]
                return torch.stack(pairs)
            def batched():return decide_endpoints(logits,ends)[0]
            if mode=='greedy':assert torch.equal(scalar(),batched())
            for name,fn in [('old',scalar),('new',batched)]:
                for _ in range(10):fn().tolist()
                torch.cuda.synchronize()
                timing=[]
                for _ in range(50):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start=time.perf_counter_ns();begin.record()
                    packet=fn();submitted=time.perf_counter_ns();end.record()
                    packet.tolist();done=time.perf_counter_ns();end.synchronize()
                    timing.append({'wall_ms':(done-start)/1e6,'submit_ms':(submitted-start)/1e6,
                                   'readback_ms':(done-submitted)/1e6,'gpu_stream_ms':begin.elapsed_time(end)})
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as trace:
                    fn().tolist()
                trace.export_chrome_trace(str(a.output/f'{name}-{mode}-b{bs}.trace.json'))
                data=json.loads((a.output/f'{name}-{mode}-b{bs}.trace.json').read_text())['traceEvents']
                kernels=[x for x in data if x.get('cat')=='kernel']
                row={'variant':name,'mode':mode,'bs':bs,'endpoints':2*bs,
                     'median':{k:statistics.median(t[k] for t in timing) for k in timing[0]},
                     'kernel_count':len(kernels),'kernel_sum_ms':sum(x['dur'] for x in kernels)/1000,'measurements':timing}
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='measurements'}),flush=True)
    (a.output/'report.json').write_text(json.dumps({'rows':rows,'reference':str(a.reference),
        'note':'fixed target logits with controlled role/proposal metadata; not a serving speedup'},indent=2)+'\n')


if __name__=='__main__':main()
