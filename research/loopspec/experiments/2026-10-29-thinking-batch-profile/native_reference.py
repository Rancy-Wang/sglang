"""Independent whole-R native path: no CDB runner, branch code or scheduler.
Generate B1 reference prefixes, calibrate the unchanged model at B16/B64.
"""
import argparse
from array import array
import json
from pathlib import Path
import sys
import torch

OLD = Path(__file__).resolve().parent.parent/'2026-10-29-cdb-v2'
sys.path.insert(0, str(OLD))
from batch_numerics import error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--first', type=int, default=16)
    p.add_argument('--model', default='/home/ruoxi/models/Ouro-2.6B-Thinking')
    p.add_argument('--dataset', type=Path, default=Path('/home/ruoxi/repo/test/datasets/aime.jsonl'))
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.models.ouro import OuroForCausalLM
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs
    OuroForCausalLM.supports_recurrent_execution=False
    options=ServerArgs(model_path=a.model,dtype='bfloat16',attention_backend='triton',
        bf16_gemm_backend='auto',max_running_requests=64,page_size=1,
        disable_overlap_schedule=True,disable_radix_cache=True,chunked_prefill_size=-1,
        disable_cuda_graph=True,context_length=2048,mem_fraction_static=.85)
    ports=PortArgs.init_new(options)
    publish(options,role='scheduler',ranks=SpawnRanks(world_rank=0,gpu_id=0))
    _init_process_global_configs(); bench,_=load_model(options,ports,0,0)
    runner=bench.torch_runner
    assert runner.recurrent_runner is None and runner.model.config.total_ut_steps==4
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    questions=[json.loads(s) for s in a.dataset.read_text().splitlines() if s.strip()][:a.first]
    prompts=[]
    for q in questions:
        text=tokenizer.apply_chat_template([{'role':'user','content':q['problem']}], tokenize=False,
            add_generation_prompt=True,enable_thinking=True)
        assert '<think>' in text.split('<|im_start|>assistant')[-1]
        prompts.append({'id':q['id'],'rendered':text,'input_ids':tokenizer.encode(text,add_special_tokens=False)})
    (a.output/'prompts.json').write_text(json.dumps({'prompts':prompts,'chat_template':tokenizer.chat_template},indent=2))
    selection=[]; captured=[]
    hook=runner.model.model.norm.register_forward_hook(lambda m,i,o:captured.append(o[selection].cpu().clone()))
    def run(indices, forced=None, padding=0):
        bench.clear(); requests=[]; offset=0; selection.clear()
        inputs=[prompts[i]['input_ids'] for i in indices]+[[1] for _ in range(padding)]
        for i,ids in enumerate(inputs):
            r=Req(rid=f'native-{i}',origin_input_text='',origin_input_ids=array('q',ids),
                sampling_params=SamplingParams(temperature=0,max_new_tokens=a.steps))
            r.full_untruncated_fill_ids=r.origin_input_ids; r.logprob_start_len=-1; r.extend_end=len(ids)
            requests.append(r); offset+=len(ids)
            if i<len(indices): selection.append(offset-1)
        records=[]; tokens=[[] for _ in indices]
        for step in range(a.steps):
            captured.clear()
            if step==0: _,logits,batch=bench.extend(requests)
            else:
                selection[:]=range(len(indices))
                chosen=[forced[i][step-1] if forced is not None else tokens[j][-1] for j,i in enumerate(indices)]
                _,logits=bench.decode(torch.tensor(chosen+[1]*padding,device=runner.device),batch)
            assert len(captured)==4
            records.append({'hidden':torch.stack(captured),'logits':logits[:len(indices)].cpu().clone()})
            for j in range(len(indices)): tokens[j].append(int(logits[j].argmax()))
        return records,tokens
    with torch.inference_mode():
        singles=[]; reference_tokens=[]
        for q in range(a.first):
            r,t=run([q]); singles.append(r); reference_tokens.append(t[0])
            print(json.dumps({'single':q+1}),flush=True)
        torch.save({'single':singles},a.output/'tensors.pt')
        (a.output/'cohort-0.json').write_text(json.dumps({'requests':[{'output_ids':t} for t in reference_tokens]}))
        errors=[]
        for padding in [0,64-a.first]:
            rows,_=run(list(range(a.first)),reference_tokens,padding)
            for step,r in enumerate(rows):
                for q in range(a.first):
                    source=singles[q][step]; x,y=source['logits'][0],r['logits'][q]
                    errors.append({'padding':padding,'question':q+1,'step':step,
                        'hidden':[error(source['hidden'][d,0],r['hidden'][d,q]) for d in range(4)],
                        'logits':error(x,y),'argmax_reference':int(x.argmax()),'argmax_candidate':int(y.argmax()),
                        'reference_margin':float(x.topk(2).values.diff().abs()[0])})
    hook.remove()
    # Register independent observed maxima BEFORE observing any LoopSpec errors.
    def bounds(items):
        return {kind:{metric:max(.001, max(r[metric] for r in items)*1.25) for metric in ['max_abs','rmse']}
                for kind in ['p99','max']}
    flips=[r for r in errors if r['argmax_reference']!=r['argmax_candidate']]
    limits={'hidden':[bounds([r['hidden'][d] for r in errors]) for d in range(4)],
        'logits':bounds([r['logits'] for r in errors]),
        'max_argmax_flip_fraction':max(.01,len(flips)/len(errors)*1.25),
        'protected_reference_margin':max(.5,max((r['reference_margin'] for r in flips),default=0)*1.25),
        'source':'Independent native full-R B16/B64 vs B1, same Thinking prefixes; 25% headroom on observed maxima.'}
    (a.output/'numerical_envelope.json').write_text(json.dumps(limits,indent=2))
    (a.output/'calibration.json').write_text(json.dumps({'records':errors,'limits':limits,'finite':all(torch.isfinite(r['logits']).all() for ss in singles for r in ss)},default=bool,indent=2))
    print(json.dumps({'completed':True,'positions':len(errors),'flips':len(flips),'limits':limits}),flush=True)

if __name__=='__main__':main()
