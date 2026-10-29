"""Checkpoint-owned HF eager, full R4, teacher-forced native prefixes."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

p=argparse.ArgumentParser()
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--model',default='/home/ruoxi/models/Ouro-2.6B-Thinking')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
ref=torch.load(a.reference/'tensors.pt',weights_only=True)
prompts=json.loads((a.reference/'prompts.json').read_text())['prompts']
tokens=json.loads((a.reference/'cohort-0.json').read_text())['requests']
model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,
    torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
assert model.config.total_ut_steps==4
records=[]
with torch.inference_mode():
    for q in [0,len(prompts)-1]:
        for step in [0,15,63]:
            ids=prompts[q]['input_ids']+tokens[q]['output_ids'][:step]
            _,depths,_=model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False)
            assert len(depths)==4
            h=torch.stack([d[0,-1] for d in depths]).cpu()
            logits=model.lm_head(depths[-1][:,-1]).float().cpu()[0]
            source=ref['single'][q][step]
            def err(x,y):
                delta=x.float()-y.float()
                return {'max_abs':float(delta.abs().max()),'rmse':float(delta.square().mean().sqrt())}
            row={'question':q+1,'step':step,'hidden':[err(h[d],source['hidden'][d,0]) for d in range(4)],
                'logits':err(logits,source['logits'][0]),'hf_argmax':int(logits.argmax()),
                'native_argmax':int(source['logits'][0].argmax()),
                'hf_margin':float(logits.topk(2).values.diff().abs()[0])}
            records.append(row); print(json.dumps(row),flush=True)
(a.output/'report.json').write_text(json.dumps({'records':records,'full_depth':4,
    'argmax_equal':all(r['hf_argmax']==r['native_argmax'] for r in records)},indent=2))
