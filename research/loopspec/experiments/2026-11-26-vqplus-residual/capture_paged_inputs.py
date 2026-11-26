"""Capture real full-depth model projections for paired kernel experiments.

This is not a generation/accuracy evaluation. Each request uses its own native
Thinking prompt. Raw pre-RoPE K and V are retained for the scalar-INT4 control.
Run in the existing Transformers-compatible training environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / '2026-11-23-loopvq-reproduction'))
from serving_probe import exact_prompt


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--book', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--length', type=int, default=2048)
    ap.add_argument('--requests', type=int, default=16)
    ap.add_argument('--layers', type=int, nargs='+', default=[23])
    ap.add_argument('--tp-rank', type=int, choices=[0, 1], default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261126)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation='sdpa').cuda().eval()
    assert model.config.total_ut_steps == 4 and model.config.num_hidden_layers == 48
    assert model.config.num_attention_heads == model.config.num_key_value_heads == 16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    capture = {}
    handles = []
    h0 = args.tp_rank * 8
    for layer in args.layers:
        for kind in ('q', 'k', 'v'):
            def hook(module, inputs, output, layer=layer, kind=kind):
                x = output[0].view(args.length, 16, 128)[:, h0:h0+8]
                if kind == 'q': x = x[-1]
                capture.setdefault((layer, kind), []).append(x.contiguous().cpu())
            handles.append(getattr(model.model.layers[layer].self_attn, kind + '_proj').register_forward_hook(hook))
    # Match serving's FP32 RoPE coefficient table, rather than inverse rotating
    # already-rounded BF16 keys. Query rounding follows the same FP32 formula.
    pos = torch.arange(args.length, device='cuda')[None]
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, args.length, 1, device='cuda', dtype=torch.float32), pos)
    rope = torch.cat((cos[0, :, :64], sin[0, :, :64]), -1).float().cpu()
    manifest = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        model=str(args.model), book=str(args.book), length=args.length, requests=args.requests,
        layers=args.layers, tp_rank=args.tp_rank, heads=list(range(h0,h0+8)), full_loops=4,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), trajectory='HF SDPA teacher projections; serving FP32 RoPE',
        book_hashes={name:hashlib.sha256((args.book/name).read_bytes()).hexdigest()
                     for name in ('codebook.pt','vq_v_codebook.pt')}, files=[])
    torch.save(rope, args.out/'rope.pt')
    for request in range(args.requests):
        capture.clear()
        fixture = exact_prompt(tokenizer, args.length, seed=request)
        ids = torch.tensor([fixture['input_ids']],device='cuda')
        model(ids, use_cache=False, logits_to_keep=1)
        for layer in args.layers:
            assert all(len(capture[layer,k]) == 4 for k in ('q','k','v'))
            values = {k:torch.stack(capture[layer,k]) for k in ('q','k','v')}
            q = values['q'].float()
            values['q'] = (q*torch.cat((rope[-1,:64],rope[-1,:64])) +
                torch.cat((-q[...,64:],q[...,:64]),-1)*torch.cat((rope[-1,64:],rope[-1,64:]))).bfloat16()
            values['input_ids'] = fixture['input_ids']
            name=f'req{request:02d}-layer{layer:02d}.pt'
            torch.save(values,args.out/name)
            manifest['files'].append(dict(request=request,layer=layer,path=name,
                sha256=hashlib.sha256((args.out/name).read_bytes()).hexdigest()))
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'captured request={request} length={args.length} layers={args.layers}',flush=True)
    for handle in handles: handle.remove()


if __name__ == '__main__':
    main()
