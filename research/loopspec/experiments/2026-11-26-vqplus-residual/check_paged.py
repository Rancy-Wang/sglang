"""Page reader boundaries and dynamic graph tests against a dense FP32 oracle."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import torch
import check_residual as reference
from sglang.kernels.ops.loopvq.residual_attention_paged_cuda import PagedResidualAttention, page_pack, page_unpack
from sglang.kernels.ops.loopvq.residual_attention_paged_triton import PagedTritonAttention


def packed_pool(original,layout=1):
    p=copy.copy(original)
    for kind in ('k','v'):
        setattr(p,kind+'_indices',page_pack(getattr(original,kind+'_indices'),layout))
        for suffix in ('residual','residual_scale'):
            x=getattr(original,kind+'_'+suffix)
            y=torch.stack([page_pack(t,layout) for t in x])
            torch.testing.assert_close(torch.stack([page_unpack(t,layout) for t in y]),x,atol=0,rtol=0)
            setattr(p,kind+'_'+suffix,y)
    p.get_key_buffer=lambda layer:p.k_indices
    p.get_value_buffer=lambda layer:p.v_indices
    p.get_hp_key_buffer=lambda layer:p.hp_k[layer]
    p.get_hp_value_buffer=lambda layer:p.hp_v[layer]
    return p


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--compact',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--staged',action='store_true')
    ap.add_argument('--remapped',action='store_true')
    ap.add_argument('--cache-mode',type=int,choices=[0,1,2],default=0)
    ap.add_argument('--triton',action='store_true')
    ap.add_argument('--layout',type=int,choices=[1,2],default=1)
    ap.add_argument('--group-size',type=int,choices=[0,1,2],default=1)
    args=ap.parse_args()
    torch.manual_seed(20261126)
    torch.backends.cuda.matmul.allow_tf32=False
    original=reference.make_pool(direct=True)
    pool=packed_pool(original,args.layout)
    opts=dict(threads=256,staged=True,cooperative=False) if args.staged else {}
    if args.remapped:
        assert not args.staged
        opts=dict(threads=256,remapped=True,cooperative=False,waves=4)
    if args.cache_mode:
        assert not args.staged and not args.remapped
        args.compact=False
        opts=dict(cooperative=False,cache_mode=args.cache_mode)
    if args.group_size != 1:
        assert args.layout == 2 and not args.triton and not opts
        args.compact = False
        attn=reference.ResidualAttention(pool,max_batch=16,max_context_len=32768,global_table=True,layout=2,group_size=args.group_size)
    elif args.triton:
        assert not opts and args.compact
        attn=PagedTritonAttention(pool,max_batch=16,layout=args.layout)
    else:
        attn=PagedResidualAttention(pool,max_batch=16,compact=args.compact,layout=args.layout,**opts)
    theta=torch.arange(32768,device='cuda').float()[:,None]/(10000**(torch.arange(64,device='cuda').float()[None]/64))
    rope=torch.cat((theta.cos(),theta.sin()),-1).contiguous()
    result={'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'group_size':args.group_size,'layout':args.layout,'compact':args.compact,'staged':args.staged,'remapped':args.remapped,'cache_mode':args.cache_mode,'triton':args.triton,'cases':[]}
    for b in (1,2,3,8,16):
        for n in (0,1,7,8,63,64,71,72,79,80,127,128,1023,1024,1025,2048,4096,8192,16384,32768):
            q=torch.randn(b,24,128,device='cuda').bfloat16()[:,:8]
            lengths=[n if r%2==0 else n//3 for r in range(b)]
            ptr,loc=reference.locations(lengths,pool.hp_global_offset,'page8')
            attn.partial.fill_(float('nan'));attn.lse.fill_(float('nan'))
            grouped=args.group_size!=1 and attn.use_grouped(b)
            if grouped:
                attn.group_partial.fill_(float('nan'));attn.group_lse.fill_(float('nan'))
            expected,lse=reference.oracle(q,original,ptr,loc,rope,return_lse=True)
            got=attn.forward(q,0,ptr,loc,rope,128**-.5)
            active=min(32,max(1,(attn.waves*attn.sm_count+b*8-1)//(b*8))) if args.compact else 32
            torch.testing.assert_close(got,expected,atol=.002,rtol=.01)
            observed_lse=attn.group_lse[:b] if grouped else attn.lse[:b,:,:active]
            torch.testing.assert_close(observed_lse.logsumexp(-1),lse,atol=2e-5,rtol=2e-5)
            result['cases'].append({'batch':b,'length':n,'max_abs':(got.float()-expected.float()).abs().max().item()})
        print('boundaries passed batch='+str(b),flush=True)
    b=16
    q=torch.randn(b,8,128,device='cuda').bfloat16()
    ptr,loc=reference.locations([8192]*b,pool.hp_global_offset,'page8')
    for _ in range(3):attn.forward(q,0,ptr,loc,rope,128**-.5)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):got=attn.forward(q,0,ptr,loc,rope,128**-.5)
    for n in (2048,1025,8,0,8192):
        p2,l2=reference.locations([n if r%2==0 else n//2 for r in range(b)],pool.hp_global_offset,'page8')
        ptr.copy_(p2);loc[:len(l2)].copy_(l2)
        attn.partial.fill_(float('nan'));attn.lse.fill_(float('nan'))
        if args.group_size!=1 and attn.use_grouped(b):
            attn.group_partial.fill_(float('nan'));attn.group_lse.fill_(float('nan'))
        graph.replay()
        torch.testing.assert_close(got,reference.oracle(q,original,ptr,loc,rope),atol=.002,rtol=.01)
    loc.zero_();graph.replay();assert not got.any()
    graph.reset()
    result.update(graph_dynamic_and_padding=True,passed=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('all page boundary/LSE/graph checks passed',flush=True)


if __name__=='__main__':main()
