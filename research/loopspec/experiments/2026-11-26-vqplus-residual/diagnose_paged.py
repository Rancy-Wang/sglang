"""Four-way real-KV controls with interleaved graph timing on SM80.

Materialization/layout conversion happen only during setup and their elapsed
times are reported separately. They are not deployment speedups. All compressed
readers retain pre-RoPE K and the existing BF16 composition boundaries.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.loopvq.encode import encode_steps
from sglang.kernels.ops.loopvq.residual_encode import encode_residual_steps
from sglang.kernels.ops.loopvq.residual_attention import ResidualAttention
from sglang.kernels.ops.loopvq.residual_attention_paged_cuda import (
    PagedResidualAttention, page_pack, page_unpack)
from sglang.srt.mem_cache.loopvq_pack import build_coupled_codebook

ROOT = Path(__file__).resolve().parents[4]


def bind(pool):
    pool.get_key_buffer = lambda layer: pool.k_indices
    pool.get_value_buffer = lambda layer: pool.v_indices
    pool.get_hp_key_buffer = lambda layer: pool.hp_k[layer]
    pool.get_hp_value_buffer = lambda layer: pool.hp_v[layer]
    return pool


def pack(x):
    x = x.float().reshape(*x.shape[:-1], 4, 32)
    scale = (x.abs().amax(-1) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    z = (x / scale.float().unsqueeze(-1)).round().clamp(-7, 7).int().flatten(-2)
    return ((z[...,::2]&15) | ((z[...,1::2]&15)<<4)).byte(), scale


def unpack(z, s):
    code = torch.stack((z&15,z>>4),-1).int().flatten(-2)
    return torch.where(code>=8,code-16,code).float()*s.float().repeat_interleave(32,-1)


def centers(ids, cb):
    return cb[torch.arange(8,device='cuda')[None,:,None],
              torch.arange(32,device='cuda')[None,None,:],ids.long()].flatten(-2).float()


def rotated(x, rope):
    c,s=rope.chunk(2,-1)
    c,s=c.repeat(1,2)[:,None],s.repeat(1,2)[:,None]
    return (x.float()*c+torch.cat((-x[...,64:].float(),x[...,:64].float()),-1)*s).bfloat16()


def metrics(x, ref):
    delta=x.float()-ref.float()
    return {'max_abs':delta.abs().max().item(),'rmse':delta.square().mean().sqrt().item()}


def transformed(pool, scalar=False):
    result=copy.copy(pool)
    for kind in ('k','v'):
        idx=getattr(pool,kind+'_indices')
        setattr(result,kind+'_indices',page_pack(idx))
        for name in ('residual','residual_scale'):
            source=getattr(pool,kind+'_'+name)
            value=torch.stack([page_pack(x) for x in source])
            for a,b in zip(value,source):
                torch.testing.assert_close(page_unpack(a),b,atol=0,rtol=0)
            setattr(result,kind+'_'+name,value)
    return bind(result)


def make_pool(rows, tables):
    b,n=len(rows),rows[0]['k'].shape[1]
    cap=8+b*n
    assert n%8==0 and n>=80
    p=SimpleNamespace(head_num=8,head_dim=128,residual_bits=4,hp_global_offset=cap)
    dst=torch.arange(8,cap,device='cuda',dtype=torch.int32)
    raw={kind:torch.cat([row[kind] for row in rows],1).cuda() for kind in ('k','v')}
    for kind in ('k','v'):
        enc=tables[kind].cuda()
        cb=torch.stack([enc[...,u::4] for u in range(4)]).bfloat16().contiguous()
        idx=torch.zeros(cap,8,32,device='cuda',dtype=torch.uint8)
        res=torch.zeros(4,cap,8,64,device='cuda',dtype=torch.uint8)
        scales=torch.full((4,cap,8,4),torch.finfo(torch.float16).tiny,device='cuda',dtype=torch.float16)
        encode_steps(list(raw[kind].unbind()),enc,idx,dst)
        encode_residual_steps(list(raw[kind].unbind()),idx,cb,res,scales,dst,0)
        hp=torch.zeros(4,b*80+8,8,128,device='cuda',dtype=torch.bfloat16)
        for r in range(b):
            x=raw[kind][:,r*n:(r+1)*n]
            hp[:,8+r*80:16+r*80]=x[:,:8]
            hp[:,16+r*80:80+r*80]=x[:,-64:]
        setattr(p,kind+'_indices',idx)
        setattr(p,kind+'_residual',res)
        setattr(p,kind+'_residual_scale',scales)
        setattr(p,kind+'_book',SimpleNamespace(cb16=cb))
        setattr(p,'hp_'+kind,hp)
    ptr=torch.arange(b+1,device='cuda',dtype=torch.int32)*n
    loc=[]
    for r in range(b):
        hp0=cap+8+r*80
        loc.append(torch.cat((torch.arange(hp0,hp0+8,device='cuda'),
            torch.arange(8+r*n+8,8+(r+1)*n-64,device='cuda'),
            torch.arange(hp0+8,hp0+72,device='cuda'))))
    q=torch.stack([row['q'] for row in rows],1).cuda()
    return bind(p),raw,q,ptr,torch.cat(loc).int()


def materialize(pool, raw, loop, rope, loc, n, *, scalar=False, original=False):
    cache=[]
    for kind in ('k','v'):
        dense=torch.zeros(pool.hp_global_offset+pool.hp_k.shape[1],8,128,device='cuda',dtype=torch.bfloat16)
        for b in range(len(loc)//n):
            slots=loc[b*n:(b+1)*n].long()
            quant=slots<pool.hp_global_offset
            ids=slots[quant]
            if original:
                x=raw[kind][loop,b*n:(b+1)*n].float()
            else:
                x=torch.empty(n,8,128,device='cuda')
                base=torch.zeros(len(ids),8,128,device='cuda') if scalar else centers(
                    getattr(pool,kind+'_indices')[ids],getattr(pool,kind+'_book').cb16[loop])
                x[quant]=base+unpack(getattr(pool,kind+'_residual')[loop,ids],getattr(pool,kind+'_residual_scale')[loop,ids])
                x[~quant]=getattr(pool,'hp_'+kind)[loop,slots[~quant]-pool.hp_global_offset].float()
            dense[slots]=rotated(x,rope) if kind=='k' else x.bfloat16()
        cache.append(dense)
    return cache


class NativeAttention:
    def __init__(self,k,v,q,ptr,loc):
        self.k,self.v,self.q,self.ptr,self.loc=k,v,q,ptr,loc
        b,h,d=q.shape
        self.out=torch.empty_like(q)
        self.partial=torch.empty(b,h,48,d,device='cuda')
        self.lse=torch.empty(b,h,48,device='cuda')
        self.splits=torch.full((b,),48,device='cuda',dtype=torch.int32)
    def __call__(self):
        decode_attention_fwd(self.q,self.k,self.v,self.out,self.ptr,self.loc,
            self.partial,self.lse,self.splits,48,128**-.5,1.,1.,enable_lean=False)
        return self.out


def oracle(q,k,v,ptr,loc):
    out=[]
    for b in range(len(q)):
        slots=loc[int(ptr[b]):int(ptr[b+1])].long()
        scores=torch.einsum('hd,nhd->hn',q[b].float(),k[slots].float())/math.sqrt(128)
        out.append(torch.einsum('hn,nhd->hd',scores.softmax(-1),v[slots].float()).bfloat16())
    return torch.stack(out)


def graph_for(fn,inner):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner): fn()
    return graph


def paired(calls,repeats,inner):
    graphs={name:graph_for(fn,inner) for name,fn in calls.items()}
    samples={name:[] for name in calls}
    for _ in range(5):
        for graph in graphs.values(): graph.replay()
    torch.cuda.synchronize()
    for repeat in range(repeats):
        order=list(graphs)
        order=order[repeat%len(order):]+order[:repeat%len(order)]
        if repeat%2: order.reverse()
        for name in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10): graphs[name].replay()
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)/(10*inner))
    for graph in graphs.values(): graph.reset()
    return {name:{'samples_ms':s,'median_ms':statistics.median(s),
                  'min_ms':min(s),'max_ms':max(s)} for name,s in samples.items()}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--book',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16])
    ap.add_argument('--layers',type=int,nargs='+',default=[23])
    ap.add_argument('--loops',type=int,nargs='+',default=[0,1,2,3])
    ap.add_argument('--repeats',type=int,default=30)
    ap.add_argument('--inner',type=int,default=16)
    ap.add_argument('--correctness-only',action='store_true')
    names=['current','old_r1','plain_reader','paged','scalar_int4','materialized',
           'page_only','coop_only','compact','compact_plain','staged','staged_scalar',
           'direct256','remapped','remapped_plain']
    ap.add_argument('--variants',nargs='+',choices=names,default=names)
    ap.add_argument('--profile-one',choices=names)
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((args.snapshot/'manifest.json').read_text())
    for name,expected in manifest['book_hashes'].items():
        assert hashlib.sha256((args.book/name).read_bytes()).hexdigest()==expected
    rope=torch.load(args.snapshot/'rope.pt',weights_only=True).cuda()
    n=manifest['length']; h0=manifest['tp_rank']*8
    tables={}
    for kind,name in [('k','codebook.pt'),('v','vq_v_codebook.pt')]:
        blob=torch.load(args.book/name,map_location='cpu',weights_only=True)
        tables[kind]=build_coupled_codebook(blob['codebooks'],48,4,4,4,16,128)[:,h0:h0+8].contiguous()
        del blob
    result=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        snapshot=str(args.snapshot),snapshot_manifest=manifest,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,records=[],tolerance={'atol':.002,'rtol':.01},
        timing='paired rotating/reversed order; warmed CUDA graphs; setup separately reported; single-layer hot-cache')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for layer in args.layers:
        files=sorted([row for row in manifest['files'] if row['layer']==layer],key=lambda row:row['request'])
        rows=[]
        for row in files[:max(args.batches)]:
            path=args.snapshot/row['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            rows.append(torch.load(path,map_location='cpu',weights_only=True))
        for batch in args.batches:
            assert batch<=len(rows)
            pool,raw,queries,ptr,loc=make_pool(rows[:batch],{k:t[layer] for k,t in tables.items()})
            torch.cuda.synchronize(); start=time.perf_counter()
            pp=transformed(pool)
            torch.cuda.synchronize(); layout_ms=(time.perf_counter()-start)*1000
            sp=copy.copy(pool)
            for kind in ('k','v'):
                res=torch.zeros_like(getattr(pool,kind+'_residual'))
                scales=torch.full_like(getattr(pool,kind+'_residual_scale'),torch.finfo(torch.float16).tiny)
                res[:,8:],scales[:,8:]=pack(raw[kind])
                setattr(sp,kind+'_residual',res);setattr(sp,kind+'_residual_scale',scales)
            bind(sp); spp=transformed(sp)
            factories={
                'current':lambda:ResidualAttention(pool,max_batch=batch,max_context_len=n,global_table=True),
                'old_r1':lambda:ResidualAttention(pool,max_batch=batch,max_context_len=n,global_table=True,group_size=1),
                'plain_reader':lambda:PagedResidualAttention(pool,max_batch=batch,layout=False,cooperative=False),
                'paged':lambda:PagedResidualAttention(pp,max_batch=batch),
                'scalar_int4':lambda:PagedResidualAttention(spp,max_batch=batch,scalar=True),
                'page_only':lambda:PagedResidualAttention(pp,max_batch=batch,cooperative=False),
                'coop_only':lambda:PagedResidualAttention(pool,max_batch=batch,layout=False),
                'compact':lambda:PagedResidualAttention(pp,max_batch=batch,compact=True),
                'compact_plain':lambda:PagedResidualAttention(pool,max_batch=batch,compact=True,layout=False,cooperative=False),
                'staged':lambda:PagedResidualAttention(pp,max_batch=batch,threads=256,compact=True,cooperative=False,staged=True),
                'staged_scalar':lambda:PagedResidualAttention(spp,max_batch=batch,threads=256,compact=True,cooperative=False,staged=True,scalar=True),
                'direct256':lambda:PagedResidualAttention(pp,max_batch=batch,threads=256,compact=True,cooperative=False,waves=4),
                'remapped':lambda:PagedResidualAttention(pp,max_batch=batch,threads=256,compact=True,cooperative=False,waves=4,remapped=True),
                'remapped_plain':lambda:PagedResidualAttention(pool,max_batch=batch,threads=256,compact=True,cooperative=False,waves=4,remapped=True,layout=False),
            }
            readers={name:factory() for name,factory in factories.items() if name in args.variants}
            for loop in args.loops:
                q=queries[loop]
                torch.cuda.synchronize();start=time.perf_counter()
                k,v=materialize(pool,raw,loop,rope,loc,n)
                torch.cuda.synchronize();materialize_ms=(time.perf_counter()-start)*1000
                native=NativeAttention(k,v,q,ptr,loc)
                ref=oracle(q,k,v,ptr,loc)
                sk,sv=materialize(sp,raw,loop,rope,loc,n,scalar=True)
                scalar_ref=oracle(q,sk,sv,ptr,loc)
                rk,rv=materialize(pool,raw,loop,rope,loc,n,original=True)
                original_ref=oracle(q,rk,rv,ptr,loc)
                calls={name:(lambda reader=reader:reader.forward(q,loop,ptr,loc,rope,128**-.5))
                       for name,reader in readers.items()}
                if 'materialized' in args.variants:calls['materialized']=native
                correctness={}
                for name,fn in calls.items():
                    got=fn();expected=scalar_ref if name in ('scalar_int4','staged_scalar') else ref
                    torch.testing.assert_close(got,expected,atol=.002,rtol=.01)
                    correctness[name]=metrics(got,expected)
                row=dict(layer=layer,loop=loop,batch=batch,length=n,correctness=correctness,
                    output_quality={'vqplus':metrics(ref,original_ref),'scalar_int4':metrics(scalar_ref,original_ref)},
                    setup={'layout_conversion_ms':layout_ms,'materialization_ms':materialize_ms},
                    logical_bytes_per_token_head={'vqplus':208,'scalar_int4':144,'bf16':512})
                if args.profile_one:
                    fn=calls[args.profile_one]
                    for _ in range(5):fn()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart();fn();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
                elif not args.correctness_only:
                    row['timing']=paired(calls,args.repeats,args.inner)
                result['records'].append(row)
                args.output.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(row),flush=True)
                del calls,native,k,v,sk,sv,rk,rv
            del readers,factories,pool,pp,sp,spp,raw,queries
    result['passed']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
