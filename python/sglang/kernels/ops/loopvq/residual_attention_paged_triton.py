# SPDX-License-Identifier: Apache-2.0
"""Tile-compiled page VQ+ attention with one reconstruction per K coordinate."""
import torch
import triton
import triton.language as tl
from .official_reduce import official_reduce


@triton.jit
def _rows(IDX, CB, RES, SCALE, HP, slots, channels, head, valid,
          H: tl.constexpr, OFFSET: tl.constexpr, SCALAR: tl.constexpr, LAYOUT:tl.constexpr):
    quant=(slots>=0)&(slots<OFFSET)
    mask=valid[:,None]&quant[:,None]
    c=channels[None,:]
    pg=(((slots[:,None]//8)*H+head)*4+c//32)*8+slots[:,None]%8
    if LAYOUT==2:
        row=((slots[:,None]//8)*H+head)*8+slots[:,None]%8
        io,ro,so=row*32+c//4,row*64+c//2,row*4+c//32
    else:
        io,ro,so=pg*8+(c//4)%8,pg*16+(c%32)//2,pg
    if SCALAR:
        center=tl.full((slots.shape[0],channels.shape[0]),0.,tl.float32)
    else:
        code=tl.load(IDX+io,mask,0).to(tl.int32)
        cp=((head*32+c//4)*256+code)*4+c%4
        center=tl.load(CB+tl.max_contiguous(cp,[1,4]),mask,0).to(tl.float32)
    packed=tl.load(RES+ro,mask,0).to(tl.int32)
    z=(packed>>((c%2)*4))&15
    z=tl.where(z>=8,z-16,z).to(tl.float32)
    scale=tl.load(SCALE+so,mask,0).to(tl.float32)
    hp=tl.load(HP+((slots[:,None]-OFFSET)*H+head)*128+c,
               valid[:,None]&~quant[:,None],0).to(tl.float32)
    return tl.where(quant[:,None],center+scale*z,hp)


@triton.jit
def _stage(Q, KI, VI, KC, VC, KR, VR, KS, VS, HK, HV, PTR, LOC, ROPE, OUT, LSE,
           H:tl.constexpr, OFFSET:tl.constexpr, QS:tl.constexpr,QH:tl.constexpr,
           ACTIVE:tl.constexpr, SCALE:tl.constexpr, SCALAR:tl.constexpr,N:tl.constexpr,LAYOUT:tl.constexpr):
    b,h,s=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    begin=tl.load(PTR+b)
    length=tl.load(PTR+b+1)-begin
    first=tl.load(LOC+begin,length>0,0)
    length=tl.where(first!=0,length,0)
    per=tl.cdiv(length,ACTIVE)
    lo,hi=s*per,tl.minimum((s+1)*per,length)
    c=tl.arange(0,64)
    d=tl.arange(0,128)
    qlo=tl.load(Q+b*QS+h*QH+c).to(tl.float32)
    qhi=tl.load(Q+b*QS+h*QH+c+64).to(tl.float32)
    maximum=tl.full((),-float('inf'),tl.float32)
    denom=tl.full((),0.,tl.float32)
    acc=tl.full((128,),0.,tl.float32)
    for start in range(lo,hi,N):
        pos=start+tl.arange(0,N)
        valid=pos<hi
        slots=tl.load(LOC+begin+pos,valid,0).to(tl.int64)
        low=_rows(KI,KC,KR,KS,HK,slots,c,h,valid,H,OFFSET,SCALAR,LAYOUT)
        high=_rows(KI,KC,KR,KS,HK,slots,c+64,h,valid,H,OFFSET,SCALAR,LAYOUT)
        co=tl.load(ROPE+pos[:,None]*128+c[None,:],valid[:,None],0)
        si=tl.load(ROPE+pos[:,None]*128+c[None,:]+64,valid[:,None],0)
        kl=(low*co-high*si).to(tl.bfloat16).to(tl.float32)
        kh=(high*co+low*si).to(tl.bfloat16).to(tl.float32)
        logits=tl.sum(kl*qlo[None,:]+kh*qhi[None,:],1)*SCALE
        logits=tl.where(valid,logits,-float('inf'))
        next_max=tl.maximum(maximum,tl.max(logits,0))
        alpha=tl.exp(maximum-next_max)
        prob=tl.exp(logits-next_max)
        v=_rows(VI,VC,VR,VS,HV,slots,d,h,valid,H,OFFSET,SCALAR,LAYOUT).to(tl.bfloat16).to(tl.float32)
        acc=acc*alpha+tl.sum(prob[:,None]*v,0)
        denom=denom*alpha+tl.sum(prob,0)
        maximum=next_max
    dst=(b*H+h)*32+s
    tl.store(OUT+dst*128+d,acc/tl.maximum(denom,1.e-30))
    tl.store(LSE+dst,tl.where(denom>0,maximum+tl.log(denom),-float('inf')))


class PagedTritonAttention:
    def __init__(self,pool,*,max_batch,max_context_len=32768,scalar=False,layout=1):
        if pool.head_dim!=128 or pool.residual_bits!=4 or not 0<max_context_len<=32768:
            raise ValueError('Page Triton requires D128/INT4/context<=32768')
        if torch.cuda.get_device_capability(pool.hp_k.device)!=(8,0):
            raise ValueError('Page Triton is currently validated on SM80')
        self.pool,self.scalar,self.max_batch=pool,scalar,max_batch
        if layout not in (1,2):raise ValueError('Unknown page layout')
        self.layout=layout
        self.waves,self.compact=4,True
        self.sm_count=torch.cuda.get_device_properties(pool.hp_k.device).multi_processor_count
        self.partial=torch.empty((max_batch,pool.head_num,32,128),device=pool.hp_k.device,dtype=torch.float32)
        self.lse=torch.empty((max_batch,pool.head_num,32),device=pool.hp_k.device,dtype=torch.float32)
    def forward(self,q,layer_id,ptr,loc,rope,scale):
        p,b=self.pool,len(q)
        if q.dtype!=torch.bfloat16 or q.shape[1:]!=(p.head_num,128) or not 0<b<=self.max_batch:
            raise ValueError('Page Triton requires BF16 [B,H,128] within capacity')
        active=min(32,max(1,(self.waves*self.sm_count+b*p.head_num-1)//(b*p.head_num)))
        _stage[(b,p.head_num,active)](q,p.get_key_buffer(layer_id),p.get_value_buffer(layer_id),
            p.k_book.cb16[layer_id],p.v_book.cb16[layer_id],p.k_residual[layer_id],p.v_residual[layer_id],
            p.k_residual_scale[layer_id],p.v_residual_scale[layer_id],p.get_hp_key_buffer(layer_id),
            p.get_hp_value_buffer(layer_id),ptr,loc,rope,self.partial,self.lse,p.head_num,
            p.hp_global_offset,q.stride(0),q.stride(1),active,float(scale),self.scalar,32,self.layout,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
        out=torch.empty_like(q)
        return official_reduce(self.partial[:b,:,:active],self.lse[:b,:,:active],out)
