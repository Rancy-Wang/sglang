"""Bitwise page encoder/flush parity with unchanged token-layout encoding."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from sglang.kernels.ops.loopvq.encode import encode_steps, encode_from_hp
from sglang.kernels.ops.loopvq.residual_encode import encode_residual_steps, encode_residual_from_hp
from sglang.kernels.ops.loopvq.residual_attention_paged_cuda import page_unpack


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.manual_seed(20261126)
    lb,h,cap,n=3,8,64,21
    enc=torch.randn(lb,h,32,256,16,device='cuda').half()
    cb=torch.stack([enc[b,...,u::4] for u in range(4) for b in range(lb)]).bfloat16().contiguous()
    hp=torch.randn(lb*4,cap,h,128,device='cuda').bfloat16()
    dst=torch.cat((torch.randperm(cap-8,device='cuda')[:n]+8,
                   torch.tensor([-1,cap+8],device='cuda'))).int()
    src=torch.cat((torch.randperm(cap,device='cuda')[:n],torch.tensor([-1,0],device='cuda'))).int()
    valid=torch.arange(n+2,device='cuda')%5!=0
    def buffers():
        return (torch.full((lb,cap,h,32),173,device='cuda',dtype=torch.uint8),
                torch.full((lb*4,cap,h,64),165,device='cuda',dtype=torch.uint8),
                torch.full((lb*4,cap,h,4),-1.,device='cuda',dtype=torch.float16))
    def same(old,page):
        for a,b in zip(old,page):
            recovered=torch.stack([page_unpack(t) for t in b])
            torch.testing.assert_close(recovered,a,atol=0,rtol=0)
    old,page=buffers(),buffers()
    for data,paged in ((old,False),(page,True)):
        idx,res,scale=data
        encode_from_hp(hp,enc,idx,src,dst,valid,paged=paged)
        encode_residual_from_hp(hp,idx,cb,res,scale,src,dst,valid,paged=paged)
    same(old,page)
    for base in range(lb):
        steps=[torch.randn(n+2,h*3,128,device='cuda').bfloat16()[:,:h] for _ in range(4)]
        for data,paged in ((old,False),(page,True)):
            idx,res,scale=data
            encode_steps(steps,enc[base],idx[base],dst,paged=paged)
            encode_residual_steps(steps,idx[base],cb,res,scale,dst,base,paged=paged)
        same(old,page)
    # Address an actual sidecar above2GiB without touching the unallocated rows.
    # 192slots *32776tokens *8heads *64bytes =3.22GB; int32 DST must still work.
    large_cap,slots=32776,192
    li=torch.zeros((1,large_cap,h,32),device='cuda',dtype=torch.uint8)
    lr=torch.empty((slots,large_cap,h,64),device='cuda',dtype=torch.uint8)
    ls=torch.empty((slots,large_cap,h,4),device='cuda',dtype=torch.float16)
    lc=torch.zeros((slots,h,32,256,4),device='cuda',dtype=torch.bfloat16)
    loc=torch.tensor([8],device='cuda',dtype=torch.int32)
    steps=[torch.ones(1,h,128,device='cuda',dtype=torch.bfloat16)*(u+1) for u in range(4)]
    for paged in (False,True):
        encode_residual_steps(steps,li[0],lc,lr,ls,loc,47,paged=paged)
        for u in range(4):
            slot=u*48+47
            r=page_unpack(lr[slot,8:16]) if paged else lr[slot,8:16]
            s=page_unpack(ls[slot,8:16]) if paged else ls[slot,8:16]
            assert (r[0]==119).all()  # both signed nibbles are+7
            torch.testing.assert_close(s[0],torch.full_like(s[0],(u+1)/7),atol=0,rtol=0)
    torch.cuda.synchronize()
    result=dict(passed=True,source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                cases=['masked_flush','invalid_destinations','three_base_layers_four_loops',
                       'strided_prefill','all_packed_arrays_bitwise','int32_locations_above_2GiB'])
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
