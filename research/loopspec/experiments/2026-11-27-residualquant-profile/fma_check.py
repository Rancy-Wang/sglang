"""Bitwise CUDA check of the exact FP16-scale x INT4 reconstruction identity."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from torch.utils.cpp_extension import load_inline


CUDA = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
__global__ void verify(const int* edges, unsigned long long* report, bool full_center) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned all = i & 65535, edge = edges[i >> 20];
    const unsigned sb = full_center ? edge : all;
    const unsigned cb = full_center ? all : edge;
    if ((sb & 0x7c00) == 0x7c00 || (cb & 0x7f80) == 0x7f80) return;
    const float scale = __half2float(__ushort_as_half((unsigned short)sb));
    const float center = __uint_as_float(cb << 16);
    const float code = (int)((i >> 16) & 15) - 8;
    const float product = __fmul_rn(scale, code);
    const float separate = __fadd_rn(center, product);
    const float fused = __fmaf_rn(scale, code, center);
    if (__float_as_uint(separate) != __float_as_uint(fused)) {
        atomicAdd(report, 1ULL);
        atomicMin(report + 1, (unsigned long long)i);
    }
}
void run(torch::Tensor edges, torch::Tensor report, bool full_center) {
    TORCH_CHECK(edges.is_cuda() && report.is_cuda());
    TORCH_CHECK(edges.numel() == 64 && edges.scalar_type() == at::kInt);
    TORCH_CHECK(report.numel() == 2 && report.scalar_type() == at::kLong);
    verify<<<(1 << 26) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        edges.data_ptr<int>(), reinterpret_cast<unsigned long long*>(report.data_ptr<int64_t>()), full_center);
    C10_CUDA_CHECK(cudaGetLastError());
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ext = load_inline(name='vqrq_fma_identity_r8',
        cpp_sources='#include <torch/extension.h>\nvoid run(torch::Tensor,torch::Tensor,bool);',
        cuda_sources=CUDA, functions=['run'],
        extra_cuda_cflags=['-O3','--fmad=false','-gencode=arch=compute_80,code=sm_80'])
    centers = [sign | (exp << 7) | mant for sign in (0,32768)
               for exp in (0,1,2,63,126,127,128,254) for mant in (0,1,63,127)]
    scales = [sign | (exp << 10) | mant for sign in (0,32768)
              for exp in (0,1,2,7,15,23,29,30) for mant in (0,1,511,1023)]
    results = []
    for full_center, edges in ((False,centers),(True,scales)):
        edges_tensor = torch.tensor(edges,device='cuda',dtype=torch.int32)
        report = torch.tensor([0,-1],device='cuda',dtype=torch.int64)
        ext.run(edges_tensor,report,full_center)
        mismatch, first = report.tolist()
        results.append(dict(full_center=full_center,mismatches=mismatch,first=first,
                            finite_cases=(65280 if full_center else 63488)*16*64,edges=edges))
        assert mismatch == 0, results[-1]
    result = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),results=results,passed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=sum(r['finite_cases'] for r in results))),flush=True)


if __name__ == '__main__':
    main()
