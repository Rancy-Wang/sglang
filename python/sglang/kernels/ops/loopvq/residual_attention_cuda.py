# SPDX-License-Identifier: Apache-2.0
"""Shared-codebook VQ+ stage 1. Compilation and smem setup precede capture."""
import torch

_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#ifndef VQPLUS_ASYNC
#define VQPLUS_ASYNC 0
#endif
#ifndef VQPLUS_GLOBAL
#define VQPLUS_GLOBAL 0
#endif

constexpr int THREADS = VQPLUS_THREADS, WARPS = THREADS / 32, TABLE = 32 * 256 * 4;
constexpr int CAP = 1024, SPLITS = 32, UNROLL = VQPLUS_UNROLL;
constexpr int TABLE_SMEM = VQPLUS_GLOBAL ? 0 : TABLE * 2;
constexpr int SMEM = TABLE_SMEM + 128 * 4 + CAP * 4 + WARPS * 128 * 4 + WARPS * 4 + 8;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    return x;
}
// One group of four channels, center loaded as one aligned 64-bit word.
__device__ __forceinline__ void row4(const uint8_t* idx, const __nv_bfloat16* table,
    const uint8_t* residual, const __half* scales, const __nv_bfloat16* hp,
    int loc, int offset, int heads, int head, int group, float* x) {
    if (loc < offset) {
        const long row = (long)loc * heads + head;
        int code = __ldg(idx + row * 32 + group);
#if VQPLUS_GLOBAL
        uint2 center = __ldg(reinterpret_cast<const uint2*>(table + (group * 256 + code) * 4));
#else
        uint2 center = *reinterpret_cast<const uint2*>(table + (group * 256 + code) * 4);
#endif
        const auto* cv = reinterpret_cast<const __nv_bfloat16*>(&center);
        unsigned z = *reinterpret_cast<const uint16_t*>(residual + row * 64 + group * 2);
        float scale = __half2float(__ldg(scales + row * 4 + group / 8));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (z >> (j * 4)) & 15;
            v = v >= 8 ? v - 16 : v;
            x[j] = __bfloat162float(cv[j]) + scale * v;
        }
    } else {
        const long row = (long)(loc - offset) * heads + head;
        uint2 value = *reinterpret_cast<const uint2*>(hp + row * 128 + group * 4);
        const auto* v = reinterpret_cast<const __nv_bfloat16*>(&value);
#pragma unroll
        for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(v[j]);
    }
}

__device__ __forceinline__ void copy_table(__nv_bfloat16* dst, const __nv_bfloat16* src) {
#if !VQPLUS_GLOBAL
    for (int i = threadIdx.x; i < TABLE / 8; i += THREADS) {
#if VQPLUS_ASYNC
        unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst + i * 8));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem), "l"(src + i * 8) : "memory");
#else
        reinterpret_cast<int4*>(dst)[i] = __ldg(reinterpret_cast<const int4*>(src) + i);
#endif
    }
#if VQPLUS_ASYNC
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
#endif
}
__device__ __forceinline__ void wait_table() {
#if VQPLUS_ASYNC
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
}

template<typename IT>
__global__ __launch_bounds__(THREADS) void stage1(
    const __nv_bfloat16* Q, const uint8_t* KI, const uint8_t* VI,
    const __nv_bfloat16* KC, const __nv_bfloat16* VC,
    const uint8_t* KR, const uint8_t* VR, const __half* KS, const __half* VS,
    const __nv_bfloat16* HK, const __nv_bfloat16* HV,
    const int* PTR, const IT* LOC, const float* ROPE, float* OUT, float* LSE,
    int H, int offset, long qs, long qh, float scale, int active_target) {
    const int b = blockIdx.x, h = blockIdx.y, split = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int begin = PTR[b];
    int length = PTR[b + 1] - begin;
    if (length && LOC[begin] == 0) length = 0; // graph dummy request
    const int active = active_target ? min(SPLITS, max((length + CAP - 1) / CAP,
        min(active_target, max(1, (length + 63) / 64)))) : SPLITS;
    const int per = (length + active - 1) / active;
    const int start = split * per, count = max(0, min(per, length - start));
    const long dst = ((long)b * H + h) * SPLITS + split;
    CUDA_KERNEL_ASSERT(per <= CAP);
    if (!count) {
        if (tid < 128) OUT[dst * 128 + tid] = 0;
        if (!tid) LSE[dst] = -INFINITY;
        return;
    }
    extern __shared__ __align__(16) char raw[];
    auto* table = reinterpret_cast<__nv_bfloat16*>(raw);
#if VQPLUS_GLOBAL
    table = const_cast<__nv_bfloat16*>(KC + h * TABLE);
#endif
    float* query = reinterpret_cast<float*>(raw + TABLE_SMEM);
    float* prob = query + 128;
    float* values = prob + CAP;
    float* reduce = values + WARPS * 128;
    float* stats = reduce + WARPS;
    if (tid < 128) query[tid] = __bfloat162float(Q[b * qs + h * qh + tid]);
    copy_table(table, KC + h * TABLE);
    wait_table();
    // Two tokens per warp, one lane owns four low/high RoPE channel pairs.
    const int g = lane & 15;
    float qlo[4], qhi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { qlo[j] = query[g * 4 + j]; qhi[j] = query[64 + g * 4 + j]; }
    #pragma unroll (UNROLL)
    for (int n0 = warp * 2; n0 < count; n0 += WARPS * 2) {
        const int n = n0 + lane / 16;
        float dot = 0;
        if (n < count) {
            float low[4], high[4];
            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low);
            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g + 16, high);
            const float* rope = ROPE + (long)(start + n) * 128 + g * 4;
            float4 co = __ldg(reinterpret_cast<const float4*>(rope));
            float4 si = __ldg(reinterpret_cast<const float4*>(rope + 64));
            const float* c = reinterpret_cast<const float*>(&co);
            const float* s = reinterpret_cast<const float*>(&si);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float kl = bf(low[j] * c[j] - high[j] * s[j]);
                float kh = bf(high[j] * c[j] + low[j] * s[j]);
                dot += fmaf(kl, qlo[j], kh * qhi[j]);
            }
        }
        for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d, 16);
        if (g == 0 && n < count) prob[n] = dot * scale;
    }
    __syncthreads();
#if VQPLUS_ASYNC
    // No QK thread still reads the K table; softmax uses separate smem.
    copy_table(table, VC + h * TABLE);
#endif
    float mx = -INFINITY;
    for (int n = tid; n < count; n += THREADS) mx = fmaxf(mx, prob[n]);
    mx = warp_max(mx);
    if (!lane) reduce[warp] = mx;
    __syncthreads();
    if (!warp) {
        mx = warp_max(lane < WARPS ? reduce[lane] : -INFINITY);
        if (!lane) stats[0] = mx;
    }
    __syncthreads();
    float sum = 0;
    for (int n = tid; n < count; n += THREADS) {
        float p = __expf(prob[n] - stats[0]); prob[n] = p; sum += p;
    }
    sum = warp_sum(sum);
    if (!lane) reduce[warp] = sum;
    __syncthreads();
    if (!warp) {
        sum = warp_sum(lane < WARPS ? reduce[lane] : 0.f);
        if (!lane) { stats[1] = sum; LSE[dst] = stats[0] + logf(sum); }
    }
    // Replace the K table after every QK consumer has finished.
#if VQPLUS_GLOBAL
    table = const_cast<__nv_bfloat16*>(VC + h * TABLE);
#endif
#if !VQPLUS_ASYNC
    copy_table(table, VC + h * TABLE);
#endif
    wait_table();
    // Four tokens per warp, eight lanes per token; 16 output channels per lane.
    const int vgroup = lane & 7;
    float acc[16] = {};
    #pragma unroll (UNROLL)
    for (int n0 = warp * 4; n0 < count; n0 += WARPS * 4) {
        const int n = n0 + lane / 8;
        if (n < count) {
            float p = prob[n];
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                float v[4];
                row4(VI, table, VR, VS, HV, (int)LOC[begin + start + n], offset, H, h, vgroup * 4 + group, v);
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[group * 4 + j] = fmaf(p, bf(v[j]), acc[group * 4 + j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], 16);
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], 8);
        if (lane < 8) values[warp * 128 + vgroup * 16 + j] = acc[j];
    }
    __syncthreads();
    if (tid < 128) {
        float total = 0;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += values[w * 128 + tid];
        OUT[dst * 128 + tid] = total / stats[1];
    }
}

void residual_init() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(stage1<int>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(stage1<int64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
}
void residual_cuda(torch::Tensor q, torch::Tensor ki, torch::Tensor vi,
    torch::Tensor kc, torch::Tensor vc, torch::Tensor kr, torch::Tensor vr,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor hk, torch::Tensor hv,
    torch::Tensor ptr, torch::Tensor loc, torch::Tensor rope, torch::Tensor out,
    torch::Tensor lse, int64_t offset, double scale, int64_t target) {
    dim3 grid(q.size(0), q.size(1), SPLITS);
#define RUN(IT) stage1<IT><<<grid, THREADS, SMEM, c10::cuda::getCurrentCUDAStream()>>>( \
    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), ki.data_ptr<uint8_t>(), vi.data_ptr<uint8_t>(), \
    reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr()), \
    kr.data_ptr<uint8_t>(), vr.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(ks.data_ptr()), \
    reinterpret_cast<const __half*>(vs.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(hk.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(hv.data_ptr()), ptr.data_ptr<int>(), loc.data_ptr<IT>(), \
    rope.data_ptr<float>(), out.data_ptr<float>(), lse.data_ptr<float>(), q.size(1), offset, q.stride(0), q.stride(1), scale, target)
    if (loc.scalar_type() == at::kLong) { RUN(int64_t); } else { RUN(int); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef RUN
}
'''

_EXT = {}
_READY = set()


def prepare(device, threads=512, unroll=1, async_table=False, global_table=False):
    if unroll not in (1, 2, 4):
        raise ValueError("Supported VQ+ token unroll: 1/2/4")
    variant = (threads, unroll, bool(async_table), bool(global_table))
    if threads not in (128, 256, 512):
        raise ValueError("Supported VQ+ CTA sizes: 128/256/512")
    device = torch.device(device)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('VQ+ CUDA must be prepared before capture')
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError('VQ+ CUDA currently validated for sm_80')
        if variant not in _EXT:
            from torch.utils.cpp_extension import load_inline
            _EXT[variant] = load_inline(
                name=f'loopvq_residual_int4_shared_t{threads}_u{unroll}_a{int(async_table)}_g{int(global_table)}_v5',
                cpp_sources='#include <torch/extension.h>\nvoid residual_init();\n'
                'void residual_cuda(' + ','.join(['torch::Tensor'] * 16) + ',int64_t,double,int64_t);',
                cuda_sources=_SRC, functions=['residual_init', 'residual_cuda'],
                extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false', f'-DVQPLUS_THREADS={threads}', f'-DVQPLUS_UNROLL={unroll}', f'-DVQPLUS_ASYNC={int(async_table)}', f'-DVQPLUS_GLOBAL={int(global_table)}', '-gencode=arch=compute_80,code=sm_80'],
            )
        if (device.index, variant) not in _READY:
            _EXT[variant].residual_init()
            _READY.add((device.index, variant))
    return _EXT[variant]
