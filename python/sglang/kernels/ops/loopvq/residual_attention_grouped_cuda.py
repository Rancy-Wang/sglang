# SPDX-License-Identifier: Apache-2.0
"""R2 VQ+: reuse shared centroids and position coefficients, never request KV."""
import torch
from .residual_attention_cuda import _SRC as _R1

# Share the exact signed-nibble reconstruction and BF16 rounding primitives.
_HEADER = _R1.split('template<typename IT>')[0].replace(
    'constexpr int CAP = 1024, SPLITS = 32, UNROLL = VQPLUS_UNROLL;',
    'constexpr int CAP = 512, SPLITS = 64, R = 2;').replace(
    'TABLE * 2 + 128 * 4 + CAP * 4 + WARPS * 128 * 4 + WARPS * 4 + 8',
    'TABLE * 2 + R * (128 * 4 + CAP * 4 + WARPS * 4 + 8) + WARPS * 128 * 4')
_SRC = _HEADER + r'''
template<typename IT>
__global__ __launch_bounds__(THREADS, 2) void residual_grouped_stage1(
    const __nv_bfloat16* Q, const uint8_t* KI, const uint8_t* VI,
    const __nv_bfloat16* KC, const __nv_bfloat16* VC,
    const uint8_t* KR, const uint8_t* VR, const __half* KS, const __half* VS,
    const __nv_bfloat16* HK, const __nv_bfloat16* HV,
    const int* PTR, const IT* LOC, const float* ROPE, float* OUT, float* LSE,
    int B, int H, int offset, long qs, long qh, float scale, int target) {
    const int first_b = blockIdx.x * R, h = blockIdx.y, split = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int begin[R], length[R], count[R];
    int longest = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int b = first_b + r;
        begin[r] = b < B ? PTR[b] : 0;
        length[r] = b < B ? PTR[b + 1] - begin[r] : 0;
        if (length[r] && LOC[begin[r]] == 0) length[r] = 0;
        longest = max(longest, length[r]);
    }
    int active = min(SPLITS, max((longest + CAP - 1) / CAP,
        min(target, max(1, (longest + 63) / 64))));
    const int per = (longest + active - 1) / active, start = split * per;
    CUDA_KERNEL_ASSERT(per <= CAP);
    int nmax = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        count[r] = max(0, min(per, length[r] - start));
        nmax = max(nmax, count[r]);
    }
    if (!nmax) {
#pragma unroll
        for (int r = 0; r < R; ++r) if (first_b + r < B) {
            long dst = ((long)(first_b + r) * H + h) * SPLITS + split;
            if (tid < 128) OUT[dst * 128 + tid] = 0;
            if (!tid) LSE[dst] = -INFINITY;
        }
        return;
    }
    extern __shared__ __align__(16) char raw[];
    auto* table = reinterpret_cast<__nv_bfloat16*>(raw);
    float* query = reinterpret_cast<float*>(table + TABLE);
    float* prob = query + R * 128;
    float* values = prob + R * CAP;
    float* reduce = values + WARPS * 128;
    float* stats = reduce + R * WARPS;
    for (int i = tid; i < R * 128; i += THREADS) {
        int r = i / 128, c = i % 128, b = first_b + r;
        query[i] = b < B ? __bfloat162float(Q[b * qs + h * qh + c]) : 0.f;
    }
    for (int i = tid; i < TABLE / 8; i += THREADS)
        reinterpret_cast<int4*>(table)[i] = __ldg(reinterpret_cast<const int4*>(KC + h * TABLE) + i);
    __syncthreads();
    const int g = lane & 15;
    float qlo[R][4], qhi[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            qlo[r][j] = query[r * 128 + g * 4 + j];
            qhi[r][j] = query[r * 128 + 64 + g * 4 + j];
        }
    for (int n0 = warp * 2; n0 < nmax; n0 += WARPS * 2) {
        const int n = n0 + lane / 16;
        float4 co = {}, si = {};
        if (n < nmax) {
            const float* rope = ROPE + (long)(start + n) * 128 + g * 4;
            co = __ldg(reinterpret_cast<const float4*>(rope));
            si = __ldg(reinterpret_cast<const float4*>(rope + 64));
        }
        const float* c = reinterpret_cast<const float*>(&co);
        const float* s = reinterpret_cast<const float*>(&si);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float dot = 0;
            if (n < count[r]) {
                float low[4], high[4];
                int loc = (int)LOC[begin[r] + start + n];
                row4(KI, table, KR, KS, HK, loc, offset, H, h, g, low);
                row4(KI, table, KR, KS, HK, loc, offset, H, h, g + 16, high);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float kl = bf(low[j] * c[j] - high[j] * s[j]);
                    float kh = bf(high[j] * c[j] + low[j] * s[j]);
                    dot += fmaf(kl, qlo[r][j], kh * qhi[r][j]);
                }
            }
            for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d, 16);
            if (g == 0 && n < count[r]) prob[r * CAP + n] = dot * scale;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float mx = -INFINITY;
        for (int n = tid; n < count[r]; n += THREADS) mx = fmaxf(mx, prob[r * CAP + n]);
        mx = warp_max(mx);
        if (!lane) reduce[r * WARPS + warp] = mx;
    }
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float mx = warp_max(lane < WARPS ? reduce[r * WARPS + lane] : -INFINITY);
            if (!lane) stats[r * 2] = count[r] ? mx : 0.f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float sum = 0;
        for (int n = tid; n < count[r]; n += THREADS) {
            float p = __expf(prob[r * CAP + n] - stats[r * 2]);
            prob[r * CAP + n] = p; sum += p;
        }
        sum = warp_sum(sum);
        if (!lane) reduce[r * WARPS + warp] = sum;
    }
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float sum = warp_sum(lane < WARPS ? reduce[r * WARPS + lane] : 0.f);
            if (!lane) {
                stats[r * 2 + 1] = count[r] ? sum : 1.f;
                if (first_b + r < B) {
                    long dst = ((long)(first_b + r) * H + h) * SPLITS + split;
                    LSE[dst] = count[r] ? stats[r * 2] + logf(sum) : -INFINITY;
                }
            }
        }
    }
    for (int i = tid; i < TABLE / 8; i += THREADS)
        reinterpret_cast<int4*>(table)[i] = __ldg(reinterpret_cast<const int4*>(VC + h * TABLE) + i);
    __syncthreads();
    const int vg = lane & 7;
    // Only one request's PV accumulators and shared reduction are live.
    // QK still reuses position coefficients across the two requests.
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc[16] = {};
        for (int n0 = warp * 4; n0 < count[r]; n0 += WARPS * 4) {
            int n = n0 + lane / 8;
            if (n < count[r]) {
                float p = prob[r * CAP + n];
                int loc = (int)LOC[begin[r] + start + n];
#pragma unroll
                for (int group = 0; group < 4; ++group) {
                    float v[4];
                    row4(VI, table, VR, VS, HV, loc, offset, H, h, vg * 4 + group, v);
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[group * 4 + j] = fmaf(p, bf(v[j]), acc[group * 4 + j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j] += __shfl_down_sync(0xffffffff, acc[j], 16);
            acc[j] += __shfl_down_sync(0xffffffff, acc[j], 8);
            if (lane < 8) values[warp * 128 + vg * 16 + j] = acc[j];
        }
        __syncthreads();
        if (tid < 128 && first_b + r < B) {
            float total = 0;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) total += values[w * 128 + tid];
            long dst = ((long)(first_b + r) * H + h) * SPLITS + split;
            OUT[dst * 128 + tid] = total / stats[r * 2 + 1];
        }
        __syncthreads();
    }
}
void grouped_init() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(residual_grouped_stage1<int>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(residual_grouped_stage1<int64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
}
void grouped_cuda(torch::Tensor q, torch::Tensor ki, torch::Tensor vi,
    torch::Tensor kc, torch::Tensor vc, torch::Tensor kr, torch::Tensor vr,
    torch::Tensor ks, torch::Tensor vs, torch::Tensor hk, torch::Tensor hv,
    torch::Tensor ptr, torch::Tensor loc, torch::Tensor rope, torch::Tensor out,
    torch::Tensor lse, int64_t offset, double scale, int64_t target) {
    dim3 grid((q.size(0) + R - 1) / R, q.size(1), SPLITS);
#define RUN(IT) residual_grouped_stage1<IT><<<grid, THREADS, SMEM, c10::cuda::getCurrentCUDAStream()>>>( \
    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), ki.data_ptr<uint8_t>(), vi.data_ptr<uint8_t>(), \
    reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr()), \
    kr.data_ptr<uint8_t>(), vr.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(ks.data_ptr()), \
    reinterpret_cast<const __half*>(vs.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(hk.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(hv.data_ptr()), ptr.data_ptr<int>(), loc.data_ptr<IT>(), \
    rope.data_ptr<float>(), out.data_ptr<float>(), lse.data_ptr<float>(), q.size(0), q.size(1), offset, q.stride(0), q.stride(1), scale, target)
    if (loc.scalar_type() == at::kLong) { RUN(int64_t); } else { RUN(int); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef RUN
}
'''
_EXT = None
_READY = set()


def prepare_grouped(device):
    global _EXT
    device = torch.device(device)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare grouped VQ+ before capture')
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError('Grouped VQ+ is validated for sm_80 only')
        if _EXT is None:
            from torch.utils.cpp_extension import load_inline
            _EXT = load_inline(
                name='loopvq_residual_int4_grouped_r2_t512_v2',
                cpp_sources='#include <torch/extension.h>\nvoid grouped_init();\n'
                'void grouped_cuda(' + ','.join(['torch::Tensor'] * 16) + ',int64_t,double,int64_t);',
                cuda_sources=_SRC, functions=['grouped_init', 'grouped_cuda'],
                extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false', '-DVQPLUS_THREADS=512', '-gencode=arch=compute_80,code=sm_80'],
            )
        if device.index not in _READY:
            _EXT.grouped_init(); _READY.add(device.index)
    return _EXT
