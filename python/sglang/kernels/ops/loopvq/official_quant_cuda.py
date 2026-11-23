# Copyright 2026 LoopVQ authors and SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Adapted from mscheong01/LoopVQ d5d58c83fe5bc9a4ce470ae96c69b55f085e2977.
"""Official G16 MODE 1 grouped CUDA stage 1; R1/R2 and PC1 only.

CUDA math/layout are from upstream vqwide_g.py. Added device bounds checks
prevent malformed split metadata from exceeding the shared-memory arena.
JIT and dynamic shared-memory setup must finish before graph capture.
"""
from dataclasses import dataclass

import torch

CAP = 2048
MAX_SPLITS = 48
GROUP_SIZE = 2

_CUDA_SRC = r'''

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>
#include <c10/macros/Macros.h>

#define NGw   VQW_GEOM_NG
#define KCw   VQW_GEOM_KC
#define NGVw  VQW_GEOM_NGV
#define KCVw  VQW_GEOM_KCV
#define Lw    VQW_GEOM_L
#define Gw    (Lw / NGw)
#define GVw   (Lw / NGVw)
#define HALFw (Lw / 2)
#define CAPw  VQW_SPLIT_CAP          /* tokens per split summed over the group */
#define THRw  VQW_THR
#define NWw   (THRw / 32)
#define UCw   VQW_UC
#define MINBK 32
#define TBLw  (NGw * KCw * Gw > NGVw * KCVw * GVw ? NGw * KCw * Gw : NGVw * KCVw * GVw)
#define RMAX  4

__device__ __forceinline__ float q2f(const __half v)        { return __half2float(v); }
__device__ __forceinline__ float q2f(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float q2f(const float v)         { return v; }

template <int GG>
__device__ __forceinline__ void load_entry(const __half* row, float* k) {
    if (GG == 4) {
        const uint2 x = *reinterpret_cast<const uint2*>(row);
        const unsigned w[2] = {x.x, x.y};
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[t]));
            k[2 * t] = a.x; k[2 * t + 1] = a.y;
        }
    } else {
        const uint4 x = *reinterpret_cast<const uint4*>(row);
        const unsigned w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[t]));
            k[2 * t] = a.x; k[2 * t + 1] = a.y;
        }
    }
}

template <typename QT, typename IT, typename ST, int R>
__global__ __launch_bounds__(THRw) void vqg_stage1(
    const QT* __restrict__ Q, const uint8_t* __restrict__ K_Idx, const __half* __restrict__ K_CB,
    const uint8_t* __restrict__ V_Idx, const __half* __restrict__ V_CB,
    const ST* __restrict__ K_SZ, const ST* __restrict__ V_SZ, const float* __restrict__ CosSin,
    const int* __restrict__ kv_indptr, const IT* __restrict__ kv_indices,
    const int* __restrict__ num_kv_splits, float* __restrict__ Att_Out, float* __restrict__ Att_Lse,
    int B, float sm_scale, int pos_offset,
    long s_q_b, long s_q_h, long s_ki_b, long s_ki_h, long s_vi_b, long s_vi_h,
    long s_ksz_b, long s_ksz_h, long s_vsz_b, long s_vsz_h, long s_cs_p,
    long s_o_b, long s_o_h, long s_o_s, long s_l_b, long s_l_h, long s_l_s)
{
    constexpr int CT = CAPw / R;                 // tokens per split per request
    constexpr int UAw = (R == 1) ? VQW_UA1 : VQW_UAG;   // tokens in flight per warp, phase A
    const int gb = blockIdx.x, kvh = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int kv0[R], len[R];
    int maxlen = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int b = gb * R + r;
        kv0[r] = b < B ? kv_indptr[b] : 0;
        len[r] = b < B ? kv_indptr[b + 1] - kv0[r] : 0;
        maxlen = max(maxlen, len[r]);
    }
    const int splits = num_kv_splits[gb * R];   // the dispatcher gives every member the group's count
    CUDA_KERNEL_ASSERT(splits >= 1 && splits <= VQW_MAX_SPLITS);
    const int per = ((maxlen + splits - 1) / splits + MINBK - 1) / MINBK * MINBK;
    CUDA_KERNEL_ASSERT(per <= CT);
    if (blockIdx.z >= splits || per * (int)blockIdx.z >= maxlen) return;

    extern __shared__ __align__(16) char smem_raw[];
    __half* tbl_s = reinterpret_cast<__half*>(smem_raw);       // TBL
    float*  q_s   = reinterpret_cast<float*>(tbl_s + TBLw);    // R*L
    int*    loc_s = reinterpret_cast<int*>(q_s + R * Lw);      // R*CT
    float*  qk_s  = reinterpret_cast<float*>(loc_s + R * CT);  // R*CT
    float*  vs_s  = qk_s + R * CT;                             // R*CT
    float*  out_s = vs_s + R * CT;                             // NW*L
    float*  red_s = out_s + NWw * Lw;                          // NW
    float*  ml_s  = red_s + NWw;                               // 2R

    for (int i = tid; i < R * Lw; i += THRw) {
        const int r = i / Lw, c = i % Lw, b = gb * R + r;
        q_s[i] = b < B ? q2f(Q[(long)b * s_q_b + (long)kvh * s_q_h + c]) : 0.f;
    }

  for (int bs = blockIdx.z; bs < splits; bs += gridDim.z) {
    const int s_start = per * bs;
    if (s_start >= maxlen) break;
    int nt[R];
    int nmax = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) { nt[r] = max(0, min(per, len[r] - s_start)); nmax = max(nmax, nt[r]); }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int n = tid; n < nt[r]; n += THRw) loc_s[r * CT + n] = (int)kv_indices[kv0[r] + s_start + n];
    {
        const int4* src = reinterpret_cast<const int4*>(K_CB + (long)kvh * ((long)NGw * KCw * Gw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGw * KCw * Gw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

    // ---- phase A: RoPE-pair lanes; one cos/sin load per position serves the R requests ----
    {
        constexpr int GG = Gw, NP = NGw / 2, LPT = NP, TPW = 32 / LPT, NF4 = GG / 4;
        const int sub = lane / LPT, ln = lane % LPT;
        float ql[R][GG], qh[R][GG];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < GG; ++j) { ql[r][j] = q_s[r * Lw + ln * GG + j]; qh[r][j] = q_s[r * Lw + HALFw + ln * GG + j]; }
        const __half* tl = tbl_s + ln * KCw * GG;
        const __half* th = tbl_s + (ln + NP) * KCw * GG;
        for (int n0 = warp * TPW * UAw; n0 < nmax; n0 += NWw * TPW * UAw) {
            float4 co[UAw][NF4], si[UAw][NF4];
            int il[UAw][R], ih[UAw][R];
            float ksv[UAw][R], vsv[UAw][R];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const int nn = n < nmax ? n : 0;
                const float* cs = CosSin + (long)(pos_offset + s_start + nn) * s_cs_p;
#pragma unroll
                for (int t = 0; t < NF4; ++t) {
                    co[u][t] = __ldg(reinterpret_cast<const float4*>(cs + ln * GG) + t);
                    si[u][t] = __ldg(reinterpret_cast<const float4*>(cs + HALFw + ln * GG) + t);
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const bool ok = n < nt[r];
                    const long loc = ok ? loc_s[r * CT + n] : 0;
                    const uint8_t* ir = K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b;
                    il[u][r] = ok ? (int)__ldg(ir + ln) : 0;
                    ih[u][r] = ok ? (int)__ldg(ir + ln + NP) : 0;
                    ksv[u][r] = vsv[u][r] = 0.f;
                    if (ok && ln == 0) {
                        ksv[u][r] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u][r] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                float c[GG], s[GG];
#pragma unroll
                for (int t = 0; t < NF4; ++t) {
                    c[4 * t] = co[u][t].x; c[4 * t + 1] = co[u][t].y; c[4 * t + 2] = co[u][t].z; c[4 * t + 3] = co[u][t].w;
                    s[4 * t] = si[u][t].x; s[4 * t + 1] = si[u][t].y; s[4 * t + 2] = si[u][t].z; s[4 * t + 3] = si[u][t].w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float kl[GG], kh[GG];
                    load_entry<GG>(tl + il[u][r] * GG, kl);
                    load_entry<GG>(th + ih[u][r] * GG, kh);
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < GG; ++j) {
                        part = fmaf(kl[j], fmaf(ql[r][j], c[j], qh[r][j] * s[j]), part);
                        part = fmaf(kh[j], fmaf(qh[r][j], c[j], -ql[r][j] * s[j]), part);
                    }
#pragma unroll
                    for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                    if (ln == 0 && n < nt[r]) { qk_s[r * CT + n] = part * ksv[u][r] * sm_scale; vs_s[r * CT + n] = vsv[u][r]; }
                }
            }
        }
    }
    __syncthreads();

    // ---- phase B: softmax per request ----
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float m = -INFINITY;
        for (int n = tid; n < nt[r]; n += THRw) m = fmaxf(m, qk_s[r * CT + n]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) red_s[warp] = m;
        __syncthreads();
        if (tid == 0) { float bm = red_s[0]; for (int w = 1; w < NWw; ++w) bm = fmaxf(bm, red_s[w]); ml_s[r] = bm; }
        __syncthreads();
        const float mh = ml_s[r];
        float l = 0.f;
        for (int n = tid; n < nt[r]; n += THRw) {
            const float e = __expf(qk_s[r * CT + n] - mh);
            qk_s[r * CT + n] = e * vs_s[r * CT + n];
            l += e;
        }
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
        if (lane == 0) red_s[warp] = l;
        __syncthreads();
        if (tid == 0) { float bl = 0.f; for (int w = 0; w < NWw; ++w) bl += red_s[w]; ml_s[R + r] = bl; }
        __syncthreads();
    }
    {
        const int4* src = reinterpret_cast<const int4*>(V_CB + (long)kvh * ((long)NGVw * KCVw * GVw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGVw * KCVw * GVw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

#if VQG_PC
    // ---- phase C per request: GPL code groups per lane (one GPL-byte code load and one p read
    // serve GPL*GV outputs), LPT lanes per token, SPW tokens per warp per step ----
    constexpr int GPL = (GVw == 4) ? 4 : 2;
    constexpr int LPC = NGVw / GPL;
    constexpr int SPW = 32 / LPC;
    constexpr int NST = NWw * SPW;
    const int lg = lane % LPC;
    const int stripe = warp * SPW + lane / LPC;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        if (nt[r] <= 0) continue;                  // uniform over the block
        const int b = gb * R + r;
        const uint8_t* vrow = V_Idx + (long)kvh * s_vi_h + lg * GPL;
        float acc[GPL * GVw];
#pragma unroll
        for (int j = 0; j < GPL * GVw; ++j) acc[j] = 0.f;
        for (int n0 = stripe; n0 < nt[r]; n0 += NST * UCw) {
            unsigned cw[UCw];
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                if (n < nt[r]) {
                    const uint8_t* rr = vrow + (long)loc_s[r * CT + n] * s_vi_b;
                    cw[u] = (GPL == 4) ? __ldg(reinterpret_cast<const unsigned*>(rr))
                                       : (unsigned)__ldg(reinterpret_cast<const unsigned short*>(rr));
                } else cw[u] = 0u;
            }
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                if (n >= nt[r]) continue;
                const float p = qk_s[r * CT + n];
#pragma unroll
                for (int a = 0; a < GPL; ++a) {
                    float v[GVw];
                    load_entry<GVw>(tbl_s + ((lg * GPL + a) * KCVw + ((cw[u] >> (8 * a)) & 0xff)) * GVw, v);
#pragma unroll
                    for (int j = 0; j < GVw; ++j) acc[a * GVw + j] = fmaf(p, v[j], acc[a * GVw + j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < GPL * GVw; ++j)
            for (int o = LPC; o < 32; o <<= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], o);
        if (lane < LPC) {
#pragma unroll
            for (int j = 0; j < GPL * GVw; ++j) out_s[warp * Lw + lg * GPL * GVw + j] = acc[j];
        }
        __syncthreads();
        float* out_base = Att_Out + (long)b * s_o_b + (long)bs * s_o_s + (long)kvh * s_o_h;
        for (int cc = tid; cc < Lw; cc += THRw) {
            float t = 0.f;
#pragma unroll
            for (int rr = 0; rr < NWw; ++rr) t += out_s[rr * Lw + cc];
            out_base[cc] = t / ml_s[R + r];
        }
        if (tid == 0) Att_Lse[(long)b * s_l_b + (long)kvh * s_l_h + (long)bs * s_l_s] = ml_s[r] + logf(ml_s[R + r]);
        __syncthreads();
    }
#else
    // ---- phase C per request (served layout: thread = (V group, token stripe)) ----
    constexpr int WPS  = NGVw >= 32 ? NGVw / 32 : 1;
    constexpr int SPW  = NGVw >= 32 ? 1 : 32 / NGVw;
    constexpr int NST  = NGVw >= 32 ? NWw / WPS : NWw * SPW;
    constexpr int ROWS = NGVw >= 32 ? NST : NWw;
    const int g      = NGVw >= 32 ? (warp % WPS) * 32 + lane : lane % NGVw;
    const int stripe = NGVw >= 32 ? warp / WPS : warp * SPW + lane / NGVw;
    const __half* vg = tbl_s + g * KCVw * GVw;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        if (nt[r] <= 0) continue;                  // uniform over the block
        const int b = gb * R + r;
        const uint8_t* vrow = V_Idx + (long)kvh * s_vi_h + g;
        float acc[GVw];
#pragma unroll
        for (int j = 0; j < GVw; ++j) acc[j] = 0.f;
        for (int n0 = stripe; n0 < nt[r]; n0 += NST * UCw) {
            int id[UCw];
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                id[u] = n < nt[r] ? (int)__ldg(vrow + (long)loc_s[r * CT + n] * s_vi_b) : -1;
            }
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                if (id[u] < 0) continue;
                const float p = qk_s[r * CT + n0 + u * NST];
                float v[GVw];
                load_entry<GVw>(vg + id[u] * GVw, v);
#pragma unroll
                for (int j = 0; j < GVw; ++j) acc[j] = fmaf(p, v[j], acc[j]);
            }
        }
        if (NGVw < 32) {
#pragma unroll
            for (int j = 0; j < GVw; ++j)
                for (int o = NGVw; o < 32; o <<= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], o);
        }
        const int row = NGVw >= 32 ? stripe : warp;
        if (NGVw >= 32 || lane < NGVw) {
#pragma unroll
            for (int j = 0; j < GVw; ++j) out_s[row * Lw + g * GVw + j] = acc[j];
        }
        __syncthreads();
        float* out_base = Att_Out + (long)b * s_o_b + (long)bs * s_o_s + (long)kvh * s_o_h;
        for (int cc = tid; cc < Lw; cc += THRw) {
            float t = 0.f;
#pragma unroll
            for (int rr = 0; rr < ROWS; ++rr) t += out_s[rr * Lw + cc];
            out_base[cc] = t / ml_s[R + r];
        }
        if (tid == 0) Att_Lse[(long)b * s_l_b + (long)kvh * s_l_h + (long)bs * s_l_s] = ml_s[r] + logf(ml_s[R + r]);
        __syncthreads();
    }
#endif
  }
}

template <int R>
static int smem_bytes_r() {
    return (int)(sizeof(__half) * TBLw + sizeof(float) * R * Lw + (sizeof(int) + 2 * sizeof(float)) * CAPw
                 + sizeof(float) * NWw * Lw + sizeof(float) * NWw + sizeof(float) * 2 * R + 64);
}

#define FOR_ST(X, R) X(__nv_bfloat16, long, __half, R) X(__nv_bfloat16, long, float, R) \
                     X(__nv_bfloat16, int, __half, R) X(__nv_bfloat16, int, float, R) \
                     X(__half, long, __half, R) X(__half, long, float, R) \
                     X(__half, int, __half, R) X(__half, int, float, R)

void vqwide_v2_init() {
#define REG(QT, IT, ST, R) TORCH_CHECK(cudaFuncSetAttribute(vqg_stage1<QT, IT, ST, R>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes_r<R>()) == cudaSuccess, "vqg: smem attr");
    FOR_ST(REG, 1) FOR_ST(REG, VQG_R)
#undef REG
}

void vqwide_v2_cuda(torch::Tensor q, torch::Tensor k_idx, torch::Tensor k_cb,
                    torch::Tensor v_idx, torch::Tensor v_cb, torch::Tensor k_sz, torch::Tensor v_sz,
                    torch::Tensor cos_sin, torch::Tensor freq_idx, torch::Tensor w_perm,
                    torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor splits,
                    torch::Tensor att_out, torch::Tensor att_lse, int64_t n_splits, double sm_scale,
                    int64_t pos_offset, int64_t mode) {
    // mode 1: one request per block; 17: groups of VQG_R (the caller sized the splits for it)
    TORCH_CHECK(mode == 1 || mode == 17, "vqwide_g: MODE 1 (plain pre-RoPE) only, got ", mode);
    const int B = q.size(0), H_KV = k_idx.size(1);
    const int R = mode == 17 ? VQG_R : 1;
    const int NB = (B + R - 1) / R;
    int dev = 0, n_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    const int zw = (int)std::max<int64_t>(1, std::min<int64_t>(n_splits, (2 * n_sm + NB * H_KV - 1) / (NB * H_KV)));
    dim3 grid(NB, H_KV, zw);
#define GO(QT, IT, ST, RR) vqg_stage1<QT, IT, ST, RR><<<grid, THRw, smem_bytes_r<RR>(), at::cuda::getCurrentCUDAStream()>>>( \
        reinterpret_cast<const QT*>(q.data_ptr()), k_idx.data_ptr<uint8_t>(),                     \
        reinterpret_cast<const __half*>(k_cb.data_ptr()), v_idx.data_ptr<uint8_t>(),              \
        reinterpret_cast<const __half*>(v_cb.data_ptr()),                                         \
        reinterpret_cast<const ST*>(k_sz.data_ptr()), reinterpret_cast<const ST*>(v_sz.data_ptr()), \
        cos_sin.data_ptr<float>(), kv_indptr.data_ptr<int>(),                                     \
        reinterpret_cast<const IT*>(kv_indices.data_ptr()), splits.data_ptr<int>(),               \
        att_out.data_ptr<float>(), att_lse.data_ptr<float>(), B, (float)sm_scale, (int)pos_offset, \
        q.stride(0), q.stride(1), k_idx.stride(0), k_idx.stride(1), v_idx.stride(0), v_idx.stride(1), \
        k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1), cos_sin.stride(0),        \
        att_out.stride(0), att_out.stride(1), att_out.stride(2),                                  \
        att_lse.stride(0), att_lse.stride(1), att_lse.stride(2))
#define PICKR(QT, IT, ST) do { if (R == 1) GO(QT, IT, ST, 1); else GO(QT, IT, ST, VQG_R); } while (0)
#define PICKS(QT, IT) do { if (k_sz.scalar_type() == at::kHalf) PICKR(QT, IT, __half); else PICKR(QT, IT, float); } while (0)
    const bool i64 = kv_indices.scalar_type() == at::kLong;
    if (q.scalar_type() == at::kBFloat16) { if (i64) PICKS(__nv_bfloat16, long); else PICKS(__nv_bfloat16, int); }
    else { if (i64) PICKS(__half, long); else PICKS(__half, int); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef GO
#undef PICKR
#undef PICKS
}

'''


@dataclass
class _Prepared:
    extension: object
    unit_scale: torch.Tensor
    dummy_float: torch.Tensor
    dummy_long: torch.Tensor
    max_context_len: int


_PREPARED = {}


def _device_key(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Official LoopVQ CUDA requires a CUDA device")
    return device.index if device.index is not None else torch.cuda.current_device()


def prepare_official_quant(*, device, max_context_len, max_splits=MAX_SPLITS):
    """Compile and configure shared memory once, outside graph capture.

    This first optimized profile deliberately fixes the validated A100 G16
    geometry and upstream CAP/R/threads/unroll settings. No environment knob
    silently changes arithmetic, capacity, or the generated source.
    """
    if max_splits != MAX_SPLITS or not 0 < max_context_len <= CAP // GROUP_SIZE * max_splits:
        raise ValueError("Official LoopVQ requires 48 splits and context <= 49152")
    key = _device_key(device)
    with torch.cuda.device(key):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare official LoopVQ CUDA before graph capture")
        if torch.cuda.get_device_capability(key) != (8, 0):
            raise ValueError("This official LoopVQ profile is validated for sm_80 only")
        if key in _PREPARED:
            _PREPARED[key].max_context_len = max_context_len
            return
        from torch.utils.cpp_extension import load_inline

        extension = load_inline(
            name="loopvq_official_g16_mode1_r2_c2048_t512_pc1_v1",
            cpp_sources=(
                "#include <torch/extension.h>\n"
                "void vqwide_v2_cuda(torch::Tensor, torch::Tensor, torch::Tensor,"
                " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
                " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
                " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
                " int64_t, double, int64_t, int64_t);\n"
                "void vqwide_v2_init();"
            ),
            cuda_sources=_CUDA_SRC,
            functions=["vqwide_v2_cuda", "vqwide_v2_init"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo", "-gencode=arch=compute_80,code=sm_80",
                "-DVQW_GEOM_NG=32", "-DVQW_GEOM_KC=256",
                "-DVQW_GEOM_NGV=32", "-DVQW_GEOM_KCV=256",
                "-DVQW_GEOM_L=128", "-DVQW_SPLIT_CAP=2048",
                "-DVQW_THR=512", "-DVQW_UA1=2", "-DVQW_UAG=1",
                "-DVQW_UC=8", "-DVQG_R=2", "-DVQG_PC=1",
                "-DVQW_MAX_SPLITS=48",
            ],
        )
        extension.vqwide_v2_init()
        _PREPARED[key] = _Prepared(
            extension, torch.ones((1, 1, 1), device=torch.device("cuda", key), dtype=torch.float16),
            torch.zeros((1, 1), device=torch.device("cuda", key), dtype=torch.float32),
            torch.zeros((1, 1), device=torch.device("cuda", key), dtype=torch.int64), max_context_len,
        )


def official_quant_attention(q, k_idx, k_cb, v_idx, v_cb, kv_indptr, kv_indices,
                             num_splits, att_out, att_lse, cos_sin_cache, sm_scale,
                             *, pos_offset, group_size=1, max_splits=MAX_SPLITS):
    """Fill normalized FP32 per-split outputs and LSEs, leaving empty splits alone.

    Caller resets LSE to -inf before every layer and supplies equal split counts
    to both members of each R2 pair. Splits are dynamically sized by metadata,
    with ceil32(ceil(max_pair_length / splits)) <= 2048 / group_size.
    """
    if group_size not in (1, 2) or max_splits != MAX_SPLITS:
        raise ValueError("Official LoopVQ supports R1/R2 and 48 quant split slots")
    if q.ndim != 3 or q.shape[1:] != (16, 128) or q.dtype != torch.bfloat16:
        raise ValueError("Official LoopVQ requires BF16 [B,16,128] queries")
    key = _device_key(q.device)
    prepared = _PREPARED.get(key)
    if prepared is None:
        raise RuntimeError("Call prepare_official_quant before using this kernel")
    b = q.shape[0]
    if pos_offset < 0 or pos_offset >= cos_sin_cache.shape[0]:
        raise ValueError("Invalid quant-prefix RoPE offset")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 128 or cos_sin_cache.dtype != torch.float32:
        raise ValueError("Official LoopVQ requires an FP32 [positions,128] RoPE table")
    if cos_sin_cache.shape[0] < prepared.max_context_len or not cos_sin_cache.is_contiguous():
        raise ValueError("RoPE table is noncontiguous or shorter than the configured context")
    for idx, cb in ((k_idx, k_cb), (v_idx, v_cb)):
        if idx.ndim != 3 or idx.shape[1:] != (16, 32) or idx.dtype != torch.uint8:
            raise ValueError("Official LoopVQ requires raw uint8 [slots,16,32] indices")
        if idx.stride(-1) != 1 or idx.stride(-2) != 32:
            raise ValueError("LoopVQ index heads/groups must be contiguous")
        if cb.shape != (16, 32, 256, 4) or cb.dtype != torch.float16 or not cb.is_contiguous():
            raise ValueError("Official LoopVQ requires contiguous FP16 G16 decode tables")
    if kv_indptr.shape != (b + 1,) or kv_indptr.dtype != torch.int32:
        raise ValueError("Quant indptr must be int32 [B+1]")
    if num_splits.shape != (b,) or num_splits.dtype != torch.int32:
        raise ValueError("Quant split counts must be int32 [B]")
    if kv_indices.ndim != 1 or kv_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quant locations must be an int32/int64 vector")
    if att_out.shape != (b, 16, max_splits, 128) or att_lse.shape != (b, 16, max_splits):
        raise ValueError("Invalid official quant split scratch shape")
    if att_out.dtype != torch.float32 or att_lse.dtype != torch.float32:
        raise ValueError("Official split scratch must be FP32")
    tensors = (k_idx, k_cb, v_idx, v_cb, kv_indptr, kv_indices, num_splits,
               att_out, att_lse, cos_sin_cache)
    if any(t.device != q.device for t in tensors):
        raise ValueError("Official LoopVQ operands must be on one CUDA device")
    if any(t.stride(-1) != 1 for t in (q, kv_indptr, kv_indices, num_splits, att_out, att_lse)):
        raise ValueError("Official LoopVQ requires contiguous innermost dimensions")
    prepared.extension.vqwide_v2_cuda(
        q, k_idx, k_cb, v_idx, v_cb,
        prepared.unit_scale.expand(k_idx.shape[0], 16, 2),
        prepared.unit_scale.expand(v_idx.shape[0], 16, 2),
        cos_sin_cache, prepared.dummy_long, prepared.dummy_float,
        kv_indptr, kv_indices, num_splits, att_out, att_lse,
        max_splits, float(sm_scale), int(pos_offset), 17 if group_size == 2 else 1,
    )
