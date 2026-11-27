# SPDX-License-Identifier: Apache-2.0
"""Packed VQ+ payload reader, preserving pre-RoPE reconstruction numerics.

FP16 scale times a signed INT4 integer is exactly representable in FP32.
Explicit reconstruction FMA therefore preserves the finite-codec FP32 result.
RoPE, BF16 rounding, and the FP32 QK/PV summation remain unchanged.
"""
import torch

from .residual_attention_paged_cuda import _SRC, _replace_once

HELPERS = r'''

__device__ __forceinline__ void pair8(const uint8_t* idx, const __nv_bfloat16* table,
    const uint8_t* residual, const __half* scales, const __nv_bfloat16* hp,
    int loc, int offset, int heads, int head, int group, float* low, float* high) {
    if (loc < offset) {
        const long row = ((long)(loc >> 3) * heads + head) * 8 + (loc & 7);
        const int code0 = __ldg(idx + row * 32 + group);
        const int code1 = __ldg(idx + row * 32 + group + 16);
        const unsigned z0 = __ldg(reinterpret_cast<const uint16_t*>(residual + row * 64 + group * 2));
        const unsigned z1 = __ldg(reinterpret_cast<const uint16_t*>(residual + row * 64 + group * 2 + 32));
        const float scale0 = __half2float(__ldg(scales + row * 4 + group / 8));
        const float scale1 = __half2float(__ldg(scales + row * 4 + group / 8 + 2));
        const uint2 c0 = __ldg(reinterpret_cast<const uint2*>(table + (group * 256 + code0) * 4));
        const uint2 c1 = __ldg(reinterpret_cast<const uint2*>(table + ((group + 16) * 256 + code1) * 4));
        const auto* v0 = reinterpret_cast<const __nv_bfloat16*>(&c0);
        const auto* v1 = reinterpret_cast<const __nv_bfloat16*>(&c1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r0 = (int)(z0 << (28 - j * 4)) >> 28;
            const int r1 = (int)(z1 << (28 - j * 4)) >> 28;
            low[j] = __bfloat162float(v0[j]) + scale0 * r0;
            high[j] = __bfloat162float(v1[j]) + scale1 * r1;
        }
    } else {
        const long row = (long)(loc - offset) * heads + head;
        const uint2 c0 = __ldg(reinterpret_cast<const uint2*>(hp + row * 128 + group * 4));
        const uint2 c1 = __ldg(reinterpret_cast<const uint2*>(hp + row * 128 + group * 4 + 64));
        const auto* v0 = reinterpret_cast<const __nv_bfloat16*>(&c0);
        const auto* v1 = reinterpret_cast<const __nv_bfloat16*>(&c1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            low[j] = __bfloat162float(v0[j]);
            high[j] = __bfloat162float(v1[j]);
        }
    }
}

__device__ __forceinline__ void pv16(const uint8_t* idx, const __nv_bfloat16* table,
    const uint8_t* residual, const __half* scales, const __nv_bfloat16* hp,
    int loc, int offset, int heads, int head, int group16, float p, float* acc) {
    if (loc < offset) {
        const long row = ((long)(loc >> 3) * heads + head) * 8 + (loc & 7);
        const unsigned codes = __ldg(reinterpret_cast<const unsigned*>(idx + row * 32 + group16 * 4));
        const uint2 packed = __ldg(reinterpret_cast<const uint2*>(residual + row * 64 + group16 * 8));
        const float scale = __half2float(__ldg(scales + row * 4 + group16 / 2));
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const unsigned z = (group < 2 ? packed.x : packed.y) >> ((group & 1) * 16);
            const int code = (codes >> (group * 8)) & 255;
            const uint2 center = __ldg(reinterpret_cast<const uint2*>(table + ((group16 * 4 + group) * 256 + code) * 4));
            const auto* cv = reinterpret_cast<const __nv_bfloat16*>(&center);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = (int)(z << (28 - j * 4)) >> 28;
                const float value = __bfloat162float(cv[j]) + scale * r;
                acc[group * 4 + j] = fmaf(p, bf(value), acc[group * 4 + j]);
            }
        }
    } else {
        const long row = (long)(loc - offset) * heads + head;
        const uint4 c0 = __ldg(reinterpret_cast<const uint4*>(hp + row * 128 + group16 * 16));
        const uint4 c1 = __ldg(reinterpret_cast<const uint4*>(hp + row * 128 + group16 * 16 + 8));
        const auto* v0 = reinterpret_cast<const __nv_bfloat16*>(&c0);
        const auto* v1 = reinterpret_cast<const __nv_bfloat16*>(&c1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = fmaf(p, bf(__bfloat162float(v0[j])), acc[j]);
            acc[j + 8] = fmaf(p, bf(__bfloat162float(v1[j])), acc[j + 8]);
        }
    }
}
'''

def _payload_source():
    mode, min_blocks, fused = "both", 3, True
    assert mode in ('qk', 'pv', 'both')
    source = _replace_once(_SRC, '__device__ __forceinline__ void copy_table',
                          HELPERS + '\n__device__ __forceinline__ void copy_table')
    if mode in ('qk', 'both'):
        source = _replace_once(source,
            '''            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low, true);
            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g + 16, high, true);''',
            '            pair8(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low, high);')
    if mode in ('pv', 'both'):
        source = _replace_once(source, '''#pragma unroll
            for (int group = 0; group < 4; ++group) {
                float v[4];
                row4(VI, table, VR, VS, HV, (int)LOC[begin + start + n], offset, H, h, vgroup * 4 + group, v);
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[group * 4 + j] = fmaf(p, bf(v[j]), acc[group * 4 + j]);
            }''', '            pv16(VI, table, VR, VS, HV, (int)LOC[begin + start + n], offset, H, h, vgroup, p, acc);')
    if fused:
        for old, new in (
            ("x[j] = __bfloat162float(cv[j]) + scale * v;", "x[j] = fmaf(scale, (float)v, __bfloat162float(cv[j]));"),
            ("low[j] = __bfloat162float(v0[j]) + scale0 * r0;", "low[j] = fmaf(scale0, (float)r0, __bfloat162float(v0[j]));"),
            ("high[j] = __bfloat162float(v1[j]) + scale1 * r1;", "high[j] = fmaf(scale1, (float)r1, __bfloat162float(v1[j]));"),
            ("const float value = __bfloat162float(cv[j]) + scale * r;", "const float value = fmaf(scale, (float)r, __bfloat162float(cv[j]));"),
        ):
            source = _replace_once(source, old, new)
    if min_blocks:
        assert min_blocks == 3
        source = _replace_once(source, "__launch_bounds__(THREADS)", "__launch_bounds__(THREADS, 3)")
    return source

def _split_policy(batch, chunks):
    """Exact integer policy, evaluated once while generating CUDA source."""
    capacity, tasks = 324, batch * 8
    waves = max(1, (chunks * tasks + capacity - 1) // capacity)
    requested = max(chunks, waves * capacity // tasks)
    if chunks > 1 and requested * tasks * 10 < waves * capacity * 9:
        waves += 1
        requested = max(chunks, waves * capacity // tasks)
    return min(32, requested)


_POLICY = "__device__ __constant__ unsigned char PAYLOAD_SPLITS[13][33] = {\n" + ",\n".join(
    "{" + ",".join(str(_split_policy(batch, chunks)) for chunks in range(33)) + "}"
    for batch in range(4, 17)
) + "\n};\n"
_SOURCE = _replace_once(_payload_source(), "template<typename IT>", _POLICY + "template<typename IT>")
_SOURCE = _replace_once(_SOURCE,
    """    const int active = active_target ? min(SPLITS, max((length + CAP - 1) / CAP,
        min(active_target, max(1, (length + 63) / 64)))) : SPLITS;""",
    """    const int requested = active_target < 0
        ? PAYLOAD_SPLITS[gridDim.x - 4][min(SPLITS, (length + CAP - 1) / CAP)]
        : active_target;
    const int active = requested ? min(SPLITS, max((length + CAP - 1) / CAP,
        min(requested, max(1, (length + 63) / 64)))) : SPLITS;""")
_SOURCE += r'''
int64_t residual_capacity() {
    int device = 0, blocks32 = 0, blocks64 = 0;
    int sm_count = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks32, stage1<int>, THREADS, SMEM));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks64, stage1<int64_t>, THREADS, SMEM));
    return std::min(blocks32, blocks64) * sm_count;
}
'''
_SOURCE = _replace_once(_SOURCE, 'void stage1(', 'void payload_stage1(')
_SOURCE = _SOURCE.replace('stage1<', 'payload_stage1<')
_EXT = {}
_READY = {}


def prepare(device, max_context_len=32768):
    from .residual_context import context_source, context_suffix
    key = max_context_len > 32768
    source = context_source(_SOURCE, max_context_len)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare payload VQ+ before graph capture")
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError("Payload VQ+ currently targets SM80")
        if key not in _EXT:
            from torch.utils.cpp_extension import load_inline
            _EXT[key] = load_inline(
                name="loopvq_payload_r2_t512_l2_smattr" + context_suffix(max_context_len),
                cpp_sources="#include <torch/extension.h>\nvoid residual_init();\n"
                    "int64_t residual_capacity();\nvoid residual_cuda(" +
                    ",".join(["torch::Tensor"] * 16) + ",int64_t,double,int64_t);",
                cuda_sources=source,
                functions=["residual_init", "residual_capacity", "residual_cuda"],
                extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false", "-Xptxas=-v",
                    "-DVQPLUS_THREADS=512", "-DVQPLUS_UNROLL=1", "-DVQPLUS_GLOBAL=1",
                    "-DVQPLUS_ASYNC=0", "-DVQPAGE_LAYOUT=2", "-DVQPAGE_COOP=0",
                    "-DVQPAGE_SCALAR=0", "-gencode=arch=compute_80,code=sm_80"],
            )
        index = (key, torch.cuda.current_device())
        if index not in _READY:
            _EXT[key].residual_init()
            _READY[index] = _EXT[key].residual_capacity()
        return _EXT[key], _READY[index]
