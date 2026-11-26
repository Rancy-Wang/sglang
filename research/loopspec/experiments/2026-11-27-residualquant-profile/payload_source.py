"""R6 token payload experiments; unchanged codec and FP32/BF16 boundaries."""
from sglang.kernels.ops.loopvq.residual_attention_paged_cuda import _SRC


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


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def source_for(mode, min_blocks=0, fused=False):
    assert mode in ('qk', 'pv', 'both')
    source = replace_once(_SRC, '__device__ __forceinline__ void copy_table',
                          HELPERS + '\n__device__ __forceinline__ void copy_table')
    if mode in ('qk', 'both'):
        source = replace_once(source,
            '''            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low, true);
            row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g + 16, high, true);''',
            '            pair8(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low, high);')
    if mode in ('pv', 'both'):
        source = replace_once(source, '''#pragma unroll
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
            source = replace_once(source, old, new)
    if min_blocks:
        assert min_blocks == 3
        source = replace_once(source, "__launch_bounds__(THREADS)", "__launch_bounds__(THREADS, 3)")
    return source


def prepare(mode, threads=512, min_blocks=0, fused=False):
    from torch.utils.cpp_extension import load_inline
    assert not min_blocks or threads == 512
    suffix = f"_r7_b{min_blocks}" if min_blocks else ""
    suffix += "_r8_fma" if fused else ""
    ext = load_inline(name=f'vqrq_payload_r6_{mode}_t{threads}{suffix}',
        cpp_sources='#include <torch/extension.h>\nvoid residual_init();\nvoid residual_cuda(' +
                    ','.join(['torch::Tensor'] * 16) + ',int64_t,double,int64_t);',
        cuda_sources=source_for(mode, min_blocks, fused), functions=['residual_init', 'residual_cuda'],
        extra_cuda_cflags=['-O3','-lineinfo','--fmad=false','-Xptxas=-v',
            f'-DVQPLUS_THREADS={threads}','-DVQPLUS_UNROLL=1','-DVQPLUS_GLOBAL=1',
            '-DVQPLUS_ASYNC=0','-DVQPAGE_LAYOUT=2','-DVQPAGE_COOP=0',
            '-DVQPAGE_SCALAR=0','-gencode=arch=compute_80,code=sm_80'])
    ext.residual_init()
    return ext
