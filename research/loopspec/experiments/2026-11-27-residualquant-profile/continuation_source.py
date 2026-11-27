"""R14+ source-level ablations against the retained R13 payload reader."""
from sglang.kernels.ops.loopvq.residual_attention_payload_cuda import _SOURCE


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


PV8 = r'''
__device__ __forceinline__ void pv8(const uint8_t* idx, const __nv_bfloat16* table,
    const uint8_t* residual, const __half* scales, const __nv_bfloat16* hp,
    int loc, int offset, int heads, int head, int group8, float p, float* acc) {
    if (loc < offset) {
        const long row = ((long)(loc >> 3) * heads + head) * 8 + (loc & 7);
        const unsigned codes = __ldg(reinterpret_cast<const uint16_t*>(idx + row * 32 + group8 * 2));
        const unsigned packed = __ldg(reinterpret_cast<const unsigned*>(residual + row * 64 + group8 * 4));
        const float scale = __half2float(__ldg(scales + row * 4 + group8 / 4));
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const unsigned z = packed >> (group * 16);
            const int code = (codes >> (group * 8)) & 255;
            const uint2 center = __ldg(reinterpret_cast<const uint2*>(table + ((group8 * 2 + group) * 256 + code) * 4));
            const auto* cv = reinterpret_cast<const __nv_bfloat16*>(&center);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = (int)(z << (28 - j * 4)) >> 28;
                const float value = fmaf(scale, (float)r, __bfloat162float(cv[j]));
                acc[group * 4 + j] = fmaf(p, bf(value), acc[group * 4 + j]);
            }
        }
    } else {
        const long row = (long)(loc - offset) * heads + head;
        const uint4 packed = __ldg(reinterpret_cast<const uint4*>(hp + row * 128 + group8 * 8));
        const auto* value = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(p, __bfloat162float(value[j]), acc[j]);
    }
}
'''


def source_for(mode):
    source = _SOURCE
    if mode.startswith('pv8'):
        assert mode in ('pv8', 'pv8_pack', 'pv8_b4')
        begin = source.index('__device__ __forceinline__ void pv16(')
        end = source.index('__device__ __forceinline__ void copy_table', begin)
        source = source[:begin] + PV8 + '\n' + source[end:]
        source = replace_once(source, 'const int vgroup = lane & 7;', 'const int vgroup = lane & 15;')
        source = replace_once(source, 'float acc[16] = {};', 'float acc[8] = {};')
        source = replace_once(source,
            'for (int n0 = warp * 4; n0 < count; n0 += WARPS * 4)',
            'for (int n0 = warp * 2; n0 < count; n0 += WARPS * 2)')
        source = replace_once(source, 'const int n = n0 + lane / 8;', 'const int n = n0 + lane / 16;')
        source = replace_once(source, '            pv16(VI, table,', '            pv8(VI, table,')
        source = replace_once(source,
            '''    for (int j = 0; j < 16; ++j) {
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], 16);
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], 8);
        if (lane < 8) values[warp * 128 + vgroup * 16 + j] = acc[j];
    }''',
            '''    for (int j = 0; j < 8; ++j) {
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], 16);
        if (lane < 16) values[warp * 128 + vgroup * 8 + j] = acc[j];
    }''')
        if mode == 'pv8_b4':
            source = replace_once(source, '__launch_bounds__(THREADS, 3)', '__launch_bounds__(THREADS, 4)')
        if mode == 'pv8_pack':
            packed = source_for('pack_pv')
            begin = packed.index('            for (int j = 0; j < 4; j += 2)')
            end = packed.index('\n            }', begin) + len('\n            }')
            begin_old = source.index('            for (int j = 0; j < 4; ++j)', source.index('void pv8('))
            end_old = source.index('\n            }', begin_old) + len('\n            }')
            source = source[:begin_old] + packed[begin:end] + source[end_old:]
    elif mode.startswith('grid'):
        grid = int(mode[4:])
        assert grid in (8, 16, 32)
        source = replace_once(source,
            'const int b = blockIdx.x, h = blockIdx.y, split = blockIdx.z;',
            'const int b = blockIdx.x, h = blockIdx.y;')
        source = replace_once(source,
            '    const int start = split * per, count = max(0, min(per, length - start));',
            '    for (int split = blockIdx.z; split < SPLITS; split += gridDim.z) {\n'
            '    const int start = split * per, count = max(0, min(per, length - start));')
        source = replace_once(source,
            '        if (!tid) LSE[dst] = -INFINITY;\n        return;',
            '        if (!tid) LSE[dst] = -INFINITY;\n        continue;')
        source = replace_once(source,
            '        OUT[dst * 128 + tid] = total / stats[1];\n    }\n}',
            '        OUT[dst * 128 + tid] = total / stats[1];\n    }\n'
            '    __syncthreads();\n    }\n}')
        source = replace_once(source, 'dim3 grid(q.size(0), q.size(1), SPLITS);',
                              f'dim3 grid(q.size(0), q.size(1), {grid});')
    elif mode in ('pack_qk', 'pack_pv', 'pack_both'):
        if mode in ('pack_qk', 'pack_both'):
            source = replace_once(source,
                '''                float kl = bf(low[j] * c[j] - high[j] * s[j]);
                float kh = bf(high[j] * c[j] + low[j] * s[j]);
                dot += fmaf(kl, qlo[j], kh * qhi[j]);''',
                '''                const float2 kv = __bfloat1622float2(__floats2bfloat162_rn(
                    low[j] * c[j] - high[j] * s[j], high[j] * c[j] + low[j] * s[j]));
                dot += fmaf(kv.x, qlo[j], kv.y * qhi[j]);''')
        if mode in ('pack_pv', 'pack_both'):
            source = replace_once(source,
                '''            for (int j = 0; j < 4; ++j) {
                const int r = (int)(z << (28 - j * 4)) >> 28;
                const float value = fmaf(scale, (float)r, __bfloat162float(cv[j]));
                acc[group * 4 + j] = fmaf(p, bf(value), acc[group * 4 + j]);
            }''',
                '''            for (int j = 0; j < 4; j += 2) {
                const int r0 = (int)(z << (28 - j * 4)) >> 28;
                const int r1 = (int)(z << (24 - j * 4)) >> 28;
                const float v0 = fmaf(scale, (float)r0, __bfloat162float(cv[j]));
                const float v1 = fmaf(scale, (float)r1, __bfloat162float(cv[j + 1]));
                const float2 value = __bfloat1622float2(__floats2bfloat162_rn(v0, v1));
                acc[group * 4 + j] = fmaf(p, value.x, acc[group * 4 + j]);
                acc[group * 4 + j + 1] = fmaf(p, value.y, acc[group * 4 + j + 1]);
            }''')
    else:
        raise ValueError(mode)
    return source


def prepare(mode):
    from torch.utils.cpp_extension import load_inline
    round_id = 16 if mode.startswith('pv8') else 14 if mode.startswith('grid') else 15
    extension = load_inline(
        name=f'vqrq_cont_r{round_id}_{mode}',
        cpp_sources='#include <torch/extension.h>\nvoid residual_init();\n'
            'int64_t residual_capacity();\nvoid residual_cuda(' +
            ','.join(['torch::Tensor'] * 16) + ',int64_t,double,int64_t);',
        cuda_sources=source_for(mode),
        functions=['residual_init', 'residual_capacity', 'residual_cuda'],
        extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false', '-Xptxas=-v',
            '-DVQPLUS_THREADS=512', '-DVQPLUS_UNROLL=1', '-DVQPLUS_GLOBAL=1',
            '-DVQPLUS_ASYNC=0', '-DVQPAGE_LAYOUT=2', '-DVQPAGE_COOP=0',
            '-DVQPAGE_SCALAR=0', '-gencode=arch=compute_80,code=sm_80'])
    extension.residual_init()
    return extension
