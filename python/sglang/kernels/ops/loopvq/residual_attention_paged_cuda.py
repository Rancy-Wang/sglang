# SPDX-License-Identifier: Apache-2.0
"""Page/group32 VQ+ reader, preserving the deployed pre-RoPE codec.

The first experiment deliberately shares the baseline arithmetic/split code.
Only packed addressing and cooperative scale reads change. Layout conversion
is a diagnostic setup operation; production encoders must write this layout.
"""
import torch

from .residual_attention_cuda import _SRC as _BASE
from .official_reduce import official_reduce


def page_pack(tensor):
    """[slots,H,4*payload] -> flat [page,H,4,token8,payload]."""
    n, h, width = tensor.shape
    if n % 8 or width % 4:
        raise ValueError("Page layout requires page8 and four channel groups")
    return tensor.reshape(n // 8, 8, h, 4, width // 4).permute(
        0, 2, 3, 1, 4).contiguous().reshape_as(tensor)


def page_unpack(tensor):
    n, h, width = tensor.shape
    return tensor.reshape(n // 8, h, 4, 8, width // 4).permute(
        0, 3, 1, 2, 4).contiguous().reshape_as(tensor)


_ROW = r'''
__device__ __forceinline__ void row4(const uint8_t* idx, const __nv_bfloat16* table,
    const uint8_t* residual, const __half* scales, const __nv_bfloat16* hp,
    int loc, int offset, int heads, int head, int group, float* x, bool qk=false) {
    if (loc < offset) {
        const long row = (long)loc * heads + head;
        const long pg = (((long)(loc >> 3) * heads + head) * 4 + group / 8) * 8 + (loc & 7);
        const long io = VQPAGE_LAYOUT ? pg * 8 + group % 8 : row * 32 + group;
        const long ro = VQPAGE_LAYOUT ? pg * 16 + (group % 8) * 2 : row * 64 + group * 2;
        const long so = VQPAGE_LAYOUT ? pg : row * 4 + group / 8;
        uint2 center = {};
#if !VQPAGE_SCALAR
        int code = __ldg(idx + io);
        center = __ldg(reinterpret_cast<const uint2*>(table + (group * 256 + code) * 4));
#endif
        const auto* cv = reinterpret_cast<const __nv_bfloat16*>(&center);
        unsigned z = *reinterpret_cast<const uint16_t*>(residual + ro);
        float scale = 0.f;
#if VQPAGE_COOP
        const int lane = threadIdx.x & 31;
        // QK: 8 lanes/group32. PV: two lanes/group32, each owns 16 channels.
        const int leader = qk ? (lane & ~7) : (lane & ~1);
        if (lane == leader) scale = __half2float(__ldg(scales + so));
        scale = __shfl_sync(__activemask(), scale, leader);
#else
        scale = __half2float(__ldg(scales + so));
#endif
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = (int)(z << (28 - j * 4)) >> 28;
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
'''

_start = _BASE.index('__device__ __forceinline__ void row4(')
_end = _BASE.index('__device__ __forceinline__ void copy_table', _start)
_SRC = _BASE[:_start] + _ROW + '\n' + _BASE[_end:]
_SRC = _SRC.replace('h, g, low);', 'h, g, low, true);').replace(
    'h, g + 16, high);', 'h, g + 16, high, true);')
_EXT = {}


def prepare(device, *, threads=512, layout=True, cooperative=True, scalar=False):
    key = (threads, bool(layout), bool(cooperative), bool(scalar))
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare page VQ+ before graph capture')
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError('Page VQ+ currently targets SM80')
        if threads not in (128, 256, 512):
            raise ValueError('Unsupported CTA size')
        if key not in _EXT:
            from torch.utils.cpp_extension import load_inline
            ext = load_inline(
                name=f'loopvq_page_r1_t{threads}_l{int(layout)}_c{int(cooperative)}_s{int(scalar)}',
                cpp_sources='#include <torch/extension.h>\nvoid residual_init();\n'
                'void residual_cuda(' + ','.join(['torch::Tensor'] * 16) + ',int64_t,double,int64_t);',
                cuda_sources=_SRC, functions=['residual_init', 'residual_cuda'],
                extra_cuda_cflags=['-O3', '-lineinfo', '--fmad=false', '-Xptxas=-v',
                    f'-DVQPLUS_THREADS={threads}', '-DVQPLUS_UNROLL=1',
                    '-DVQPLUS_GLOBAL=1', '-DVQPLUS_ASYNC=0',
                    f'-DVQPAGE_LAYOUT={int(layout)}', f'-DVQPAGE_COOP={int(cooperative)}',
                    f'-DVQPAGE_SCALAR={int(scalar)}', '-gencode=arch=compute_80,code=sm_80'],
            )
            _EXT[key] = ext
        _EXT[key].residual_init()
    return _EXT[key]


class PagedResidualAttention:
    def __init__(self, pool, *, max_batch, threads=512, layout=True, cooperative=True, scalar=False):
        self.pool = pool
        self.splits = 32
        self.extension = prepare(pool.hp_k.device, threads=threads, layout=layout,
                                 cooperative=cooperative, scalar=scalar)
        self.sm_count = torch.cuda.get_device_properties(pool.hp_k.device).multi_processor_count
        self.partial = torch.empty((max_batch, pool.head_num, 32, 128), device=pool.hp_k.device)
        self.lse = torch.empty((max_batch, pool.head_num, 32), device=pool.hp_k.device)

    def forward(self, q, layer_id, ptr, loc, rope, scale):
        p, b = self.pool, q.shape[0]
        if q.dtype != torch.bfloat16 or q.shape[1:] != (p.head_num, 128) or b > self.partial.shape[0]:
            raise ValueError('Page VQ+ requires BF16 [B,H,128] within capacity')
        out = torch.empty_like(q)
        self.extension.residual_cuda(q, p.get_key_buffer(layer_id), p.get_value_buffer(layer_id),
            p.k_book.cb16[layer_id], p.v_book.cb16[layer_id],
            p.k_residual[layer_id], p.v_residual[layer_id],
            p.k_residual_scale[layer_id], p.v_residual_scale[layer_id],
            p.get_hp_key_buffer(layer_id), p.get_hp_value_buffer(layer_id),
            ptr, loc, rope, self.partial, self.lse, p.hp_global_offset, float(scale),
            max(1, (2 * self.sm_count + b * p.head_num - 1) // (b * p.head_num)))
        return official_reduce(self.partial[:b], self.lse[:b], out)
