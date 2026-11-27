"""R14+ source-level ablations against the retained R13 payload reader."""
from sglang.kernels.ops.loopvq.residual_attention_payload_cuda import _SOURCE


def replace_once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def source_for(mode):
    source = _SOURCE
    if mode.startswith('grid'):
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
    else:
        raise ValueError(mode)
    return source


def prepare(mode):
    from torch.utils.cpp_extension import load_inline
    extension = load_inline(
        name=f'vqrq_cont_r14_{mode}',
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
