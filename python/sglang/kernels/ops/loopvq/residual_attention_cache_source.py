# SPDX-License-Identifier: Apache-2.0
"""Explicit L2-only streaming loads while retaining cached centroid gathers."""


def cache_source(source, mode, replace):
    source=replace(source,'float4 co = __ldg(reinterpret_cast<const float4*>(rope));',
                   'float4 co = __ldcg(reinterpret_cast<const float4*>(rope));')
    source=replace(source,'float4 si = __ldg(reinterpret_cast<const float4*>(rope + 64));',
                   'float4 si = __ldcg(reinterpret_cast<const float4*>(rope + 64));')
    if mode==2:
        source=replace(source,'int code = __ldg(idx + io);','int code = __ldcg(idx + io);')
        source=replace(source,'unsigned z = *reinterpret_cast<const uint16_t*>(residual + ro);',
                       'unsigned z = __ldcg(reinterpret_cast<const uint16_t*>(residual + ro));')
        assert source.count('__half2float(__ldg(scales + so))')==2
        source=source.replace('__half2float(__ldg(scales + so))',
                              '__half2float(__ushort_as_half(__ldcg(reinterpret_cast<const unsigned short*>(scales + so))))')
    return source
