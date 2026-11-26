# SPDX-License-Identifier: Apache-2.0
"""SM80 double-buffered compressed payload staging for the compact reader."""

_STAGE = r'''
// Each tile buffer: locations, indices, residual bytes, FP16 group32 scales.
constexpr int PAYLOAD_BYTES = CAP * (4 + 32 + 64 + 8);
template<typename IT>
__device__ __forceinline__ void stage_payload(char* buffer,
    const uint8_t* idx, const uint8_t* residual, const __half* scales,
    const IT* locs, int begin, int start, int count, int heads, int head, int offset) {
    auto* slots = reinterpret_cast<int*>(buffer);
    auto* ti = reinterpret_cast<uint8_t*>(buffer + CAP * 4);
    auto* tr = ti + CAP * 32;
    auto* ts = reinterpret_cast<__half*>(tr + CAP * 64);
    for (int item = threadIdx.x; item < count * 4; item += THREADS) {
        const int n = item / 4, group = item % 4;
        const int loc = (int)locs[begin + start + n];
        if (!group) slots[n] = loc;
        if (loc < offset) {
            const long pg = (((long)(loc >> 3) * heads + head) * 4 + group) * 8 + (loc & 7);
            unsigned si = static_cast<unsigned>(__cvta_generic_to_shared(ti + n * 32 + group * 8));
            unsigned sr = static_cast<unsigned>(__cvta_generic_to_shared(tr + n * 64 + group * 16));
#if !VQPAGE_SCALAR
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(si), "l"(idx + pg * 8) : "memory");
#endif
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sr), "l"(residual + pg * 16) : "memory");
            ts[n * 4 + group] = __ldg(scales + pg);
        }
    }
}
'''


def staged_source(source, replace):
    source=replace(source,'constexpr int CAP = 1024,','constexpr int CAP = 128,')
    source=replace(source,'float* x, bool qk=false)', 'float* x, bool qk=false, int token=0)')
    before='''        const long row = VQPAGE_LAYOUT == 2 ? ((long)(loc >> 3) * heads + head) * 8 + (loc & 7) : (long)loc * heads + head;
        const long pg = (((long)(loc >> 3) * heads + head) * 4 + group / 8) * 8 + (loc & 7);
        const long io = VQPAGE_LAYOUT == 1 ? pg * 8 + group % 8 : row * 32 + group;
        const long ro = VQPAGE_LAYOUT == 1 ? pg * 16 + (group % 8) * 2 : row * 64 + group * 2;
        const long so = VQPAGE_LAYOUT == 1 ? pg : row * 4 + group / 8;'''
    source=replace(source,before,'''        const long io = token * 32 + group;
        const long ro = token * 64 + group * 2;
        const long so = token * 4 + group / 8;''')
    source=replace(source,'int code = __ldg(idx + io);','int code = idx[io];')
    # Both conditional branches must read shared memory, not use __ldg.
    assert source.count('__half2float(__ldg(scales + so))')==2
    source=source.replace('__half2float(__ldg(scales + so))','__half2float(scales[so])')
    source=replace(source,'template<typename IT>\n__global__',_STAGE+'\ntemplate<typename IT>\n__global__')
    source=replace(source,'''    for (int tile = 0; tile < span; tile += CAP) {
    const int start = range_start + tile, count = min(CAP, span - tile);
    table = const_cast<__nv_bfloat16*>(KC + h * TABLE);''','''    char* staging = raw + ((SMEM + 15) / 16) * 16;
    stage_payload(staging, KI, KR, KS, LOC, begin, range_start, min(CAP, span), H, h, offset);
    stage_payload(staging + PAYLOAD_BYTES, VI, VR, VS, LOC, begin, range_start, min(CAP, span), H, h, offset);
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (int tile = 0; tile < span; tile += CAP) {
    const int start = range_start + tile, count = min(CAP, span - tile);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int current = (tile / CAP) & 1;
    char* kb = staging + current * 2 * PAYLOAD_BYTES;
    char* vb = kb + PAYLOAD_BYTES;
    const int* tile_locs = reinterpret_cast<const int*>(kb);
    const auto* tki = reinterpret_cast<const uint8_t*>(kb + CAP * 4);
    const auto* tvi = reinterpret_cast<const uint8_t*>(vb + CAP * 4);
    const auto* tkr = tki + CAP * 32;
    const auto* tvr = tvi + CAP * 32;
    const auto* tks = reinterpret_cast<const __half*>(tkr + CAP * 64);
    const auto* tvs = reinterpret_cast<const __half*>(tvr + CAP * 64);
    if (tile + CAP < span) {
        char* next = staging + (1 - current) * 2 * PAYLOAD_BYTES;
        stage_payload(next, KI, KR, KS, LOC, begin, start + CAP, min(CAP, span - tile - CAP), H, h, offset);
        stage_payload(next + PAYLOAD_BYTES, VI, VR, VS, LOC, begin, start + CAP, min(CAP, span - tile - CAP), H, h, offset);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    table = const_cast<__nv_bfloat16*>(KC + h * TABLE);''')
    source=replace(source,'row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g, low, true);',
                   'row4(tki, table, tkr, tks, HK, tile_locs[n], offset, H, h, g, low, true, n);')
    source=replace(source,'row4(KI, table, KR, KS, HK, (int)LOC[begin + start + n], offset, H, h, g + 16, high, true);',
                   'row4(tki, table, tkr, tks, HK, tile_locs[n], offset, H, h, g + 16, high, true, n);')
    source=replace(source,'row4(VI, table, VR, VS, HV, (int)LOC[begin + start + n], offset, H, h, vgroup * 4 + group, v);',
                   'row4(tvi, table, tvr, tvs, HV, tile_locs[n], offset, H, h, vgroup * 4 + group, v, false, n);')
    assert source.count('cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)')==2
    source=source.replace('cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)',
                         'cudaFuncAttributeMaxDynamicSharedMemorySize, ((SMEM + 15) / 16) * 16 + 4 * PAYLOAD_BYTES)')
    source=replace(source,'<<<grid, THREADS, SMEM,',
                   '<<<grid, THREADS, ((SMEM + 15) / 16) * 16 + 4 * PAYLOAD_BYTES,')
    return source
