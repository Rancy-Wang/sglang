"""R11: code-major shared K table and PV-specific group permutation."""


def shared_source(source, replace_once):
    for old, new in (
        ("__ldg(reinterpret_cast<const uint2*>(table + (group * 256 + code0) * 4))",
         "*reinterpret_cast<const uint2*>(table + (code0 * 32 + group) * 4)"),
        ("__ldg(reinterpret_cast<const uint2*>(table + ((group + 16) * 256 + code1) * 4))",
         "*reinterpret_cast<const uint2*>(table + (code1 * 32 + group + 16) * 4)"),
        ("__ldg(reinterpret_cast<const uint2*>(table + ((group16 * 4 + group) * 256 + code) * 4))",
         "*reinterpret_cast<const uint2*>(table + (code * 32 + group * 8 + group16) * 4)"),
        ("__ldg(reinterpret_cast<const uint2*>(table + (group * 256 + code) * 4))",
         "*reinterpret_cast<const uint2*>(table + (code * 32 + group) * 4)"),
    ):
        source = replace_once(source, old, new)
    start = source.index("__device__ __forceinline__ void copy_table(")
    end = source.index("__device__ __forceinline__ void wait_table()", start)
    source = source[:start] + r'''
__device__ __forceinline__ void copy_table(__nv_bfloat16* dst,
    const __nv_bfloat16* src, bool value_layout) {
    const int lane = threadIdx.x & 31;
    for (int tile = threadIdx.x >> 5; tile < 256; tile += WARPS) {
        const int group = (tile & 3) * 8 + (lane & 7);
        const int code = (tile >> 2) * 4 + (lane >> 3);
        const int stored_group = value_layout ? (group & 3) * 8 + (group >> 2) : group;
        unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst + (code * 32 + stored_group) * 4));
        const auto* global = src + (group * 256 + code) * 4;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(smem), "l"(global) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
''' + source[end:]
    source = replace_once(source, "copy_table(table, KC + h * TABLE);",
                          "copy_table(table, KC + h * TABLE, false);")
    assert source.count("copy_table(table, VC + h * TABLE);") == 2
    return source.replace("copy_table(table, VC + h * TABLE);",
                          "copy_table(table, VC + h * TABLE, true);")
