# SPDX-License-Identifier: Apache-2.0
"""Resource-derived geometry and device-length scheduling for packed VQ+.

The payload arithmetic is the retained FP32 reconstruction/RoPE/PV contract.
Only CTA ownership and the FP32 reduction grouping vary with geometry.
The surrounding serving backend still has its own architecture restrictions.
"""
import torch

from .residual_attention_payload_cuda import _POLICY, _SOURCE, _replace_once


def split_policy(batch, heads, capacity, chunks):
    """First sufficiently packed resident wave, bounded by scratch capacity."""
    if min(batch, heads, capacity) < 1 or not 0 <= chunks <= 32:
        raise ValueError("Invalid payload scheduling geometry")
    tasks = batch * heads
    waves = max(1, (max(1, chunks) * tasks + capacity - 1) // capacity)
    while True:
        splits = min(32, max(1, chunks, waves * capacity // tasks))
        if splits == 32 or splits * tasks * 10 >= waves * capacity * 9:
            return splits
        waves += 1


def geometry_for(batch, heads, capacities):
    eligible = [t for t, c in capacities.items() if 0 < c <= batch * heads * 32]
    if not eligible:
        return None
    # Fewer partials usually offset a wider CTA's larger reduction. Prefer the
    # narrower CTA when both need the same number of partials to fill waves.
    return min(eligible, key=lambda t: (split_policy(batch, heads, capacities[t], 1), t))


def source_for(min_blocks):
    source = _replace_once(_SOURCE, _POLICY, "")
    source = _replace_once(source, "__launch_bounds__(THREADS, 3)",
                           f"__launch_bounds__(THREADS, {min_blocks})")
    source = _replace_once(source, "float scale, int active_target) {",
                           "float scale, int active_target, const uint8_t* POLICY) {")
    source = _replace_once(source,
        "PAYLOAD_SPLITS[gridDim.x - 4][min(SPLITS, (length + CAP - 1) / CAP)]",
        "__ldg(POLICY + gridDim.x * 33 + min(SPLITS, (length + CAP - 1) / CAP))")
    source = _replace_once(source, "int64_t offset, double scale, int64_t target) {",
                           "int64_t offset, double scale, int64_t target, torch::Tensor policy) {")
    source = _replace_once(source, "q.stride(1), scale, target)",
                           "q.stride(1), scale, target, policy.data_ptr<uint8_t>())")
    helper = r'''
__device__ __forceinline__ float signed_nibble_float(unsigned word, int shift) {
    const unsigned nibble = ((word >> shift) & 15u) ^ 8u;
    return __fsub_rn(__uint_as_float(0x4b000000u | nibble), 8388616.0f);
}
'''
    source = _replace_once(source, '__device__ __forceinline__ void pair8(',
                           helper + '\n__device__ __forceinline__ void pair8(')
    for name, word in (('r0', 'z0'), ('r1', 'z1'), ('r', 'z')):
        source = _replace_once(source,
            f'const int {name} = (int)({word} << (28 - j * 4)) >> 28;',
            f'const float {name} = signed_nibble_float({word}, j * 4);')
    return source.replace("payload_stage1", "adaptive_stage1")


_EXTENSIONS = {}
_CAPACITIES = {}
_SCHEDULES = {}


class AdaptivePayload:
    def __init__(self, extensions, capacities, policies, heads, max_batch):
        self.extensions, self.capacities, self.policies = extensions, capacities, policies
        self.geometries = [None] + [geometry_for(b, heads, capacities) for b in range(1, max_batch + 1)]

    def supports(self, batch):
        return 0 < batch < len(self.geometries) and self.geometries[batch] is not None

    def capacity_for(self, batch):
        return self.capacities[self.geometries[batch]]

    def residual_cuda(self, *args):
        threads = self.geometries[args[0].shape[0]]
        self.extensions[threads].residual_cuda(*args, self.policies[threads])


def prepare(device, max_batch, heads):
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare adaptive VQ+ before graph capture")
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8:
            raise ValueError("Packed BF16 VQ+ requires NVIDIA SM80 or newer")
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        arch = major * 10 + minor
        extensions, capacities, policies = {}, {}, {}
        for threads in (128, 256):
            blocks = max(1, (props.max_threads_per_multi_processor * 3 // 4) // threads)
            key = (arch, threads, blocks)
            if key not in _EXTENSIONS:
                from torch.utils.cpp_extension import load_inline
                _EXTENSIONS[key] = load_inline(
                    name=f"loopvq_adaptive_r2_sm{arch}_t{threads}_b{blocks}",
                    cpp_sources="#include <torch/extension.h>\nvoid residual_init();\n"
                        "int64_t residual_capacity();\nvoid residual_cuda(" +
                        ",".join(["torch::Tensor"] * 16) +
                        ",int64_t,double,int64_t,torch::Tensor);",
                    cuda_sources=source_for(blocks),
                    functions=["residual_init", "residual_capacity", "residual_cuda"],
                    extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false", "-Xptxas=-v",
                        f"-DVQPLUS_THREADS={threads}", "-DVQPLUS_UNROLL=1", "-DVQPLUS_GLOBAL=1",
                        "-DVQPLUS_ASYNC=0", "-DVQPAGE_LAYOUT=2", "-DVQPAGE_COOP=0",
                        "-DVQPAGE_SCALAR=0", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                )
            extension = _EXTENSIONS[key]
            device_key = (index, key)
            if device_key not in _CAPACITIES:
                extension.residual_init()
                _CAPACITIES[device_key] = extension.residual_capacity()
            capacity = _CAPACITIES[device_key]
            if capacity <= 0:
                continue
            schedule_key = (device_key, heads, max_batch, capacity)
            if schedule_key not in _SCHEDULES:
                rows = [[1] * 33] + [[split_policy(b, heads, capacity, n) for n in range(33)]
                                     for b in range(1, max_batch + 1)]
                _SCHEDULES[schedule_key] = torch.tensor(rows, dtype=torch.uint8, device=device)
            extensions[threads], capacities[threads] = extension, capacity
            policies[threads] = _SCHEDULES[schedule_key]
        return AdaptivePayload(extensions, capacities, policies, heads, max_batch)
