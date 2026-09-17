"""Check native SM80 tuning, state restoration, and packed MoE numerics."""

import json
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires the native CUDA MoE runtime"
)


def test_constraints_restore_on_exception_and_preserve_large_prefill():
    from sglang.srt.layers.moe.fused_moe_triton import triton_kernels_moe as moe
    from sglang.srt.runtime_context import override_platform

    flags = moe._opt_flags._opt_flags_constraints
    saved = flags.copy()
    x = SimpleNamespace(shape=(1, 2880))
    w = SimpleNamespace(dtype=moe.FP4)
    md = SimpleNamespace(expected_slice_size=None, n_slices=128)
    try:
        flags.pop("num_warps", None)
        with override_platform(device_sm=80):
            with (
                pytest.raises(RuntimeError, match="kernel failure"),
                moe._small_ampere_mxfp4_warps(x, w, md, 4),
            ):
                assert flags["num_warps"] == 4
                raise RuntimeError("kernel failure")
            assert "num_warps" not in flags
            with moe._small_ampere_mxfp4_warps(
                SimpleNamespace(shape=(8192, 2880)), w, md, 4
            ):
                assert "num_warps" not in flags
            flags["num_warps"] = 8
            with moe._small_ampere_mxfp4_warps(x, w, md, 4):
                assert flags["num_warps"] == 8
            assert flags["num_warps"] == 8
    finally:
        flags.clear()
        flags.update(saved)


def test_native_mxfp4_outputs_and_cuda_graph_timing():
    import triton
    from sglang.srt.layers.moe.fused_moe_triton import triton_kernels_moe as moe
    from sglang.srt.layers.moe.topk import routing
    from triton_kernels.matmul import PrecisionConfig
    from triton_kernels.tensor import FP4, wrap_torch_tensor

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("A800 configuration regression")
    torch.manual_seed(42)
    experts, hidden, intermediate = 128, 2880, 1440
    weights = []
    for k, n in [(hidden, intermediate * 2), (intermediate, hidden)]:
        w = torch.randint(
            0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda"
        ).transpose(-2, -1)
        scale = torch.full(
            (experts, n, k // 32), 121, dtype=torch.uint8, device="cuda"
        ).transpose(-2, -1)
        weights.append(
            (
                wrap_torch_tensor(w, dtype=FP4),
                PrecisionConfig(b_mx_scale=wrap_torch_tensor(scale)),
                torch.zeros((experts, n), device="cuda", dtype=torch.float32),
            )
        )
    flags = moe._opt_flags._opt_flags_constraints
    saved = flags.copy()
    try:
        for rows in (1, 32):
            x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
            md, gather, scatter, gates, active = routing(
                torch.randn((rows, experts), device="cuda"), 4
            )

            def run(x=x, md=md, gather=gather, scatter=scatter, gates=gates, active=active):
                return moe.triton_kernel_fused_experts_with_bias(
                    x,
                    *weights[0],
                    *weights[1],
                    md,
                    gather,
                    scatter,
                    gates,
                    active,
                    gemm1_alpha=1.702,
                    gemm1_clamp_limit=7.0,
                )

            flags["num_warps"] = 1  # exact upstream small-tile configuration
            reference = run().clone()
            original_ms = triton.testing.do_bench_cudagraph(run, rep=100)
            flags.pop("num_warps")
            actual = run()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            tuned_ms = triton.testing.do_bench_cudagraph(run, rep=100)
            assert "num_warps" not in flags
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "original_ms": original_ms,
                        "tuned_ms": tuned_ms,
                        "bit_exact": True,
                    }
                ),
                flush=True,
            )
    finally:
        flags.clear()
        flags.update(saved)
