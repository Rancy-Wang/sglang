"""CUDA numerical and graph checks for the explicitly selected linear backend."""

import json

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization import tinygemm, unquant


@torch.inference_mode()
def main():
    torch.manual_seed(11907)
    tinygemm.initialize()
    unquant._BF16_GEMM_BACKEND = unquant.Bf16GemmBackend.TINYGEMM
    results = []
    # Includes Ouro projection dimensions, an LM-head-like shape, and fallback.
    for m, n, k, biased in [
        (1, 2048, 2048, False),
        (7, 5632, 2048, True),
        (4, 49152, 2048, False),
        (17, 128, 64, True),
    ]:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k**0.5
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if biased else None
        assert tinygemm.eligible(x, weight, bias) == (m <= 16)
        assert not tinygemm.eligible(x.float(), weight, bias)
        assert not tinygemm.eligible(x[:, ::2], weight[:, ::2], bias)
        actual = unquant._bf16_gemm_dispatch_impl(x, weight, bias)
        oracle = F.linear(
            x.float(), weight.float(), None if bias is None else bias.float()
        )
        # Fixed BF16 tolerance before running, alongside error measurements.
        torch.testing.assert_close(actual.float(), oracle, rtol=0.02, atol=0.04)
        error = actual.float() - oracle
        if m > 16:
            torch.testing.assert_close(
                actual, F.linear(x, weight, bias), rtol=0, atol=0
            )
        # Replay must read changed input, rather than captured/stale activations.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                unquant._bf16_gemm_dispatch_impl(x, weight, bias)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = unquant._bf16_gemm_dispatch_impl(x, weight, bias)
        x.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            captured, unquant._bf16_gemm_dispatch_impl(x, weight, bias), rtol=0, atol=0
        )
        results.append(
            {
                "shape": [m, n, k],
                "bias": biased,
                "tinygemm": m <= 16,
                "max_abs_error": error.abs().max().item(),
                "rmse": error.square().mean().sqrt().item(),
                "graph": "passed",
            }
        )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
