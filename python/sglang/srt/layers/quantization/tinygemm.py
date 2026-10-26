"""Explicit FlashInfer TinyGEMM backend; no process-wide method replacement."""

import logging

import torch

_kernel = None


def initialize():
    global _kernel
    from flashinfer.gemm import tinygemm_bf16

    major, minor = torch.cuda.get_device_capability()
    if not tinygemm_bf16.is_compute_capability_supported(major * 10 + minor):
        raise ValueError(
            f"--bf16-gemm-backend tinygemm does not support SM{major}{minor}"
        )
    x = torch.zeros((2, 64), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((16, 64), dtype=x.dtype, device=x.device)
    output = torch.empty((2, 16), dtype=x.dtype, device=x.device)
    tinygemm_bf16(x, weight, output)
    torch.cuda.synchronize()
    _kernel = tinygemm_bf16
    logging.getLogger(__name__).info(
        "FlashInfer TinyGEMM enabled on SM%d%d", major, minor
    )


def eligible(x, weight, bias=None):
    if x.ndim == 0 or x.shape[-1] == 0 or weight.ndim != 2:
        return False
    n, k = weight.shape
    rows = x.numel() // x.shape[-1]
    return (
        x.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == k
        and 1 <= rows <= 16
        and n > 0
        and n % 16 == 0
        and k % 64 == 0
        and x.is_contiguous()
        and weight.is_contiguous()
        and (
            bias is None
            or (
                bias.shape == (n,)
                and bias.device == x.device
                and bias.dtype == x.dtype
                and bias.is_contiguous()
            )
        )
    )


def linear(x, weight, bias=None):
    if _kernel is None:
        raise RuntimeError("TinyGEMM must be initialized before model execution")
    output = torch.empty(
        (x.numel() // x.shape[-1], weight.shape[0]), dtype=x.dtype, device=x.device
    )
    _kernel(x.view(-1, x.shape[-1]), weight, output, bias)
    return output.view(*x.shape[:-1], weight.shape[0])
