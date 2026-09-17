"""Opt-in CUDA differential tests against the fixed mini production kernel."""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from test_ir import ROOT, load_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1",
    reason="set RUN_CONTEXT_GPU=1 on the isolated test GPU",
)


@pytest.fixture(scope="module")
def kernels():
    source = os.environ.get("MINI_SGLANG_REFERENCE")
    if not source:
        pytest.fail("MINI_SGLANG_REFERENCE is required for GPU differential checks")
    package = types.ModuleType("mini_rope_reference")
    package.__path__ = [str(Path(source) / "python/minisgl/kernel")]
    sys.modules[package.__name__] = package
    mini = importlib.import_module(package.__name__ + ".reposition_kv")
    actual = load_file(
        "context_rope_kernel",
        ROOT / "python/sglang/kernels/ops/attention/context_reposition.py",
    )
    return mini.reposition_kv_with_rope_delta, actual.reposition_kv_layers


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("neox_style", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_native_layer_pointers_match_mini(kernels, dtype, neox_style, head_dim):
    reference, actual = kernels
    device = "cuda"
    torch.manual_seed(17)
    layers, slots, heads = 3, 256, 2
    base_k = torch.randn((layers, slots, heads, head_dim), device=device, dtype=dtype)
    base_v = torch.randn_like(base_k)

    k_buffers = [layer.clone() for layer in base_k]
    v_buffers = [layer.clone() for layer in base_v]
    k_ptrs = torch.tensor(
        [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
    )
    v_ptrs = torch.tensor(
        [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
    )
    source = torch.tensor(
        [1, 3, 15, 16, 31, 63, 64, 95], dtype=torch.int32, device=device
    )
    destination = torch.tensor(
        [129, 130, 143, 144, 159, 191, 192, 223], dtype=torch.int32, device=device
    )
    positions = torch.tensor(
        [
            [9, 9],
            [100, 3],
            [512, 4],
            [128, 1],
            [1023, 32],
            [9, 0],
            [2047, 2047],
            [2047, 8],
        ],
        dtype=torch.int32,
        device=device,
    )
    theta = torch.arange(2048, device=device)[:, None] * torch.linspace(
        0.01, 1, head_dim // 2, device=device
    )
    rope = (
        torch.cat((theta.cos(), theta.sin()), dim=-1) * 1.37
    )  # YaRN magnitude must cancel.

    def to_neox(tensor):
        if neox_style:
            return tensor.clone()
        return torch.cat((tensor[..., ::2], tensor[..., 1::2]), dim=-1)

    expected_k = to_neox(base_k)
    expected_v = base_v.clone()
    reference(expected_k, expected_v, source, destination, positions, rope)
    actual(
        k_ptrs,
        v_ptrs,
        k_buffers[0],
        v_buffers[0],
        source,
        destination,
        positions,
        rope,
        is_neox_style=neox_style,
    )

    output_k = torch.stack(k_buffers)
    output_v = torch.stack(v_buffers)
    assert torch.equal(to_neox(output_k), expected_k)
    assert torch.equal(output_v, expected_v)
    assert torch.equal(output_k[:, source.long()], base_k[:, source.long()])
    assert torch.equal(output_v[:, source.long()], base_v[:, source.long()])
    same = positions[:, 0] == positions[:, 1]
    assert torch.equal(
        output_k[:, destination[same].long()], base_k[:, source[same].long()]
    )


def test_reposition_rejects_large_page_before_device_access(kernels):
    _, actual = kernels
    with pytest.raises(ValueError, match="page_size=1"):
        actual(*([None] * 8), page_size=16)


def test_swa_missing_copy_sources_and_destinations_survive_graph_replay(kernels):
    _, actual = kernels
    k = torch.randn((32, 2, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    original_k, original_v = k.clone(), v.clone()
    kptr = torch.tensor([k.data_ptr()], device="cuda", dtype=torch.uint64)
    vptr = torch.tensor([v.data_ptr()], device="cuda", dtype=torch.uint64)
    src = torch.tensor([0, 2, 3, 4], device="cuda", dtype=torch.int32)
    dst = torch.tensor([17, 0, 19, 20], device="cuda", dtype=torch.int32)
    pairs = torch.tensor([[1, 1]] * 4, device="cuda", dtype=torch.int32)
    rope = torch.cat((torch.ones(4, 32), torch.zeros(4, 32)), dim=1).cuda()

    def run():
        actual(kptr, vptr, k, v, src, dst, pairs, rope, skip_unmapped=True)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        k.copy_(original_k)
        v.copy_(original_v)
        graph.replay()
        torch.cuda.synchronize()
        expected_k, expected_v = original_k.clone(), original_v.clone()
        expected_k[19:21] = original_k[3:5]
        expected_v[19:21] = original_v[3:5]
        assert torch.equal(k, expected_k)
        assert torch.equal(v, expected_v)
