"""Opt-in CUDA checks against independent inverse/forward RoPE references."""

import os

import pytest
import torch
from test_ir import ROOT, load_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1",
    reason="set RUN_CONTEXT_GPU=1 on the isolated test GPU",
)


@pytest.fixture(scope="module")
def kernels():
    return load_file(
        "context_rope_kernel",
        ROOT / "python/sglang/kernels/ops/attention/context_reposition.py",
    ).reposition_kv_layers


def inverse_forward(k, positions, rope, neox_style, dtype=torch.float32):
    """Independent PyTorch inverse/forward; no production kernel or delta angle."""
    part = k.to(dtype)
    a, b = part.chunk(2, -1) if neox_style else (part[..., ::2], part[..., 1::2])
    oc, os_ = rope[positions[:, 0].long()].to(dtype).chunk(2, -1)
    nc, ns = rope[positions[:, 1].long()].to(dtype).chunk(2, -1)
    oc, os_, nc, ns = (x[None, :, None] for x in (oc, os_, nc, ns))
    scale = oc.square() + os_.square()
    x, y = (a * oc + b * os_) / scale, (b * oc - a * os_) / scale
    first, second = x * nc - y * ns, y * nc + x * ns
    rotated = torch.cat((first, second), -1) if neox_style else torch.stack((first, second), -1).flatten(-2)
    result = rotated.to(k.dtype)
    same = positions[:, 0] == positions[:, 1]
    result[:, same] = k[:, same]
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("neox_style", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_native_layer_pointers_match_independent_inverse_forward(kernels, dtype, neox_style, head_dim):
    actual = kernels
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

    expected_k = base_k.clone()
    expected_v = base_v.clone()
    expected_k[:, destination.long()] = inverse_forward(
        base_k[:, source.long()], positions, rope, neox_style
    )
    expected_v[:, destination.long()] = base_v[:, source.long()]
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
    assert torch.equal(output_k, expected_k)
    assert torch.equal(output_v, expected_v)
    assert torch.equal(output_k[:, source.long()], base_k[:, source.long()])
    assert torch.equal(output_v[:, source.long()], base_v[:, source.long()])
    same = positions[:, 0] == positions[:, 1]
    assert torch.equal(
        output_k[:, destination[same].long()], base_k[:, source[same].long()]
    )


def test_reposition_rejects_large_page_before_device_access(kernels):
    actual = kernels
    with pytest.raises(ValueError, match="page_size=1"):
        actual(*([None] * 8), page_size=16)


def test_swa_missing_copy_sources_and_destinations_survive_graph_replay(kernels):
    actual = kernels
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("neox_style", [True, False])
@pytest.mark.parametrize("heads", [1, 2, 8])
def test_partial_rope_preserves_tail_values_sources_and_graph(dtype, neox_style, heads):
    """Independent FP32 inverse/forward reference, including scaled native cache."""
    actual = load_file("partial_context_rope", ROOT / "python/sglang/kernels/ops/attention/context_reposition.py").reposition_kv_layers
    torch.manual_seed(29)
    device = "cuda"
    k = torch.randn((3, 32, heads, 128), device=device, dtype=dtype)
    v = torch.randn_like(k)
    source = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    destination = torch.tensor([17, 19, 21, 23], device=device, dtype=torch.int32)
    positions = torch.tensor([[3, 3], [1023, 1], [100, 0], [200, 22]], device=device, dtype=torch.int32)
    theta = torch.arange(1024, device=device, dtype=torch.float32)[:, None] / (5000000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
    rope = (torch.cat((theta.cos(), theta.sin()), -1) * 1.37).to(dtype)
    before_k, before_v = k.clone(), v.clone()
    pointers_k = torch.tensor([x.data_ptr() for x in k], device=device, dtype=torch.uint64)
    pointers_v = torch.tensor([x.data_ptr() for x in v], device=device, dtype=torch.uint64)
    x = k[:, source.long(), :, :64].float()
    a, b = (x[..., :32], x[..., 32:]) if neox_style else (x[..., ::2], x[..., 1::2])
    old_cos, old_sin = rope[positions[:, 0].long()].float().chunk(2, -1)
    new_cos, new_sin = rope[positions[:, 1].long()].float().chunk(2, -1)
    oc, os_ = old_cos[None, :, None], old_sin[None, :, None]
    nc, ns = new_cos[None, :, None], new_sin[None, :, None]
    unrotated_a, unrotated_b = (a * oc + b * os_) / (oc.square() + os_.square()), (b * oc - a * os_) / (oc.square() + os_.square())
    expected_a, expected_b = unrotated_a * nc - unrotated_b * ns, unrotated_b * nc + unrotated_a * ns
    expected = torch.empty_like(x)
    if neox_style:
        expected[..., :32], expected[..., 32:] = expected_a, expected_b
    else:
        expected[..., ::2], expected[..., 1::2] = expected_a, expected_b
    def run():
        actual(pointers_k, pointers_v, k[0], v[0], source, destination, positions, rope,
               rotary_dim=64, is_neox_style=neox_style)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    k[:, destination.long()] = 0
    v[:, destination.long()] = 0
    graph.replay()
    torch.cuda.synchronize()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    rounded = expected.to(dtype)
    rounded[:, 0] = before_k[:, source[0].long(), :, :64]
    assert torch.equal(k[:, destination.long(), :, :64], rounded)
    precise = inverse_forward(before_k[:, source.long(), :, :64], positions, rope, neox_style, torch.float64)
    torch.testing.assert_close(k[:, destination.long(), :, :64], precise, atol=tolerance, rtol=tolerance)
    assert torch.equal(k[:, destination.long(), :, 64:], before_k[:, source.long(), :, 64:])
    assert torch.equal(v[:, destination.long()], before_v[:, source.long()])
    assert torch.equal(k[:, source.long()], before_k[:, source.long()])
    assert torch.equal(k[:, destination[0].long()], before_k[:, source[0].long()])
    assert torch.equal(v[:, source.long()], before_v[:, source.long()])
    for invalid in (0, 63, 130):
        with pytest.raises(ValueError, match="rotary_dim"):
            actual(pointers_k, pointers_v, k[0], v[0], source, destination, positions, rope, rotary_dim=invalid)
