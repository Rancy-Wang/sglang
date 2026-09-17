"""Native Triton kernel checks, without importing the serving runtime.

The frozen upstream kernel is the numerical oracle here. Dense masks are test
fixtures only; they are not a production implementation or a model oracle.
"""

import os
import subprocess
import sys
import types

import pytest
import torch
from test_ir import ROOT, load_file

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1",
    reason="set RUN_CONTEXT_GPU=1 on the test GPU",
)


@pytest.fixture(scope="module")
def native_attention(tmp_path_factory):
    # Load the real kernel dependencies. Only unrelated SRT bootstrap and
    # platform detection are isolated, so this also runs in the mini oracle env.
    with pytest.MonkeyPatch.context() as patch:
        for name, path in (
            ("sglang", "python/sglang"),
            ("sglang.srt", "python/sglang/srt"),
            ("sglang.kernels", "python/sglang/kernels"),
            ("sglang.kernels.ops", "python/sglang/kernels/ops"),
            ("sglang.kernels.ops.attention", "python/sglang/kernels/ops/attention"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / path)]
            patch.setitem(sys.modules, name, package)
        utils = types.ModuleType("sglang.srt.utils")
        utils.is_cuda = lambda: torch.version.cuda is not None
        utils.is_hip = lambda: False
        utils.is_gfx95_supported = lambda: False
        utils.is_gfx1250_supported = lambda: False
        utils.get_device_core_count = lambda device=0: (
            torch.cuda.get_device_properties(device).multi_processor_count
        )
        patch.setitem(sys.modules, utils.__name__, utils)
        # Restore every real dependency too; don't change later unit fixtures.
        paths = [
            ("sglang.srt.environ", ROOT / "python/sglang/srt/environ.py"),
            *[
                (
                    f"sglang.kernels.ops.attention.{name}",
                    ROOT / f"python/sglang/kernels/ops/attention/{name}.py",
                )
                for name in ("score_mod", "decode_attention", "prefill_attention")
            ],
        ]
        for name, path in paths:
            patch.setitem(sys.modules, name, None)
            load_file(name, path)
        current = load_file(
            "context_native_extend",
            ROOT / "python/sglang/kernels/ops/attention/extend_attention.py",
        )
        baseline_path = (
            tmp_path_factory.mktemp("native-baseline") / "extend_attention.py"
        )
        baseline_path.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    (
                        "9d0a8d75364ea4571e05ba2e37227ec2579324f2:"
                        "python/sglang/kernels/ops/attention/extend_attention.py"
                    ),
                ],
                cwd=ROOT,
            )
        )
        baseline = load_file("context_frozen_extend", baseline_path)
        yield current.extend_attention_fwd, baseline.extend_attention_fwd


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
@pytest.mark.parametrize("window", [-1, 16])
@pytest.mark.parametrize("has_sink", [False, True])
def test_native_position_windows(native_attention, dtype, page_size, window, has_sink):
    actual, reference = native_attention
    torch.manual_seed(104)
    device = "cuda"
    q_lens, prefix_lens = [131, 17, 1], [69, 31, 0]
    nq, nk, heads, kv_heads, dim = sum(q_lens), sum(prefix_lens), 4, 2, 64
    q = torch.randn(nq, heads, dim, device=device, dtype=dtype)
    k = torch.randn(nq, kv_heads, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    pool_k = torch.randn(256, kv_heads, dim, device=device, dtype=dtype)
    pool_v = torch.randn_like(pool_k)

    def native_view(pool):
        if page_size == 1:
            return pool
        return (
            pool.reshape(-1, page_size, kv_heads, dim).permute(0, 2, 1, 3).contiguous()
        )

    pool_k, pool_v = native_view(pool_k), native_view(pool_v)
    indices = torch.randperm(256, device=device)[:nk]
    qo = torch.tensor([0, 131, 148, 149], device=device, dtype=torch.int32)
    ki = torch.tensor([0, 69, 100, 100], device=device, dtype=torch.int32)
    q_positions, k_positions, masks, mask_offsets = [], [], [], [0]
    for n_query, n_prefix in zip(q_lens, prefix_lens):
        positions = torch.randint(1, 6, (n_prefix + n_query,)).cumsum(0)
        p_pos, q_pos = positions[:n_prefix], positions[n_prefix:]
        q_positions.append(q_pos)
        k_positions.append(p_pos)
        # Causality uses true order. Position gaps narrow only the SWA window.
        visible = positions[None, :] <= q_pos[:, None]
        if window > 0:
            visible &= q_pos[:, None] <= positions[None, :] + window
        masks.append(visible.flatten())
        mask_offsets.append(mask_offsets[-1] + visible.numel())
    q_positions = torch.cat(q_positions).to(device=device, dtype=torch.int32)
    k_positions = torch.cat(k_positions).to(device=device, dtype=torch.int32)
    mask = torch.cat(masks).to(device)
    mask_ptr = torch.tensor(mask_offsets, device=device, dtype=torch.int64)
    sinks = torch.randn(heads, device=device, dtype=torch.float32) if has_sink else None

    def run(kernel, **kwargs):
        output = torch.empty_like(q)
        kernel(
            q,
            k,
            v,
            output,
            pool_k,
            pool_v,
            qo,
            ki,
            indices,
            kwargs.pop("custom_mask", None),
            True,
            kwargs.pop("mask_indptr", None),
            max(q_lens),
            1.0,
            1.0,
            sinks=sinks,
            page_size=page_size,
            extend_seq_lens_cpu=q_lens,
            **kwargs,
        )
        return output

    # No-feature numerical regression against an independent frozen module.
    assert torch.equal(
        run(actual, sliding_window_size=window),
        run(reference, sliding_window_size=window),
    )
    expected = run(
        reference, custom_mask=mask, mask_indptr=mask_ptr, skip_prefix_custom_mask=False
    )
    result = run(
        actual,
        sliding_window_size=window,
        context_q_positions=q_positions,
        context_kv_positions=k_positions,
    )
    assert torch.isfinite(result).all()
    assert torch.equal(result, expected)
