"""Native checkpoint boundaries and real GDN snapshots with raw Full KV holes."""

from types import SimpleNamespace as NS
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT requires Linux")


@pytest.mark.parametrize("length", [1, 3, 8, 63, 64, 65, 136, 625])
@pytest.mark.parametrize("holes", [0, 16, 40])
@pytest.mark.parametrize("branch", [None, 128])
def test_context_checkpoint_matches_native_raw_boundary(length, holes, branch):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    args = ServerArgs(model_path="dummy", page_size=1)
    args._mamba_cache_chunk_size = 64
    publish(args, role="scheduler")
    try:
        def prepare(context):
            prefix = torch.arange(64)
            prefix[:holes] = -1
            kv = NS(mamba_ping_pong_track_buffer=torch.tensor([2, 3]),
                    mamba_next_track_idx=0, mamba_last_track_seqlen=64)
            req = NS(context_program=object() if context else None, prefix_indices=prefix,
                     kv=kv, extend_range=NS(start=64, end=64+length, length=length),
                     mamba_branching_seqlen=branch)
            batch = NS(model_config=NS(hf_text_config=NS(mamba_chunk_size=64)),
                       tree_cache=NS(page_size=1), req_to_token_pool=NS(
                           get_mamba_ping_pong_other_idx=lambda i: 1-i))
            track = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
            return track, kv.mamba_last_track_seqlen
        native, native_boundary = prepare(False)
        context, boundary = prepare(True)
        assert context == native
        expected = 64 + length // 64 * 64
        if branch is not None and 64 < branch < 64 + length:
            expected = branch
        assert boundary == native_boundary == expected
        assert context.track_mask == (length >= 64)
    finally:
        reset_context()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA recurrent kernel required")
@pytest.mark.parametrize("tail", [1, 3, 8, 63, 64, 65])
def test_cuda_native_checkpoint_matches_actual_prefix(tail):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    torch.manual_seed(1041)
    start, n = 192, 256 + tail
    length = n-start
    boundary = start + length // 64 * 64
    q, k, v = [torch.randn(1, n, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(1, n, 1, device="cuda") * .05
    beta = torch.sigmoid(torch.randn(1, n, 1, device="cuda"))
    states = torch.zeros(4, 1, 64, 64, dtype=torch.float32, device="cuda")
    kernel = TritonGDNKernel()

    def run(a, b, slot):
        return kernel.extend(q=q[:, a:b], k=k[:, a:b], v=v[:, a:b],
                             g=g[:, a:b], beta=beta[:, a:b], ssm_states=states,
                             cache_indices=torch.tensor([slot], device="cuda", dtype=torch.int32),
                             query_start_loc=torch.tensor([0, b-a], device="cuda", dtype=torch.int32))

    run(0, boundary, 0)  # independently stop exactly at the checkpoint
    run(0, start, 1)
    _, _, h = run(start, n, 1)
    backend = NS(device="cuda", mamba_chunk_size=64, conv_states_shape=(1,64,3))
    forward = NS(mamba_track_mask=torch.tensor([True], device="cuda"),
                 extend_seq_lens=torch.tensor([length], device="cuda"),
                 mamba_track_indices=torch.tensor([2], device="cuda"),
                 mamba_track_seqlens=torch.tensor([n], device="cuda"),
                 extend_prefix_lens=torch.tensor([start], device="cuda"),
                 mamba_track_aligned_lens=lambda: torch.tensor([boundary-start], device="cuda"))
    values = MambaAttnBackendBase._init_track_ssm_indices(backend, torch.tensor([1], device="cuda"), forward)
    _, hs, hd, _, fs, fd, _, _, rd = values
    meta = NS(has_mamba_track_mask=True, track_ssm_h_src=hs, track_ssm_h_dst=hd,
              track_ssm_recompute_dst=rd, track_ssm_final_src=fs, track_ssm_final_dst=fd)
    MambaAttnBackendBase._track_mamba_state_extend(backend, forward, h, states, meta)
    torch.testing.assert_close(states[2], states[0], atol=.03, rtol=.03)
    saved = states[2].clone()
    run(0, 1, 1)
    assert torch.equal(states[2], saved)

    # Use the actual native conv selector, then compare with convolution stopped at L.
    channels, width = 64, 4
    conv = torch.zeros(3, channels, width-1, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(channels, length, device="cuda", dtype=torch.bfloat16)
    indices = MambaAttnBackendBase._init_track_conv_indices(
        backend, torch.tensor([0,length],device="cuda"), forward)
    conv[2] = x[:, indices].transpose(0,1)[0]
    causal_conv1d_fn(x[:, :boundary-start].contiguous(),
                    torch.randn(channels, width, device="cuda", dtype=torch.bfloat16), None,
                    conv_states=conv, query_start_loc=torch.tensor([0,boundary-start],device="cuda",dtype=torch.int32),
                    seq_lens_cpu=[boundary-start], cache_indices=torch.tensor([0],device="cuda",dtype=torch.int32),
                    has_initial_state=torch.tensor([False],device="cuda"))
    assert torch.equal(conv[2], conv[0])
    conv[0].zero_()
    assert torch.equal(conv[2], x[:, boundary-start-(width-1):boundary-start])
