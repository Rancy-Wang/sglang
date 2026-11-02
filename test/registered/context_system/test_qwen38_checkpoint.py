"""Raw checkpoint identity and final-state capture, independent of Full KV holes."""

from types import SimpleNamespace as NS
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT requires Linux")


@pytest.mark.parametrize("length", [1, 3, 63, 64, 65, 136])
@pytest.mark.parametrize("holes", [0, 16, 40])
def test_context_checkpoint_uses_raw_extend_end(length, holes):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = 64
    publish(server_args, role="scheduler")
    try:
        prefix = torch.arange(64)
        prefix[:holes] = -1
        kv = NS(mamba_ping_pong_track_buffer=torch.tensor([2, 3]), mamba_next_track_idx=0)
        req = NS(context_program=object(), prefix_indices=prefix, kv=kv,
                 extend_range=NS(start=64, end=64 + length, length=length),
                 mamba_branching_seqlen=None)
        batch = NS(model_config=NS(hf_text_config=NS(mamba_chunk_size=64)),
                   tree_cache=NS(page_size=1),
                   req_to_token_pool=NS(get_mamba_ping_pong_other_idx=lambda i: 1-i))
        track = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
        assert track.track_mask and track.track_index == 2
        assert track.track_seqlen == kv.mamba_last_track_seqlen == 64 + length
    finally:
        reset_context()


@pytest.mark.parametrize("length", [1, 3, 63, 64, 65, 136])
def test_endpoint_capture_reads_final_state_and_survives_live_updates(length):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase

    backend = NS(device="cpu", mamba_chunk_size=64)
    # A Context row and a native row share a batch: only Context opts into
    # the exact endpoint; the ordinary row retains its native chunk snapshot.
    forward = NS(mamba_track_mask=torch.tensor([True, True]),
                 extend_seq_lens=torch.tensor([length, 136]),
                 mamba_track_indices=torch.tensor([2, 3]),
                 mamba_track_seqlens=torch.tensor([64 + length, 200]),
                 extend_prefix_lens=torch.tensor([64, 64]),
                 context_mamba_endpoint_mask=torch.tensor([True, False]))
    values = MambaAttnBackendBase._init_track_ssm_indices(
        backend, torch.tensor([0, 1]), forward
    )
    _, h_src, h_dst, _, final_src, final_dst, _, _, recompute_dst = values
    assert final_src.tolist() == [0] and final_dst.tolist() == [2]
    assert h_dst.tolist() == [3]
    # Synthetic recurrence has an independent exact state at every raw step.
    state = torch.zeros(4, 2, 2)
    expected = torch.arange(4).reshape(2, 2).float() + 64 + length
    state[0] = expected
    h = torch.full((int(h_src.max()) + 1, 2, 2), -192.0)
    metadata = NS(has_mamba_track_mask=True, track_ssm_h_src=h_src,
                  track_ssm_h_dst=h_dst, track_ssm_recompute_dst=recompute_dst,
                  track_ssm_final_src=final_src, track_ssm_final_dst=final_dst)
    MambaAttnBackendBase._track_mamba_state_extend(backend, forward, h, state, metadata)
    assert torch.equal(state[2], expected)
    assert torch.all(state[3] == -192)
    state[0].add_(1)  # later decode must not mutate the published checkpoint
    assert torch.equal(state[2], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA recurrent kernel required")
@pytest.mark.parametrize("tail", [1, 3, 8, 63, 65])
def test_cuda_endpoint_state_matches_unsplit_recurrence(tail):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    torch.manual_seed(1041)
    n = 192 + tail
    q, k, v = [torch.randn(1, n, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(1, n, 1, device="cuda") * .05
    beta = torch.sigmoid(torch.randn(1, n, 1, device="cuda"))
    states = torch.zeros(4, 1, 64, 64, dtype=torch.float32, device="cuda")
    kernel = TritonGDNKernel()

    def run(start, end, slot):
        return kernel.extend(q=q[:, start:end], k=k[:, start:end], v=v[:, start:end],
                             g=g[:, start:end], beta=beta[:, start:end], ssm_states=states,
                             cache_indices=torch.tensor([slot], device="cuda", dtype=torch.int32),
                             query_start_loc=torch.tensor([0, end-start], device="cuda", dtype=torch.int32))

    run(0, n, 0)
    run(0, 192, 1)
    earlier = states[1].clone()
    _, _, h = run(192, n, 1)
    torch.testing.assert_close(states[0], states[1], atol=.03, rtol=.03)
    assert not torch.allclose(earlier, states[1])
    backend = NS(device="cuda", mamba_chunk_size=64)
    forward = NS(mamba_track_mask=torch.tensor([True], device="cuda"),
                 extend_seq_lens=torch.tensor([tail], device="cuda"),
                 mamba_track_indices=torch.tensor([2], device="cuda"),
                 mamba_track_seqlens=torch.tensor([n], device="cuda"),
                 extend_prefix_lens=torch.tensor([192], device="cuda"),
                 context_mamba_endpoint_mask=torch.tensor([True], device="cuda"))
    values = MambaAttnBackendBase._init_track_ssm_indices(backend, torch.tensor([1], device="cuda"), forward)
    _, hs, hd, _, fs, fd, _, _, rd = values
    meta = NS(has_mamba_track_mask=True, track_ssm_h_src=hs, track_ssm_h_dst=hd,
              track_ssm_recompute_dst=rd, track_ssm_final_src=fs, track_ssm_final_dst=fd)
    MambaAttnBackendBase._track_mamba_state_extend(backend, forward, h, states, meta)
    assert torch.equal(states[2], states[1])
    saved = states[2].clone()
    run(0, 1, 1)
    assert torch.equal(states[2], saved)

    # The final conv window includes prior history for 1/3-token extensions.
    channels, width = 64, 4
    conv = torch.randn(3, channels, width-1, device="cuda", dtype=torch.bfloat16)
    initial = conv[1].clone()
    x = torch.randn(channels, tail, device="cuda", dtype=torch.bfloat16)
    expected = torch.cat((initial, x), dim=-1)[:, -(width-1):]
    causal_conv1d_fn(x, torch.randn(channels, width, device="cuda", dtype=torch.bfloat16), None,
                    conv_states=conv, query_start_loc=torch.tensor([0,tail],device="cuda",dtype=torch.int32),
                    seq_lens_cpu=[tail], cache_indices=torch.tensor([1],device="cuda",dtype=torch.int32),
                    has_initial_state=torch.tensor([True],device="cuda"))
    GDNAttnBackend._track_context_endpoint_conv(backend, forward, conv, torch.tensor([1],device="cuda"))
    assert torch.equal(conv[2], expected)
    conv[1].zero_()
    assert torch.equal(conv[2], expected)
