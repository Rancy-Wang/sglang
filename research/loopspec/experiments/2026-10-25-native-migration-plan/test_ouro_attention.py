"""GPU oracle tests: ragged/paged GQA, chunked prefix, and dynamic graph replay."""

import torch

from sglang.kernels.ops.attention.ouro_attention import ouro_attention


def run(decode, graph=False):
    torch.manual_seed(14)
    device = "cuda"
    dtype = torch.bfloat16
    lengths = [1, 1] if decode else [7, 35]
    prefixes = [19, 67]
    qptr = torch.tensor([0, lengths[0], sum(lengths)], device=device, dtype=torch.int32)
    kptr = torch.tensor(
        [0, prefixes[0], sum(prefixes)], device=device, dtype=torch.int32
    )
    indices = torch.randperm(256, device=device).to(torch.int32)
    q = torch.randn(sum(lengths), 8, 128, device=device, dtype=dtype)
    k = torch.randn(sum(lengths), 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    kc = torch.randn(256, 2, 128, device=device, dtype=dtype)
    vc = torch.randn_like(kc)
    out = torch.empty_like(q)

    def invoke():
        ouro_attention(
            q,
            k,
            v,
            kc,
            vc,
            out,
            kptr,
            indices,
            qptr,
            2,
            max(lengths),
            128**-0.5,
            decode=decode,
        )

    def check():
        for i, length in enumerate(lengths):
            start = sum(lengths[:i])
            prefix_start = sum(prefixes[:i])
            slots = indices[prefix_start : prefix_start + prefixes[i]].long()
            keys, values = kc[slots], vc[slots]
            if not decode:
                keys = torch.cat((keys, k[start : start + length]))
                values = torch.cat((values, v[start : start + length]))
            keys, values = [
                x.repeat_interleave(4, dim=1).transpose(0, 1) for x in (keys, values)
            ]
            query = q[start : start + length].transpose(0, 1)
            scores = torch.matmul(query, keys.transpose(1, 2)) * 128**-0.5
            if not decode:
                mask = torch.arange(keys.shape[1], device=device)[None, :] > (
                    prefixes[i] + torch.arange(length, device=device)[:, None]
                )
                scores.masked_fill_(mask, float("-inf"))
            expected = (
                scores.softmax(-1, dtype=torch.float32).to(dtype) @ values
            ).transpose(0, 1)
            actual = out[start : start + length]
            torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.01)
            assert (actual.float() - expected.float()).square().mean().sqrt() < 0.001

    invoke()
    check()
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke()
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            invoke()
        # Replay changes lengths, page mappings, and query data without recapture.
        prefixes[:] = [11, 43]
        kptr.copy_(torch.tensor([0, 11, 54], device=device, dtype=torch.int32))
        indices.copy_(indices.flip(0))
        q.mul_(0.5)
        g.replay()
        check()


if __name__ == "__main__":
    run(False)
    run(True)
    run(True, graph=True)
    print("Ouro paged attention: extend, decode, dynamic graph passed")
