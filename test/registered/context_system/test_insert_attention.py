"""Insert logical read sets through native attention and graph decode gather."""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
from test_insert_ir import all_reads, program

pytest_plugins = ("test_rules", "test_native_attention", "test_context_attention_plan")
pytestmark = pytest.mark.skipif(os.environ.get("RUN_CONTEXT_GPU") != "1", reason="isolated CUDA device required")


@pytest.mark.parametrize("repos", [False, True])
@pytest.mark.parametrize("window_size", [-1, 16])
@pytest.mark.parametrize("long_insert", [False, True])
def test_insert_segment_batch(context_modules, attention_plan, native_attention, repos, window_size, long_insert):
    p = program(context_modules, nested=True, repos=repos, width=19)
    if long_insert:
        ms = [{"role": "user", "content": "x", **({"index": -1} if i == 1 else {})}
              for i in range(4)]
        trace = SimpleNamespace(input_ids=list(range(86)), owners=[0]+[1]*80+[2]*2+[3]*3)
        p = context_modules[2].compile_chat_program(ms, trace, None, [-1] if repos else [], {2: -1})
    occ = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    w = occ.compile_occurrence_window(p.layout, p.visible_until, p.layout.positions,
                                      query_start=0, query_end=len(p.input_ids))
    plan = attention_plan.ContextAttentionPlan.merge([attention_plan.ContextSequence.from_window(w)])
    torch.manual_seed(44)
    slots = torch.randperm(plan.occurrence_count + 32, device="cuda")[:plan.occurrence_count]
    metadata = plan.bind(slots)
    pool_k = torch.randn(plan.occurrence_count + 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    pool_v = torch.randn_like(pool_k)
    q = torch.randn(len(p.input_ids), 4, 64, device="cuda", dtype=torch.bfloat16)
    pairs = list(zip(w.occurrence_raw_tokens.tolist(), w.occurrence_positions.tolist()))
    index = {pair: i for i, pair in enumerate(pairs)}
    births = [index[(i, int(pos))] for i, pos in enumerate(p.layout.birth_positions)]
    k, v = pool_k[slots[births]], pool_v[slots[births]]
    actual, frozen = native_attention
    output = metadata.forward(actual, q, k, v, torch.empty_like(q), pool_k, pool_v,
                              page_size=1, sliding_window_size=window_size)
    prefix, offsets = [], [0]
    for row in all_reads(context_modules, p).values():
        pos = row[-1][1]
        prefix.extend(index[pair] for pair in row[:-1] if window_size < 0 or pos <= pair[1]+window_size)
        offsets.append(len(prefix))
    expected = torch.empty_like(q)
    frozen(q, k, v, expected, pool_k, pool_v,
           torch.arange(len(q)+1, device="cuda", dtype=torch.int64),
           torch.tensor(offsets, device="cuda"), slots[torch.tensor(prefix, device="cuda")],
           None, True, None, 1, 1.0, 1.0, page_size=1, extend_seq_lens_cpu=[1]*len(q))
    torch.testing.assert_close(output, expected, rtol=.0078125, atol=.0078125)


def test_nonmonotone_swa_gather_graph(context_modules):
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry
    occ = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    p = program(context_modules, nested=True, repos=False, width=7)
    view = occ.ContextDecodeLayout.from_layout(p.layout, "cuda")
    assert any(a > b for a, b in zip(view.positions[:-1], view.positions[1:]))
    table = torch.arange(512, dtype=torch.int32, device="cuda").reshape(2, 256)
    registry = ContextDecodeRegistry(SimpleNamespace(page_size=1, defer_read_translate=False,
        req_to_token=table, is_translating=False), SimpleNamespace())
    req = SimpleNamespace(context_program=p, context_decode_layout=view)
    rows = torch.tensor([1], device="cuda")
    _, lens, _, _ = registry.bind_batch([req], [len(p.input_ids)+1], rows)
    ptr = torch.zeros(2, device="cuda", dtype=torch.int32)
    out = torch.empty(256, device="cuda", dtype=torch.int64)
    def run():
        registry.fill_window(rows, lens, ptr, 9, out)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for generated in (1, 3, 7):
        _, new_lens, _, _ = registry.bind_batch([req], [len(p.input_ids)+generated], rows)
        lens.copy_(new_lens)
        graph.replay()
        lower = p.layout.next_position + generated - 1 - 9
        raw = [r for r, pos in zip(view.raw_indices.tolist(), view.positions.tolist()) if pos >= lower]
        raw += [len(p.input_ids)+i for i in range(generated) if p.layout.next_position+i >= lower]
        assert ptr[-1].item() == len(raw)
        assert out[:len(raw)].tolist() == [256+r for r in raw]


@pytest.mark.skipif(not os.environ.get("CONTEXT_INSERT_KV_TRACES"), reason="saved model KV probes required")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("layer", range(4))
@pytest.mark.parametrize("mode", ["warm", "exact", "cold"])
@pytest.mark.parametrize("kind", ["insert_no_r", "insert_r"])
def test_insert_saved_model_attention(rank, layer, mode, kind):
    """Independent FP32 attention over actual GPT-OSS query and cached KV.

    The token state-machine tests separately verify visibility. Here the real
    model's sinks, GQA, physical prefix reads and BF16 outputs are checked on
    both tensor-parallel ranks, with and without a post-Insert Reposition.
    """
    import json
    import math
    from pathlib import Path

    from safetensors import safe_open

    root = Path(os.environ["CONTEXT_INSERT_KV_TRACES"])
    model = Path(os.environ["CONTEXT_SERVER_MODEL"])
    data = torch.load(root / f"{kind}-{mode}.pt.layer{layer}.rank{rank}.0.pt", weights_only=True)
    torch.testing.assert_close(data["stored_k"], data["k"].reshape_as(data["stored_k"]), rtol=0, atol=0)
    torch.testing.assert_close(data["stored_v"], data["v"].reshape_as(data["stored_v"]), rtol=0, atol=0)
    name = f"model.layers.{layer}.self_attn.sinks"
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model / index[name]), framework="pt", device="cpu") as weights:
        sinks = weights.get_tensor(name)
    dim = data["stored_k"].shape[-1]
    q = data["q"].reshape(len(data["q"]), -1, dim).float().cuda()
    k = data["k"].reshape(len(q), -1, dim).float().cuda()
    v = data["v"].reshape_as(k).float().cuda()
    pk, pv = data["prefix_k"].float().cuda(), data["prefix_v"].float().cuda()
    qp, pp = data["query_positions"].cuda(), data["kv_positions"].cuda()
    sinks = sinks[rank*q.shape[1]:(rank+1)*q.shape[1]].float().cuda()
    expected = torch.empty_like(q)
    for a, b, x, y in zip(data["qo_indptr"][:-1], data["qo_indptr"][1:],
                          data["kv_indptr"][:-1], data["kv_indptr"][1:]):
        a, b, x, y = map(int, (a, b, x, y))
        kk = torch.cat((pk[x:y], k[a:b])).repeat_interleave(q.shape[1]//k.shape[1], dim=1)
        vv = torch.cat((pv[x:y], v[a:b])).repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        positions = torch.cat((pp[x:y], qp[a:b]))
        for start in range(a, b, 128):
            end = min(start+128, b)
            scores = torch.einsum("qhd,khd->hqk", q[start:end], kk) / math.sqrt(dim)
            visible = torch.arange(len(kk), device="cuda")[None, :] <= (
                y-x+torch.arange(start-a, end-a, device="cuda"))[:, None]
            if data["window"] is not None and data["window"] >= 0:
                visible &= qp[start:end, None] <= positions[None, :]+data["window"]
            scores.masked_fill_(~visible[None, :, :], -float("inf"))
            scores = torch.cat((scores, sinks[:, None, None].expand(-1, end-start, 1)), dim=-1)
            expected[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1)[..., :-1], vv)
    torch.testing.assert_close(data["output"].reshape_as(expected).float().cuda(),
                               expected, rtol=.0078125, atol=.0078125)
