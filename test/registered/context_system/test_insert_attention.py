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
                              birth_slots=slots[births], page_size=1, sliding_window_size=window_size)
    prefix, offsets = [], [0]
    for row in all_reads(context_modules, p).values():
        pos = row[-1][1]
        chosen = [pair for pair in row if window_size < 0 or pos <= pair[1]+window_size]
        if window_size >= 0:
            chosen = chosen[-(window_size+1):]
        prefix.extend(index[pair] for pair in chosen if pair != row[-1])
        offsets.append(len(prefix))
    expected = torch.empty_like(q)
    frozen(q, k, v, expected, pool_k, pool_v,
           torch.arange(len(q)+1, device="cuda", dtype=torch.int64),
           torch.tensor(offsets, device="cuda"), slots[torch.tensor(prefix, device="cuda")],
           None, True, None, 1, 1.0, 1.0, page_size=1, extend_seq_lens_cpu=[1]*len(q))
    torch.testing.assert_close(output, expected, rtol=.0078125, atol=.0078125)

    # Reordering whole independent Q segments changes storage order only. Each
    # segment keeps its own logical KV order, causal extent, positions and sinks.
    # Reordering Q without those associations would describe a different mask.
    from dataclasses import replace
    import numpy as np

    qo, kv, occurrences, qpos, kpos = metadata.cpu_fields
    order = list(reversed(range(len(qo)-1)))
    qtake = np.concatenate([np.arange(qo[i], qo[i+1]) for i in order])
    ktake = np.concatenate([np.arange(kv[i], kv[i+1]) for i in order])
    qptr = np.r_[0, np.cumsum([qo[i+1]-qo[i] for i in order])]
    kptr = np.r_[0, np.cumsum([kv[i+1]-kv[i] for i in order])]
    qi, ki = torch.from_numpy(qtake).cuda(), torch.from_numpy(ktake).cuda()
    reordered = replace(metadata, qo_indptr=torch.from_numpy(qptr).cuda(),
                        kv_indptr=torch.from_numpy(kptr).cuda(),
                        kv_indices=metadata.kv_indices[ki], query_positions=metadata.query_positions[qi],
                        kv_positions=metadata.kv_positions[ki],
                        query_lengths=tuple(int(x) for x in np.diff(qptr)),
                        cpu_fields=(qptr, kptr, occurrences[ktake], qpos[qtake], kpos[ktake]),
                        native_plans={})
    permuted = reordered.forward(actual, q[qi], k[qi], v[qi], torch.empty_like(q), pool_k, pool_v,
                                 birth_slots=slots[births][qi], page_size=1, sliding_window_size=window_size)
    torch.testing.assert_close(permuted, output[qi], rtol=.0078125, atol=.0078125)


@pytest.mark.parametrize("tail_dropped", [False, True])
def test_nonmonotone_swa_gather_graph(context_modules, tail_dropped):
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry
    occ = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    p = program(context_modules, nested=True, repos=False, width=7)
    window_size = 9
    if tail_dropped:
        ir = importlib.import_module(context_modules[2].__package__ + ".ir")
        layout = ir.compile_execution_layout(
            list(range(10)), [(2, 3, 1, 0), (10, 4, 0, 0), (10, 1, 1, 2)],
            [0, 9, *range(1, 9)], [(2, 10, 2)])
        expiry = torch.full((10,), 2**31-1, dtype=torch.int32)
        expiry[1] = 10
        p = context_modules[2].ContextProgram(layout, expiry)
        window_size = 3
    view = occ.ContextDecodeLayout.from_layout(p.layout, "cuda")
    assert not view.positions_monotone
    table = torch.arange(512, dtype=torch.int32, device="cuda").reshape(2, 256)
    registry = ContextDecodeRegistry(SimpleNamespace(page_size=1, defer_read_translate=False,
        req_to_token=table, is_translating=False), SimpleNamespace())
    req = SimpleNamespace(context_program=p, context_decode_layout=view)
    rows = torch.tensor([1], device="cuda")
    _, lens, _, _ = registry.bind_batch([req], [len(p.input_ids)+1], rows)
    ptr = torch.zeros(2, device="cuda", dtype=torch.int32)
    out = torch.empty(256, device="cuda", dtype=torch.int64)
    def run():
        registry.fill_window(rows, lens, ptr, window_size, out)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for generated in (1, 3, 7):
        _, new_lens, _, _ = registry.bind_batch([req], [len(p.input_ids)+generated], rows)
        lens.copy_(new_lens)
        graph.replay()
        lower = p.layout.next_position + generated - 1 - window_size
        raw = [r for r, pos in zip(view.raw_indices.tolist(), view.positions.tolist()) if pos >= lower]
        raw += [len(p.input_ids)+i for i in range(generated) if p.layout.next_position+i >= lower]
        raw = raw[-(window_size+1):]
        assert ptr[-1].item() == len(raw)
        assert out[:len(raw)].tolist() == [256+r for r in raw]


@pytest.mark.parametrize("window_size", [0, 3, 128])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_random_swa_graph_matches_logical_tail(window_size, length_dtype):
    from sglang.srt.context_system.occurrence import ContextDecodeLayout
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry

    torch.manual_seed(931)
    table = torch.randperm(4096, device="cuda", dtype=torch.int32).reshape(2, 2048)
    registry = ContextDecodeRegistry(SimpleNamespace(page_size=1, defer_read_translate=False,
        req_to_token=table, is_translating=False), SimpleNamespace())
    rows = torch.tensor([1], device="cuda")
    for _ in range(20):
        layout = SimpleNamespace(positions=torch.randint(0, 160, (257,), dtype=torch.int32),
                                 keep_mask=torch.rand(257) > .2,
                                 logical_order=torch.randperm(257, dtype=torch.int32), next_position=80)
        view = ContextDecodeLayout.from_layout(layout, "cuda")
        req = SimpleNamespace(context_program=SimpleNamespace(layout=layout), context_decode_layout=view)
        _, lengths, _, _ = registry.bind_batch([req], [258], rows)
        lengths = lengths.to(length_dtype)
        ptr = torch.zeros(2, device="cuda", dtype=torch.int32)
        output = torch.empty(256, device="cuda", dtype=torch.int64)
        def run():
            registry.fill_window(rows, lengths, ptr, window_size, output)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for generated in (1, 2, 4, 129, 150):
            _, updated, _, _ = registry.bind_batch([req], [257+generated], rows)
            lengths.copy_(updated)
            graph.replay()
            lower = 80 + generated - 1 - window_size
            raw = [int(r) for r, pos in zip(view.raw_indices, view.positions) if pos >= lower]
            raw += [257+i for i in range(generated) if 80+i >= lower]
            expected = table[1, raw[-(window_size+1):]].tolist()
            assert output[:int(ptr[-1])].tolist() == expected


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
                from_right = visible.flip(-1).int().cumsum(-1).flip(-1)
                visible &= from_right <= data["window"]+1
            scores.masked_fill_(~visible[None, :, :], -float("inf"))
            scores = torch.cat((scores, sinks[:, None, None].expand(-1, end-start, 1)), dim=-1)
            expected[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1)[..., :-1], vv)
    torch.testing.assert_close(data["output"].reshape_as(expected).float().cuda(),
                               expected, rtol=.0078125, atol=.0078125)


@pytest.mark.parametrize("mixed", [False, True])
def test_native_decode_adapters_oversized_window(context_modules, mixed):
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterDecode,
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    occ = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    positions = [0] + list(range(1, 513))*2
    layout = SimpleNamespace(keep_mask=torch.ones(1025, dtype=torch.bool),
                             positions=torch.tensor(positions, dtype=torch.int32),
                             logical_order=torch.arange(1025, dtype=torch.int32), next_position=513)
    view = occ.ContextDecodeLayout.from_layout(layout, "cuda")
    table = torch.arange(5*2048, device="cuda", dtype=torch.int32).reshape(5, 2048)
    translator = SimpleNamespace(page_size=1, defer_read_translate=False,
                                 req_to_token=table, is_translating=False, reads_are_translated=False)
    registry = ContextDecodeRegistry(translator, SimpleNamespace())
    req = SimpleNamespace(context_program=object(), context_decode_layout=view)
    reqs = [req]*3 + [SimpleNamespace(context_program=None) if mixed else req]
    rows = torch.arange(4, device="cuda")
    cpu, lens, _, _ = registry.bind_batch(reqs, [1026]*4, rows)
    expected = []
    for i in range(4):
        raw = (list(range(897, 1026)) if mixed and i == 3 else
               [j for j, p in enumerate(positions) if p >= 385] + [1025])
        expected.extend(i*2048+j for j in raw[-129:])
    assert len(expected) == 4*129
    backend = SimpleNamespace(context_decode_registry=registry, sliding_window_size=128,
                              cuda_graph_kv_indices=torch.empty(8192),
                              cuda_graph_window_kv_indices=torch.empty(516))
    assert TritonAttnBackend.context_decode_fits_graph(backend, 4)
    backend.cuda_graph_window_kv_indices = torch.empty(len(expected))
    assert TritonAttnBackend.context_decode_fits_graph(backend, 4)
    metadata = SimpleNamespace(page_table=torch.empty((4, 2048)), pa_swa_page_table=torch.empty((4, 129)))
    backend.decode_cuda_graph_metadata = {4: metadata}
    assert FlashAttentionBackend.context_decode_fits_graph(backend, 4)
    metadata.pa_swa_page_table = torch.empty((4, 257))
    assert FlashAttentionBackend.context_decode_fits_graph(backend, 4)
    # Exercise the real FlashInfer updater, stopping at the native wrapper's plan
    # boundary. Compare its physical gather to an independent logical oracle.
    captured = []
    def begin(ptr, indices, *args, **kwargs):
        captured.append(indices[:int(ptr[-1])].clone())
    updater = object.__new__(FlashInferIndicesUpdaterDecode)
    updater.attn_backend = SimpleNamespace(context_decode_registry=registry, kv_index_translator=translator)
    updater.sliding_window_size = 128
    updater._swa_kv_pool = None
    updater.kv_indptr = [torch.zeros(5, device="cuda", dtype=torch.int32) for _ in range(2)]
    updater.kv_last_page_len = torch.ones(4, device="cuda", dtype=torch.int32)
    updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = 4, 2, 64
    updater.data_type = updater.q_data_type = torch.bfloat16
    wrappers = [SimpleNamespace(is_cuda_graph_enabled=False, begin_forward=begin) for _ in range(2)]
    updater.update_sliding_window(lens, cpu, int(cpu.sum()), wrappers, None, None, req_pool_indices=rows)
    assert captured[0].tolist() == expected
    assert len(captured[1]) == 4*1026
    # Use the FA table and Triton packed adapters with sufficient buffers.
    out = torch.full((4, 257), -1, device="cuda", dtype=torch.int32)
    lengths = registry.fill_table(rows, lens, out, window=128)
    actual = [x for row, n in zip(out.tolist(), lengths.tolist()) for x in row[:n]]
    assert actual == expected
    ptr = torch.zeros(5, device="cuda", dtype=torch.int32)
    packed = torch.empty(len(expected), device="cuda", dtype=torch.int64)
    registry.fill_window(rows, lens, ptr, 128, packed)
    assert packed.tolist() == expected
