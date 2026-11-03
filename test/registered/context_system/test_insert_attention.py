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
def test_insert_segment_batch(context_modules, attention_plan, native_attention, repos, window_size):
    p = program(context_modules, nested=True, repos=repos, width=19)
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
