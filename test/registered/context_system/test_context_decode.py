"""Final-position decode read sets, mixed lanes, graph replay and raw ownership."""

from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, load_file

ContextDecodeLayout = load_file(
    "context_decode_occurrence", ROOT / "python/sglang/srt/context_system/occurrence.py"
).ContextDecodeLayout


def layout(raw, positions, prompt, next_pos, device="cpu"):
    keep = torch.zeros(prompt, dtype=torch.bool)
    keep[raw] = True
    pos = torch.zeros(prompt, dtype=torch.int32)
    pos[raw] = torch.tensor(positions, dtype=torch.int32)
    program = SimpleNamespace(keep_mask=keep, positions=pos, next_position=next_pos)
    return ContextDecodeLayout.from_layout(program, device), program


def test_raw_swa_floor_preserves_position_window_and_overlap():
    view, _ = layout([0, 7, 8, 19], [0, 1, 2, 3], 20, 4)
    assert view.swa_raw_floor(10, 2) == 0  # unfinished prefill
    assert view.swa_raw_floor(20, 2) == 7  # inclusive positions 1,2,3
    assert view.swa_raw_floor(21, 2) == 8  # positions 2,3,4
    assert view.swa_raw_floor(25, 2) == 22  # only generated KV remains
    assert view.device_indices.tolist() == [0, 7, 8, 19]


def test_empty_active_prompt_still_marks_context():
    view, _ = layout([], [], 9, 19)
    assert len(view.raw_indices) == 0
    assert view.device_indices.data_ptr() != 0
    assert view.swa_raw_floor(12, 2) == 9


def test_mixed_extend_composes_queries_positions_and_slots():
    module = load_file(
        "context_mixed_backend",
        ROOT / "python/sglang/srt/layers/attention/context_backend.py",
    )
    view, program = layout([0, 7, 8, 19], [0, 1, 2, 3], 20, 4)
    req = SimpleNamespace(
        context_program=SimpleNamespace(layout=program),
        context_decode_layout=view,
        kv=SimpleNamespace(req_pool_idx=1),
    )
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)
    batch = SimpleNamespace(
        reqs=[req],
        seq_lens_cpu=torch.tensor([22]),
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        out_cache_loc=table[1, 21:22],
    )
    decode = module.ContextPrefillInput.from_prepared_batch(batch, decode=True)
    plain_seq = module.ContextSequence.ordinary(2, 3)
    empty = torch.empty(0, dtype=torch.int32)
    plain = module.ContextPrefillInput(
        module.ContextAttentionPlan.merge([plain_seq]),
        torch.arange(5, dtype=torch.int32),
        empty,
        empty,
        empty.reshape(0, 2),
        None,
    )
    mixed = module.ContextPrefillInput.concatenate((plain, decode))
    metadata = mixed.attention_plan.bind(mixed.occurrence_slots)
    assert metadata.query_positions.tolist() == [2, 3, 4, 5]
    assert metadata.kv_positions.tolist() == [0, 1, 0, 1, 2, 3, 4]
    assert metadata.kv_indices.tolist() == [0, 1, 32, 39, 40, 51, 52]
    assert metadata.qo_indptr.tolist() == [0, 3, 4]
    assert metadata.kv_indptr.tolist() == [0, 2, 7]


def test_decode_receipt_counts_completed_work_once():
    occurrence = load_file(
        "context_decode_receipt",
        ROOT / "python/sglang/srt/context_system/occurrence.py",
    )
    usage_module = load_file(
        "context_decode_usage", ROOT / "python/sglang/srt/context_system/usage.py"
    )
    usage = usage_module.ContextUsage(
        torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool)
    )
    receipt = occurrence.ContextDecodeCompletion(usage)
    assert usage.snapshot().actual_decode_tokens == 0
    receipt.complete(None)
    receipt.complete(None)
    assert usage.snapshot().actual_decode_tokens == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("translated", [False, True])
def test_decode_gather_keeps_native_raw_row_and_replays_mutable_registry(translated):
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry

    device = "cuda"
    table = torch.arange(4 * 64, device=device, dtype=torch.int32).reshape(4, 64)
    before = table.clone()
    mapping = torch.arange(256, device=device, dtype=torch.int32).flip(0)
    translator = SimpleNamespace(
        page_size=1,
        defer_read_translate=False,
        req_to_token=table,
        is_translating=translated,
        _full_v2p_table=mapping,
        _swa_v2p_table=mapping,
        _full_page_multiplier=2,
        _swa_page_multiplier=3,
    )
    registry = ContextDecodeRegistry(translator, SimpleNamespace())
    view, program = layout([0, 7, 8, 19], [0, 1, 2, 3], 20, 4, device)
    reqs = [
        SimpleNamespace(
            context_program=SimpleNamespace(layout=program), context_decode_layout=view
        ),
        SimpleNamespace(context_program=None),
    ]
    rows = torch.tensor([2, 1], device=device)
    cpu_lens, lens, positions, refs = registry.bind_batch(reqs, [21, 7], rows)
    assert cpu_lens.tolist() == [5, 7]
    assert positions.tolist() == [4, 6]
    assert refs == (view,)
    indptr = torch.zeros(3, device=device, dtype=torch.int32)
    windptr = torch.zeros_like(indptr)
    output = torch.empty(128, device=device, dtype=torch.int64)
    window = torch.empty_like(output)

    def run():
        indptr[1:] = lens.cumsum(0)
        registry.fill(rows, lens, indptr, output)
        registry.fill_window(rows, lens, windptr, 2, window)

    def check(raw_full, raw_window):
        full = torch.tensor(raw_full, device=device, dtype=torch.int64)
        swa = torch.tensor(raw_window, device=device, dtype=torch.int64)
        if translated:
            full, swa = mapping[full] * 2, mapping[swa] * 3
        torch.testing.assert_close(output[: len(full)], full.to(output.dtype))
        torch.testing.assert_close(window[: len(swa)], swa.to(window.dtype))

    run()
    check([128, 135, 136, 147, 148, *range(64, 71)], [136, 147, 148, 69, 70])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Same graph, changed ordinary length and Context positions with a gap.
    view2, program2 = layout([1, 7, 9], [0, 6, 8], 20, 9, device)
    reqs[0].context_decode_layout = view2
    reqs[0].context_program = SimpleNamespace(layout=program2)
    _, new_lens, _, _ = registry.bind_batch(reqs, [22, 6], rows)
    lens.copy_(new_lens)
    graph.replay()
    check([129, 135, 137, 148, 149, *range(64, 70)], [137, 148, 149, 68, 69])
    torch.testing.assert_close(table, before)
    # Reuse a former Context request slot for an ordinary request.
    reqs[0].context_program = None
    _, new_lens, _, _ = registry.bind_batch(reqs, [3, 4], rows)
    lens.copy_(new_lens)
    graph.replay()
    check([128, 129, 130, 64, 65, 66, 67], [129, 130, 66, 67])
