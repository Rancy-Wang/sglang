"""Raw-history overflow: request ownership, model-position bounds and GPU IO."""

from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, load_file

storage = load_file(
    "context_raw_storage", ROOT / "python/sglang/srt/context_system/request_storage.py"
)


def pool_and_request(device="cpu"):
    pool = SimpleNamespace(
        req_to_token=torch.zeros((3, 16), dtype=torch.int32, device=device),
        device=device,
    )
    req = SimpleNamespace(
        context_program=object(),
        origin_input_ids=list(range(40)),
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=SimpleNamespace(req_pool_idx=1),
    )
    return pool, req


def test_overflow_ownership_decode_reserve_and_slot_reuse():
    pool, req = pool_and_request()
    original = pool.req_to_token.data_ptr()
    storage.prepare_request_row(pool, req)
    row = storage.request_row(pool, 1)
    assert len(row) == 52 and pool.req_to_token.data_ptr() == original
    storage.prepare_request_row(pool, req)
    assert storage.request_row(pool, 1) is row
    storage.write_request_slots(pool, (1, slice(35, 40)), torch.arange(5))
    storage.write_request_slots(
        pool, (torch.tensor([1, 2]), torch.tensor([40, 3])), torch.tensor([55, 66])
    )
    assert row[35:41].tolist() == [0, 1, 2, 3, 4, 55]
    assert pool.req_to_token[2, 3] == 66
    assert pool.req_to_token[1].count_nonzero() == 0
    storage.release_request_row(pool, 1)
    assert storage.row_pointers(pool) is None
    assert storage.request_row(pool, 1).data_ptr() == pool.req_to_token[1].data_ptr()
    req.context_program = None
    storage.prepare_request_row(pool, req)
    assert not pool._context_rows


def test_positions_validate_intermediate_rope_not_raw_history():
    # Raw length 40, every real materialization below model limit 24.
    layout = SimpleNamespace(
        birth_positions=torch.arange(40) % 20,
        transition_old_positions=torch.tensor([19]),
        transition_new_positions=torch.tensor([3]),
        positions=torch.arange(40) % 20,
        keep_mask=torch.arange(40) >= 20,
        next_position=20,
    )
    program = SimpleNamespace(layout=layout)
    assert storage.validate_positions(program, 24) == 20
    layout.transition_old_positions[0] = 24
    with pytest.raises(ValueError, match="execution position"):
        storage.validate_positions(program, 24)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_overflow_fused_prefix_decode_scatter_and_graph_gather():
    from sglang.kernels.ops.memory.common import write_req_to_token_pool_triton
    from sglang.srt.layers.attention.context_backend import ContextDecodeRegistry
    from test_context_decode import layout

    pool, req = pool_and_request("cuda")
    storage.prepare_request_row(pool, req)
    prefix = torch.arange(100, 130, dtype=torch.int64, device="cuda")
    out = torch.arange(130, 140, dtype=torch.int64, device="cuda")
    rows = torch.tensor([1], device="cuda")
    pointers = torch.tensor([prefix.data_ptr()], dtype=torch.uint64, device="cuda")
    lens = [torch.tensor([v], device="cuda") for v in (30, 40, 10)]
    write_req_to_token_pool_triton[(1,)](
        pool.req_to_token, rows, pointers, *lens, out, 16, storage.row_pointers(pool)
    )
    storage.write_request_slots(
        pool,
        (rows, torch.tensor([40], device="cuda")),
        torch.tensor([140], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        storage.request_row(pool, 1)[:41],
        torch.arange(100, 141, device="cuda", dtype=torch.int32),
    )
    view, program = layout([0, 31, 39], [0, 1, 2], 40, 3, "cuda")
    req.context_program = SimpleNamespace(layout=program)
    req.context_decode_layout = view
    translator = SimpleNamespace(
        page_size=1,
        defer_read_translate=False,
        req_to_token=pool.req_to_token,
        is_translating=False,
    )
    registry = ContextDecodeRegistry(translator, SimpleNamespace(), pool)
    _, lengths, _, refs = registry.bind_batch([req], [41], rows)
    indptr = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    output = torch.empty(4, device="cuda", dtype=torch.int64)
    registry.fill(rows, lengths, indptr, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        registry.fill(rows, lengths, indptr, output)
    graph.replay()
    assert output.tolist() == [100, 131, 139, 140]
    assert pool.req_to_token[1].count_nonzero() == 0
    # Reuse the same slot for native input without stale pointer reads.
    storage.release_request_row(pool, 1)
    pool.req_to_token[1, :4] = torch.tensor([7, 8, 9, 10], device="cuda")
    req.context_program = None
    registry.bind_batch([req], [4], rows)
    graph.replay()
    assert output.tolist() == [7, 8, 9, 10]
