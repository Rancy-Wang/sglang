"""GPU check of real occurrence -> ragged metadata -> native Triton execution."""

import os

import pytest
import torch
from test_ir import args
from test_planner import query_visibility

pytest_plugins = (
    "test_ir",
    "test_planner",
    "test_native_attention",
    "test_context_attention_plan",
)
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_GPU") != "1", reason="requires the test GPU"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("window_size", [-1, 16])
def test_segmented_native_batch(
    compiler,
    occurrence,
    attention_plan,
    native_attention,
    dtype,
    page_size,
    window_size,
):
    actual, reference = native_attention
    tokens, drops, repos = list(range(141)), {60: [(0, 16)], 100: [(20, 40)]}, [59, 99]
    expected, expiry = query_visibility(tokens, drops, repos)
    layout = compiler(*args(tokens, drops, repos))
    query_start = 22
    win = occurrence(
        layout, expiry, layout.positions, query_start=query_start, query_end=len(tokens)
    )
    plain = attention_plan.ContextSequence.ordinary(7, 3)
    plan = attention_plan.ContextAttentionPlan.merge(
        [plain, attention_plan.ContextSequence.from_window(win), plain]
    )
    torch.manual_seed(101)
    device, heads, kv_heads, dim = "cuda", 4, 2, 64
    pool_size = ((plan.occurrence_count + 63) // 64 + 1) * 64
    slots = torch.randperm(pool_size, device=device)[: plan.occurrence_count]
    metadata = plan.bind(slots)
    q = torch.randn(metadata.query_count, heads, dim, device=device, dtype=dtype)
    pk = torch.randn(pool_size, kv_heads, dim, device=device, dtype=dtype)
    pv = torch.randn_like(pk)
    # The layer has already stored all birth KV; transformed occurrence values
    # are distinct fixtures. The production RoPE kernel has its own mini oracle.
    births = (
        [7, 8, 9]
        + list(range(10 + query_start, 10 + len(tokens)))
        + list(range(plan.occurrence_count - 3, plan.occurrence_count))
    )
    birth_slots = slots[torch.tensor(births, device=device)]
    k, v = pk[birth_slots], pv[birth_slots]
    sinks = torch.randn(heads, device=device)
    result = metadata.forward(
        actual,
        q,
        k,
        v,
        torch.empty_like(q),
        pk,
        pv,
        sinks=sinks,
        page_size=page_size,
        sliding_window_size=window_size,
    )
    # Independent one-query subsequences materialize the staged oracle's visible
    # sets. This deliberately expensive expansion exists only in this small test.
    pair_to_id = {
        pair: i + 10
        for i, pair in enumerate(
            zip(win.occurrence_raw_tokens.tolist(), win.occurrence_positions.tolist())
        )
    }
    oracle_prefix, oracle_offsets = [], [0]
    visibility = (
        [[(i, i) for i in range(8 + q)] for q in range(3)]
        + expected[query_start:]
        + [[(i, i) for i in range(8 + q)] for q in range(3)]
    )
    for query, visible in enumerate(visibility):
        pos = visible[-1][1]
        visible = visible[:-1]
        if window_size > 0:
            visible = [pair for pair in visible if pos <= pair[1] + window_size]
        if query < 3:
            prefix_ids = [raw for raw, _ in visible]
        elif query >= len(visibility) - 3:
            prefix_ids = [plan.occurrence_count - 10 + raw for raw, _ in visible]
        else:
            prefix_ids = [pair_to_id[pair] for pair in visible]
        oracle_prefix.extend(prefix_ids)
        oracle_offsets.append(len(oracle_prefix))
    oracle_result = torch.empty_like(q)
    reference(
        q,
        k,
        v,
        oracle_result,
        pk,
        pv,
        torch.arange(len(q) + 1, device=device, dtype=torch.int64),
        torch.tensor(oracle_offsets, device=device),
        slots[torch.tensor(oracle_prefix, device=device)],
        None,
        True,
        None,
        1,
        1.0,
        1.0,
        sinks=sinks,
        page_size=page_size,
        extend_seq_lens_cpu=[1] * len(q),
    )
    # Segmentation changes reduction order, so bound by one output rounding step
    # (the separate native kernel test measures error against FP64 arithmetic).
    torch.testing.assert_close(
        result, oracle_result, rtol=torch.finfo(dtype).eps, atol=torch.finfo(dtype).eps
    )
