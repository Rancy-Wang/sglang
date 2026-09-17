"""Validate merged metadata against an independent sequential visibility oracle."""

import pytest
import torch
from test_ir import ROOT, args, cases, load_file
from test_planner import query_visibility

pytest_plugins = ("test_ir", "test_planner")


@pytest.fixture(scope="module")
def attention_plan():
    return load_file(
        "context_test_attention_plan",
        ROOT / "python/sglang/srt/layers/attention/context_backend.py",
    )


def test_mixed_segments(compiler, occurrence, attention_plan):
    for tokens, drops, reposition in cases():
        layout = compiler(*args(tokens, drops, reposition))
        expected, expiry = query_visibility(tokens, drops, reposition)
        start = len(tokens) // 2
        window = occurrence(
            layout, expiry, layout.positions, query_start=start, query_end=len(tokens)
        )
        seq = attention_plan.ContextSequence.from_window(window)
        plain = attention_plan.ContextSequence.ordinary(7, 3)
        plan = attention_plan.ContextAttentionPlan.merge([plain, seq, plain])
        # Unique shuffled physical addresses prevent confusing occurrence IDs,
        # raw IDs, active offsets, and physical pool slots.
        slots = torch.randperm(plan.occurrence_count) * 4 + 16
        bound = plan.bind(slots)
        inverse = {
            slot: occurrence_id for occurrence_id, slot in enumerate(slots.tolist())
        }
        raw = window.occurrence_raw_tokens.tolist()
        q_ptr, kv_ptr = bound.qo_indptr.tolist(), bound.kv_indptr.tolist()
        for segment in range(1, len(q_ptr) - 2):
            qa, qb = q_ptr[segment : segment + 2]
            ka, kb = kv_ptr[segment : segment + 2]
            prefix_ids = [
                inverse[slot] - 10 for slot in bound.kv_indices[ka:kb].tolist()
            ]
            prefix = list(
                zip((raw[i] for i in prefix_ids), bound.kv_positions[ka:kb].tolist())
            )
            for q in range(qa, qb):
                query_raw = start + q - 3
                births = list(
                    zip(
                        range(start + qa - 3, query_raw + 1),
                        bound.query_positions[qa : q + 1].tolist(),
                    )
                )
                assert prefix + births == expected[query_raw]
                for window_size in (1, 4, 128):
                    pos = int(bound.query_positions[q])
                    assert [
                        pair for pair in prefix + births if pos <= pair[1] + window_size
                    ] == [
                        pair
                        for pair in expected[query_raw]
                        if pos <= pair[1] + window_size
                    ]
        assert bound.query_positions[:3].tolist() == [7, 8, 9]
        assert bound.query_positions[-3:].tolist() == [7, 8, 9]
        assert bound.kv_indices[:7].tolist() == slots[:7].tolist()
        assert bound.kv_indices[-7:].tolist() == slots[-10:-3].tolist()


def test_invalid_binding(attention_plan):
    plan = attention_plan.ContextAttentionPlan.merge(
        [attention_plan.ContextSequence.ordinary(3, 2)]
    )
    with pytest.raises(ValueError, match="binding"):
        plan.bind(torch.arange(4))
    with pytest.raises(ValueError, match="empty"):
        attention_plan.ContextAttentionPlan.merge([])


def test_large_page_rejected_before_kernel(attention_plan):
    plan = attention_plan.ContextAttentionPlan.merge(
        [attention_plan.ContextSequence.ordinary(3, 2)]
    )
    bound = plan.bind(torch.arange(5))

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("unsupported page size reached kernel")

    with pytest.raises(ValueError, match="page_size=1"):
        bound.forward(
            unexpected_kernel, None, None, None, None, None, None, page_size=16
        )
