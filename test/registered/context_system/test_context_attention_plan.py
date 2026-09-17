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


@pytest.mark.parametrize("shared_swa_pool", [False, True])
def test_native_model_pool_and_rope_bind_once(attention_plan, shared_swa_pool):
    from types import SimpleNamespace

    class Pool:
        def __init__(self):
            self.calls = []
            self.buffers = {
                i: (
                    torch.empty(128, 2, 64, dtype=torch.bfloat16),
                    torch.empty(128, 2, 64, dtype=torch.bfloat16),
                )
                for i in (7, 13)
            }

        def get_kv_buffer(self, layer_id):
            self.calls.append(layer_id)
            return self.buffers[layer_id]

    class Translator:
        def translate_full_attn_ids(self, ids):
            return ids + 10

        def sliding_window_write_loc_for(self, ids):
            return None if shared_swa_pool else ids + 50

    ropes = [torch.randn(512, 64) for _ in range(2)]
    modules = [
        SimpleNamespace(
            attn=SimpleNamespace(layer_id=i, sliding_window_size=window),
            rotary_emb=SimpleNamespace(cos_sin_cache=rope, is_neox_style=style),
        )
        for i, window, rope, style in zip((7, 13), (-1, 128), ropes, (True, False))
    ]
    model = SimpleNamespace(modules=lambda: iter(modules))
    pool = Pool()
    binding = attention_plan.ContextModelBinding(model, pool, Translator(), page_size=1)
    plan = attention_plan.ContextAttentionPlan.merge(
        [attention_plan.ContextSequence.ordinary(3, 2)]
    )
    inputs = attention_plan.ContextPrefillInput(
        plan,
        torch.arange(1, 6),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        torch.tensor([[17, 3]], dtype=torch.int32),
        binding,
    )
    for _ in range(2):
        metadata = inputs.bind()
        assert metadata.full.kv_indices.tolist() == [11, 12, 13]
        offset = 0 if shared_swa_pool else 50
        assert metadata.sliding_window.kv_indices.tolist() == [
            11 + offset,
            12 + offset,
            13 + offset,
        ]
        assert (metadata.sliding_window is metadata.full) == shared_swa_pool
        assert metadata.layer_copies[7].source_slots.tolist() == [11]
        assert metadata.layer_copies[13].source_slots.tolist() == [11 + offset]
        assert metadata.layer_copies[7].destination_slots.tolist() == [19]
        assert metadata.layer_copies[13].destination_slots.tolist() == [19 + offset]
        assert metadata.layer_copies[13].skip_unmapped == (not shared_swa_pool)
        assert metadata.layer_copies[13].cos_sin_cache is ropes[1]
        assert not metadata.layer_copies[13].is_neox_style
    assert pool.calls == [7, 13]
    assert inputs.occurrence_slots.tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="page_size=1"):
        attention_plan.ContextModelBinding(model, pool, Translator(), page_size=16)
