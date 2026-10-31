"""Qwen3.5/3.8 Context admission, native text template and hybrid routing."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_ir import ROOT, load_file


def test_hybrid_admission():
    capabilities = load_file("qwen38_capabilities", ROOT / "python/sglang/srt/context_system/capabilities.py")
    args = NS(page_size=1, prefill_attention_backend=None, decode_attention_backend=None,
              attention_backend="triton", kv_cache_dtype="auto", disaggregation_mode="null",
              pp_size=1, attn_cp_size=1, dcp_size=1, speculative_algorithm=None,
              dllm_algorithm=None, enable_deterministic_inference=False, radix_cache_backend=None)
    text = NS(layer_types=["linear_attention"] * 3 + ["full_attention"], num_hidden_layers=4)
    model = NS(hf_config=NS(architectures=["Qwen3_5ForConditionalGeneration"], text_config=text),
               is_multimodal=False, dtype=torch.bfloat16)
    capabilities.validate_context_config(args, model)
    model.is_multimodal = True
    with pytest.raises(ValueError, match="text-only"):
        capabilities.validate_context_config(args, model)
    model.is_multimodal = False
    text.layer_types[-1] = "qwen_sparse_attention"
    with pytest.raises(ValueError, match="Full/Linear"):
        capabilities.validate_context_config(args, model)
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
    model.hf_config.text_config = Qwen3_5TextConfig(
        num_hidden_layers=4, full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    capabilities.validate_context_config(args, model)


@pytest.fixture(scope="module")
def frontend():
    path = os.environ.get("QWEN38_TOKENIZER_PATH")
    if not path:
        pytest.skip("set QWEN38_TOKENIZER_PATH to local model directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    provenance = load_file("qwen38_provenance", ROOT / "python/sglang/srt/context_system/provenance.py")
    load_file("sglang.srt.context_system.thinking_template", ROOT / "python/sglang/srt/context_system/thinking_template.py")
    return tokenizer, provenance


@pytest.mark.parametrize("preserve", [False, True])
def test_native_template_and_reasoning(frontend, preserve):
    tokenizer, provenance = frontend
    messages = [{"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "OLD_QUESTION"},
                {"role": "assistant", "reasoning_content": "OLD_REASONING", "content": "ANSWER"},
                {"role": "user", "content": "NEW_QUESTION 中文🙂"}]
    original = copy.deepcopy(messages)
    template = tokenizer.chat_template
    actual = provenance.build_template_token_provenance(
        tokenizer, messages, tools=None, add_generation_prompt=True, enable_thinking=False,
        template_kwargs={"preserve_thinking": False, "preserve_thinking_history": preserve})
    native = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False, preserve_thinking=preserve)
    assert actual.rendered_text == native
    assert actual.input_ids == tokenizer(native, add_special_tokens=False)["input_ids"]
    assert ("OLD_REASONING" in native) == preserve
    assert tokenizer.chat_template == template and messages == original
    for sentinel, owner in [("SYSTEM", 0), ("ANSWER", 2), ("NEW_QUESTION", 3)]:
        start = native.index(sentinel)
        assert set(actual.char_owners[start:start + len(sentinel)]) == {owner}


def test_full_registry_does_not_reach_linear_backend():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
    wrapper = object.__new__(HybridLinearAttnBackend)
    wrapper.full_attn_backend = NS()
    wrapper.linear_attn_backend = NS()
    registry = object()
    wrapper.context_decode_registry = registry
    assert wrapper.full_attn_backend.context_decode_registry is registry
    assert not hasattr(wrapper.linear_attn_backend, "context_decode_registry")
    wrapper.context_decode_registry = None
    assert wrapper.full_attn_backend.context_decode_registry is None


def test_binding_selects_full_layers_and_partial_rope():
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from torch import nn
    model = nn.ModuleList([nn.Module() for _ in range(8)])
    buffers = {}
    for i, layer in enumerate(model):
        layer.attn = NS(layer_id=i, sliding_window_size=-1)
        if i % 4 == 3:
            layer.rotary_emb = NS(cos_sin_cache=torch.zeros(32, 64), rotary_dim=64, is_neox_style=True)
            buffers[i] = (torch.zeros(32, 2, 256, dtype=torch.bfloat16),
                          torch.zeros(32, 2, 256, dtype=torch.bfloat16))
    binding = ContextModelBinding(model, NS(get_kv_buffer=lambda i: buffers[i]), NS(), page_size=1)
    assert [layer.layer_id for layer in binding.layers] == [3, 7]
    assert all(layer.rotary_dim == 64 for layer in binding.layers)


def test_hybrid_publication_does_not_relabel_native_state():
    from sglang.srt.context_system.request_storage import context_publish_length

    req = NS(
        context_state=NS(terminal_rows=torch.tensor([1, 2, -1, -1, 5])),
        context_recompute_program=None,
        context_program=NS(
            layout=NS(keep_mask=torch.tensor([True, True, False, False, True])),
            visible_until=torch.tensor([5, 5, 4, 4, 5]),
        ),
    )
    assert context_publish_length(req, 4) == 2
    assert context_publish_length(req, 4, preserve_state_boundary=True) == 0
    assert context_publish_length(req, 5, preserve_state_boundary=True) == 5
