"""Official MiniMax tokenizer regression; no CUDA/model weights required.

Set MINIMAX_TOKENIZER_PATH to the pinned M2.7 snapshot. Missing fixtures fail,
so this opt-in file cannot be mistaken for completed model coverage.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frontend():
    path = os.environ.get("MINIMAX_TOKENIZER_PATH")
    if not path:
        pytest.fail("Set MINIMAX_TOKENIZER_PATH to the verified M2.7 tokenizer snapshot")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    provenance = load("minimax_provenance", "python/sglang/srt/context_system/provenance.py")
    thinking = load("sglang.srt.context_system.thinking_template", "python/sglang/srt/context_system/thinking_template.py")
    return tokenizer, provenance, thinking


TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a key", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]


def trace(frontend, messages, **kwargs):
    tokenizer, provenance, _ = frontend
    return provenance.build_template_token_provenance(tokenizer, messages, tools=TOOLS,
        add_generation_prompt=True, enable_thinking=None, template_kwargs=kwargs)


@pytest.mark.parametrize("content", ["", "FINAL_TEXT", "<think>INLINE_REASONING</think>FINAL_TEXT",
    [{"type": "text", "text": "<think>INLINE_REASONING</think>FINAL_TEXT"}], "中文🙂 café"])
@pytest.mark.parametrize("later_user", [False, True])
def test_native_text_tokens_and_owner(frontend, content, later_user):
    tokenizer, _, _ = frontend
    messages = [{"role": "system", "content": "SYSTEM_SENTINEL"}, {"role": "user", "content": "QUESTION"}, {"role": "assistant", "content": content}]
    if later_user:
        messages.append({"role": "user", "content": "NEXT_QUESTION"})
    original = copy.deepcopy(messages)
    actual = trace(frontend, messages)
    native = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    assert actual.rendered_text == native
    assert actual.input_ids == tokenizer(native, add_special_tokens=False)["input_ids"]
    assert messages == original
    assert actual.owners[-1] == len(messages)
    for sentinel, owner in [("SYSTEM_SENTINEL", 0), ("FINAL_TEXT", 2), ("中文🙂 café", 2)]:
        start = native.find(sentinel)
        if start >= 0:
            assert set(actual.char_owners[start:start + len(sentinel)]) == {owner}


def test_parallel_tools_and_cross_owner_tokens(frontend):
    messages = [{"role": "user", "content": "Find A and B"},
        {"role": "assistant", "content": "", "reasoning_content": "REASONING", "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": {"q": "A"}}},
            {"type": "function", "function": {"name": "lookup", "arguments": {"q": "B"}}}]},
        {"role": "tool", "content": "TOOL_A"}, {"role": "tool", "content": "TOOL_B"}]
    actual = trace(frontend, messages)
    tokenizer, _, _ = frontend
    assert actual.rendered_text == tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    for sentinel, owner in [("TOOL_A", 2), ("TOOL_B", 3), ("REASONING", 1)]:
        start = actual.rendered_text.index(sentinel)
        assert set(actual.char_owners[start:start + len(sentinel)]) == {owner}
    for owner, (start, end) in zip(actual.owners, actual.offsets):
        if end > start:
            assert owner == actual.char_owners[start]


@pytest.mark.parametrize("inline", [False, True])
def test_history_retention_is_request_local(frontend, inline):
    tokenizer, _, thinking = frontend
    original = tokenizer.chat_template
    assistant = ({"role": "assistant", "content": "<think>OLDER_REASONING</think>ANSWER"} if inline
                 else {"role": "assistant", "content": "ANSWER", "reasoning_content": "OLDER_REASONING"})
    messages = [{"role": "user", "content": "old"}, assistant, {"role": "user", "content": "new"}]
    assert "OLDER_REASONING" not in trace(frontend, messages).rendered_text
    actual = trace(frontend, messages, preserve_thinking_history=True)
    start = actual.rendered_text.index("OLDER_REASONING")
    assert set(actual.char_owners[start:start + len("OLDER_REASONING")]) == {1}
    patched, family = thinking.retained_template(original)
    assert family == "minimax"
    assert thinking.retained_template(patched) == (patched, family)
    assert tokenizer.chat_template == original
    assert "OLDER_REASONING" not in trace(frontend, messages).rendered_text


def test_unrecognized_macro_and_retention_guard_fail_closed(frontend):
    tokenizer, provenance, thinking = frontend
    modified = tokenizer.chat_template.replace("{{ content }}", "{{ content + 'changed' }}", 1)
    assert modified != tokenizer.chat_template
    with pytest.raises(ValueError, match="visible_text macro"):
        provenance._compile_traced_template(modified)
    modified = tokenizer.chat_template.replace("reasoning_content and loop.index0", "reasoning_content and 1 + loop.index0")
    with pytest.raises(ValueError, match="thinking history guard"):
        thinking.retained_template(modified)


def test_rolling_drop_96k_uses_full_template_boundaries(frontend):
    from minimax_context_fixture import RollingDrop96K, TOOLS, tool_history

    tokenizer, _, _ = frontend
    messages = tool_history()[:-1]
    state = RollingDrop96K(
        tokenizer, provenance_builder=frontend[1].build_template_token_provenance
    )
    first = state.extend(messages[:26], TOOLS)  # 12 tool responses, no Drop.
    assert first == {"drop_message": {}, "reposition": []}
    second = state.extend(messages, TOOLS)
    assert second["drop_message"] == {"27": [3], "29": [5], "31": [7], "33": [9]}
    assert second["reposition"] == []  # No eager compaction below 96K.
    assistant = copy.deepcopy(messages[-2])
    assistant["tool_calls"][0]["id"] = "long_call"
    messages += [
        assistant,
        {
            "role": "tool",
            "tool_call_id": "long_call",
            "content": " token" * (96 * 1024),
        },
    ]
    third = state.extend(messages, TOOLS)
    assert third["reposition"] == [35]
    assert all(
        check["before"] >= 96 * 1024 and check["after"] < check["before"]
        for check in state.checks
    )
    assert state.extend(messages, TOOLS) == third
    changed = copy.deepcopy(messages)
    changed[0]["content"] += "changed"
    with pytest.raises(AssertionError, match="Historical messages changed"):
        state.extend(changed, TOOLS)


def test_output_validation_rejects_corruption_and_malformed_calls():
    from minimax_context_fixture import output_findings

    assert output_findings({"content": "中文🙂 legitimate answer"}) == []
    assert "content:invalid_unicode" in output_findings({"content": "broken\ufffd"})
    assert "reasoning_content:repeated_block" in output_findings(
        {"reasoning_content": "A long repeated sentence with forty characters. " * 8}
    )
    assert "invalid_tool_call" in output_findings(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "id": "x",
                    "function": {"name": "bash", "arguments": "not JSON"},
                }
            ]
        }
    )
