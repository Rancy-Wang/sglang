"""Request semantics and one-render provenance without a server or model."""

import importlib
import os
import random
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from test_ir import ROOT, load_file


@pytest.fixture(scope="module")
def context_modules():
    package = types.ModuleType("context_boundary_test")
    package.__path__ = [str(ROOT / "python/sglang/srt/context_system")]
    sys.modules[package.__name__] = package
    rules = importlib.import_module(package.__name__ + ".rules")
    provenance = importlib.import_module(package.__name__ + ".provenance")
    planner = importlib.import_module(package.__name__ + ".planner")
    kernel = load_file(
        "context_boundary_kernel",
        ROOT / "python/sglang/kernels/ops/attention/context_plan.py",
    )
    rules._load_text_match_module = kernel.load_context_text_match
    ir = importlib.import_module(package.__name__ + ".ir")
    ir._load_module = kernel.load_context_plan
    return rules, provenance, planner


class CharacterTokenizer:
    is_fast = True
    special_tokens_map: ClassVar[dict] = {}
    chat_template = (
        "{% for message in messages %}"
        "{{ '<' + message['role'] + '>' + message['content'] }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )

    def __init__(self):
        self.calls = 0

    def get_chat_template(self, *, tools):
        return self.chat_template

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        self.calls += 1
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_native_utf8_overlapping_and_ordered(context_modules):
    rules, _, _ = context_modules
    rng = random.Random(20260917)
    for _ in range(250):
        text = "".join(rng.choices("aa界😀ba", k=80))
        patterns = [text[i : i + rng.randrange(1, 5)] for i in rng.sample(range(70), 8)]
        assert rules.find_all(text, patterns) == rules.find_all_reference(
            text, patterns
        )
        sources = [text[i : i + 12] for i in range(0, 60, 12)]
        selected = sorted(rng.sample(range(len(sources)), 3))
        targets = [sources[i][3:6] for i in selected]
        keys = [i % 2 for i in range(len(sources))]
        target_keys = [keys[i] for i in selected]
        assert rules.find_ordered_latest(
            sources, targets, source_keys=keys, pattern_keys=target_keys
        ) == rules.find_ordered_latest_reference(
            sources, targets, source_keys=keys, pattern_keys=target_keys
        )
    with pytest.raises(ValueError, match="maximum"):
        rules.find_all("aaaa", ["a"], max_matches=2)


def test_compiler_failure_is_not_silent_fallback(context_modules, monkeypatch):
    rules, _, _ = context_modules

    def unavailable():
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(rules, "_load_text_match_module", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        rules.find_all("test", ["test"])
    assert rules.find_all("test", ["test"], allow_fallback=True) == [[(0, 4)]]


def test_message_trigger_after_whole_message_and_reposition(context_modules):
    rules, provenance, planner = context_modules
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new"},
    ]
    tokenizer = CharacterTokenizer()
    trace = provenance.build_template_token_provenance(
        tokenizer,
        messages,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=None,
    )
    assert tokenizer.calls == 1
    assert trace.rendered_text == "<user>old<assistant>answer<user>new<assistant>"
    rule = rules.parse_drop_rule(None, messages, legacy_drop_message={1: [0]})
    program = planner.compile_chat_program(messages, trace, rule, [1])
    boundary = len("<user>old<assistant>answer")
    removed = len("<user>old")
    assert program.visible_until[:removed].tolist() == [boundary] * removed
    assert program.layout.birth_positions[boundary] == boundary - removed
    assert program.layout.next_position == len(trace.input_ids) - removed
    assert program.layout.drop_ranges.tolist() == [0, removed]
    assert program.layout.records[program.layout.token_to_key[removed], 3] == 0
    assert trace.owners[-len("<assistant>") :] == [3] * len("<assistant>")


def test_text_occurrence_and_protocol_aware_keep(context_modules):
    rules, _, _ = context_modules
    messages = [{"role": "user", "content": "界界界"}]
    rule = rules.parse_drop_rule(
        {
            "type": "text_drop",
            "drop_messages": [{"role": "user", "content": "界界", "occurrence": 2}],
        },
        messages,
    )
    assert rule.selections[0].spans == ((1, 3),)
    full = [
        {"role": "tool", "tool_call_id": "one", "content": "target"},
        {"role": "tool", "tool_call_id": "two", "content": "target"},
    ]
    kept = rules.parse_drop_rule(
        {"type": "keep_text_drop", "full_messages": full}, [full[0]]
    )
    assert kept.keep_spans == ((0, 6), None)
    assert rules.parse_drop_rule(kept.to_wire(), full, allow_internal=True) == kept
    with pytest.raises(ValueError, match="reserved"):
        rules.parse_drop_rule(kept.to_wire(), full)


def test_partial_keep_retains_template_and_crossing_token(context_modules):
    rules, provenance, planner = context_modules
    trace = provenance.TemplateTokenProvenance(
        input_ids=[10, 11, 12, 13, 14],
        owners=[0, 0, 0, 0, 1],
        offsets=[(0, 3), (3, 8), (8, 18), (18, 22), (22, 25)],
        rendered_text="[U]alpha assistant[/U][A]",
        char_owners=[0] * 22 + [1] * 3,
        cross_owner_tokens=0,
    )
    messages = [{"role": "user", "content": "alpha assistant"}]
    rule = rules.KeepTextDropRule(tuple(messages), ((6, 12),))
    program = planner.compile_chat_program(messages, trace, rule, None)
    assert program.layout.drop_ranges.tolist() == [1, 2]
    assert program.layout.keep_mask.tolist() == [True, False, True, True, True]


def test_thinking_field_is_retained_and_dropped_at_its_owner(context_modules):
    rules, provenance, planner = context_modules
    messages = [
        {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
        {"role": "user", "content": "next"},
    ]
    tokenizer = CharacterTokenizer()
    tokenizer.chat_template = "{% for m in messages %}{{ '<'+m.role+'>' }}{% if m.reasoning_content %}{{ m.reasoning_content }}{% endif %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    trace = provenance.build_template_token_provenance(
        tokenizer,
        messages,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=None,
    )
    rule = rules.parse_drop_rule({"type": "thinking_drop"}, messages)
    program = planner.compile_chat_program(messages, trace, rule, None)
    assert program.layout.drop_ranges.tolist() == [11, 17]
    assert program.visible_until[11:17].tolist() == [23] * 6


def test_reordered_owners_cannot_silently_move_event(context_modules):
    _, _, planner = context_modules
    with pytest.raises(RuntimeError, match="reordered"):
        planner.TokenEventCompiler._query_epochs_from_owners([0, 2, 1, 3], 3)


def test_original_mini_provenance_differential(context_modules):
    source = os.environ.get("MINI_SGLANG_REFERENCE")
    if not source:
        pytest.skip("requires fixed mini checkout")
    _, provenance, _ = context_modules
    reference = load_file(
        "mini_boundary_provenance",
        Path(source) / "python/minisgl/tokenizer/template_provenance.py",
    )
    messages = [
        {"role": "user", "content": "界界"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "result"},
    ]
    kwargs = {"tools": None, "add_generation_prompt": True, "enable_thinking": None}
    expected = reference.build_template_token_provenance(
        CharacterTokenizer(), messages, **kwargs
    )
    actual = provenance.build_template_token_provenance(
        CharacterTokenizer(), messages, **kwargs
    )
    assert vars(actual) == vars(expected)
