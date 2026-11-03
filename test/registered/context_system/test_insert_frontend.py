"""Insert validation and the one-tokenization public frontend contract."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_rules import CharacterTokenizer

pytest_plugins = ("test_rules", "test_context_frontend")


def messages(explicit=False):
    ids = [0, 1, -2, -1, 2, 3]
    return [dict(role="user", content=str(i), **({"index": i} if explicit or i < 0 else {}))
            for i in ids]


@pytest.mark.parametrize("explicit", [False, True])
def test_single_tokenization_and_default_indices(context_modules, explicit):
    rules, provenance, planner = context_modules
    ms = messages(explicit)
    tokenizer = CharacterTokenizer()
    trace = provenance.build_template_token_provenance(
        tokenizer, ms, tools=None, add_generation_prompt=True, enable_thinking=None)
    p = planner.compile_chat_program(ms, trace, None, [-1], {3: -1, -1: -2})
    assert tokenizer.calls == 1
    assert rules.normalize_message_indices(ms) == (0, 1, -2, -1, 2, 3)
    assert sorted(p.input_ids) == sorted(trace.input_ids)
    assert p.to_wire()["version"] == 5
    assert p.to_json_wire()["version"] == 6


@pytest.mark.parametrize("indices,insert,error", [
    ([0, None], {}, "all specify"),
    ([1, 2], {}, "must be 0"),
    ([None, -2, None], {1: -2}, "without gaps"),
    ([None, -1, -1, None], {1: -1}, "unique"),
    ([None, -1, None], {}, "exactly one"),
    ([None, -1, None], {9: -1}, "missing"),
    ([None, -1, None], {0: -1, 1: -1}, "more than once"),
    ([None, -1, None], {-1: -1}, "cycle"),
    ([None, -1, None], {1: "-1"}, "negative integer"),
    ([True, None], {}, "integer"),
])
def test_bad_format(context_modules, indices, insert, error):
    ms = [dict(role="user", content="x", **({"index": i} if i is not None else {})) for i in indices]
    with pytest.raises(ValueError, match=error):
        context_modules[0].InsertRule.from_payload(insert, ms)


def test_unborn_anchor_and_drop_rejected(context_modules):
    rules, _, planner = context_modules
    ms = messages()
    trace = SimpleNamespace(input_ids=list(range(6)), owners=list(range(6)))
    with pytest.raises(ValueError, match="anchor"):
        planner.compile_chat_program(ms, trace, None, [], {0: -1, -1: -2})
    drop = rules.parse_drop_rule(None, ms, legacy_drop_message={1: [-1]})
    with pytest.raises(ValueError, match="unborn"):
        planner.compile_chat_program(ms, trace, drop, [], {3: -1, -1: -2})


@pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime requires Linux")
def test_native_frontend_and_qwen_rejection(chat):
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    ms = messages()
    req = ChatCompletionRequest(messages=ms, insert={"3": -1, "-1": -2}, reposition=[-1])
    # Native model dump roundtrips preserve omitted optional indices.
    req = ChatCompletionRequest.model_validate(req.model_dump())
    tokenizer = chat.tokenizer_manager.tokenizer
    original = tokenizer.__class__.__call__
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    with patch.object(type(tokenizer), "__call__", counted):
        result = chat._process_messages(req, False)
    assert len(calls) == 1
    p = ContextProgram.from_wire(result.context_program, result.prompt_ids)
    assert len(p.layout.insert_spans) == 6
    assert req.context_index_metadata()["context_message_indices"] == [0, 1, -2, -1, 2, 3]
    config = chat.tokenizer_manager.model_config
    config.hf_config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config._resolved_model_arch = "Qwen3_5ForConditionalGeneration"
    with pytest.raises(ValueError, match="Insert|insert"):
        chat._process_messages(req, False)
