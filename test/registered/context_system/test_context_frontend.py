"""Native chat preprocessing and native IPC; no model/GPU is initialized."""

import os
import sys
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest
from test_ir import ROOT, load_file

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native SRT runtime requires Linux"
)


@pytest.fixture
def chat():
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    fixture = load_file(
        "context_native_chat_fixture",
        ROOT / "test/registered/unit/entrypoints/openai/test_serving_chat.py",
    )
    reset_context()
    publish(ServerArgs(model_path="dummy", page_size=1), role="tokenizer")
    manager = fixture._MockTokenizerManager()
    manager._config_overrides["page_size"] = 1
    template = fixture._MockTemplateManager()
    template.chat_template_name = None
    template.jinja_template_content_format = "string"
    serving = OpenAIServingChat(manager, template)
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "[BOS]",
                "user",
                "assistant",
                "tool",
                "old",
                "answer",
                "new",
                "<",
                ">",
                "reason",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    manager.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]"
    )
    manager.tokenizer.chat_template = (
        "{% for message in messages %}{{ '< ' + message.role + ' > ' }}"
        "{% if message.reasoning_content %}{{ message.reasoning_content + ' ' }}{% endif %}"
        "{{ message.content + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '< assistant > ' }}{% endif %}"
    )
    serving._tokenizer_auto_adds_specials = False
    try:
        yield serving
    finally:
        reset_context()


def messages():
    return [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new"},
    ]


def request(**kwargs):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    return ChatCompletionRequest(messages=kwargs.pop("messages", messages()), **kwargs)


def test_no_feature_uses_original_render(chat):
    with patch(
        "sglang.srt.context_system.provenance.build_template_token_provenance",
        side_effect=AssertionError("ordinary request compiled Context"),
    ):
        result = chat._process_messages(request(), False)
    assert result.context_program is None
    assert result.prompt_ids == chat.tokenizer_manager.tokenizer.apply_chat_template(
        messages(), tokenize=True, add_generation_prompt=True, return_dict=False
    )


@pytest.mark.parametrize("repos", [None, [1]])
def test_chat_program_and_native_ipc(chat, repos):
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.io_struct import (
        TokenizedGenerateReqInput,
        msgpack_decode,
        msgpack_encode,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    result = chat._process_messages(
        request(drop_message={"1": [0]}, reposition=repos), False
    )
    assert result.prompt_ids == chat.tokenizer_manager.tokenizer.apply_chat_template(
        messages(), tokenize=True, add_generation_prompt=True, return_dict=False
    )
    obj = TokenizedGenerateReqInput(
        input_text=None,
        input_ids=array("q", result.prompt_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        context_program=result.context_program,
    )
    decoded = msgpack_decode(msgpack_encode(obj))
    program = ContextProgram.from_wire(decoded.context_program, decoded.input_ids)
    assert program.layout.drop_ranges.tolist() == [0, 4]
    assert program.visible_until[:4].tolist() == [8] * 4
    assert program.layout.next_position == len(result.prompt_ids) - (4 if repos else 0)


def test_continuation_preserves_native_ids(chat):
    from sglang.srt.context_system.planner import ContextProgram

    raw = messages()[:2]
    plain = chat._process_messages(
        request(messages=raw, continue_final_message=True), False
    )
    feature = chat._process_messages(
        request(
            messages=raw,
            continue_final_message=True,
            drop_message={"1": [0]},
            reposition=[1],
        ),
        False,
    )
    assert feature.prompt_ids == plain.prompt_ids
    program = ContextProgram.from_wire(feature.context_program, feature.prompt_ids)
    assert program.layout.drop_insert_offsets.tolist() == [len(feature.prompt_ids)]


def test_keep_projection_renders_native_full_history(chat):
    from sglang.srt.context_system.planner import ContextProgram

    full = messages()
    result = chat._process_messages(
        request(
            messages=[full[-1]],
            drop_rule={"type": "keep_text_drop", "full_messages": full},
        ),
        False,
    )
    plain = chat._process_messages(request(messages=full), False)
    assert result.prompt_ids == plain.prompt_ids
    program = ContextProgram.from_wire(result.context_program, result.prompt_ids)
    assert not bool(program.layout.keep_mask[0])
    assert bool(program.layout.keep_mask[-1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_ids": [1, 2], "drop_message": {"1": [0]}},
        {"reposition": [True]},
        {"reposition": ["1"]},
        {"drop_rule": {"type": "thinking_drop"}, "drop_message": {}},
    ],
)
def test_invalid_context_request_is_not_ignored(chat, kwargs):
    with pytest.raises(ValueError):
        chat._process_messages(request(**kwargs), False)


@pytest.mark.parametrize(
    "model_path",
    [
        path
        for path in os.environ.get("CONTEXT_TOKENIZER_PATHS", "").split(os.pathsep)
        if path
    ]
    or [None],
)
def test_real_model_template_token_ids(chat, model_path):
    if model_path is None:
        pytest.skip("set CONTEXT_TOKENIZER_PATHS for mandatory-model tokenizers")
    from sglang.srt.context_system.planner import ContextProgram
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    chat.tokenizer_manager.tokenizer = tokenizer
    chat._tokenizer_auto_adds_specials = bool(tokenizer.encode(""))
    histories = [
        messages(),
        [
            {"role": "user", "content": "Find the answer"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Use a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":"test"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "old"},
            {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
            {"role": "user", "content": "new"},
        ],
    ]
    for history in histories:
        plain = chat._process_messages(request(messages=history), False)
        result = chat._process_messages(
            request(
                messages=history,
                drop_message={str(len(history) - 1): [0]},
                reposition=[len(history) - 1],
            ),
            False,
        )
        assert result.prompt_ids == plain.prompt_ids, Path(model_path).name
        program = ContextProgram.from_wire(result.context_program, result.prompt_ids)
        assert len(program.layout.drop_ranges) > 0
