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
    import torch
    manager.server_args = ServerArgs(model_path="dummy", page_size=1, attention_backend="triton")
    manager.model_config.hf_config.architectures = ["Qwen2ForCausalLM"]
    manager.model_config.hf_config.dual_chunk_attention_config = None
    manager.model_config._resolved_model_arch = "Qwen2ForCausalLM"
    manager.model_config.dtype = torch.bfloat16
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


@pytest.mark.parametrize(
    "reported", [None,
                 {"cached_tokens": 0, "repos_tokens": 0, "drop_skipped_tokens": 0, "actual_prefill_tokens": 11, "actual_decode_tokens": 3},
                 {"cached_tokens": 12, "repos_tokens": 20, "drop_skipped_tokens": 30, "actual_prefill_tokens": 38, "actual_decode_tokens": 1}]
)
@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
def test_context_stream_reports_native_usage(chat, reported, include_usage, continuous):
    import asyncio
    import json
    from unittest.mock import Mock

    async def generate(*args):
        yield {
            "text": "answer",
            "meta_info": {
                "id": "context-stream",
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "cached_tokens": 70,
                "finish_reason": {"type": "length", "length": 2},
                "context_usage": reported,
            },
        }

    async def collect():
        return [
            chunk
            async for chunk in chat._generate_chat_stream(
                Mock(),
                request(stream=True, stream_options={"include_usage": include_usage,
                                                     "continuous_usage_stats": continuous}),
                None,
            )
        ]

    chat.tokenizer_manager.generate_request = generate
    chunks = asyncio.run(collect())
    events = [json.loads(chunk[6:]) for chunk in chunks if chunk != "data: [DONE]\n\n"]
    assert all("context_usage" not in (event.get("sglext") or {}) for event in events)
    terminal = [event["usage"] for event in events
                if not event["choices"] and event.get("usage") is not None]
    assert len(terminal) == int(include_usage or reported is not None)
    for event in events:
        if event.get("usage") is not None:
            details = event["usage"]["prompt_tokens_details"]
            if reported is not None:
                assert details == {key: reported[key] for key in
                                   ("cached_tokens", "repos_tokens", "drop_skipped_tokens")}
            else:
                assert details is None


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


@pytest.mark.parametrize(
    "model_path",
    [
        path
        for path in os.environ.get("CONTEXT_TOKENIZER_PATHS", "").split(os.pathsep)
        if path
    ]
    or [None],
)
def test_real_thinking_retention_and_exact_drop(chat, model_path):
    if model_path is None:
        pytest.skip("mandatory-model tokenizer paths required")
    from sglang.srt.context_system.planner import ContextProgram
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    original_template = tokenizer.chat_template
    chat.tokenizer_manager.tokenizer = tokenizer
    chat._tokenizer_auto_adds_specials = bool(tokenizer.encode(""))
    history = [
        {"role": "user", "content": "Find the answer"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "First investigate the source.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"test"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "source answer"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "Now verify the answer.",
        },
        {"role": "user", "content": "Explain the answer"},
    ]
    kwargs = {"preserve_thinking_history": True}
    retained = chat._process_messages(
        request(messages=history, chat_template_kwargs=kwargs), False
    )
    dropped = chat._process_messages(
        request(messages=history, drop_rule={"type": "thinking_drop"}), False
    )
    assert retained.prompt_ids == dropped.prompt_ids
    text = tokenizer.decode(retained.prompt_ids)
    for message in history:
        reasoning = message.get("reasoning_content")
        if reasoning:
            assert text.count(reasoning) == 1, (Path(model_path).name, text)
    program = ContextProgram.from_wire(dropped.context_program, dropped.prompt_ids)
    assert not program.layout.keep_mask.all()
    assert tokenizer.chat_template == original_template
    # A retention preference must not modify messages with no reasoning.
    plain = chat._process_messages(request(), False)
    kept = chat._process_messages(request(chat_template_kwargs=kwargs), False)
    assert plain.prompt_ids == kept.prompt_ids
    # Frozen upstream can consume the same native-template adapter through
    # --chat-template, without any modified Python preprocessing code.
    from sglang.srt.context_system.thinking_template import retained_template

    template, family = retained_template(original_template)
    assert retained_template(template) == (template, family)
    tokenizer.chat_template = template
    with patch(
        "sglang.srt.context_system.thinking_template.prepare_thinking_history",
        side_effect=lambda tokenizer, messages, tools, kwargs: (messages, kwargs),
    ):
        upstream_render = chat._process_messages(
            request(messages=history, chat_template_kwargs=kwargs), False
        )
    assert upstream_render.prompt_ids == retained.prompt_ids


@pytest.mark.parametrize("repos", [None, [1]])
def test_req_decode_key_and_native_cache_lifecycle(chat, repos):
    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import match_prefix_for_req
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.sampling.sampling_params import SamplingParams

    result = chat._process_messages(
        request(drop_message={"1": [0]}, reposition=repos), False
    )

    def make_req(rid):
        return Req(
            rid,
            "",
            array("q", result.prompt_ids),
            SamplingParams(temperature=0, max_new_tokens=8),
            context_program=result.context_program,
        )

    req = make_req("writer")
    program = req.context_program
    prompt_len = len(req.origin_input_ids)
    prompt_key = req.make_prefix_key(req.origin_input_ids)
    prompt_hash = prompt_key.hash_page(0, len(prompt_key))
    req.output_ids.extend((1, 2, 3))
    req._refresh_fill_ids()
    # The final sampled token does not yet own computed KV.
    computed_len = prompt_len + 2
    key = req.make_prefix_key(req.full_untruncated_fill_ids, limit=computed_len)
    assert len(key) == computed_len
    assert len(req.context_key_data.positions) == computed_len
    assert list(req.context_key_data.positions[-2:]) == [
        program.layout.next_position,
        program.layout.next_position + 1,
    ]
    assert prompt_key.hash_page(0, len(prompt_key)) == prompt_hash
    req.reset_for_retract()
    assert req.context_program is program
    assert req.context_key_data is key.context
    assert req.context_source_positions is None

    allocator = TokenToKVPoolAllocator(
        size=256, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
    )
    pool = ReqToTokenPool(4, 128, "cpu", False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    pool.alloc([req])
    slots = allocator.alloc(computed_len)
    pool.write((req.kv.req_pool_idx, slice(0, computed_len)), slots.to(torch.int32))
    req.kv.kv_committed_len = computed_len
    req.last_node = cache.root_node_handle()
    cache.cache_finished_req(req, kv_len_to_handle=computed_len)
    assert (
        cache.match_prefix(MatchPrefixParams(key=key)).device_indices.tolist()
        == slots.tolist()
    )
    assert allocator.available_size() == 256 - computed_len

    reader = make_req("reader")
    matched = match_prefix_for_req(cache, reader)
    assert len(matched.device_indices) == prompt_len
    reader.init_next_round_input(cache)
    assert len(reader.prefix_indices) == prompt_len - 1
    assert reader.context_source_positions is not None
    # No request keeps a cache lock after finished insertion or read-only match.
    cache.evict(EvictParams(num_tokens=256))
    assert allocator.available_size() == 256
    assert len(torch.unique(allocator.get_all_free_pages())) == 256
    cache.sanity_check()


def test_retract_rebuilds_generated_occurrences_and_retains_usage(chat):
    import torch
    from sglang.srt.context_system.usage import ContextUsage
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    result = chat._process_messages(
        request(drop_message={"1": [0]}, reposition=[1]), False
    )
    req = Req(
        "retract",
        "",
        array("q", result.prompt_ids),
        SamplingParams(max_new_tokens=8),
        context_program=result.context_program,
    )
    req.context_usage = ContextUsage(
        torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    )
    req.context_usage.record_prefill(
        torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 5
    )
    req.context_usage.record_decode(2)
    usage, original = req.context_usage, req.context_program
    req.output_ids.extend((2, 3, 4))
    req.reset_for_retract()
    req.init_next_round_input()
    end = len(req.origin_input_ids) + 3
    req.plan_context_prefill(end)
    assert req.context_usage is usage
    assert req.context_program is original
    expanded = req.context_recompute_program
    assert expanded.layout.positions[-3:].tolist() == list(
        range(original.layout.next_position, original.layout.next_position + 3)
    )
    assert (
        expanded.visible_until[original.layout.keep_mask.tolist() + [True] * 3].min()
        == torch.iinfo(torch.int32).max
    )
    window, plan = req.context_window_plan
    assert int(window.segment_query_ends[-1]) == end
    usage.record_prefill(plan.read_cached, plan.repositioned_cached, end)
    assert usage.snapshot().actual_prefill_tokens == end + 5


def test_context_admission_limits_preserve_native_tp_and_overlap(chat):
    from types import SimpleNamespace

    import torch
    from sglang.srt.context_system.capabilities import validate_context_request
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import ServerArgs

    processed = chat._process_messages(request(drop_message={"1": [0]}), False)
    obj = GenerateReqInput(
        input_ids=processed.prompt_ids, context_program=processed.context_program
    )
    obj.normalize_batch_and_arguments()
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        is_multimodal=False,
        dtype=torch.bfloat16,
    )
    args = ServerArgs(
        model_path="dummy", page_size=1, attention_backend="triton", tp_size=2
    )
    validate_context_request(args, config, obj)
    for field, value, error in (
        ("page_size", 16, "page_size=1"),
        ("attention_backend", "torch_native", "Triton"),
        ("attention_backend", "fa4", "validated backend"),
        ("attention_backend", "trtllm_mha", "validated backend"),
        ("enable_hierarchical_cache", True, "hierarchical"),
        ("speculative_algorithm", "EAGLE", "non-speculative"),
    ):
        previous = getattr(args, field)
        setattr(args, field, value)
        with pytest.raises(ValueError, match=error):
            validate_context_request(args, config, obj)
        setattr(args, field, previous)
    for mode in ("prefill", "decode"):
        args.disaggregation_mode = mode
        for backup in (None, "cpu_tensor", "host_pool"):
            args.disaggregation_decode_retraction_backup = backup
            validate_context_request(args, config, obj)
        args.disaggregation_decode_enable_radix_cache = True
        validate_context_request(args, config, obj)
        args.disaggregation_decode_enable_radix_cache = False


def test_context_usage_streams_scalar_snapshot_in_mixed_batch():
    from types import SimpleNamespace

    import torch
    from sglang.srt.context_system.usage import ContextUsage
    from sglang.srt.managers.io_struct import unwrap_from_pickle
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    fixture = load_file(
        "context_native_output_fixture",
        ROOT / "test/registered/unit/managers/test_output_streamer_customized_info.py",
    )
    context = fixture._FakeReq("context", [11, 12], finished=True)
    context.context_usage = ContextUsage(
        torch.ones(2, dtype=torch.bool), torch.tensor([False, True])
    )
    context.context_usage.record_prefill(
        torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool), 3
    )
    context.context_usage.record_decode()
    accumulator = fixture._accumulator()
    for req in (
        fixture._FakeReq("before", [9], finished=True),
        context,
        fixture._FakeReq("after", [10], finished=True),
    ):
        accumulator.accept(req=req)
    payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
    values = unwrap_from_pickle(payload.customized_info)
    manager = TokenizerManager.__new__(TokenizerManager)
    state = SimpleNamespace(customized_info_accumulated={})
    for index in range(3):
        meta = {}
        manager.update_request_meta_info(meta, state, values, index, {"type": "length"})
        if index == 1:
            assert meta["context_usage"] == {
                "cached_tokens": 1,
                "repos_tokens": 0,
                "drop_skipped_tokens": 1,
                "actual_prefill_tokens": 3,
                "actual_decode_tokens": 1,
            }
        else:
            assert "context_usage" not in meta
    assert state.customized_info_accumulated == {}


@pytest.mark.parametrize("fields", [{"reposition": []}, {"drop_message": None}])
def test_explicit_empty_feature_keeps_program_and_prompt(chat, fields):
    plain = chat._process_messages(request(), False)
    feature = chat._process_messages(request(**fields), False)
    assert feature.prompt_ids == plain.prompt_ids
    assert feature.context_program is not None


def test_unsupported_model_rejected_before_template_compilation(chat):
    chat.tokenizer_manager.model_config._resolved_model_arch = "LlamaForCausalLM"
    with patch("sglang.srt.context_system.provenance.build_template_token_provenance", side_effect=AssertionError("compiled unsupported model")):
        assert chat._process_messages(request(), False).context_program is None
        with pytest.raises(ValueError, match="not supported by Llama"):
            chat._process_messages(request(reposition=[]), False)


def test_nonstream_reports_zero_context_usage_without_meta_flag(chat):
    values = {"cached_tokens": 0, "repos_tokens": 0, "drop_skipped_tokens": 0,
              "actual_prefill_tokens": 4, "actual_decode_tokens": 1}
    ret = [{"text": "answer", "meta_info": {"id": "context-zero", "weight_version": "test", "prompt_tokens": 4,
            "completion_tokens": 1, "cached_tokens": 0,
            "finish_reason": {"type": "length", "length": 1}, "context_usage": values}}]
    response = chat._build_chat_response(request(), ret, 0)
    assert response.sglext is None
    assert response.usage.prompt_tokens_details.model_dump() == {
        "cached_tokens": 0, "repos_tokens": 0, "drop_skipped_tokens": 0,
    }


@pytest.mark.parametrize("cache_report", [False, True])
def test_native_usage_context_aggregation_counts_prompt_once(cache_report):
    from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

    contexts = [dict(cached_tokens=7, repos_tokens=11, drop_skipped_tokens=13),
                dict(cached_tokens=70, repos_tokens=110, drop_skipped_tokens=130),
                dict(cached_tokens=2, repos_tokens=3, drop_skipped_tokens=5),
                dict(cached_tokens=20, repos_tokens=30, drop_skipped_tokens=50)]
    # Two prompts, two choices each. D's native cache count must not replace P's
    # Context snapshot, nor may prompt counters be summed across choices.
    responses = [{"meta_info": dict(prompt_tokens=40, completion_tokens=i + 1,
                                    cached_tokens=999, context_usage=context)}
                 for i, context in enumerate(contexts)]
    ordinary = UsageProcessor.calculate_response_usage(
        responses, n_choices=2, enable_cache_report=cache_report)
    streamed = UsageProcessor.calculate_streaming_usage(
        {i: 40 for i in range(4)}, {}, {i: i + 1 for i in range(4)},
        {i: 999 for i in range(4)}, n_choices=2, enable_cache_report=cache_report,
        context_usage=dict(enumerate(contexts)))
    assert ordinary.model_dump() == streamed.model_dump()
    assert ordinary.prompt_tokens == 80 and ordinary.completion_tokens == 10
    assert ordinary.prompt_tokens_details.model_dump() == {
        "cached_tokens": 9, "repos_tokens": 14, "drop_skipped_tokens": 18,
    }
    for response in responses:
        del response["meta_info"]["context_usage"]
    native = UsageProcessor.calculate_response_usage(
        responses, n_choices=2, enable_cache_report=cache_report)
    assert (native.prompt_tokens_details.model_dump() if native.prompt_tokens_details else None) == (
        {"cached_tokens": 1998} if cache_report else None)


@pytest.mark.parametrize("architecture", ["QWenLMHeadModel", "Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "GptOssForCausalLM"])
def test_context_model_permissions_are_checked_per_request(chat, architecture):
    chat.tokenizer_manager.model_config._resolved_model_arch = architecture
    chat.tokenizer_manager.model_config.hf_config.architectures = [architecture]
    result = chat._process_messages(request(drop_message={"1": [0]}, reposition=[1]), False)
    assert result.context_program is not None
    assert result.prompt_ids


def test_slow_tokenizer_exact_bytes_preserve_unicode_boundaries():
    from types import SimpleNamespace
    from sglang.srt.context_system.provenance import _encode_with_offsets, append_assistant_prefix, TemplateTokenProvenance
    tokenizer = SimpleNamespace(is_fast=False, bos_token_id=None,
        encode=lambda text, **_: list(text.encode("utf-8")),
        tokenizer=SimpleNamespace(decode_single_token_bytes=lambda value: bytes([value])))
    ids, offsets = _encode_with_offsets(tokenizer, "你a", add_special_tokens=False)
    assert ids == list("你a".encode("utf-8"))
    assert offsets == [(0, 1), (0, 1), (0, 1), (1, 2)]
    trace = TemplateTokenProvenance([], [], [], "", [], 0)
    extended = append_assistant_prefix(trace, tokenizer, "你a", owner=2)
    assert extended.input_ids == ids and extended.offsets == offsets
    assert extended.owners == [2] * 4
    tokenizer.tokenizer.decode_single_token_bytes = lambda _: b"bad"
    with pytest.raises(RuntimeError, match="bytes do not reproduce"):
        _encode_with_offsets(tokenizer, "你a", add_special_tokens=False)


def test_compiler_warmup_failure_leaves_ordinary_requests_available(chat):
    chat.tokenizer_manager.context_warmup_error = "compiler unavailable"
    assert chat._process_messages(request(), False).context_program is None
    with patch("sglang.srt.context_system.provenance.build_template_token_provenance",
               side_effect=AssertionError("unavailable compiler must be rejected before rendering")):
        with pytest.raises(ValueError, match="compiler unavailable"):
            chat._process_messages(request(reposition=[]), False)


@pytest.mark.parametrize("continue_final", [False, True])
def test_native_chatml_context_matches_original_prompt(chat, continue_final):
    from sglang.srt.context_system.planner import ContextProgram

    chat.template_manager.chat_template_name = "chatml"
    history = [{"role": "system", "content": "Be concise."},
               {"role": "user", "content": "你你好 repeated repeated"},
               {"role": "assistant", "content": "hello"}]
    native = chat._process_messages(request(messages=history, continue_final_message=continue_final), False)
    context = chat._process_messages(request(messages=history, continue_final_message=continue_final,
                                             drop_message={"2": [1]}, reposition=[2]), False)
    assert context.prompt_ids == native.prompt_ids
    program = ContextProgram.from_wire(context.context_program, context.prompt_ids)
    assert not program.layout.keep_mask.all()


@pytest.mark.parametrize("preserve_thinking", [False, True])
def test_harmony_tools_share_exact_provenance(preserve_thinking):
    pytest.importorskip("openai_harmony")
    from sglang.srt.context_system.provenance import TemplateTokenProvenance
    from sglang.srt.parser.gpt_oss_encoding import HarmonyEncoder

    messages = [
        {"role": "system", "content": "Answer with evidence."},
        {"role": "user", "content": "查天气😀"},
        {"role": "assistant", "content": "", "reasoning_content": "先查工具",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "lookup", "arguments": '{"city":"北京"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "晴天"},
        {"role": "assistant", "content": "晴天。"},
        {"role": "user", "content": "谢谢"},
    ]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    encoder = HarmonyEncoder(preserve_thinking=preserve_thinking)
    ids, owners, _ = encoder.render_tokens(messages, tools=tools)
    trace = encoder.render(messages, tools=tools)
    assert isinstance(trace, TemplateTokenProvenance)
    assert trace.input_ids == ids and trace.owners == owners
    assert len(trace.offsets) == len(ids)
    assert len(trace.char_owners) == len(trace.rendered_text)
    for owner, text in [(1, "查天气😀"), (3, "晴天"), (5, "谢谢")]:
        start = trace.rendered_text.index(text)
        assert trace.char_owners[start:start + len(text)] == [owner] * len(text)
    assert ("先查工具" in trace.rendered_text) is preserve_thinking
