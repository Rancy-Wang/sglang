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
        ("attention_backend", "flashinfer", "Triton"),
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
        with pytest.raises(ValueError, match="radix_cache"):
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
