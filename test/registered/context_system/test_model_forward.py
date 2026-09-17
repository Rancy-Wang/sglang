"""Opt-in real-model forward checks, using native models and CUDA graphs.

This layer tests the ForwardBatch occurrence consumer. HTTP scheduler lifetime,
mini differential generation and PD are separate acceptance requirements.
"""

import os
from array import array

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_MODEL_PATH"), reason="requires an isolated model GPU"
)


@pytest.fixture(scope="module")
def runtime():
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    args = ServerArgs(
        model_path=os.environ["CONTEXT_MODEL_PATH"],
        page_size=1,
        max_total_tokens=4096,
        max_running_requests=8,
        context_length=2048,
        mem_fraction_static=0.75,
        attention_backend=os.environ.get("CONTEXT_TEST_ATTENTION_BACKEND"),
        cuda_graph_config={"decode": {"bs": [1, 2], "max_bs": 2}},
    )
    args.resolve_once()
    _set_envs_and_config(args)
    publish(args, role="scheduler")
    wrapper, tokenizer = load_model(args, PortArgs.init_new(args), 0, 0)
    runner = wrapper.torch_runner
    if runner.prefill_attention_backend_str != "triton":
        pytest.fail("This consumer test currently requires the Context Triton backend")
    binding = ContextModelBinding(
        runner.model, runner.token_to_kv_pool, runner.kv_index_translator, page_size=1
    )
    print(
        "MODEL",
        args.model_path,
        "BACKENDS",
        runner.prefill_attention_backend_str,
        runner.decode_attention_backend_str,
        flush=True,
    )
    return wrapper, tokenizer, binding


def prepare_batch(runner, token_lists):
    from sglang.benchmark.one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    reqs = []
    for i, tokens in enumerate(token_lists):
        req = Req(
            str(i),
            "",
            array("q", tokens),
            SamplingParams(temperature=0, max_new_tokens=8),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_range(0, len(tokens))
        reqs.append(req)
    cache = TreeCacheNamespace(
        page_size=1,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=cache,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
    batch.prefill_input_ids_cpu = None
    return batch


def forward(runner, batch):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    fb = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    result = runner.forward(fb)
    return result.logits_output.next_token_logits.clone()


@torch.no_grad()
def test_ordinary_context_consumer_matches_native_model(runtime):
    from sglang.srt.layers.attention.context_backend import (
        ContextAttentionPlan,
        ContextPrefillInput,
        ContextSequence,
    )

    wrapper, tokenizer, binding = runtime
    runner = wrapper.torch_runner
    tokens = [
        tokenizer.encode(text)
        for text in (
            "The library contains red books and blue notebooks. What color are the books?",
            "A machine adds two and three, then multiplies the sum by four. The result is",
        )
    ]
    wrapper.clear()
    plain = prepare_batch(runner, tokens)
    expected = forward(runner, plain)
    # Isolated allocator contents, same loaded model and token paths.
    wrapper.clear()
    context = prepare_batch(runner, tokens)
    context.context_prefill_input = ContextPrefillInput(
        ContextAttentionPlan.merge(
            [ContextSequence.ordinary(0, len(t)) for t in tokens]
        ),
        context.out_cache_loc,
        torch.empty(0, dtype=torch.int32, device=runner.device),
        torch.empty(0, dtype=torch.int32, device=runner.device),
        torch.empty(0, 2, dtype=torch.int32, device=runner.device),
        binding,
    )
    actual = forward(runner, context)
    error = (actual.float() - expected.float()).abs()
    print(
        "NO_EVENT_LOGITS",
        {"max_abs": error.max().item(), "mean_abs": error.mean().item()},
        flush=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
