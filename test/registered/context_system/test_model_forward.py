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
        cuda_graph_config={
            "decode": {"bs": [1, 2], "max_bs": 2},
            "prefill": {"bs": [16, 32, 64, 96, 128, 256], "max_bs": 256},
        },
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
    runner.context_model_binding = binding
    print(
        "MODEL",
        args.model_path,
        "BACKENDS",
        runner.prefill_attention_backend_str,
        runner.decode_attention_backend_str,
        flush=True,
    )
    return wrapper, tokenizer, binding


def prepare_batch(
    runner, token_lists, *, programs=None, states=None, ends=None, usages=None
):
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
            context_program=programs[i].to_wire() if programs else None,
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        end = ends[i] if ends else len(tokens)
        if states is not None:
            req.prefix_indices = states[i].terminal_slots()
            req.context_state = states[i]
            req.context_usage = usages[i]
        req.set_extend_range(len(req.prefix_indices), end)
        if programs:
            req.plan_context_prefill(end)
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
    from unittest.mock import patch

    from sglang.srt.layers.attention.context_backend import ContextAttentionMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    fb = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    context_calls = 0
    original = ContextAttentionMetadata.forward

    def observe(metadata, *args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        return original(metadata, *args, **kwargs)

    with patch.object(ContextAttentionMetadata, "forward", observe):
        result = runner.forward(fb)
    if batch.context_prefill_input is not None:
        assert context_calls == len(runner.context_model_binding.layers)
    else:
        assert context_calls == 0
    print(
        "FORWARD_PATH",
        len(batch.input_ids),
        result.can_run_graph,
        context_calls,
        flush=True,
    )
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


@torch.no_grad()
def test_drop_reposition_real_model_chunk_lifetime(runtime):
    from sglang.srt.context_system.ir import compile_context_layout
    from sglang.srt.context_system.planner import ContextProgram
    from test_ir import args
    from test_occurrence_ownership import expiry_for

    wrapper, tokenizer, _ = runtime
    runner = wrapper.torch_runner
    tokens = tokenizer.encode(
        "The library contains red books and blue notebooks. The librarian records every visit. "
        * 12
    )[:96]
    assert len(tokens) == 96
    cuts = [17, 31, 57, 74, 96]

    def run(drops, repos, ends):
        layout = compile_context_layout(*args(tokens, drops, repos))
        program = ContextProgram(layout, expiry_for(len(tokens), drops))
        wrapper.clear()
        allocator = runner.token_to_kv_pool_allocator
        available_before = allocator.available_size()
        state = usage = None
        for end in ends:
            batch = prepare_batch(
                runner,
                [tokens],
                programs=[program],
                ends=[end],
                states=[state] if state is not None else None,
                usages=[usage] if usage is not None else None,
            )
            logits = forward(runner, batch)
            completed = torch.cuda.Event()
            completed.record()
            completed.synchronize()
            # Same receipts as BatchResultProcessor, including duplicate callback.
            for receipt in batch.copy().context_completions:
                receipt.complete(allocator)
                receipt.complete(allocator)
            req = batch.reqs[0]
            state, usage = req.context_state, req.context_usage
            runner.req_to_token_pool.free(req)
        assert usage.snapshot().actual_prefill_tokens == len(tokens)
        saved = []
        terminal = runner.kv_index_translator.translate_full_attn_ids(
            state.terminal_slots()
        )
        for layer in runner.context_model_binding.layers[:4]:
            saved.append(
                (layer.k_buffer[terminal].clone(), layer.v_buffer[terminal].clone())
            )
        allocator.free(state.private_slots())
        assert allocator.available_size() == available_before
        return logits, saved

    # Calibrate the native model's batch-shape rounding on the same token path.
    (plain_full, _), (plain_chunks, _) = run({}, [], [96]), run({}, [], cuts)
    baseline_error = (plain_full.float() - plain_chunks.float()).abs()
    drops, repos = {24: [(4, 12)], 56: [(16, 36)]}, [23, 55]
    (drop_full, _), (drop_chunks, _) = run(drops, [], [96]), run(drops, [], cuts)
    print(
        "DROP_ONLY_LOGITS",
        (drop_full.float() - drop_chunks.float()).abs().max().item(),
        flush=True,
    )
    (full, full_kv), (chunks, chunk_kv) = (
        run(drops, repos, [96]),
        run(drops, repos, cuts),
    )
    for layer, (expected_kv, actual_kv) in enumerate(zip(full_kv, chunk_kv)):
        for name, a, b in zip(("K", "V"), expected_kv, actual_kv):
            per_raw = (a.float() - b.float()).abs().flatten(1).max(1).values
            print(
                "CHUNK_KV",
                layer,
                name,
                {
                    "max": per_raw.max().item(),
                    "argmax_raw": per_raw.argmax().item(),
                    "per_raw": per_raw.tolist(),
                },
                flush=True,
            )
    error = (full.float() - chunks.float()).abs()
    maximum = max(0.04, 2 * baseline_error.max().item())
    mean = max(0.002, 2 * baseline_error.mean().item())
    print(
        "CHUNK_LOGITS",
        {
            "baseline_max": baseline_error.max().item(),
            "baseline_mean": baseline_error.mean().item(),
            "context_max": error.max().item(),
            "context_mean": error.mean().item(),
            "max_threshold": maximum,
            "mean_threshold": mean,
        },
        flush=True,
    )
    assert error.max().item() <= maximum
    assert error.mean().item() <= mean


@torch.no_grad()
def test_context_decode_native_graph_and_position_window(runtime):
    from sglang.srt.context_system.ir import compile_context_layout
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from test_ir import args
    from test_occurrence_ownership import expiry_for

    wrapper, tokenizer, _ = runtime
    runner = wrapper.torch_runner
    tokens = tokenizer.encode(
        "The library contains red books and blue notebooks. " * 40
    )[:256]
    drops = {60: [(4, 14)], 140: [(70, 90)]}
    layout = compile_context_layout(*args(tokens, drops, [59, 139]))
    program = ContextProgram(layout, expiry_for(len(tokens), drops))
    wrapper.clear()
    batch = prepare_batch(runner, [tokens], programs=[program])
    logits = forward(runner, batch)
    torch.cuda.synchronize()
    req = batch.reqs[0]
    for receipt in batch.context_completions:
        receipt.complete(runner.token_to_kv_pool_allocator)
    # The HTTP scheduler publishes this terminal row via cache_unfinished_req.
    terminal = req.context_state.terminal_slots()
    runner.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), terminal)
    generated = []
    for step in range(4):
        token = int(logits.argmax(-1)[0])
        generated.append(token)
        req.output_ids.append(token)
        batch.input_ids = torch.tensor([token], device=runner.device, dtype=torch.int64)
        batch.prepare_for_decode()
        fb = ForwardBatch.init_new(
            batch, runner, return_hidden_states_before_norm=False
        )
        assert int(fb.positions[0]) == layout.next_position + step
        assert int(fb.seq_lens[0]) == int(layout.keep_mask.sum()) + step + 1
        assert int(batch.seq_lens[0]) == len(tokens) + step + 1
        result = runner.forward(fb)
        assert result.can_run_graph, "Context decode must replay the native model graph"
        logits = result.logits_output.next_token_logits.clone()
        # Inspect the graph's actual read indices once, beyond GPT-OSS's SWA window.
        if step == 3:
            backend = runner.decode_attn_backend
            metadata = backend.forward_metadata
            active_raw = torch.cat(
                (
                    layout.keep_mask.nonzero().flatten(),
                    torch.arange(len(tokens), len(tokens) + 4),
                )
            ).to(runner.device)
            raw_slots = runner.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, active_raw
            ]
            expected = runner.kv_index_translator.translate_full_attn_ids(raw_slots)
            count = int(metadata.kv_indptr[1])
            torch.testing.assert_close(
                metadata.kv_indices[:count], expected.to(torch.int64)
            )
            if runner.sliding_window_size is not None:
                positions = torch.cat(
                    (
                        layout.positions[layout.keep_mask],
                        torch.arange(layout.next_position, layout.next_position + 4),
                    )
                ).to(runner.device)
                visible = (
                    positions >= layout.next_position + 3 - runner.sliding_window_size
                )
                swa = runner.kv_index_translator.sliding_window_write_loc_for(
                    expected[visible]
                )
                count = int(metadata.window_kv_indptr[1])
                torch.testing.assert_close(
                    metadata.window_kv_indices[:count], swa.to(torch.int64)
                )
                print(
                    "DECODE_SWA_COUNT",
                    count,
                    "WINDOW_DISTANCE",
                    runner.sliding_window_size,
                    flush=True,
                )
    torch.cuda.synchronize()
    print("DECODE_GENERATED", generated, "GRAPH_REPLAYS", 4, flush=True)
    assert torch.isfinite(logits).all()
    wrapper.clear()
