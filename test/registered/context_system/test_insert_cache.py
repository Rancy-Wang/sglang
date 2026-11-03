"""Insert records through native radix splitting, retrigger and publication."""

import os
import sys
from pathlib import Path

import pytest
import torch
from test_insert_ir import program

pytest_plugins = ("test_rules", "test_serving_runtime")
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT requires Linux")


def create_insert_kv_probe(config):
    """Opt-in RadixAttention forward hook; never enabled in timing runs.

    Use the native --forward-hooks API with target ``*.self_attn.attn`` and
    custom_params.context_kv_probe=true on a single-request diagnostic run.
    Disable graphs so every request, including exact hits, invokes the hook.
    """
    from sglang.srt.model_executor.forward_context import get_attn_backend

    counts = {}
    layers = set(config.get("layers", [0, 1, 2, 3]))

    def hook(layer, args, output):
        if layer.layer_id not in layers or len(args) < 4:
            return
        q, k, v, batch = args[:4]
        params = getattr(batch.sampling_info, "custom_params", None)
        if not params or len(params) != 1 or not params[0].get("context_kv_probe"):
            return
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        context = batch.context_attention
        window = layer.sliding_window_size
        is_swa = window is not None and window >= 0
        slots = metadata.swa_out_cache_loc if is_swa else None
        if slots is None:
            slots = backend.kv_index_translator.translate_full_attn_ids(batch.out_cache_loc)
        kp, vp = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        data = dict(
            input_ids=batch.input_ids.detach().cpu(), positions=batch.positions.detach().cpu(),
            q=q.detach().cpu(), k=k.detach().cpu(), v=v.detach().cpu(),
            slots=slots.detach().cpu(), stored_k=kp[slots.long()].detach().cpu(),
            stored_v=vp[slots.long()].detach().cpu(), output=output.detach().cpu(),
            window=window,
        )
        if context is not None:
            cm = context.for_layer(layer)
            for name in ("qo_indptr", "kv_indptr", "kv_indices", "query_positions", "kv_positions"):
                data[name] = getattr(cm, name).detach().cpu()
            data["prefix_k"] = kp[cm.kv_indices.long()].detach().cpu()
            data["prefix_v"] = vp[cm.kv_indices.long()].detach().cpu()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = params[0]["context_trace_path"]
        key = (path, layer.layer_id)
        n = counts.get(key, 0)
        counts[key] = n + 1
        torch.save(data, f"{path}.layer{layer.layer_id}.rank{rank}.{n}.pt")

    return hook


@pytest.mark.parametrize("swa", [False, True])
def test_insert_match_publication_and_branch_isolation(context_modules, monkeypatch, swa):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import reset_context
    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    components = (ComponentType.FULL, ComponentType.SWA) if swa else (ComponentType.FULL,)
    cache, allocator, _ = native.build_fixture(native.CacheConfig(components=components, kv_size=512, sliding_window_size=16 if swa else None))
    try:
        p = program(context_modules, nested=True, width=7)
        target = RadixKey.from_context(p.layout)
        # Only final TOKEN positions differ: this is precisely retrigger's
        # legal rotation case. Virtual history and birth-depth stay immutable.
        source = RadixKey.from_context(p.layout)
        from dataclasses import replace
        records = p.layout.records.clone()
        records[p.layout.token_to_key, 2] += 100
        source = RadixKey.from_context(replace(p.layout, records=records, positions=p.layout.positions + 100))
        saved_source = source.context.records[:]
        slots = allocator.alloc(len(source))
        cache.insert(InsertParams(key=source, value=slots))
        match = cache.match_prefix(MatchPrefixParams(key=target[:-1], context_retry=True))
        assert len(match.device_indices) == len(target)-1
        assert match.context_retry
        assert match.context_source_positions.tolist() == (p.layout.positions[:-1]+100).tolist()
        # A separate target allocation models the scheduler's mandatory COW.
        new_slots = allocator.alloc(len(target))
        cache.insert(InsertParams(key=target, value=new_slots))
        for key, expected in ((source, slots), (target, new_slots)):
            hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
            assert len(hit.device_indices) == len(key)
            assert not hit.context_retry
            assert torch.equal(hit.device_indices, expected)
        assert source.context.records == saved_source
        # End markers, including a prefix ending immediately after IE, survive
        # page-one edge splits and re-publication.
        for end in p.layout.insert_spans.reshape(-1, 3)[:, 1].tolist():
            hit = cache.match_prefix(MatchPrefixParams(key=target[:end], context_retry=True))
            assert len(hit.device_indices) == end
    finally:
        cache.evict(EvictParams(num_tokens=512, swa_num_tokens=512))
        assert allocator.available_size() == 512
        reset_context()


@pytest.mark.skipif(not os.environ.get("CONTEXT_SERVER_MODEL"), reason="isolated model server required")
def test_insert_http_summary_lifecycle(tmp_path, request):
    """Warm/retrigger, exact, nested later-turn insertion and cold restoration."""
    import requests
    from serving_logits_probe import serialized_probe
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark/context_system"))
    from insert_summary48k import logit_metrics, native_logit_bounds
    server = os.environ.get("CONTEXT_EXISTING_SERVER") or request.getfixturevalue("server")
    root = Path(os.environ.get("CONTEXT_TRACE_DIR", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": "Remember the conversation. Answer briefly."}]
    for i in range(1, 10):
        messages.append({"role": "user" if i % 2 else "assistant",
                             "content": f"Message {i}. " + "red blue green amber " * 24})
    counter = 0
    def call(ms, **ops):
        nonlocal counter
        counter += 1
        path = root / f"insert-{counter}.pt"
        payload = dict(model=os.environ["CONTEXT_SERVER_MODEL"], messages=ms, temperature=0,
                       max_tokens=1, ignore_eos=True, return_meta_info=True,
                       return_input_ids_in_sglext=True, return_output_ids_in_sglext=True,
                       custom_logit_processor=serialized_probe(), custom_params={
                           "context_trace_path": str(path), "context_trace_count": 1}, **ops)
        response = requests.post(server+"/v1/chat/completions", json=payload, timeout=600)
        (root / f"insert-{counter}.json").write_text(response.text)
        assert response.status_code == 200, response.text
        return response.json(), torch.load(path, weights_only=True)
    requests.post(server+"/flush_cache", timeout=30).raise_for_status()
    call(messages[:9])
    _, native_warm = call(messages)
    _, native_exact = call(messages)
    requests.post(server+"/flush_cache", timeout=30).raise_for_status()
    _, native_cold = call(messages)
    bounds = native_logit_bounds([logit_metrics(x, native_cold) for x in (native_warm, native_exact)])
    print("INSERT_NATIVE_BOUNDS", bounds, flush=True)
    requests.post(server+"/flush_cache", timeout=30).raise_for_status()
    call(messages[:9], drop_message={"3": [1, 2]}, reposition=[3])
    summary = {"index": -1, "role": "user", "content": "Summary: the earlier messages discussed red, blue, green and amber."}
    logical = messages[:6] + [summary] + messages[6:]
    ops = {"drop_message": {"3": [1, 2], "8": [1, 2, 3, 4, 5]}, "insert": {"9": -1}, "reposition": [3, -1]}
    warm, a = call(logical, **ops)
    exact, b = call(logical, **ops)
    requests.post(server+"/flush_cache", timeout=30).raise_for_status()
    _cold, c = call(logical, **ops)
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert exact["usage"]["prompt_tokens_details"]["cached_tokens"] >= warm["usage"]["prompt_tokens_details"]["cached_tokens"]
    for actual in (a, b):
        metrics = logit_metrics(actual, c)
        print("INSERT_LOGITS", metrics, flush=True)
        assert all(metrics[k] <= bounds[k] for k in bounds)
    # A second Insert targets the prior inserted token range by birth depth.
    logical = messages[:6] + [{"index": -2, "role": "user", "content": "Additional summary: remember amber."}, summary] + messages[6:]
    logical += [{"role": "assistant", "content": "Understood."}, {"role": "user", "content": "Which colors?"}]
    ops["insert"]["11"] = -2
    ops["reposition"].append(-2)
    later, d = call(logical, **ops)
    requests.post(server+"/flush_cache", timeout=30).raise_for_status()
    _, e = call(logical, **ops)
    metrics = logit_metrics(d, e)
    print("INSERT_LATER_LOGITS", metrics, flush=True)
    assert all(metrics[k] <= bounds[k] for k in bounds)
    assert later["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


@pytest.mark.skipif(not os.environ.get("CONTEXT_INSERT_BCP_OUTPUT"), reason="explicit ten-case BCP output required")
def test_insert_bcp48k(request, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmark/context_system"))
    from insert_summary48k import CASES, run
    run(SimpleNamespace(
        url=request.getfixturevalue("server"), model=os.environ["CONTEXT_SERVER_MODEL"],
        data=os.environ["CONTEXT_BCP_ROOT"], output=os.environ["CONTEXT_INSERT_BCP_OUTPUT"],
        cases=list(CASES), keep_recent=20000, summary_tokens=4096, timeout=1800,
        probe_tokens=int(os.environ.get("CONTEXT_INSERT_PROBE_TOKENS", "0")), skip_numeric=False,
    ))
