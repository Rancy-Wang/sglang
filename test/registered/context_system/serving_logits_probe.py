"""Opt-in semantic/numerical instrumentation, excluded from throughput measurements."""


def independent_policy_decode_pairs(spec, generated, position, window):
    """Expected logical readers from the input event tape, never candidate IR."""
    from cache_policy_reference import simulate_compact

    if generated < 1:
        raise ValueError("Decode probe requires an already sampled input token")
    reference = simulate_compact(
        spec["token_count"] + generated,
        {int(k): v for k, v in spec["drops"].items()},
        spec["reposition"], policy=spec["policy"],
    )
    query = len(reference.raw) - 1
    if int(reference.compute_positions[query]) != position:
        raise AssertionError("Decode position differs from independent event replay")
    return reference.row(query, window)


def policy_structure_errors(program, spec):
    from cache_policy_reference import simulate_compact

    reference = simulate_compact(
        spec["token_count"], {int(k): v for k, v in spec["drops"].items()},
        spec["reposition"], policy=spec["policy"],
    )
    count = len(reference.raw)
    return [name for name, actual, expected in (
        ("projection", program.execution_to_full[:count].tolist(), reference.raw.tolist()),
        ("stored_positions", program.layout.positions[:count].tolist(), reference.positions.tolist()),
        ("compute_positions", program.layout.birth_positions[:count].tolist(), reference.compute_positions.tolist()),
        ("visibility", program.visible_until[:count].clamp(max=spec["token_count"] + 1).tolist(), reference.expiry.clip(max=spec["token_count"] + 1).tolist()),
        ("keep_mask", program.layout.keep_mask[:count].tolist(), (reference.expiry > spec["token_count"]).tolist()),
    ) if actual != expected]


def install_cache_policy_graph_probe():
    """Inspect buffers outside capture, immediately before native graph replay."""
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    if getattr(TritonAttnBackend, "_cache_policy_graph_probe", False):
        return
    original = TritonAttnBackend.init_forward_metadata_out_graph

    def metadata(backend, batch, in_capture=False):
        result = original(backend, batch, in_capture=in_capture)
        params = getattr(batch.sampling_info, "custom_params", None)
        if (in_capture or not batch.forward_mode.is_decode() or not params
                or len(params) != 1 or not params[0]
                or not params[0].get("context_graph_read_probe")):
            return result
        import json
        from pathlib import Path

        import torch
        from sglang.srt.context_system.request_storage import request_row

        param = params[0]
        generated = len(param.get("_context_probe_rows", []))
        if generated >= param["context_trace_count"]:
            return result  # Native overlap may execute a discarded final step.
        generated += param.get("context_forced_offset", 0)
        req = param["__req__"]
        position = int(batch.positions[0])
        assert generated >= 1
        assert int(batch.input_ids[0]) == param["context_forced_tokens"][generated - 1]
        row = request_row(backend.req_to_token_pool, req.kv.req_pool_idx)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        windows = [-1]
        if backend.sliding_window_size is not None and backend.sliding_window_size > 0:
            windows.append(backend.sliding_window_size)
        for window in windows:
            pairs = independent_policy_decode_pairs(param["context_policy_spec"], generated, position, window)
            raw = torch.tensor([i for i, _ in pairs], device=row.device, dtype=torch.long)
            wanted = row[raw]
            if window >= 0:
                wanted = backend.token_to_kv_pool.translate_loc_from_full_to_swa(wanted)
                indptr, indices = backend.window_kv_indptr, backend.cuda_graph_window_kv_indices
            else:
                wanted = backend.kv_index_translator.translate_full_attn_ids(wanted)
                indptr, indices = backend.kv_indptr, backend.cuda_graph_kv_indices
            actual = indices[int(indptr[0]):int(indptr[1])]
            report = {"rank": rank, "position": position, "generated": generated,
                      "window": window, "expected_count": len(pairs), "read_count": len(actual),
                      "slots_ok": torch.equal(actual, wanted), "source": "native-graph-replay-buffers"}
            with Path(param["context_trace_path"] + f".graph-reads.rank{rank}.jsonl").open("a") as out:
                out.write(json.dumps(report) + "\n")
            if not report["slots_ok"]:
                raise AssertionError(f"Graph physical reader mismatch: {report}")
        return result

    TritonAttnBackend.init_forward_metadata_out_graph = metadata
    TritonAttnBackend._cache_policy_graph_probe = True


def install_cache_policy_eviction_probe():
    """Explicit diagnostics may evict via the native API after releasing leases."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if getattr(UnifiedRadixCache, "_cache_policy_eviction_probe", False):
        return
    original = UnifiedRadixCache.cache_finished_req

    def finish(cache, req, *args, **kwargs):
        import dataclasses
        import json
        from pathlib import Path

        import torch

        params = req.sampling_params.custom_params or {}
        quota = params.get("context_cache_policy_evict_tokens", 0)
        program = req.context_program
        if quota and (type(quota) is not int or quota < 1 or program is None
                      or program.layout.cache_policy not in (1, 2)):
            raise ValueError("Eviction probe requires a new policy and positive quota")
        result = original(cache, req, *args, **kwargs)
        if quota:
            evicted = cache.evict(EvictParams(num_tokens=quota))
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                Path(params["context_trace_path"] + ".eviction.json").write_text(
                    json.dumps(dataclasses.asdict(evicted)) + "\n"
                )
        return result

    UnifiedRadixCache.cache_finished_req = finish
    UnifiedRadixCache._cache_policy_eviction_probe = True


def install_cache_policy_attention_probe():
    """Attach read-only inspection only for explicit new-policy diagnostics."""
    from sglang.srt.layers.radix_attention import RadixAttention

    if getattr(RadixAttention, "_cache_policy_attention_probe", False):
        return
    original = RadixAttention.forward
    hooks = {}

    def forward(layer, q, k, v, forward_batch, *args, **kwargs):
        batch = forward_batch
        output = original(layer, q, k, v, batch, *args, **kwargs)
        params = getattr(batch.sampling_info, "custom_params", None)
        if (params and len(params) == 1 and params[0]
                and params[0].get("context_policy_spec")
                and params[0].get("context_layer_probe")):
            if len(params[0].get("_context_probe_rows", [])) >= params[0]["context_trace_count"]:
                # Native overlap may execute a discarded extra sample after
                # the requested fixed continuation has already completed.
                return output
            model = params[0]["context_policy_model"]
            if model not in hooks:
                hooks[model] = create_decode_read_probe(
                    {"model": model, "layers": list(range(36))}
                )
            hooks[model](layer, (q, k, v, batch), output)
        return output

    RadixAttention.forward = forward
    RadixAttention._cache_policy_attention_probe = True


def decode_attention_fp32(q, k, v, scaling, sinks):
    import torch

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        groups = q.shape[1] // k.shape[1]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(groups, 1)) * scaling
        scores = torch.cat((scores, sinks.float()[:, None, None]), -1)
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1)[..., :-1],
                            v.float().repeat_interleave(groups, 1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def create_decode_read_probe(config):
    """Audit actual Triton decode gathers and FP32 attention, outside timing runs.

    Single-request, page-one GPT-OSS diagnostics only. No production selector
    supplies the expected window. Save metrics, not a full historical KV dump.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from sglang.srt.context_system.request_storage import request_row
    from sglang.srt.model_executor.forward_context import get_attn_backend

    model = Path(config["model"])
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = set(config.get("layers", [0, 1, 2, 3]))
    sinks_cache = {}
    read_cache, read_epoch = {}, None

    def hook(layer, args, output):
        nonlocal read_epoch
        if layer.layer_id not in layers or len(args) < 4:
            return
        q, k, v, batch = args[:4]
        params = getattr(batch.sampling_info, "custom_params", None)
        if (not batch.forward_mode.is_decode() or not params or len(params) != 1
                or not params[0].get("context_layer_probe")):
            return
        req = params[0]["__req__"]
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        window = layer.sliding_window_size
        window = -1 if window is None else window
        pos = int(batch.positions[0])
        epoch = (req.rid, pos)
        if epoch != read_epoch:
            read_cache.clear()
            read_epoch = epoch
        if window not in read_cache:
            if req.context_program is None:
                pairs = [(i, i) for i in range(pos+1)]
                if window >= 0:
                    # Native Triton treats window as total capacity. Keep its
                    # unchanged control distinct from Context's distance rule.
                    pairs = pairs[-window:] if window else []
            elif params[0].get("context_policy_spec"):
                # Sampler rows are committed in model execution order. The
                # scheduler's output_ids can lag with native overlap enabled.
                generated = len(params[0].get("_context_probe_rows", [])) + params[0].get("context_forced_offset", 0)
                forced = params[0]["context_forced_tokens"]
                if generated < 1 or forced[generated - 1] != int(batch.input_ids[0]):
                    raise AssertionError("Decode did not consume the frozen input")
                pairs = independent_policy_decode_pairs(
                    params[0]["context_policy_spec"], generated, pos, window
                )
            else:
                from insert_state_audit import brute_decode_reads

                layout = req.context_program.layout
                pairs = brute_decode_reads(layout, pos-int(layout.next_position)+1, window)
            row = request_row(backend.req_to_token_pool, req.kv.req_pool_idx)
            raw = torch.tensor([i for i, _ in pairs], device=row.device, dtype=torch.long)
            wanted = row[raw]
            if window >= 0:
                wanted = backend.token_to_kv_pool.translate_loc_from_full_to_swa(wanted)
            else:
                wanted = backend.kv_index_translator.translate_full_attn_ids(wanted)
            # One independent scan per query/pool, shared by all sampled layers.
            # The cache never outlives this query's read-only physical mapping.
            read_cache[window] = (len(pairs), wanted)
        expected_count, wanted = read_cache[window]
        if window >= 0:
            indptr, indices = metadata.window_kv_indptr, metadata.window_kv_indices
        else:
            indptr, indices = metadata.kv_indptr, metadata.kv_indices
        actual = indices[int(indptr[0]):int(indptr[1])]
        slots_ok = torch.equal(actual, wanted)
        kp, vp = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        kk, vv = kp[actual.long()].float(), vp[actual.long()].float()
        qq = q.reshape(1, layer.tp_q_head_num, layer.qk_head_dim).float()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if layer.layer_id not in sinks_cache:
            name = f"model.layers.{layer.layer_id}.self_attn.sinks"
            with safe_open(str(model / weights[name]), framework="pt", device="cpu") as source:
                sinks_cache[layer.layer_id] = source.get_tensor(name)[
                    rank*layer.tp_q_head_num:(rank+1)*layer.tp_q_head_num
                ].float().to(q.device)
        # Explicit FP32 dot/softmax/reduction, with the model's attention sink.
        expected = decode_attention_fp32(qq, kk, vv, layer.scaling, sinks_cache[layer.layer_id])
        error = (output.reshape_as(expected).float()-expected).abs()
        stored_k_ok = torch.equal(kp[actual[-1].long()].reshape_as(k[-1]), k[-1])
        stored_v_ok = torch.equal(vp[actual[-1].long()].reshape_as(v[-1]), v[-1])
        report = dict(layer=layer.layer_id, rank=rank, position=pos,
                      rule="native-capacity" if req.context_program is None else "context-inclusive-distance",
                      window=window,
                      read_count=len(actual), expected_count=expected_count, slots_ok=slots_ok,
                      stored_k_ok=stored_k_ok, stored_v_ok=stored_v_ok,
                      max_abs=float(error.max()), mean_abs=float(error.mean()),
                      fp32_close=bool(torch.all(error <= .0078125 + .0078125*expected.abs())))
        destination = Path(params[0]["context_trace_path"]+f".reads.rank{rank}.jsonl")
        with destination.open("a") as target:
            target.write(json.dumps(report)+"\n")
        if not slots_ok or not stored_k_ok or not stored_v_ok:
            raise AssertionError(f"Decode physical read oracle failed: {report}")

    return hook


def create_bounded_attention_probe(config):
    """Sample final query rows at each layer, never dump a long prefix KV pool.

    Register on ``*.self_attn.attn`` with native ``--forward-hooks``. Set
    custom_params.context_layer_probe=true and context_trace_path per request;
    disable CUDA graphs for the diagnostic run. Samples include each chunk,
    so compare the last extend call before decode, not a different chunk end.
    This localizes layer divergence; it is not itself a semantic KV oracle.
    """
    from pathlib import Path

    import torch

    layers = set(config["layers"]) if "layers" in config else None
    rows = int(config.get("rows", 4))
    if not 1 <= rows <= 32:
        raise ValueError("Layer probe rows must be between 1 and 32")
    counts = {}

    def hook(layer, args, output):
        if len(args) < 4 or (layers is not None and layer.layer_id not in layers):
            return
        q, k, v, batch = args[:4]
        params = getattr(batch.sampling_info, "custom_params", None)
        if not params or len(params) != 1 or not params[0].get("context_layer_probe"):
            return
        if not len(q):
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = params[0]["context_trace_path"]
        key = (path, layer.layer_id)
        call = counts.get(key, 0)
        counts[key] = call + 1
        sample = {"mode": str(batch.forward_mode), "query_count": len(q),
                  "layer": layer.layer_id, "rank": rank, "call": call,
                  "input_ids": batch.input_ids[-rows:].detach().cpu(),
                  "positions": batch.positions[-rows:].detach().cpu(),
                  "q": q[-rows:].detach().cpu(), "output": output[-rows:].detach().cpu()}
        for name, value in (("k", k), ("v", v)):
            if value is not None:
                sample[name] = value[-rows:].detach().cpu()
        destination = Path(f"{path}.layer{layer.layer_id}.rank{rank}.{call}.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample, destination)

    return hook


def rewind_retracted_probe(req):
    params = req.sampling_params.custom_params or {}
    rows = params.get("_context_probe_rows")
    if rows is not None:
        committed = len(req.output_ids) - params.get("context_forced_offset", 0)
        # Native overlap discards in-flight samples when the request retracts.
        del rows[max(0, committed) :]


def install_pd_retraction_barrier():
    """Batch two diagnostic transfers without changing native KV ownership."""
    import time

    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.disaggregation.decode import DecodeTransferQueue
    from sglang.srt.managers.schedule_batch import Req

    if getattr(DecodeTransferQueue, "_context_probe_barrier", False):
        return
    native_poll = DecodeTransferQueue._poll_with_metadata_gate
    native_reset = Req.reset_for_retract
    started, released = {}, set()

    def observed_reset(req):
        rewind_retracted_probe(req)
        return native_reset(req)

    def grouped_poll(queue):
        polls = native_poll(queue)
        groups = {}
        for i, item in enumerate(queue.queue):
            params = item.req.sampling_params.custom_params or {}
            group = params.get("context_retraction_group")
            if group and group not in released:
                groups.setdefault(group, []).append(i)
        for group, indices in groups.items():
            started.setdefault(group, time.monotonic())
            ready = len(indices) == 2 and all(
                polls[i] == KVPoll.Success for i in indices
            )
            failed = any(polls[i] == KVPoll.Failed for i in indices)
            if ready or failed or time.monotonic() - started[group] > 90:
                released.add(group)
                print("CONTEXT_RETRACTION_BARRIER", group, ready, flush=True)
            else:
                for i in indices:
                    if polls[i] == KVPoll.Success:
                        polls[i] = KVPoll.Transferring
        return polls

    DecodeTransferQueue._poll_with_metadata_gate = grouped_poll
    DecodeTransferQueue._context_probe_barrier = True
    Req.reset_for_retract = observed_reset


def serialized_probe(*, graph_metadata=False):
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class Probe(CustomLogitProcessor):
        def __init__(self):
            import os

            if graph_metadata:
                # Explicit diagnostics can add this read-only observer to an
                # already running graph server after a Git-synced probe update.
                import importlib

                import serving_logits_probe

                importlib.reload(serving_logits_probe).install_cache_policy_graph_probe()

            from serving_logits_probe import (
                install_cache_policy_attention_probe,
                install_cache_policy_eviction_probe,
            )

            install_cache_policy_eviction_probe()
            install_cache_policy_attention_probe()

            if os.environ.get("CONTEXT_PD_RETRACT") == "1":
                from serving_logits_probe import install_pd_retraction_barrier

                install_pd_retraction_barrier()
            self.rows = {}
            # Installed only by this explicit diagnostic processor. Preserve
            # raw model logits before the native grammar changes them in place.
            from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

            if not getattr(SamplingBatchInfo, "_context_probe_installed", False):
                native_bias = SamplingBatchInfo.apply_logits_bias

                def observed_bias(info, logits):
                    for i, params in enumerate(info.custom_params or []):
                        if params and "context_trace_path" in params:
                            params["_context_probe_raw_logits"] = (
                                logits[i].detach().clone()
                            )
                    return native_bias(info, logits)

                SamplingBatchInfo.apply_logits_bias = observed_bias
                SamplingBatchInfo._context_probe_installed = True

        def __call__(self, logits, custom_param_list):
            import torch

            for i, params in enumerate(custom_param_list):
                req = params["__req__"]
                # Native chunk sampling is discarded; retain generation only.
                if req.extend_range.end < len(req.origin_input_ids):
                    continue
                key = params["context_trace_path"]
                if params.get("context_cache_policy_probe") and not params.get("_cache_policy_recorded"):
                    import json
                    from pathlib import Path

                    from serving_logits_probe import policy_structure_errors

                    program = req.context_recompute_program or req.context_program
                    resident = req.context_resident
                    recovery = req.context_recovery_plan
                    matched = len(req.context_source_positions) if req.context_source_positions is not None else 0
                    errors = policy_structure_errors(program, params["context_policy_spec"]) if params.get("context_policy_spec") else None
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        Path(key + ".policy.json").write_text(json.dumps({
                            "policy": program.layout.cache_policy if program else 0,
                            "matched": matched,
                            "exact_prefix": req.context_exact_prefix_len,
                            "retry_tokens": max(0, matched-req.context_exact_prefix_len),
                            "resident": int(resident.sum()) if resident is not None else 0,
                            "holes": int((~resident).sum()) if resident is not None else 0,
                            "recovery_intervals": recovery.intervals if recovery else [],
                            "execution_tokens": len(program.input_ids) if program else len(req.origin_input_ids),
                            "structure_errors": errors,
                        }))
                    if errors:
                        raise AssertionError(f"Independent policy structure mismatch: {errors}")
                    params["_cache_policy_recorded"] = True
                reuse = getattr(req, "context_decode_reuse", None)
                if reuse is not None and not params.get("_context_reuse_recorded"):
                    import json
                    from pathlib import Path

                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        Path(key + ".reuse.json").write_text(json.dumps({
                            "active": len(reuse.reusable),
                            "reused": int(reuse.reusable.sum()),
                            "borrowed": int(reuse.borrowed.sum()),
                            "copied": len(reuse.copy_indices),
                            "missing": int((~reuse.reusable).sum()),
                        }))
                    params["_context_reuse_recorded"] = True
                rows = params.setdefault("_context_probe_rows", [])
                self.rows[key] = rows
                step = len(rows)
                if step >= params["context_trace_count"]:
                    self.rows.pop(key, None)
                    continue
                raw = params.pop("_context_probe_raw_logits", None)
                rows.append(raw if raw is not None else logits[i].detach().clone())
                forced = params.get("context_forced_tokens")
                if forced is not None:
                    logits[i].fill_(-float("inf"))
                    logits[i, forced[step + params.get("context_forced_offset", 0)]] = 0
                if len(rows) == params["context_trace_count"]:
                    # One transfer at request completion; no per-token D2H.
                    snapshot = torch.stack(rows).float().cpu()
                    if (
                        not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    ):
                        torch.save(snapshot, key)
                    del self.rows[key]
                    # Retain rows until native completion: an overlapped final
                    # sample may still be discarded by retraction and replayed.
            return logits

    return Probe.to_str()
