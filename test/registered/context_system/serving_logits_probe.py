"""Opt-in numerical instrumentation, excluded from throughput measurements."""


def create_decode_read_probe(config):
    """Audit actual Triton decode gathers and FP32 attention, outside timing runs.

    Single-request, page-one GPT-OSS diagnostics only. No production selector
    supplies the expected window. Save metrics, not a full historical KV dump.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from insert_state_audit import brute_decode_reads
    from sglang.srt.context_system.request_storage import request_row
    from sglang.srt.model_executor.forward_context import get_attn_backend

    model = Path(config["model"])
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = set(config.get("layers", [0, 1, 2, 3]))
    sinks_cache = {}

    def hook(layer, args, output):
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
        if req.context_program is None:
            pairs = [(i, i) for i in range(pos+1)]
            if window >= 0:
                pairs = pairs[-(window+1):]
        else:
            layout = req.context_program.layout
            pairs = brute_decode_reads(layout, pos-int(layout.next_position)+1, window)
        row = request_row(backend.req_to_token_pool, req.kv.req_pool_idx)
        raw = torch.tensor([i for i, _ in pairs], device=row.device, dtype=torch.long)
        wanted = row[raw]
        if window >= 0:
            wanted = backend.token_to_kv_pool.translate_loc_from_full_to_swa(wanted)
            indptr, indices = metadata.window_kv_indptr, metadata.window_kv_indices
        else:
            wanted = backend.kv_index_translator.translate_full_attn_ids(wanted)
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
        groups = layer.tp_q_head_num // kk.shape[1]
        # Explicit FP32 dot/softmax/reduction, with the model's attention sink.
        scores = torch.einsum("qhd,khd->hqk", qq, kk.repeat_interleave(groups, 1)) * layer.scaling
        scores = torch.cat((scores, sinks_cache[layer.layer_id][:, None, None]), -1)
        expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1)[..., :-1],
                                vv.repeat_interleave(groups, 1))
        error = (output.reshape_as(expected).float()-expected).abs()
        report = dict(layer=layer.layer_id, rank=rank, position=pos,
                      read_count=len(actual), expected_count=len(pairs), slots_ok=slots_ok,
                      max_abs=float(error.max()), mean_abs=float(error.mean()),
                      fp32_close=bool(torch.all(error <= .0078125 + .0078125*expected.abs())))
        destination = Path(params[0]["context_trace_path"]+f".reads.rank{rank}.jsonl")
        with destination.open("a") as target:
            target.write(json.dumps(report)+"\n")
        if not slots_ok:
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


def serialized_probe():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class Probe(CustomLogitProcessor):
        def __init__(self):
            import os

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
