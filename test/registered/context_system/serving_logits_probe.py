"""Opt-in numerical instrumentation, excluded from throughput measurements."""


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
