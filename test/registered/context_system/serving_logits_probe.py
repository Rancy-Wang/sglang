"""Opt-in numerical instrumentation, excluded from throughput measurements."""


def serialized_probe():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class Probe(CustomLogitProcessor):
        def __init__(self):
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
                    del params["_context_probe_rows"]
            return logits

    return Probe.to_str()
