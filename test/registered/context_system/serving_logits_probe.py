"""Opt-in numerical instrumentation, excluded from throughput measurements."""


def serialized_probe():
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    class Probe(CustomLogitProcessor):
        def __init__(self):
            self.rows = {}

        def __call__(self, logits, custom_param_list):
            import torch

            for i, params in enumerate(custom_param_list):
                req = params["__req__"]
                # Native chunk sampling is discarded; retain generation only.
                if req.extend_range.end < len(req.origin_input_ids):
                    continue
                key = params["context_trace_path"]
                rows = self.rows.setdefault(key, [])
                step = len(rows)
                rows.append(logits[i].detach().clone())
                forced = params.get("context_forced_tokens")
                if forced is not None:
                    logits[i].fill_(-float("inf"))
                    logits[i, forced[step]] = 0
                if len(rows) == params["context_trace_count"]:
                    # One transfer at request completion; no per-token D2H.
                    snapshot = torch.stack(rows).float().cpu()
                    if (
                        not torch.distributed.is_initialized()
                        or torch.distributed.get_rank() == 0
                    ):
                        torch.save(snapshot, key)
                    del self.rows[key]
            return logits

    return Probe.to_str()
