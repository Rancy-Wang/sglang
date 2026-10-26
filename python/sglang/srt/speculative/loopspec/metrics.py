"""LoopSpec counters use actual proposals/verifications, not reserved KV width."""


class LoopSpecMetrics:
    def __init__(self, model_name, *, registry=None):
        # Import only after SGLang initializes PROMETHEUS_MULTIPROC_DIR.
        from prometheus_client import Counter, Histogram, REGISTRY

        registry = REGISTRY if registry is None else registry
        self.labels = {"model_name": model_name}
        self.events = Counter(
            "sglang:loopspec_events",
            "Actual LoopSpec pipeline events.",
            ["model_name", "event"],
            registry=registry,
        )
        self.graphs = Counter(
            "sglang:loopspec_graph_replays",
            "LoopSpec CUDA graph replays by role.",
            ["model_name", "role"],
            registry=registry,
        )
        self.peak_slots = Histogram(
            "sglang:loopspec_peak_physical_timeline_slots",
            "Peak live role-depth-token slots per pipeline request, including its prefix.",
            ["model_name"],
            buckets=(16, 64, 256, 1024, 4096, 16384, 65536),
            registry=registry,
        )

    def finish(self, executor, pipeline, graphs, natural_stop):
        counts = {
            "pipeline_requests": 1,
            "matched_eos_requests" if natural_stop else "other_finished_requests": 1,
            "verified_tokens": sum(executor.accepted) + executor.rejected,
            "accepted_first": executor.accepted[0],
            "accepted_second": executor.accepted[1],
            "residual_tokens": executor.rejected,
            "proposed_first": executor.proposed[0],
            "proposed_second": executor.proposed[1],
            "cycles": pipeline.cycles,
            "reused_states": pipeline.reused,
            "restarted_states": pipeline.restarts,
        }
        for event, count in counts.items():
            self.events.labels(**self.labels, event=event).inc(count)
        for role, count in graphs.items():
            self.graphs.labels(**self.labels, role=role).inc(count)
        self.peak_slots.labels(**self.labels).observe(executor.peak_slots)
