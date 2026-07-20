import atexit
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class _StageTimer:
    def __init__(self, profiler: "DDTreeProfiler", stage: str, use_cuda_event: bool):
        self.profiler = profiler
        self.stage = stage
        self.use_cuda_event = use_cuda_event
        self.start_ns = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        if self.use_cuda_event:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_cuda_event:
            assert self.end_event is not None
            self.end_event.record()
            self.profiler._pending_cuda_events.append(
                (self.stage, self.start_event, self.end_event)
            )

        elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1_000_000.0
        self.profiler._record_cpu_ms(self.stage, elapsed_ms)
        return False


class DDTreeProfiler:
    """Low-overhead, opt-in profiler for DFlash/DDTree speculative decode stages."""

    def __init__(
        self,
        *,
        enabled: bool,
        name: str,
        rank: int,
        warmup: int,
        interval: int,
        output: Optional[str],
    ):
        self.enabled = bool(enabled)
        self.name = name
        self.rank = int(rank)
        self.warmup = max(0, int(warmup))
        self.interval = max(1, int(interval))
        self.output = output

        self.round_idx = 0
        self._profiled_rounds_since_flush = 0
        self._last_flush_round_idx = 0

        self._cpu_sum_ms: Dict[str, float] = defaultdict(float)
        self._cpu_count: Dict[str, int] = defaultdict(int)
        self._cpu_max_ms: Dict[str, float] = defaultdict(float)
        self._gpu_sum_ms: Dict[str, float] = defaultdict(float)
        self._gpu_count: Dict[str, int] = defaultdict(int)
        self._gpu_max_ms: Dict[str, float] = defaultdict(float)
        self._meta_sum: Dict[str, float] = defaultdict(float)
        self._meta_count: Dict[str, int] = defaultdict(int)
        self._pending_cuda_events = []

        if self.enabled:
            atexit.register(self.flush)

    @classmethod
    def from_server_args(
        cls,
        server_args,
        *,
        name: str,
        rank: int,
    ) -> "DDTreeProfiler":
        return cls(
            enabled=bool(getattr(server_args, "speculative_ddtree_profile", False))
            and int(rank) == 0,
            name=name,
            rank=rank,
            warmup=int(getattr(server_args, "speculative_ddtree_profile_warmup", 0)),
            interval=int(
                getattr(server_args, "speculative_ddtree_profile_interval", 50)
            ),
            output=getattr(server_args, "speculative_ddtree_profile_output", None),
        )

    def _should_profile_current_round(self) -> bool:
        return self.enabled and self.round_idx >= self.warmup

    def cpu(self, stage: str):
        if not self._should_profile_current_round():
            return nullcontext()
        return _StageTimer(self, stage, use_cuda_event=False)

    def gpu(self, stage: str):
        if not self._should_profile_current_round():
            return nullcontext()
        use_cuda_event = torch.cuda.is_available()
        return _StageTimer(self, stage, use_cuda_event=use_cuda_event)

    def _record_cpu_ms(self, stage: str, elapsed_ms: float) -> None:
        self._cpu_sum_ms[stage] += elapsed_ms
        self._cpu_count[stage] += 1
        if elapsed_ms > self._cpu_max_ms[stage]:
            self._cpu_max_ms[stage] = elapsed_ms

    def _record_gpu_ms(self, stage: str, elapsed_ms: float) -> None:
        self._gpu_sum_ms[stage] += elapsed_ms
        self._gpu_count[stage] += 1
        if elapsed_ms > self._gpu_max_ms[stage]:
            self._gpu_max_ms[stage] = elapsed_ms

    def _drain_cuda_events(self) -> None:
        pending = self._pending_cuda_events
        if not pending:
            return
        self._pending_cuda_events = []
        for stage, start_event, end_event in pending:
            end_event.synchronize()
            self._record_gpu_ms(stage, float(start_event.elapsed_time(end_event)))

    def record_round(self, **metadata: Any) -> None:
        should_profile = self._should_profile_current_round()
        if should_profile:
            self._profiled_rounds_since_flush += 1
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    self._meta_sum[key] += float(value)
                    self._meta_count[key] += 1

        self.round_idx += 1

        if (
            self.enabled
            and should_profile
            and self._profiled_rounds_since_flush >= self.interval
        ):
            self.flush()

    def _build_record(self) -> Dict[str, Any]:
        self._drain_cuda_events()

        stages: Dict[str, Dict[str, float]] = {}
        for stage in sorted(set(self._cpu_count.keys()) | set(self._gpu_count.keys())):
            entry: Dict[str, float] = {}
            cpu_count = self._cpu_count.get(stage, 0)
            if cpu_count:
                entry["cpu_count"] = cpu_count
                entry["cpu_sum_ms"] = self._cpu_sum_ms[stage]
                entry["cpu_avg_ms"] = self._cpu_sum_ms[stage] / cpu_count
                entry["cpu_max_ms"] = self._cpu_max_ms[stage]
            gpu_count = self._gpu_count.get(stage, 0)
            if gpu_count:
                entry["gpu_count"] = gpu_count
                entry["gpu_sum_ms"] = self._gpu_sum_ms[stage]
                entry["gpu_avg_ms"] = self._gpu_sum_ms[stage] / gpu_count
                entry["gpu_max_ms"] = self._gpu_max_ms[stage]
            stages[stage] = entry

        meta = {
            key: {
                "count": count,
                "sum": self._meta_sum[key],
                "avg": self._meta_sum[key] / count,
            }
            for key, count in sorted(self._meta_count.items())
            if count > 0
        }

        return {
            "type": "sglang_speculative_tree_profile",
            "name": self.name,
            "rank": self.rank,
            "round_start": self._last_flush_round_idx,
            "round_end": self.round_idx,
            "profiled_rounds": self._profiled_rounds_since_flush,
            "warmup": self.warmup,
            "interval": self.interval,
            "meta": meta,
            "stages": stages,
        }

    def _reset_since_flush(self) -> None:
        self._last_flush_round_idx = self.round_idx
        self._profiled_rounds_since_flush = 0
        self._cpu_sum_ms.clear()
        self._cpu_count.clear()
        self._cpu_max_ms.clear()
        self._gpu_sum_ms.clear()
        self._gpu_count.clear()
        self._gpu_max_ms.clear()
        self._meta_sum.clear()
        self._meta_count.clear()

    def flush(self) -> None:
        if not self.enabled or self._profiled_rounds_since_flush <= 0:
            return

        record = self._build_record()
        self._emit_log(record)

        if self.output:
            output_dir = os.path.dirname(self.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(self.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

        self._reset_since_flush()

    def _emit_log(self, record: Dict[str, Any]) -> None:
        stages = record["stages"]
        top_cpu = sorted(
            (
                (stage, values.get("cpu_avg_ms", 0.0))
                for stage, values in stages.items()
                if "cpu_avg_ms" in values
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        top_gpu = sorted(
            (
                (stage, values.get("gpu_avg_ms", 0.0))
                for stage, values in stages.items()
                if "gpu_avg_ms" in values
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        meta = record["meta"]
        mean_accept_len = meta.get("mean_accept_len", {}).get("avg", None)
        output_tokens = meta.get("round_output_tokens", {}).get("avg", None)
        logger.info(
            "%s profile rounds=[%s,%s) profiled=%s mean_accept_len=%s output_tokens/round=%s top_cpu_ms=%s top_gpu_ms=%s",
            self.name,
            record["round_start"],
            record["round_end"],
            record["profiled_rounds"],
            f"{mean_accept_len:.3f}" if mean_accept_len is not None else "n/a",
            f"{output_tokens:.3f}" if output_tokens is not None else "n/a",
            ", ".join(f"{s}:{v:.3f}" for s, v in top_cpu),
            ", ".join(f"{s}:{v:.3f}" for s, v in top_gpu),
        )
