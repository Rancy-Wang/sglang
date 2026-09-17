"""Request-lifetime Context accounting from completed full-attention read sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ContextUsageSnapshot:
    cached_tokens: int
    repos_tokens: int
    drop_skipped_tokens: int
    actual_prefill_tokens: int
    actual_decode_tokens: int


class ContextUsage:
    """Union cached reads across chunks; count actual computation with repeats.

    ``resident`` describes physical KV at initial match, excluding holes. Inputs
    are CPU metadata, never page-index tensors copied back from the GPU. Only
    full-attention demand contributes; SWA's window trimming does not constitute
    Drop-skipped. Cacheback copies and CUDA-graph padding are not model compute.
    """

    def __init__(self, resident: torch.Tensor, dropped: torch.Tensor):
        for value in (resident, dropped):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.bool
                or value.ndim != 1
            ):
                raise ValueError("Context usage requires CPU bool vectors")
        if resident.shape != dropped.shape:
            raise ValueError("Drop proof must cover the initial cache match")
        self.resident = resident.numpy().copy()
        self.dropped = dropped.numpy().copy()
        self.read = np.zeros(len(resident), dtype=np.bool_)
        self.transformed = np.zeros(len(resident), dtype=np.bool_)
        self.prefill_queries = 0
        self.decode_queries = 0
        self.recomputing = False
        self._cache_counts = None

    def begin_recompute(self) -> None:
        """Freeze initial cache provenance; subsequent work still costs tokens."""
        self.recomputing = True

    def record_prefill(
        self,
        read_raw: torch.Tensor,
        transformed_raw: torch.Tensor,
        query_count: int,
        *,
        initial_match: bool | None = None,
    ) -> None:
        """Call once after a forward completes, including any recovery queries."""
        if type(query_count) is not int or query_count < 1:
            raise ValueError("Completed prefill must contain actual model queries")
        if initial_match is None:
            initial_match = not self.recomputing
        if not initial_match:
            self.prefill_queries += query_count
            return
        n = len(self.resident)
        for value in (read_raw, transformed_raw):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.bool
                or value.ndim != 1
                or len(value) < n
            ):
                raise ValueError("Context read sets must cover the initial match")
        used = read_raw.numpy()[:n]
        self.read |= used
        # A scheduled transform that no query reads never contributes to repos.
        self.transformed |= transformed_raw.numpy()[:n] & used
        self.prefill_queries += query_count
        self._cache_counts = None

    def record_decode(self, query_count: int = 1) -> None:
        if type(query_count) is not int or query_count < 1:
            raise ValueError("Completed decode must contain actual model queries")
        self.decode_queries += query_count

    def snapshot(self) -> ContextUsageSnapshot:
        # Cache provenance changes only on prefill completion. Streaming decode
        # reports counters in O(1), without rescanning the historical prompt.
        if self._cache_counts is None:
            used = self.resident & self.read
            self._cache_counts = (
                int(np.count_nonzero(used & ~self.transformed)),
                int(np.count_nonzero(used & self.transformed)),
                int(np.count_nonzero(self.resident & self.dropped & ~self.read)),
            )
        return ContextUsageSnapshot(
            *self._cache_counts,
            actual_prefill_tokens=self.prefill_queries,
            actual_decode_tokens=self.decode_queries,
        )
