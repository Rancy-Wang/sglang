# MIT License
#
# Copyright (c) 2026 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch


class DropEvictionCandidates:
    """Persistent leaf-first heaps with bounded lazy invalidation.

    The tree updates a candidate only when its locks, children, residency or
    recency change. Reclaim never scans the tree or reads device page indices.
    ``kind=0`` denotes a leaf and ``kind=1`` a proven Drop internal edge.
    """

    def __init__(self):
        self.heaps = ([], [])
        self.entries = {}
        self.version = 0

    def update(self, node, kind, priority=None):
        self.entries.pop(node.id, None)
        if kind is not None:
            self.version += 1
            entry = (priority, node.id, self.version, node)
            self.entries[node.id] = (kind, entry)
            heapq.heappush(self.heaps[kind], entry)
        if sum(map(len, self.heaps)) > 2 * len(self.entries) + 64:
            self.heaps = tuple(
                [entry for k, entry in self.entries.values() if k == kind]
                for kind in (0, 1)
            )
            for heap in self.heaps:
                heapq.heapify(heap)

    def pop(self):
        for kind, heap in enumerate(self.heaps):
            while heap:
                entry = heapq.heappop(heap)
                if self.entries.get(entry[1]) == (kind, entry):
                    del self.entries[entry[1]]
                    return kind, entry[-1]
        return None


def mask_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def proven_skip_ranges(
    matched_records: torch.Tensor, required_raw: torch.Tensor
) -> list[tuple[int, int]]:
    """Only a Delta on this matched path authorizes releasing an ancestor KV."""
    if len(matched_records) == 0:
        return []
    records = matched_records.numpy()
    if records.ndim != 2 or records.shape[1] != 4:
        return []
    real = records[:, 0] == 0
    raw_keys = np.flatnonzero(real)
    count = len(raw_keys)
    required = required_raw.numpy()
    if required.dtype != np.bool_ or len(required) != count:
        raise ValueError("Drop lock demand must cover the matched raw prefix.")
    dropped = np.zeros(count, dtype=np.bool_)
    raw_before = np.cumsum(real)
    for key in np.flatnonzero(records[:, 0] == 1):
        start, end = -int(records[key, 1]) - 1, -int(records[key, 2]) - 1
        if not 0 <= start < end <= int(raw_before[key]):
            raise ValueError("Matched Delta references a non-ancestor token range.")
        dropped[start:end] = True
    skip = np.zeros(len(records), dtype=np.bool_)
    skip[raw_keys] = dropped & ~required
    return mask_ranges(skip)


@dataclass(frozen=True)
class RecoveryPlan:
    # Continuous intervals use the existing prefill attention path. Resident
    # gaps between intervals are reused, including the matched suffix.
    intervals: tuple[tuple[int, int], ...]
    required_prefix: torch.Tensor
    matched_length: int
    # Initial resident tokens actually reused, excluding restored source versions.
    reusable_prefix: torch.Tensor

    @property
    def start(self) -> int:
        return self.intervals[0][0]

    def remaining_queries(self, cursor: int) -> int:
        return sum(
            end - max(start, cursor) for start, end in self.intervals if cursor < end
        )

    def next_interval(self, cursor: int) -> tuple[int, int]:
        for start, end in self.intervals:
            if cursor < end:
                return max(start, cursor), end
        raise ValueError("Recovery cursor is past the query stream.")


def plan_recovery(
    resident: torch.Tensor,
    visible_until: torch.Tensor,
    input_length: int,
    rewind_sources: torch.Tensor | None = None,
    incompatible_sources: torch.Tensor | None = None,
) -> RecoveryPlan:
    """Close missing KV dependencies backwards without scanning the Radix tree.

    A missing token t is needed iff a planned later query q < expiry[t]
    reads it. A reverse scan over missing tokens closes this relation: once a
    token is needed it becomes the earliest query for all preceding tokens.
    """
    if (
        resident.device.type != "cpu"
        or resident.dtype != torch.bool
        or resident.ndim != 1
    ):
        raise ValueError("Recovery residency must be a CPU bool vector.")
    if (
        visible_until.device.type != "cpu"
        or visible_until.dtype != torch.int32
        or visible_until.ndim != 1
    ):
        raise ValueError("Recovery visibility must be a CPU int32 vector.")
    for source_mask in (rewind_sources, incompatible_sources):
        if source_mask is not None and (
            source_mask.device.type != "cpu"
            or source_mask.dtype != torch.bool
            or source_mask.shape != resident.shape
        ):
            raise ValueError(
                "Recovery source masks must match the CPU residency vector."
            )
    present = resident.numpy()
    expiry = visible_until.numpy()
    matched = len(present)
    if not 0 <= matched < input_length or len(expiry) < input_length:
        raise ValueError("Recovery metadata must leave an uncached query suffix.")
    if np.any(expiry[:input_length] <= np.arange(input_length)):
        raise ValueError("A token must remain visible to its own birth query.")
    rewind = (
        np.zeros(matched, dtype=np.bool_)
        if rewind_sources is None
        else rewind_sources.numpy()
    )
    incompatible = (
        np.zeros(matched, dtype=np.bool_)
        if incompatible_sources is None
        else incompatible_sources.numpy()
    )
    suffix_demand = expiry[:matched] > matched
    if not np.any((~present | incompatible) & suffix_demand):
        # No suffix query needs an absent/version-incompatible KV. Rewind-only
        # sources are irrelevant until a historical query actually needs repair.
        return RecoveryPlan(
            ((matched, input_length),),
            torch.from_numpy(suffix_demand),
            matched,
            torch.from_numpy(present.copy()),
        )
    missing = np.flatnonzero(~present | rewind | incompatible)
    needed = np.zeros(input_length, dtype=np.bool_)
    needed[matched:] = True
    earliest = matched
    for raw in missing[::-1]:
        # Rebuilding an earlier query must not use a lossy inverse rotation of
        # a later-position source. Ordinary suffix queries still reuse it.
        if present[raw] and not incompatible[raw] and earliest == matched:
            continue
        if expiry[raw] > earliest:
            needed[raw] = True
            earliest = int(raw)
    # A resident token is held only when some planned query can read it.
    next_query = np.minimum.accumulate(
        np.where(needed, np.arange(input_length), input_length)[::-1]
    )[::-1]
    required = expiry[:matched] > next_query[1 : matched + 1]
    required |= needed[:matched]
    return RecoveryPlan(
        tuple(mask_ranges(needed)),
        torch.from_numpy(required),
        matched,
        torch.from_numpy(present & ~needed[:matched]),
    )
