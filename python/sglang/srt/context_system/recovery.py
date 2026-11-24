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
    """Persistent Drop-first heaps with bounded lazy invalidation.

    The tree updates a candidate only when its locks, children, residency or
    recency change. Reclaim never scans the tree or reads device page indices.
    ``kind=0`` denotes an ordinary leaf and ``kind=1`` a proven Drop edge,
    including a Drop edge that has become a leaf. ``kind=2`` is a B historical
    edge: still reclaimable, but competes with ordinary leaves by priority.
    """

    def __init__(self):
        self.heaps = ([], [], [])
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
                for kind in (0, 1, 2)
            )
            for heap in self.heaps:
                heapq.heapify(heap)

    def pop(self):
        for kind in (0, 1, 2):
            heap = self.heaps[kind]
            while heap and self.entries.get(heap[0][1]) != (kind, heap[0]):
                heapq.heappop(heap)
        if self.heaps[1]:
            kind = 1
        else:
            candidates = [k for k in (0, 2) if self.heaps[k]]
            if not candidates:
                return None
            kind = min(candidates, key=lambda k: self.heaps[k][0][:3])
        entry = heapq.heappop(self.heaps[kind])
        del self.entries[entry[1]]
        return kind, entry[-1]


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
    if records.ndim != 2 or records.shape[1] != 3:
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


def _insert_window(layout, expiry, start=0):
    from .occurrence import compile_occurrence_window
    return compile_occurrence_window(layout, expiry, layout.positions,
                                     query_start=start, query_end=len(expiry))


def _history_layout(layout):
    """B's single fixed-position stage; other layouts keep the general oracle."""
    return (
        layout.cache_policy == 2
        and not layout.insert_spans.numel()
        and not layout.logical_order.numel()
        and not layout.transition_raw_tokens.numel()
        and torch.equal(layout.birth_positions, layout.positions)
    )


def _history_read_plan(expiry, selected, resident, *, recover):
    from sglang.kernels.ops.attention.cache_policy_plan import load_cache_policy_plan

    full = torch.empty(len(resident), dtype=torch.bool)
    load_cache_policy_plan().history_recovery(
        expiry.contiguous(), resident.contiguous(), selected, full, recover,
    )
    return full


def _insert_read_mask(layout, expiry, intervals, prefix_length):
    """Union shared-page readers in occurrence order, without a dense mask."""
    full = np.zeros(prefix_length, dtype=np.bool_)
    if not intervals or not prefix_length:
        return torch.from_numpy(full)
    selected = np.zeros(len(expiry), dtype=np.bool_)
    for a, b in intervals:
        selected[a:b] = True
    if _history_layout(layout):
        return _history_read_plan(
            expiry, torch.from_numpy(selected),
            torch.ones(prefix_length, dtype=torch.bool), recover=False,
        )
    view = _insert_window(layout, expiry, min(a for a, b in intervals))
    raw = view.occurrence_raw_tokens.numpy()
    keys, offsets = view.segment_key_occurrences.numpy(), view.segment_key_offsets.numpy()
    for a, b, x, y in zip(view.segment_query_starts.numpy(), view.segment_query_ends.numpy(), offsets[:-1], offsets[1:]):
        queries = np.flatnonzero(selected[a:b]) + a
        if not len(queries):
            continue
        source = raw[keys[x:y-(b-a)]]
        full[source[source < prefix_length]] = True
        full[a:min(queries[-1], prefix_length)] = True
    return torch.from_numpy(full)


def _plan_insert_recovery(resident, expiry, layout, rewind, incompatible, source_positions):

    n, matched = len(expiry), len(resident)
    if not 0 <= matched < n:
        raise ValueError("Insert recovery must leave an uncached query suffix")
    if not matched:
        return RecoveryPlan(((0, n),), resident.clone(), 0, resident.clone())
    present = resident.numpy()
    rewind = np.zeros(matched, dtype=bool) if rewind is None else rewind.numpy()
    incompatible = np.zeros(matched, dtype=bool) if incompatible is None else incompatible.numpy()
    suffix = ((matched, n),)
    if (
        _history_layout(layout)
        and not rewind.any()
        and not incompatible.any()
        and (source_positions is None or torch.equal(
            source_positions, layout.positions[:matched]
        ))
    ):
        selected = torch.arange(n) >= matched
        full = _history_read_plan(expiry, selected, resident, recover=True)
        return RecoveryPlan(
            tuple(mask_ranges(selected.numpy())), full | selected[:matched],
            matched, resident & ~selected[:matched],
        )
    full = _insert_read_mask(layout, expiry, suffix, matched)
    if not np.any((~present | incompatible) & full.numpy()):
        return RecoveryPlan(suffix, full, matched, resident.clone())

    view = _insert_window(layout, expiry)
    raw, positions = view.occurrence_raw_tokens.numpy(), view.occurrence_positions.numpy()
    keys, offsets = view.segment_key_occurrences.numpy(), view.segment_key_offsets.numpy()
    birth = layout.birth_positions.numpy()
    stored_positions = (layout.positions.numpy()[:matched] if source_positions is None
                        else source_positions.numpy())
    # Each source's readers are intervals of execution queries. Shared prefixes
    # cost one interval per segment, not a dense query-by-key matrix.
    readers = [[] for _ in range(matched)]
    old_version_readers = [[] for _ in range(matched)]
    for a, b, x, y in zip(view.segment_query_starts.numpy(), view.segment_query_ends.numpy(), offsets[:-1], offsets[1:]):
        a, b = int(a), int(b)
        for occurrence in keys[x:y-(b-a)]:
            source = int(raw[occurrence])
            if source >= matched:
                continue
            readers[source].append((a, b))
            if a < matched and positions[occurrence] != stored_positions[source]:
                old_version_readers[source].append((a, min(b, matched)))
        for source in range(a, min(b, matched)):
            if source+1 < b:
                readers[source].append((source+1, b))
                if source+1 < matched and birth[source] != stored_positions[source]:
                    old_version_readers[source].append((source+1, min(b, matched)))

    # A Fenwick tree answers whether any selected query lies in a read interval.
    tree = np.zeros(n+1, dtype=np.int32)
    for i in range(1, n+1):
        tree[i] = max(0, i-max(i-(i & -i), matched))

    def total(end):
        value = 0
        while end:
            value += int(tree[end])
            end -= end & -end
        return value

    def reads(ranges):
        return any(total(b) != total(a) for a, b in ranges)

    needed = np.zeros(n, dtype=bool)
    needed[matched:] = True
    required = np.zeros(matched, dtype=bool)
    for source in range(matched-1, -1, -1):
        full_needed = reads(readers[source])
        required[source] = full_needed
        # A repaired query after R can read the already-correct final Full KV.
        # Rewind only when a selected historical query needs another version;
        # an unrelated missing page must not invalidate the rotated prefix.
        # Queries at/after the match may apply new R transitions by COW. Those
        # future rotations do not invalidate the source used by repaired queries.
        needs_old_version = (rewind[source] or incompatible[source]) and reads(old_version_readers[source])
        full_missing = not present[source] or needs_old_version
        if full_needed and full_missing:
            needed[source] = required[source] = True
            i = source+1
            while i <= n:
                tree[i] += 1
                i += i & -i
    return RecoveryPlan(tuple(mask_ranges(needed)), torch.from_numpy(required),
                        matched, torch.from_numpy(present & ~needed[:matched]))


def plan_recovery(
    resident: torch.Tensor,
    visible_until: torch.Tensor,
    input_length: int,
    rewind_sources: torch.Tensor | None = None,
    incompatible_sources: torch.Tensor | None = None,
    *,
    layout=None,
    source_positions: torch.Tensor | None = None,
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
    for source_mask in (
        rewind_sources,
        incompatible_sources,
    ):
        if source_mask is not None and (
            source_mask.device.type != "cpu"
            or source_mask.dtype != torch.bool
            or source_mask.shape != resident.shape
        ):
            raise ValueError(
                "Recovery source masks must match the CPU residency vector."
            )
    if layout is not None and (layout.insert_spans.numel() or layout.cache_policy):
        if input_length != len(visible_until) or input_length != len(layout.positions):
            raise ValueError("Insert recovery metadata must cover the complete input")
        if source_positions is not None and (
            source_positions.device.type != "cpu" or source_positions.dtype != torch.int32
            or source_positions.shape != resident.shape
        ):
            raise ValueError("Insert source positions must match CPU residency")
        return _plan_insert_recovery(resident, visible_until, layout, rewind_sources,
                                     incompatible_sources, source_positions)
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
        full_missing = (
            not present[raw]
            or incompatible[raw]
            or (earliest < matched and rewind[raw])
        )
        full_needed = full_missing and expiry[raw] > earliest
        if full_needed:
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
