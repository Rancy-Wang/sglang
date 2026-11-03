"""Logical SWA selection after the native position predicate.

Consecutive-position runs share an O(N) immutable maximum index. A tail query
visits only qualifying runs and emits at most its budget, rather than scanning
the prompt again for each query. The packed representation is also read by the
decode Triton gather; it contains logical indices, never physical KV addresses.
"""

from __future__ import annotations

import numpy as np


class LogicalSWAWindowIndex:
    def __init__(self, positions):
        positions = np.asarray(positions, dtype=np.int32)
        self.length = len(positions)
        self.starts = (
            np.r_[0, np.flatnonzero(np.diff(positions) != 1) + 1].astype(np.int32)
            if len(positions)
            else np.empty(0, dtype=np.int32)
        )
        self.ends = (
            np.r_[self.starts[1:], len(positions)].astype(np.int32)
            if len(positions)
            else self.starts.copy()
        )
        self.first = positions[self.starts]
        self.size = 1 << max(0, (max(1, len(self.starts)) - 1).bit_length())
        self.tree = np.full(2 * self.size, np.iinfo(np.int32).min, dtype=np.int32)
        self.tree[self.size : self.size + len(self.starts)] = (
            self.first + self.ends - self.starts - 1
        )
        width = self.size
        while width > 1:
            self.tree[width // 2 : width] = np.maximum(
                self.tree[width : 2 * width : 2], self.tree[width + 1 : 2 * width : 2]
            )
            width //= 2
        self.packed = np.concatenate(
            (
                np.array([len(self.starts), self.size], dtype=np.int32),
                self.starts,
                self.ends,
                self.first,
                self.tree,
            )
        )

    def _previous(self, end, lower):
        if end <= 0:
            return -1
        node = self.size + end - 1
        while True:
            if self.tree[node] >= lower:
                while node < self.size:
                    right = node * 2 + 1
                    node = right if self.tree[right] >= lower else right - 1
                return node - self.size
            while node > 1 and node % 2 == 0:
                node //= 2
            if node == 1:
                return -1
            node -= 1

    def tail(self, lower, count):
        """Last ``count`` qualifying logical indices, returned in forward order."""
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        pieces, end = [], len(self.starts)
        while count and end:
            run = self._previous(end, lower)
            if run < 0:
                break
            hi = int(self.ends[run])
            lo = max(
                int(self.starts[run]),
                int(self.starts[run]) + int(lower) - int(self.first[run]),
            )
            take = min(count, hi - lo)
            pieces.append(np.arange(hi - take, hi, dtype=np.int64))
            count -= take
            end = run
        return np.concatenate(pieces[::-1]) if pieces else np.empty(0, dtype=np.int64)

    def prefix_reads(self, query_position, query_count, capacity):
        """Prefix reads for a consecutive query run, including query in capacity.

        After ``capacity`` new queries no old prefix can be selected. Earlier
        candidates are reconsidered at each position: an initially excluded key
        can re-enter when later logical keys expire from the position window.
        """
        for offset in range(min(query_count, capacity - 1)):
            yield offset, self.tail(
                query_position + offset - (capacity - 1), capacity - offset - 1
            )


def terminal_swa_mask(layout, capacity):
    """Prompt KV needed by any future decode query, before its window fills."""
    order = getattr(layout, "logical_order", None)
    raw = (
        order.numpy()
        if order is not None and order.numel()
        else np.arange(len(layout.positions))
    )
    raw = raw[layout.keep_mask.numpy()[raw]]
    result = np.zeros(len(layout.positions), dtype=np.bool_)
    positions = layout.positions.numpy()[raw]
    if capacity <= 1 or not len(raw):
        return result
    if np.all(np.diff(positions) > 0):
        # Ordinary summaries after R: future decode reads are subsets of this
        # first window. No run index or W separate selections are needed.
        start = max(
            int(np.searchsorted(positions, layout.next_position - capacity + 1)),
            len(raw) - capacity + 1,
        )
        result[raw[start:]] = True
        return result
    index = LogicalSWAWindowIndex(positions)
    for _, selected in index.prefix_reads(layout.next_position, capacity - 1, capacity):
        result[raw[selected]] = True
    return result
