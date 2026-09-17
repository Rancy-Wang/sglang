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

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ir import ContextLayout


@dataclass(frozen=True)
class OccurrenceWindow:
    """Lazy attention/KV plan for one post-match raw query window.

    Every ``(raw token, RoPE position)`` pair owns a distinct occurrence.  The
    scheduler later assigns one page from the ordinary KV page allocator to
    each required occurrence; K and V share a slot. The initial supported page size is one token; allocations remain under
    the native allocator ownership contract.
    """

    occurrence_raw_tokens: torch.Tensor
    occurrence_positions: torch.Tensor
    birth_occurrences: torch.Tensor
    terminal_occurrences: torch.Tensor
    segment_query_starts: torch.Tensor
    segment_query_ends: torch.Tensor
    segment_key_offsets: torch.Tensor
    segment_key_occurrences: torch.Tensor

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrence_raw_tokens)

    @property
    def segment_count(self) -> int:
        return len(self.segment_query_starts)


def compile_occurrence_window(
    layout: ContextLayout,
    full_token_visible_until: torch.Tensor,
    terminal_positions: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
) -> OccurrenceWindow:
    """Compile the same ordered plan using request-local native CPU arrays.

    All inputs are read-only views. Working arrays belong to this invocation;
    there is no mutable global scratch or trusted-validation flag. In particular
    the final materialization still covers *all* raw tokens, including dropped
    tokens needed by the final-position Radix cache.
    """
    tensors = (
        layout.birth_positions,
        layout.birth_stages,
        layout.transition_offsets,
        layout.transition_raw_tokens,
        layout.transition_old_positions,
        layout.transition_new_positions,
        full_token_visible_until,
        terminal_positions,
    )
    if any(
        t.device.type != "cpu" or t.dtype != torch.int32 or t.ndim != 1 for t in tensors
    ):
        raise ValueError(
            "Compact occurrence inputs must be one-dimensional CPU int32 tensors."
        )
    birth, stages, offsets, changed_raw, old, new, expiry, terminal = (
        t.numpy() for t in tensors
    )
    n = len(birth)
    if n < 1 or len(stages) != n:
        raise ValueError(
            "Compact occurrence birth metadata must cover a nonempty prompt."
        )
    if len(expiry) != n or len(terminal) != n:
        raise ValueError(
            "Occurrence visibility and terminal positions must cover the prompt."
        )
    if not 0 <= query_start < query_end <= n:
        raise ValueError("Occurrence query window is outside the raw prompt.")
    if len(offsets) < 1:
        raise ValueError("Occurrence transition offsets require an initial zero.")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(
            "Occurrence transition offsets must start at zero and be monotonic."
        )
    if not offsets[-1] == len(changed_raw) == len(old) == len(new):
        raise ValueError(
            "Occurrence transition offsets do not cover the transition arrays."
        )
    if np.any(birth < 0) or np.any(new < 0):
        raise ValueError("Occurrence positions must be non-negative.")
    stage_count = len(offsets) - 1
    if np.any(stages < 0) or np.any(stages > stage_count):
        raise ValueError("Occurrence birth stages are outside the Reposition program.")
    if np.any(stages[1:] < stages[:-1]):
        raise ValueError("Occurrence birth stages must preserve raw-token order.")
    raw = np.arange(n, dtype=np.int32)
    if np.any(expiry <= raw):
        raise ValueError("A token cannot become invisible before it has been computed.")
    bounds = np.searchsorted(stages, np.arange(stage_count + 2))
    current_ids = raw.copy()
    current_pos = birth.copy()
    materialized_pos = birth.copy()
    raw_parts, pos_parts = [raw], [birth]
    next_id = n
    starts, ends, keys, key_offsets = [], [], [], [0]

    def materialize(indices):
        nonlocal next_id
        stale = indices[materialized_pos[indices] != current_pos[indices]]
        count = len(stale)
        if not count:
            return
        if next_id + count > np.iinfo(np.int32).max:
            raise ValueError("Occurrence IDs exceed int32 capacity.")
        raw_parts.append(stale)
        pos_parts.append(current_pos[stale])
        current_ids[stale] = np.arange(next_id, next_id + count, dtype=np.int32)
        materialized_pos[stale] = current_pos[stale]
        next_id += count

    covered = query_start
    for stage in range(stage_count + 1):
        if stage:
            begin, end = int(offsets[stage - 1]), int(offsets[stage])
            ids = changed_raw[begin:end]
            if np.any(ids < 0) or np.any(ids >= n):
                raise ValueError(
                    "Reposition transition references an invalid raw token."
                )
            if np.any(ids[1:] <= ids[:-1]) and len(np.unique(ids)) != len(ids):
                raise ValueError(
                    "One Reposition stage cannot transition a raw token twice."
                )
            if not np.array_equal(current_pos[ids], old[begin:end]):
                raise ValueError(
                    "Reposition transition old positions do not match current state."
                )
            current_pos[ids] = new[begin:end]
        local_start = max(query_start, int(bounds[stage]))
        local_end = min(query_end, int(bounds[stage + 1]))
        if local_start >= local_end:
            continue
        if local_start != covered:
            raise RuntimeError(
                "Occurrence stages do not cover the requested query window."
            )
        values = expiry[:local_end]
        cuts = [
            local_start,
            *np.unique(values[(values > local_start) & (values < local_end)]),
            local_end,
        ]
        for start, end in pairwise(cuts):
            active = raw[:start][expiry[:start] > start]
            materialize(active)
            selected = np.concatenate((current_ids[active], raw[start:end]))
            if not len(selected) or selected[-1] != end - 1:
                raise RuntimeError(
                    "Occurrence segment does not end at its final query token."
                )
            starts.append(start)
            ends.append(end)
            keys.append(selected)
            key_offsets.append(key_offsets[-1] + len(selected))
        covered = local_end
    if covered != query_end:
        raise RuntimeError("Occurrence stages do not cover the requested query window.")
    if not np.array_equal(current_pos, terminal):
        raise ValueError(
            "Compact occurrence transitions disagree with terminal Radix positions."
        )
    materialize(raw)
    if key_offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Occurrence segment offsets exceed int32 capacity.")
    all_raw = np.concatenate(raw_parts)
    if not np.array_equal(all_raw[current_ids], raw):
        raise RuntimeError("Terminal occurrences do not cover the raw stream in order.")
    return OccurrenceWindow(
        occurrence_raw_tokens=torch.from_numpy(all_raw),
        occurrence_positions=torch.from_numpy(np.concatenate(pos_parts)),
        birth_occurrences=torch.from_numpy(raw),
        terminal_occurrences=torch.from_numpy(current_ids),
        segment_query_starts=torch.from_numpy(np.asarray(starts, dtype=np.int32)),
        segment_query_ends=torch.from_numpy(np.asarray(ends, dtype=np.int32)),
        segment_key_offsets=torch.from_numpy(np.asarray(key_offsets, dtype=np.int32)),
        segment_key_occurrences=torch.from_numpy(np.concatenate(keys)),
    )


@dataclass(frozen=True)
class OccurrenceMaterialization:
    """CPU ownership decision, made before asking the native allocator for KV.

    Source rows address canonical slots first, then existing terminal slots.
    A negative source row denotes a fresh allocation. Query birth slots have
    already been reserved by native admission and occupy canonical source rows.
    Unused occurrences stay unbound; no physical page is allocated for them.
    """

    source_rows: torch.Tensor
    allocated_occurrences: torch.Tensor
    copy_occurrences: torch.Tensor
    copy_source_rows: torch.Tensor
    copy_position_pairs: torch.Tensor
    terminal_occurrences: torch.Tensor
    read_cached: torch.Tensor
    repositioned_cached: torch.Tensor

    @property
    def extra_page_count(self) -> int:
        return len(self.allocated_occurrences)

    def bind(self, canonical_slots, terminal_slots, allocated_slots):
        """Bind device indices without copying them back to the scheduler CPU."""
        n = len(self.read_cached)
        if (
            canonical_slots.ndim != 1
            or terminal_slots.ndim != 1
            or allocated_slots.ndim != 1
            or any(
                value.dtype not in (torch.int32, torch.int64)
                for value in (canonical_slots, terminal_slots, allocated_slots)
            )
            or canonical_slots.dtype != terminal_slots.dtype
            or canonical_slots.dtype != allocated_slots.dtype
            or len(canonical_slots) != n
            or len(terminal_slots) != n
            or len(allocated_slots) != self.extra_page_count
            or canonical_slots.device != terminal_slots.device
            or canonical_slots.device != allocated_slots.device
        ):
            raise ValueError("Occurrence slot buffers do not match the ownership plan")
        # CPU-only index preparation; one transfer covers reuse, fresh slots and
        # copy metadata. In particular, no CUDA nonzero/item is used here.
        rows = self.source_rows.numpy()
        reused = np.flatnonzero(rows >= 0)
        fields = (
            reused,
            rows[reused],
            self.allocated_occurrences.numpy(),
            self.copy_occurrences.numpy(),
            self.copy_source_rows.numpy(),
            self.copy_position_pairs.numpy().reshape(-1),
        )
        offsets = np.r_[0, np.cumsum([len(field) for field in fields])]
        packed = torch.from_numpy(np.concatenate(fields).astype(np.int64, copy=False))
        packed = packed.to(canonical_slots.device, non_blocking=True)
        reuse_ids, reuse_rows, fresh_ids, copy_ids, copy_rows, pairs = (
            packed[a:b] for a, b in pairwise(offsets)
        )
        sources = torch.cat((canonical_slots, terminal_slots))
        slots = canonical_slots.new_full((len(rows),), -1)
        slots[reuse_ids] = sources[reuse_rows]
        slots[fresh_ids] = allocated_slots
        return (
            slots,
            sources[copy_rows].to(torch.int32),
            slots[copy_ids].to(torch.int32),
            pairs.reshape(-1, 2).to(torch.int32),
        )


def plan_occurrence_materialization(
    window: OccurrenceWindow,
    canonical_positions: torch.Tensor,
    canonical_present: torch.Tensor,
    canonical_owned: torch.Tensor,
    terminal_present: torch.Tensor,
    terminal_keep: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
    exact_prefix_len: int,
) -> OccurrenceMaterialization:
    """Select minimum occurrence allocations under explicit native ownership.

    Canonical metadata covers the prefix before this forward. A retained terminal
    slot already belongs to the target version. A borrowed Retry source after the
    *selected source's* exact prefix must be copied before target publication,
    even at the same position (a bit-copy, not another RoPE round trip).

    ``terminal_keep`` is a raw-token mask supplied by the cache policy. Holes may
    remain holes only when neither this forward's read set nor publication needs
    them. Missing required sources are returned as an error for recovery planning;
    they must never be bound to the padded slot 0.
    """
    n = len(window.birth_occurrences)
    if not 0 <= exact_prefix_len <= query_start < query_end <= n:
        raise ValueError("Invalid occurrence ownership/query boundaries")
    for value, size, dtype in (
        (canonical_positions, query_start, torch.int32),
        (canonical_present, query_start, torch.bool),
        (canonical_owned, query_start, torch.bool),
        (terminal_present, query_start, torch.bool),
        (terminal_keep, query_end, torch.bool),
    ):
        if value.device.type != "cpu" or value.dtype != dtype or value.shape != (size,):
            raise ValueError("Occurrence ownership must use aligned CPU metadata")
    if (
        int(window.segment_query_starts[0]) != query_start
        or int(window.segment_query_ends[-1]) != query_end
    ):
        raise ValueError("Occurrence ownership does not cover the planned queries")
    raw = window.occurrence_raw_tokens.numpy()
    position = window.occurrence_positions.numpy()
    birth = window.birth_occurrences.numpy()
    terminal = window.terminal_occurrences.numpy()
    count = len(raw)
    present = np.zeros(n, dtype=np.bool_)
    present[:query_start] = canonical_present.numpy()
    present[query_start:query_end] = True
    owned = np.zeros(n, dtype=np.bool_)
    owned[:query_start] = canonical_owned.numpy()
    owned[query_start:query_end] = True
    canonical_pos = position[birth].copy()
    canonical_pos[:query_start] = canonical_positions.numpy()
    terminal_live = np.zeros(n, dtype=np.bool_)
    terminal_live[:query_start] = terminal_present.numpy()
    terminal_pos = position[terminal]

    read = np.zeros(count, dtype=np.bool_)
    read[window.segment_key_occurrences.numpy()] = True
    required = read.copy()
    required[birth[query_start:query_end]] = True
    publish_raw = np.flatnonzero(terminal_keep.numpy())
    publish_ids = terminal[publish_raw]
    required[publish_ids] = True
    ids = np.flatnonzero(required)
    selected_raw = raw[ids]
    selected_pos = position[ids]
    terminal_match = terminal_live[selected_raw] & (
        terminal_pos[selected_raw] == selected_pos
    )
    canonical_match = present[selected_raw] & (
        canonical_pos[selected_raw] == selected_pos
    )
    # A page cannot become owned by two unrelated native Radix branches. Same
    # position does not imply same ownership; exact prefix pages remain borrowed.
    publishing = np.zeros(count, dtype=np.bool_)
    publishing[publish_ids] = True
    force_own = (
        publishing[ids]
        & (selected_raw >= exact_prefix_len)
        & ~owned[selected_raw]
        & ~terminal_match
    )
    reuse = terminal_match | (canonical_match & ~force_own)
    sources = np.where(terminal_match, n + selected_raw, selected_raw)
    rows = np.full(count, -1, dtype=np.int64)
    rows[ids[reuse]] = sources[reuse]
    allocated = ids[~reuse]
    copied_raw = raw[allocated]
    # Prefer the canonical source, keeping each copy independent of other copies
    # in this layer. Newly computed queries use their native birth write slots.
    copy_from_terminal = ~present[copied_raw] & terminal_live[copied_raw]
    missing = ~present[copied_raw] & ~terminal_live[copied_raw]
    if np.any(missing):
        raise ValueError(
            "Occurrence recovery required for raw tokens "
            + repr(np.unique(copied_raw[missing]).tolist())
        )
    copy_rows = np.where(copy_from_terminal, n + copied_raw, copied_raw)
    old = np.where(
        copy_from_terminal, terminal_pos[copied_raw], canonical_pos[copied_raw]
    )
    pairs = np.column_stack((old, position[allocated])).astype(np.int32)
    retained = np.full(query_end, -1, dtype=np.int32)
    retained[publish_raw] = publish_ids
    read_cached = np.zeros(n, dtype=np.bool_)
    read_raw = raw[np.flatnonzero(read)]
    read_cached[read_raw[read_raw < query_start]] = True
    rotated_cached = np.zeros(n, dtype=np.bool_)
    changed_reads = (
        read[ids]
        & (selected_pos != canonical_pos[selected_raw])
        & (selected_raw < query_start)
    )
    rotated_cached[selected_raw[changed_reads]] = True
    return OccurrenceMaterialization(
        torch.from_numpy(rows),
        torch.from_numpy(allocated),
        torch.from_numpy(allocated.copy()),
        torch.from_numpy(copy_rows),
        torch.from_numpy(pairs),
        torch.from_numpy(retained),
        torch.from_numpy(read_cached),
        torch.from_numpy(rotated_cached),
    )
