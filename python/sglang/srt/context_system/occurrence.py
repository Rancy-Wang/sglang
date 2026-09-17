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
    each required occurrence; K and V share a slot. Physical pages may contain
    multiple slots, and remain under the native allocator ownership contract.
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
    if len(offsets) < 2:
        raise ValueError(
            "Paged-occurrence requires at least one effective Reposition stage."
        )
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
