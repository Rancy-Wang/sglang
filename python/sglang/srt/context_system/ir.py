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

import time
from dataclasses import dataclass

import torch

TOKEN_KIND = 0
DELTA_KIND = 1
REPOSITION_KIND = 2


@dataclass(frozen=True)
class ContextLayout:
    drop_insert_offsets: torch.Tensor
    drop_range_offsets: torch.Tensor
    drop_ranges: torch.Tensor
    records: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor
    positions: torch.Tensor
    repos_info: torch.Tensor
    keep_mask: torch.Tensor
    materialized_stage: torch.Tensor
    birth_positions: torch.Tensor
    birth_stages: torch.Tensor
    transition_offsets: torch.Tensor
    transition_raw_tokens: torch.Tensor
    transition_old_positions: torch.Tensor
    transition_new_positions: torch.Tensor
    effective_reposition_stages: torch.Tensor
    drop_event_to_key: torch.Tensor
    effective_repositions: torch.Tensor
    ignored_repositions: torch.Tensor
    next_position: int
    current_reposition: int
    compile_ns: int

    @property
    def keys(self) -> torch.Tensor:
        return self.records


def _load_module():
    from sglang.kernels.ops.attention.context_plan import load_context_plan

    return load_context_plan()


def compile_context_layout(
    token_ids: torch.Tensor,
    drop_insert_offsets: torch.Tensor,
    drop_range_offsets: torch.Tensor,
    drop_ranges: torch.Tensor,
    reposition_raw_boundaries: torch.Tensor,
    reposition_insert_offsets: torch.Tensor,
) -> ContextLayout:
    """Compile raw-token events, preserving mini-sglang final-position semantics.

    Drop offsets are insertion points (the next query); intervals are half-open
    raw-token ranges. Reposition boundaries are the preceding raw token. Drop
    runs before Reposition at a shared insertion point. Physical KV/page IDs
    never occur in this immutable request program.

    No-feature requests must bypass this compiler at the serving boundary.
    Tensor fields are read-only by convention, including across TP processes.
    """

    vectors = {
        "token_ids": token_ids,
        "drop_insert_offsets": drop_insert_offsets,
        "drop_range_offsets": drop_range_offsets,
        "drop_ranges": drop_ranges,
        "reposition_raw_boundaries": reposition_raw_boundaries,
        "reposition_insert_offsets": reposition_insert_offsets,
    }
    for name, value in vectors.items():
        if (
            value.device.type != "cpu"
            or value.ndim != 1
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{name} must be a one-dimensional CPU integer tensor.")
        # Validate before narrowing: a wrapped offset can address unrelated KV.
        if (
            value.dtype == torch.int64
            and len(value)
            and (int(value.min()) < -(2**31) or int(value.max()) >= 2**31)
        ):
            raise ValueError(f"{name} exceeds the signed int32 range.")
    token_ids = token_ids.contiguous()
    drop_insert_offsets = drop_insert_offsets.to(torch.int32).contiguous()
    drop_range_offsets = drop_range_offsets.to(torch.int32).contiguous()
    drop_ranges = drop_ranges.to(torch.int32).contiguous()
    reposition_raw_boundaries = reposition_raw_boundaries.to(torch.int32).contiguous()
    reposition_insert_offsets = reposition_insert_offsets.to(torch.int32).contiguous()

    compile_started_ns = time.perf_counter_ns()
    token_count = len(token_ids)
    reposition_count = len(reposition_raw_boundaries)
    transition_counts = torch.zeros(reposition_count, dtype=torch.int32, device="cpu")
    count_status = torch.zeros(2, dtype=torch.int64, device="cpu")
    _load_module().count_radix_reposition_transitions(
        token_count,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        transition_counts,
        count_status,
    )
    if int(count_status[0]) == 1:
        boundary = int(count_status[1])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    transition_count = int(transition_counts.sum().item())
    range_count = len(drop_ranges) // 2
    capacity = token_count + range_count + reposition_count
    records = torch.empty((capacity, 4), dtype=torch.int32, device="cpu")
    virtual_mask = torch.empty(capacity, dtype=torch.bool, device="cpu")
    key_to_token = torch.empty(capacity, dtype=torch.int64, device="cpu")
    token_to_key = torch.empty(token_count, dtype=torch.int64, device="cpu")
    positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    repos_info = torch.empty(token_count, dtype=torch.int32, device="cpu")
    keep_mask = torch.empty(token_count, dtype=torch.bool, device="cpu")
    materialized_stage = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_stages = torch.empty(token_count, dtype=torch.int32, device="cpu")
    transition_offsets = torch.empty(
        reposition_count + 1, dtype=torch.int32, device="cpu"
    )
    transition_raw_tokens = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    transition_old_positions = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    transition_new_positions = torch.empty(
        transition_count, dtype=torch.int32, device="cpu"
    )
    effective_reposition_stages = torch.full(
        (reposition_count,), -1, dtype=torch.int32, device="cpu"
    )
    drop_event_to_key = torch.full(
        (len(drop_insert_offsets),), -1, dtype=torch.int64, device="cpu"
    )
    effective = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    ignored = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    status = torch.zeros(6, dtype=torch.int64, device="cpu")

    _load_module().compile_radix_reposition_layout(
        token_ids,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        records,
        virtual_mask,
        key_to_token,
        token_to_key,
        positions,
        repos_info,
        keep_mask,
        materialized_stage,
        birth_positions,
        birth_stages,
        transition_offsets,
        transition_raw_tokens,
        transition_old_positions,
        transition_new_positions,
        effective_reposition_stages,
        drop_event_to_key,
        effective,
        ignored,
        status,
    )

    if int(status[0]) == 1:
        boundary = int(status[4])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    if int(status[0]) == 2:
        token_id = int(status[4])
        raise ValueError(
            f"Reposition Radix token ID {token_id} is outside the non-negative int32 range."
        )

    key_len = int(status[1])
    effective_stage_count = int(status[2])
    return ContextLayout(
        drop_insert_offsets=drop_insert_offsets,
        drop_range_offsets=drop_range_offsets,
        drop_ranges=drop_ranges,
        records=records[:key_len],
        virtual_mask=virtual_mask[:key_len],
        key_to_token=key_to_token[:key_len],
        token_to_key=token_to_key,
        positions=positions,
        repos_info=repos_info,
        keep_mask=keep_mask,
        materialized_stage=materialized_stage,
        birth_positions=birth_positions,
        birth_stages=birth_stages,
        transition_offsets=transition_offsets[: effective_stage_count + 1],
        transition_raw_tokens=transition_raw_tokens,
        transition_old_positions=transition_old_positions,
        transition_new_positions=transition_new_positions,
        effective_reposition_stages=effective_reposition_stages,
        drop_event_to_key=drop_event_to_key,
        effective_repositions=effective,
        ignored_repositions=ignored,
        next_position=int(status[3]),
        current_reposition=int(status[5]),
        compile_ns=time.perf_counter_ns() - compile_started_ns,
    )


def prewarm_context_layout() -> None:
    """Load and execute the structured Radix compiler before accepting work."""

    empty = torch.empty(0, dtype=torch.int32, device="cpu")
    compile_context_layout(
        torch.tensor([0], dtype=torch.int32, device="cpu"),
        empty,
        torch.zeros(1, dtype=torch.int32, device="cpu"),
        empty,
        empty,
        empty,
    )
