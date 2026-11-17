"""Compile ablations into existing occurrence execution; never change legacy IR."""

import time
from dataclasses import replace

import torch

from .ir import COMPUTE_KIND, REPOSITION_KIND, TOKEN_KIND, compile_context_layout

TEXT_DROP_WITH_PE = "text_drop_with_pe"
MESSAGE_DROP_WITHOUT_REPOS = "message_drop_without_repos"
POLICIES = {TEXT_DROP_WITH_PE: 1, MESSAGE_DROP_WITHOUT_REPOS: 2}


def compile_cache_policy(tokens, events, repos, policy):
    """Return execution layout, visibility and execution-to-full provenance.

    Text mode uses the existing linear event compiler only as a position tape.
    Its pre-reset computation graph and dropped KV are not part of execution.
    KV mode has one computation stage, including historical dropped tokens.
    """
    started = time.perf_counter_ns()
    if policy not in POLICIES:
        raise ValueError("Unknown cache policy")
    if policy == MESSAGE_DROP_WITHOUT_REPOS and len(repos.insert_offsets):
        raise ValueError("message_drop_without_repos does not accept reposition")
    full = compile_context_layout(
        tokens,
        events.event_insert_offsets,
        events.range_offsets,
        events.raw_ranges,
        repos.raw_boundaries,
        repos.insert_offsets,
    )
    n = len(tokens)
    if not bool(full.keep_mask.any()):
        raise ValueError("Cache policy requires at least one active token for logits")
    if policy == MESSAGE_DROP_WITHOUT_REPOS:
        from sglang.kernels.ops.attention.cache_policy_plan import (
            load_cache_policy_plan,
        )

        pos = torch.empty(n, dtype=torch.int32)
        load_cache_policy_plan().frozen_death_positions(
            events.full_token_visible_until, pos
        )
        records = full.records.clone()
        records[full.token_to_key, 2] = pos
        return (
            replace(
                full,
                records=records,
                positions=pos,
                birth_positions=pos.clone(),
                next_position=int(full.keep_mask.sum()),
                cache_policy=2,
                compile_ns=time.perf_counter_ns() - started,
            ),
            events.full_token_visible_until,
            torch.arange(n, dtype=torch.int32),
        )

    active = torch.nonzero(full.keep_mask).flatten()
    count = len(active)
    # TokenEventCompiler already removes repeated/empty Drop events.
    reset = (
        int(events.event_insert_offsets[-1]) if len(events.event_insert_offsets) else 0
    )
    before = (repos.insert_offsets < reset) & full.effective_repositions
    prior_stages = int(before.sum())
    birth = full.birth_positions.clone()
    # Apply all pre-reset rotations in order; reset-prefix queries use this tape.
    offsets = full.transition_offsets
    for stage in range(prior_stages):
        a, b = offsets[stage : stage + 2].tolist()
        birth[full.transition_raw_tokens[a:b].long()] = full.transition_new_positions[
            a:b
        ]
    birth = birth[active]
    stages = torch.clamp(full.birth_stages[active] - prior_stages, min=0)
    final = full.positions[active]
    inverse = torch.full((n,), -1, dtype=torch.int32)
    inverse[active] = torch.arange(count, dtype=torch.int32)
    cut = int(offsets[prior_stages])
    changed = inverse[full.transition_raw_tokens[cut:].long()]
    if bool(torch.any(changed < 0)):
        raise RuntimeError("Post-reset transition references removed text")
    remaining = full.effective_repositions & ~before
    boundaries = torch.searchsorted(active, repos.insert_offsets[remaining].long())
    # Merge interleaved C/T pairs and R markers in execution coordinates.
    rcount = len(boundaries)
    token_keys = 2 * torch.arange(count) + 1
    token_keys += torch.searchsorted(boundaries, torch.arange(count), right=True)
    records = torch.empty((2 * count + rcount, 3), dtype=torch.int32)
    records[token_keys - 1] = torch.stack(
        (torch.full_like(birth, COMPUTE_KIND), birth, torch.full_like(birth, -1)), 1
    )
    records[token_keys] = torch.stack(
        (torch.zeros_like(final), tokens[active].int(), final), 1
    )
    rkeys = 2 * boundaries + torch.arange(rcount)
    records[rkeys] = torch.stack(
        (
            torch.full((rcount,), REPOSITION_KIND, dtype=torch.int32),
            (boundaries - 1).int(),
            torch.full((rcount,), -1, dtype=torch.int32),
        ),
        1,
    )
    key_to_token = torch.full((len(records),), -1, dtype=torch.int64)
    key_to_token[token_keys] = torch.arange(count)
    empty = torch.empty(0, dtype=torch.int32)
    layout = replace(
        full,
        drop_insert_offsets=empty,
        drop_range_offsets=torch.zeros(1, dtype=torch.int32),
        drop_ranges=empty,
        records=records,
        virtual_mask=records[:, 0] != TOKEN_KIND,
        key_to_token=key_to_token,
        token_to_key=token_keys,
        positions=final,
        keep_mask=torch.ones(count, dtype=torch.bool),
        materialized_stage=torch.clamp(
            full.materialized_stage[active] - prior_stages, min=0
        ),
        birth_positions=birth,
        birth_stages=stages,
        transition_offsets=offsets[prior_stages:] - cut,
        transition_raw_tokens=changed,
        transition_old_positions=full.transition_old_positions[cut:],
        transition_new_positions=full.transition_new_positions[cut:],
        effective_reposition_stages=torch.arange(1, rcount + 1, dtype=torch.int32),
        effective_repositions=torch.ones(rcount, dtype=torch.bool),
        ignored_repositions=torch.zeros(rcount, dtype=torch.bool),
        drop_event_to_key=torch.empty(0, dtype=torch.int64),
        cache_policy=1,
        compile_ns=time.perf_counter_ns() - started,
    )
    return layout, torch.full((count,), 2**31 - 1, dtype=torch.int32), active.int()
