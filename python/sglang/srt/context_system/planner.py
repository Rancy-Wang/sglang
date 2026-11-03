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

from bisect import bisect_right
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any

import torch

from .ir import ContextLayout, compile_context_layout
from .provenance import TemplateTokenProvenance
from .rules import (
    DropCompileContext,
    DropRule,
    TokenDropEvents,
    _matchable_content,
    find_all,
)


@dataclass(frozen=True)
class TokenRepositionEvents:
    raw_boundaries: torch.Tensor
    insert_offsets: torch.Tensor


def resolve_reposition_token_boundaries(
    reposition_ids: list[int] | None,
    owner_ranges: dict[int, list[tuple[int, int]]],
    public_to_normalized_owner: dict[int, int],
) -> TokenRepositionEvents:
    """Translate public message IDs into exact raw-token boundaries."""

    raw_ids = reposition_ids or []
    if any(
        isinstance(raw_id, bool) or not isinstance(raw_id, int) for raw_id in raw_ids
    ):
        raise ValueError("reposition must contain integer message IDs.")
    if raw_ids != sorted(set(raw_ids)):
        raise ValueError(
            "reposition must contain strictly increasing unique message IDs."
        )
    boundaries: list[int] = []
    for raw_id in raw_ids:
        owner = public_to_normalized_owner.get(raw_id)
        if owner is None:
            raise ValueError(
                f"reposition message ID {raw_id} is outside the conversation."
            )
        ranges = owner_ranges.get(owner)
        if not ranges:
            raise ValueError(
                f"Cannot map reposition message ID {raw_id} into the token stream."
            )
        boundaries.append(max(end for _, end in ranges) - 1)
    if boundaries != sorted(set(boundaries)):
        raise ValueError(
            "Chat template ownership does not preserve Reposition boundary order."
        )
    raw_boundaries = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    return TokenRepositionEvents(raw_boundaries, raw_boundaries + 1)


class TokenEventCompiler:
    @staticmethod
    def _build_owner_position_ranges(
        owners: list[int] | torch.Tensor,
    ) -> dict[int, list[tuple[int, int]]]:
        """Return exact full-token ranges for every provenance owner."""

        ranges: dict[int, list[tuple[int, int]]] = {}
        owner_tensor = torch.as_tensor(owners, dtype=torch.int32, device="cpu")
        if owner_tensor.ndim != 1:
            raise ValueError(
                "Token provenance owners must be a one-dimensional vector."
            )
        if len(owner_tensor) == 0:
            return ranges
        changes = (
            torch.nonzero(owner_tensor[1:] != owner_tensor[:-1], as_tuple=False).view(
                -1
            )
            + 1
        )
        starts = torch.cat((torch.zeros(1, dtype=torch.int64), changes.to(torch.int64)))
        ends = torch.cat(
            (
                changes.to(torch.int64),
                torch.tensor([len(owner_tensor)], dtype=torch.int64),
            )
        )
        range_owners = owner_tensor[starts].tolist()
        for owner, start, end in zip(
            range_owners,
            starts.tolist(),
            ends.tolist(),
            strict=True,
        ):
            ranges.setdefault(owner, []).append((start, end))
        return ranges

    @staticmethod
    def _canonicalize_position_ranges(
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not ranges:
            return []
        normalized = sorted((int(start), int(end)) for start, end in ranges)
        merged: list[tuple[int, int]] = []
        for start, end in normalized:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid token-position Drop range: [{start}, {end})")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    def _build_position_range_drop_plan(
        cls,
        event_ranges: dict[int, list[tuple[int, int]]],
        query_epoch: list[int],
    ) -> TokenDropEvents:
        """Compile rule-produced absolute ranges into the generic delta wire."""

        delta_by_pos: dict[int, list[tuple[int, int]]] = {}
        covered_ranges: list[tuple[int, int]] = []
        for event_n in sorted(event_ranges):
            canonical = cls._canonicalize_position_ranges(event_ranges[event_n])
            newly_effective = cls._subtract_position_ranges(canonical, covered_ranges)
            covered_ranges = cls._canonicalize_position_ranges(
                covered_ranges + canonical
            )
            if not newly_effective:
                continue
            insertion_pos = bisect_right(query_epoch, event_n)
            if any(end > insertion_pos for _, end in newly_effective):
                raise ValueError(
                    "A Drop event cannot hide a token before that token has been computed: "
                    f"event={event_n}, insertion_pos={insertion_pos}, ranges={newly_effective}"
                )
            delta_by_pos.setdefault(insertion_pos, []).extend(newly_effective)

        event_positions: list[int] = []
        range_offsets: list[int] = [0]
        flat_ranges: list[tuple[int, int]] = []
        visible_until = torch.full(
            (len(query_epoch),),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device="cpu",
        )
        for insertion_pos in sorted(delta_by_pos):
            canonical = cls._canonicalize_position_ranges(delta_by_pos[insertion_pos])
            if not canonical:
                continue
            event_positions.append(insertion_pos)
            flat_ranges.extend(canonical)
            range_offsets.append(len(flat_ranges))
            for start, end in canonical:
                visible_until[start:end] = torch.minimum(
                    visible_until[start:end],
                    torch.tensor(insertion_pos, dtype=torch.int32, device="cpu"),
                )
        return TokenDropEvents(
            event_insert_offsets=torch.tensor(
                event_positions, dtype=torch.int32, device="cpu"
            ),
            range_offsets=torch.tensor(range_offsets, dtype=torch.int32, device="cpu"),
            raw_ranges=torch.tensor(
                flat_ranges, dtype=torch.int32, device="cpu"
            ).reshape(-1),
            full_token_visible_until=visible_until,
        )

    @staticmethod
    def _subtract_position_ranges(
        ranges: list[tuple[int, int]],
        covered: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for start, end in ranges:
            fragments = [(start, end)]
            for cover_start, cover_end in covered:
                next_fragments: list[tuple[int, int]] = []
                for frag_start, frag_end in fragments:
                    if cover_end <= frag_start or cover_start >= frag_end:
                        next_fragments.append((frag_start, frag_end))
                        continue
                    if frag_start < cover_start:
                        next_fragments.append((frag_start, cover_start))
                    if cover_end < frag_end:
                        next_fragments.append((cover_end, frag_end))
                fragments = next_fragments
            result.extend(fragments)
        return result

    @staticmethod
    def _rendered_source_start(
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        source: str,
        field: str,
        prefer_latest: bool = False,
    ) -> int:
        candidates = []
        for start, end in find_all(provenance.rendered_text, [source])[0]:
            if all(
                candidate == owner for candidate in provenance.char_owners[start:end]
            ):
                candidates.append(start)
        if prefer_latest and candidates:
            return candidates[-1]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot map {field} for messages[{owner}] uniquely into the "
                "canonical chat template"
            )
        return candidates[0]

    @classmethod
    def _token_ranges_for_char_spans(
        cls,
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        spans: list[tuple[int, int]],
        field: str,
        boundary_mode: str = "contained",
        allow_empty: bool = False,
    ) -> list[tuple[int, int]]:
        if boundary_mode not in {"contained", "overlap"}:
            raise ValueError(f"Unsupported token boundary mode: {boundary_mode}")
        selected: list[int] = []
        for token_id, (start, end) in enumerate(provenance.offsets):
            if provenance.owners[token_id] != owner or start == end:
                continue
            if boundary_mode == "contained":
                matched = any(
                    start >= span_start and end <= span_end
                    for span_start, span_end in spans
                )
            else:
                matched = any(
                    start < span_end and end > span_start
                    for span_start, span_end in spans
                )
            if matched:
                selected.append(token_id)
        if not selected:
            if allow_empty:
                return []
            raise ValueError(
                f"{field} for messages[{owner}] contains no complete token; "
                "boundary-crossing tokens are kept"
            )
        return cls._position_ranges_from_ids(selected)

    @classmethod
    def _position_ranges_from_ids(cls, token_ids: list[int]) -> list[tuple[int, int]]:
        if not token_ids:
            return []
        token_ids = sorted(set(token_ids))
        ranges: list[tuple[int, int]] = []
        start = previous = token_ids[0]
        for token_id in token_ids[1:]:
            if token_id == previous + 1:
                previous = token_id
                continue
            ranges.append((start, previous + 1))
            start = previous = token_id
        ranges.append((start, previous + 1))
        return cls._canonicalize_position_ranges(ranges)

    @staticmethod
    def _query_epochs_from_owners(
        owners: list[int] | torch.Tensor, message_count: int
    ) -> list[int]:
        owner_tensor = torch.as_tensor(owners, dtype=torch.int32, device="cpu")
        if owner_tensor.ndim != 1:
            raise ValueError(
                "Token provenance owners must be a one-dimensional vector."
            )
        epochs = torch.clamp(owner_tensor, min=0, max=message_count)
        if len(epochs) > 1 and bool(torch.any(epochs[1:] < epochs[:-1]).item()):
            raise RuntimeError(
                "Chat template reordered messages; cannot construct monotonic Drop events."
            )
        return epochs.tolist()


@dataclass(frozen=True)
class ContextProgram:
    layout: ContextLayout
    visible_until: torch.Tensor

    @property
    def input_ids(self):
        return self.layout.records[self.layout.token_to_key, 1].tolist()

    def with_generated(self, token_ids) -> ContextProgram:
        from .ir import append_generated_layout

        layout = append_generated_layout(self.layout, token_ids)
        if layout is self.layout:
            return self
        # Survivors remain visible throughout generation. A trailing Drop at
        # the prompt boundary stays effective for the very first output query.
        expiry = torch.where(
            self.layout.keep_mask, torch.iinfo(torch.int32).max, self.visible_until
        )
        return ContextProgram(
            layout,
            torch.cat(
                (
                    expiry,
                    torch.full(
                        (len(token_ids),),
                        torch.iinfo(torch.int32).max,
                        dtype=torch.int32,
                    ),
                )
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        """Use SGLang's native tensor-buffer IPC, without per-token Python lists.

        Compilation happens once at the tokenizer boundary. Scheduler/TP workers
        reconstruct views of the transported CPU tensors, not the event program.
        The tensors are immutable by convention throughout the request lifetime.
        """
        return {
            "version": 5 if self.layout.insert_spans.numel() else 3,
            "layout": {name: value for name, value in vars(self.layout).items()
                       if self.layout.insert_spans.numel() or name not in
                       {"logical_order", "logical_ranks", "insert_spans"}},
            "visible_until": self.visible_until,
        }

    def to_json_wire(self) -> dict[str, Any]:
        """Bounded binary JSON for rare native PD rebootstrap HTTP requests."""
        import base64

        def encode(value):
            if not isinstance(value, torch.Tensor):
                return value
            return {
                "shape": list(value.shape),
                "data": base64.b64encode(value.numpy().tobytes()).decode("ascii"),
            }

        return {
            "version": 6 if self.layout.insert_spans.numel() else 4,
            "layout": {
                name: encode(value) for name, value in self.to_wire()["layout"].items()
            },
            "visible_until": encode(self.visible_until),
        }

    @classmethod
    def from_wire(cls, wire: dict[str, Any], input_ids) -> ContextProgram:
        """Check the internal payload before binding it to a scheduler request."""
        if (
            isinstance(wire, dict)
            and type(wire.get("version")) is int
            and wire["version"] in (4, 6)
        ):
            import base64
            import math

            scalar = {"next_position", "compile_ns"}
            boolean = {
                "virtual_mask",
                "keep_mask",
                "effective_repositions",
                "ignored_repositions",
            }
            wide = {"key_to_token", "token_to_key", "drop_event_to_key"}

            def decode(name, value):
                if name in scalar:
                    return value
                if not isinstance(value, dict) or set(value) != {"shape", "data"}:
                    raise ValueError("Invalid Context JSON tensor")
                shape = value["shape"]
                if (
                    not isinstance(shape, list)
                    or len(shape) != (2 if name == "records" else 1)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(value["data"], str)
                ):
                    raise ValueError("Invalid Context JSON tensor shape")
                dtype = (
                    torch.bool
                    if name in boolean
                    else torch.int64
                    if name in wide
                    else torch.int32
                )
                raw = base64.b64decode(value["data"], validate=True)
                width = torch.empty((), dtype=dtype).element_size()
                if len(raw) != math.prod(shape) * width:
                    raise ValueError("Context JSON tensor byte length mismatch")
                if dtype == torch.bool and any(b > 1 for b in raw):
                    raise ValueError("Invalid Context JSON boolean")
                return (
                    torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
                    if raw
                    else torch.empty(shape, dtype=dtype)
                )

            if set(wire) != {"version", "layout", "visible_until"} or not isinstance(
                wire["layout"], dict
            ):
                raise ValueError("Invalid Context JSON wire")
            wire = {
                "version": wire["version"] - 1,
                "layout": {
                    name: decode(name, value) for name, value in wire["layout"].items()
                },
                "visible_until": decode("visible_until", wire["visible_until"]),
            }
        if (
            not isinstance(wire, dict)
            or set(wire) != {"version", "layout", "visible_until"}
            or type(wire["version"]) is not int
            or wire["version"] not in (3, 5)
        ):
            raise ValueError("Unsupported Context program wire version")
        data = wire["layout"]
        expected = {field.name for field in fields(ContextLayout)}
        if wire["version"] == 3:
            expected -= {"logical_order", "logical_ranks", "insert_spans"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Context layout fields do not match the wire schema")
        layout = ContextLayout(**data)
        n = len(input_ids)
        scalar_names = {"next_position", "compile_ns"}
        bool_names = {
            "virtual_mask",
            "keep_mask",
            "effective_repositions",
            "ignored_repositions",
        }
        i64_names = {"key_to_token", "token_to_key", "drop_event_to_key"}
        for name, value in data.items():
            if name in scalar_names:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid Context scalar {name}")
                continue
            dtype = (
                torch.bool
                if name in bool_names
                else (torch.int64 if name in i64_names else torch.int32)
            )
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or (
                    value.dtype != dtype
                    or not value.is_contiguous()
                    or value.ndim != (2 if name == "records" else 1)
                )
            ):
                raise ValueError(f"Invalid Context tensor {name}")
        records = layout.records
        if records.shape[1] != 3:
            raise ValueError("Context key records must have three columns")
        max_kind = 4 if wire["version"] == 5 else 2
        if bool(torch.any(records[:, 0] < 0) | torch.any(records[:, 0] > max_kind)):
            raise ValueError("Context key record kind is invalid for its wire version")
        for name in (
            "token_to_key",
            "positions",
            "keep_mask",
            "materialized_stage",
            "birth_positions",
            "birth_stages",
        ):
            if len(data[name]) != n:
                raise ValueError(f"Context {name} does not cover the input tokens")
        real_keys = torch.nonzero(records[:, 0] == 0, as_tuple=True)[0]
        if not torch.equal(real_keys, layout.token_to_key) or not torch.equal(
            records[real_keys, 1].to(torch.int64),
            torch.as_tensor(input_ids, dtype=torch.int64),
        ):
            raise ValueError(
                "Context program does not describe the canonical input IDs"
            )
        expiry = wire["visible_until"]
        if (
            not isinstance(expiry, torch.Tensor)
            or expiry.device.type != "cpu"
            or (
                expiry.dtype != torch.int32
                or expiry.ndim != 1
                or len(expiry) != n
                or not expiry.is_contiguous()
            )
            or bool(torch.any(expiry <= torch.arange(n)))
        ):
            raise ValueError("Invalid Context visibility metadata")
        if not torch.equal(layout.positions, records[real_keys, 2]) or bool(
            torch.any(layout.positions < 0) | torch.any(layout.birth_positions < 0)
        ):
            raise ValueError("Context positions disagree with the Radix key")
        if wire["version"] == 5 and not layout.insert_spans.numel():
            raise ValueError("Insert wire requires Insert spans")
        if layout.insert_spans.numel():
            if layout.insert_spans.numel() % 3 or len(layout.logical_order) != n or len(layout.logical_ranks) != n:
                raise ValueError("Invalid Context Insert permutation or spans")
            order, ranks = layout.logical_order.to(torch.int64), layout.logical_ranks.to(torch.int64)
            if (bool(torch.any(order < 0) | torch.any(order >= n))
                    or not torch.equal(ranks[order], torch.arange(n))):
                raise ValueError("Context logical and execution maps are not inverse permutations")
            for start, end, depth in layout.insert_spans.reshape(-1, 3).tolist():
                if not 0 < depth <= start < end <= n:
                    raise ValueError("Context Insert anchor is not an ancestor token")
        return cls(layout, expiry)


def _compile_insert_program(raw_messages, provenance, drop_rule, reposition, insert):
    """Schedule message events, then permute one complete logical tokenization."""
    import numpy as np

    from .ir import compile_execution_layout
    from .rules import InsertRule, MessageDropRule

    rule = InsertRule.from_payload(insert, raw_messages)
    ids = rule.message_ids
    owners = {public: ordinal for ordinal, public in enumerate(ids)}
    count, n = len(ids), len(provenance.input_ids)
    compiler = TokenEventCompiler
    token_owners = np.clip(np.asarray(provenance.owners), 0, count)
    owner_ranges = compiler._build_owner_position_ranges(token_owners.tolist())
    repos = reposition or []
    if any(type(i) is not int or i not in owners for i in repos) or len(set(repos)) != len(repos):
        raise ValueError("reposition must contain unique existing message indices")
    repo_set = set(repos)

    # Each negative message is born before following its child trigger. Children
    # run before the parent's Drop/Reposition hooks, independent of JSON order.
    schedule = []
    for ordinary in (i for i in ids if i >= 0):
        stack = [("post", ordinary), ("tokens", ordinary)]
        if ordinary in rule.triggers:
            stack.append(("insert", rule.triggers[ordinary]))
        while stack:
            kind, public = stack.pop()
            if kind == "insert":
                stack.extend([("post", public)])
                if public in rule.triggers:
                    stack.append(("insert", rule.triggers[public]))
                stack.extend([("end", public), ("tokens", public), ("begin", public)])
            else:
                schedule.append((kind, public))

    execution = []
    for kind, public in schedule:
        if kind == "tokens":
            for a, b in owner_ranges.get(owners[public], ()):
                execution.extend(range(a, b))
    for a, b in owner_ranges.get(count, ()):
        execution.extend(range(a, b))
    if len(execution) != n or len(set(execution)) != n:
        raise ValueError("Insert template ownership does not cover the logical token sequence")
    ranks = np.asarray(execution, dtype=np.int32)
    inverse = np.empty(n, dtype=np.int32)
    inverse[ranks] = np.arange(n, dtype=np.int32)

    other_events = {}
    if drop_rule is not None and not isinstance(drop_rule, MessageDropRule):
        context = DropCompileContext(
            raw_messages=raw_messages, owner_ranges=owner_ranges,
            provenance=provenance, normalized_message_count=count,
            normalize_content=lambda content: _matchable_content({"content": content}, field="messages"),
            rendered_source_start=compiler._rendered_source_start,
            token_ranges_for_char_spans=compiler._token_ranges_for_char_spans,
            canonicalize_ranges=compiler._canonicalize_position_ranges,
            position_ranges_from_ids=compiler._position_ranges_from_ids,
        )
        other_events = {ids[k]: v for k, v in drop_rule.position_events(context).items()}

    cursor, commands, spans = 0, [], []
    born, covered, live = set(), [], []
    expiry = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    insertion = None
    for kind, public in schedule:
        ordinal = owners[public]
        ranges = owner_ranges.get(ordinal, ())
        if kind == "begin":
            if not ranges:
                raise ValueError(f"insert message {public} owns no tokens")
            first = ranges[0][0]
            # Live ranges, not message IDs, resolve the first born successor.
            successor = next((a for a, b in live if a > first), None)
            if successor is None:
                raise ValueError(f"insert message {public} has no already-born logical token anchor")
            target = int(inverse[successor])
            commands.append((cursor, 3, target, 0))
            insertion = (cursor, target+1)
        elif kind == "tokens":
            born.add(public)
            cursor += sum(b-a for a, b in ranges)
            live = compiler._canonicalize_position_ranges([*live, *ranges])
        elif kind == "end":
            start, depth = insertion
            commands.append((cursor, 4, 0, 0))
            spans.append((start, cursor, depth))
            insertion = None
        else:
            selected = []
            if isinstance(drop_rule, MessageDropRule):
                for target in drop_rule.drop_messages.get(public, ()):
                    if target not in born:
                        raise ValueError(f"drop_message[{public}] cannot drop unborn message {target}")
                    selected.extend(owner_ranges.get(owners[target], ()))
            else:
                selected = other_events.get(public, [])
            canonical = compiler._canonicalize_position_ranges(selected)
            delta = compiler._subtract_position_ranges(canonical, covered)
            if delta:
                raw = np.sort(np.concatenate([inverse[a:b] for a, b in delta]))
                if len(raw) and raw[-1] >= cursor:
                    raise ValueError(f"drop_message[{public}] references unborn tokens")
                cuts = np.r_[0, np.flatnonzero(np.diff(raw) != 1)+1, len(raw)]
                for a, b in pairwise(cuts):
                    commands.append((cursor, 1, int(raw[a]), int(raw[b-1])+1))
                expiry[raw] = cursor
                covered = compiler._canonicalize_position_ranges([*covered, *delta])
                live = compiler._subtract_position_ranges(live, delta)
            if public in repo_set:
                if not live:
                    raise ValueError(f"reposition {public} has no active tokens")
                commands.append((cursor, 2, 0, 0))

    tokens = np.asarray(provenance.input_ids, dtype=np.int32)[ranks]
    layout = compile_execution_layout(tokens, commands, ranks, spans)
    return ContextProgram(layout, torch.from_numpy(expiry))


def compile_chat_program(
    raw_messages: list[dict[str, Any]],
    provenance: TemplateTokenProvenance,
    drop_rule: DropRule | None,
    reposition: list[int] | None,
    insert: dict | None = None,
) -> ContextProgram:
    """Compile one canonical native-template result; never render a prefix again.

    The caller supplies messages in public ID order and provenance with those
    same IDs. Keep-text callers supply the full history selected by the rule.
    This function is called only after the no-feature fast-path decision.
    """
    if any(message.get("index") is not None for message in raw_messages):
        from .rules import InsertRule
        InsertRule.from_payload(insert, raw_messages)
    if insert:
        return _compile_insert_program(raw_messages, provenance, drop_rule, reposition, insert)
    compiler = TokenEventCompiler
    message_count = len(raw_messages)
    epochs = compiler._query_epochs_from_owners(provenance.owners, message_count)
    owner_ranges = compiler._build_owner_position_ranges(provenance.owners)
    event_ranges = {}
    if drop_rule is not None:

        context = DropCompileContext(
            raw_messages=raw_messages,
            owner_ranges=owner_ranges,
            provenance=provenance,
            normalized_message_count=message_count,
            normalize_content=lambda content: _matchable_content(
                {"content": content}, field="messages"
            ),
            rendered_source_start=compiler._rendered_source_start,
            token_ranges_for_char_spans=compiler._token_ranges_for_char_spans,
            canonicalize_ranges=compiler._canonicalize_position_ranges,
            position_ranges_from_ids=compiler._position_ranges_from_ids,
        )
        event_ranges = drop_rule.position_events(context)
    events = compiler._build_position_range_drop_plan(event_ranges, epochs)
    repos = resolve_reposition_token_boundaries(
        reposition, owner_ranges, {i: i for i in range(message_count)}
    )
    layout = compile_context_layout(
        torch.tensor(provenance.input_ids, dtype=torch.int32),
        events.event_insert_offsets,
        events.range_offsets,
        events.raw_ranges,
        repos.raw_boundaries,
        repos.insert_offsets,
    )
    return ContextProgram(layout, events.full_token_visible_until)
