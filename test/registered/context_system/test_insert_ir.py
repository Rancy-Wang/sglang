"""Independent Insert histories; native compiler and occurrence planning, no model."""
import importlib
import random
from types import SimpleNamespace

import pytest
import torch

pytest_plugins = ("test_rules",)


def program(context_modules, *, nested=False, repos=True, redundant=False, width=1):
    rules, _, planner = context_modules
    ids = [*range(6), *([-2] if nested else []), -1, *range(6, 10)]
    messages = [{"role": "user", "content": str(i), **({"index": i} if i < 0 else {})}
                for i in ids]
    tokens = [i if i >= 0 else 99-i for i in ids for _ in range(width)]
    trace = SimpleNamespace(input_ids=tokens, owners=[i for i in range(len(ids)) for _ in range(width)])
    drops = {3: [1, 2], 8: ([1, 2, 3, 4, 5, 3] if redundant else [3, 4, 5])}
    rule = rules.parse_drop_rule(None, messages, legacy_drop_message=drops)
    insert = {9: -1, **({-1: -2} if nested else {})}
    return planner.compile_chat_program(messages, trace, rule, [3, -1] if repos else [], insert)


def all_reads(context_modules, compiled):
    occurrence = importlib.import_module(context_modules[2].__package__ + ".occurrence")
    layout = compiled.layout
    window = occurrence.compile_occurrence_window(
        layout, compiled.visible_until, layout.positions, query_start=0,
        query_end=len(layout.positions))
    pairs = list(zip(window.occurrence_raw_tokens.tolist(), window.occurrence_positions.tolist()))
    rows = {}
    for a, b, x, y in zip(window.segment_query_starts.tolist(), window.segment_query_ends.tolist(),
                          window.segment_key_offsets[:-1].tolist(), window.segment_key_offsets[1:].tolist()):
        selected = window.segment_key_occurrences[x:y].tolist()
        prefix = len(selected) - (b-a)
        for q in range(a, b):
            rows[q] = [pairs[k] for k in selected[:prefix+q-a+1]]
    return rows


def test_summary_history(context_modules):
    p = program(context_modules)
    assert p.input_ids == [*range(9), 100, 9]
    assert p.layout.insert_spans.tolist() == [9, 10, 7]
    assert p.layout.drop_ranges.tolist() == [1, 3, 3, 6]
    assert [r for r in p.layout.records.tolist() if r[0] in (3, 4)] == [[3, 7, 0], [4, 0, 0]]
    rows = all_reads(context_modules, p)
    assert rows[8] == [(0, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6)]
    assert rows[9] == [(0, 0), (9, 4)]
    assert rows[10] == [(0, 0), (9, 1), (6, 2), (7, 3), (8, 4), (10, 5)]


def test_negative_chain_and_delta(context_modules):
    p = program(context_modules, nested=True)
    assert p.input_ids == [*range(9), 100, 101, 9]
    assert p.layout.insert_spans.tolist() == [9, 10, 7, 10, 11, 10]
    rows = all_reads(context_modules, p)
    assert rows[9] == [(0, 0), (9, 4)]
    assert rows[10] == [(0, 0), (10, 4)]
    assert rows[11] == [(0, 0), (10, 1), (9, 2), (6, 3), (7, 4), (8, 5), (11, 6)]
    repeated = program(context_modules, nested=True, redundant=True)
    assert torch.equal(p.layout.records, repeated.layout.records)
    assert torch.equal(p.visible_until, repeated.visible_until)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("repos", [False, True])
@pytest.mark.parametrize("width", [1, 7])
def test_decode_order_wire_and_chunks(context_modules, nested, repos, width):
    p = program(context_modules, nested=nested, repos=repos, width=width)
    _, _, planner = context_modules
    restored = planner.ContextProgram.from_wire(p.to_wire(), p.input_ids)
    binary = planner.ContextProgram.from_wire(p.to_json_wire(), p.input_ids)
    assert torch.equal(restored.layout.logical_order, binary.layout.logical_order)
    occurrence = importlib.import_module(planner.__package__ + ".occurrence")
    decoded = occurrence.ContextDecodeLayout.from_layout(p.layout, "cpu")
    assert p.layout.logical_ranks[decoded.raw_indices].tolist() == sorted(p.layout.logical_ranks[decoded.raw_indices].tolist())
    expected = all_reads(context_modules, p)
    n = len(p.input_ids)
    for a, b in [(0, 2), (n//2, n), (n-2, n)]:
        w = occurrence.compile_occurrence_window(p.layout, p.visible_until, p.layout.positions,
                                                query_start=a, query_end=b)
        pairs = list(zip(w.occurrence_raw_tokens.tolist(), w.occurrence_positions.tolist()))
        for start, end, x, y in zip(w.segment_query_starts.tolist(), w.segment_query_ends.tolist(),
                                    w.segment_key_offsets[:-1].tolist(), w.segment_key_offsets[1:].tolist()):
            selected = w.segment_key_occurrences[x:y].tolist()
            prefix = len(selected)-(end-start)
            for q in range(start, end):
                assert [pairs[k] for k in selected[:prefix+q-start+1]] == expected[q]


def test_exact_and_retrigger_positions(context_modules):
    ir = importlib.import_module(context_modules[2].__package__ + ".ir")
    before = program(context_modules, repos=False)
    after = program(context_modules, repos=True)
    # Different historical R operations must not become equivalent by retry.
    a, b = ir.ContextKeyData.from_layout(before.layout), ir.ContextKeyData.from_layout(after.layout)
    assert a.match(b, 0, 0, len(before.input_ids), retry=True) < len(before.input_ids)


@pytest.mark.parametrize("seed", range(60))
def test_random_mixed_turns_against_state_machine(context_modules, seed):
    """Independent mutable list oracle, including nested and repeated summaries."""
    rules, _, planner = context_modules
    ir = importlib.import_module(planner.__package__ + ".ir")
    rng = random.Random(seed)
    logical, active, token_owner, positions, reads = [], [], [], {}, {}
    widths, inserts, drops, repos = {}, {}, {}, []
    negative, cursor, previous = -1, 0, None

    def birth(public, before=None):
        nonlocal cursor
        widths[public] = rng.randrange(1, 5)
        raw = len(token_owner)
        location = len(active) if before is None else active.index(before)
        p = cursor if before is None else positions[before]
        for offset in range(widths[public]):
            q = raw + offset
            positions[q] = p + offset
            active.insert(location + offset, q)
            token_owner.append(public)
            reads[q] = [(r, positions[r]) for r in active[:location + offset + 1]]
        if before is None:
            cursor = p + widths[public]
        return raw

    for turn in range(18):
        if turn > 2 and active and rng.random() < .6:
            # Pick the first active token of an existing owner, never split
            # the middle of a message in this public-interface oracle.
            anchor_owner = rng.choice(list(dict.fromkeys(token_owner[r] for r in active)))
            anchor = next(r for r in active if token_owner[r] == anchor_owner)
            logical.insert(logical.index(anchor_owner), negative)
            inserts[turn] = negative
            born_anchor = birth(negative, anchor)
            parent = negative
            negative -= 1
            if rng.random() < .4:
                logical.insert(logical.index(parent), negative)
                inserts[parent] = negative
                birth(negative, born_anchor)
                negative -= 1
            if rng.random() < .5:
                repos.append(parent)
                changed = any(positions[r] != p for p, r in enumerate(active))
                for p, r in enumerate(active):
                    positions[r] = p
                if changed:
                    cursor = len(active)
        logical.append(turn)
        birth(turn)
        if rng.random() < .65:
            chosen = rng.sample(list(widths), rng.randrange(min(3, len(widths)) + 1))
            drops[turn] = chosen + chosen[:1]  # repeated entries accepted
            active[:] = [r for r in active if token_owner[r] not in chosen]
        if active and rng.random() < .6:
            repos.append(turn)
            changed = any(positions[r] != p for p, r in enumerate(active))
            for p, r in enumerate(active):
                positions[r] = p
            if changed:
                cursor = len(active)
        ms = [dict(role="user", content=str(i), **({"index": i} if i < 0 else {})) for i in logical]
        tokens = [1000+i for i in logical for _ in range(widths[i])]
        owners = [o for o, i in enumerate(logical) for _ in range(widths[i])]
        trace = SimpleNamespace(input_ids=tokens, owners=owners)
        rule = rules.parse_drop_rule(None, ms, legacy_drop_message=drops)
        p = planner.compile_chat_program(ms, trace, rule, repos, inserts)
        assert all_reads(context_modules, p) == reads, (seed, turn)
        assert p.layout.positions.tolist() == [positions[r] for r in range(len(token_owner))]
        key = ir.ContextKeyData.from_layout(p.layout)
        if previous is not None:
            # A future final-position rewrite may reduce exact matching,
            # but cannot shorten operation-compatible retrigger matching.
            assert previous.match(key, 0, 0, len(previous.positions), retry=True) == len(previous.positions), (seed, turn)
        previous = key
