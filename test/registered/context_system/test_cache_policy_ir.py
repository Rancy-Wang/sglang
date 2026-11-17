"""Independent event simulation against production compilation and occurrence reads."""

import importlib
import random
from array import array
from types import SimpleNamespace

import pytest
import torch
from cache_policy_reference import simulate
from test_ir import ROOT, load_file

pytest_plugins = ("test_rules", "test_context_radix_key")
POLICIES = ("text_drop_with_pe", "message_drop_without_repos")


@pytest.fixture(scope="module")
def policy_modules(context_modules):
    load_file(
        "sglang.kernels.ops.attention.cache_policy_plan",
        ROOT / "python/sglang/kernels/ops/attention/cache_policy_plan.py",
    )
    return context_modules


def compile_policy(modules, policy, tokens, drops, repos=()):
    rules, _, planner = modules
    messages = [{"role": "user", "content": str(t)} for t in tokens]
    rule = rules.parse_drop_rule(
        policy,
        messages,
        legacy_drop_message={
            b - 1: [i for a, z in spans for i in range(a, z)]
            for b, spans in drops.items()
        },
    )
    return planner.compile_chat_program(
        messages,
        SimpleNamespace(input_ids=tokens, owners=list(range(len(tokens)))),
        rule,
        [b - 1 for b in repos],
    )


def reads(program, modules):
    occurrence = importlib.import_module(modules[2].__package__ + ".occurrence")
    n = len(program.input_ids)
    w = occurrence.compile_occurrence_window(
        program.layout,
        program.visible_until,
        program.layout.positions,
        query_start=0,
        query_end=n,
    )
    raw, pos = w.occurrence_raw_tokens.tolist(), w.occurrence_positions.tolist()
    result = []
    for a, b, x, y in zip(
        w.segment_query_starts.tolist(),
        w.segment_query_ends.tolist(),
        w.segment_key_offsets[:-1].tolist(),
        w.segment_key_offsets[1:].tolist(),
    ):
        ids = w.segment_key_occurrences[x:y].tolist()
        prefix = len(ids) - (b - a)
        for q in range(a, b):
            result.append([(raw[k], pos[k]) for k in ids[: prefix + q - a + 1]])
    return result


@pytest.mark.parametrize("policy", POLICIES)
def test_random_history_positions_and_every_attention_reader(policy_modules, policy):
    rng = random.Random(20261117)
    for _ in range(400):
        n = rng.randrange(3, 40)
        tokens = [rng.randrange(5) for _ in range(n)]
        drops = {
            b: [(i, i + 1) for i in range(b - 1) if rng.random() < 0.22]
            for b in range(2, n + 1)
            if rng.random() < 0.3
        }
        repos = (
            [b for b in range(2, n + 1) if rng.random() < 0.25]
            if policy == POLICIES[0]
            else []
        )
        ref = simulate(tokens, drops, repos, policy=policy)
        p = compile_policy(policy_modules, policy, tokens, drops, repos)
        projection = p.execution_to_full.tolist()
        assert projection == (ref.active if policy == POLICIES[0] else list(range(n)))
        assert p.layout.positions.tolist() == [ref.positions[i] for i in projection]
        assert p.layout.birth_positions.tolist() == [
            ref.compute_positions[i] for i in projection
        ]
        assert p.layout.next_position == ref.next_position
        actual = [
            [(projection[j], pos) for j, pos in row] for row in reads(p, policy_modules)
        ]
        assert actual == [ref.readers[i] for i in projection]
        for wire in (p.to_wire(), p.to_json_wire()):
            restored = policy_modules[2].ContextProgram.from_wire(wire, p.input_ids)
            assert torch.equal(restored.layout.records, p.layout.records)
            assert restored.allow_retry == (policy == POLICIES[0])


def test_examples_and_compute_identity(policy_modules, key_types):
    from test_context_radix_key import make_key

    a = compile_policy(policy_modules, POLICIES[0], [0, 1, 2, 3], {3: [(1, 2)]})
    rotated = compile_policy(
        policy_modules, POLICIES[0], [0, 1, 2, 3], {3: [(1, 2)]}, [4]
    )
    assert a.layout.positions.tolist() == [0, 2, 3]
    assert rotated.layout.positions.tolist() == [0, 1, 2]
    assert rotated.layout.birth_positions.tolist() == [0, 2, 3]
    ka, kb = (make_key(key_types, p.layout) for p in (a, rotated))
    assert ka.match(kb) == 1
    assert ka.match_at(kb, 0, context_retry=True) == 3
    compact = compile_policy(policy_modules, POLICIES[0], [0, 2, 3], {})
    # Identical token IDs and stored positions do NOT mean equal hidden states.
    kc = make_key(key_types, compact.layout)
    assert kb.match(kc) == 1
    assert kb.match_at(kc, 0, context_retry=True) == 1
    b = compile_policy(policy_modules, POLICIES[1], [0, 1, 2, 3], {4: [(1, 2)]})
    assert b.layout.positions.tolist() == [0, 1, 1, 2]
    assert reads(b, policy_modules)[3] == [(0, 0), (1, 1), (2, 1), (3, 2)]
    assert reads(b.with_generated([4]), policy_modules)[4] == [
        (0, 0),
        (2, 1),
        (3, 2),
        (4, 3),
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_generated_key_retraction_and_namespace(policy_modules, key_types, policy):
    from test_context_radix_key import make_key

    p = compile_policy(
        policy_modules,
        policy,
        [0, 1, 2, 3],
        {3: [(1, 2)]},
        [4] if policy == POLICIES[0] else [],
    )
    extended = p.with_generated([6, 7, 8])
    data = key_types[1].from_layout(p.layout)
    data.append_tokens(array("q", [6, 7, 8]), next_position=p.layout.next_position)
    target = key_types[1].from_layout(extended.layout)
    assert data.records == target.records
    assert data.retry_records == target.retry_records
    assert data.token_to_record == target.token_to_record
    policy_modules[2].ContextProgram.from_wire(
        extended.to_json_wire(), extended.input_ids
    )
    plain = key_types[0](array("q", p.input_ids))
    key = make_key(key_types, p.layout)
    assert key.match(plain) == 0
    assert key.child_key() != plain.child_key()
    assert key.context_retry_child_key() != plain.context_retry_child_key()


def test_deleted_text_not_in_identity_and_repeated_drop_no_reset(policy_modules):
    a = compile_policy(
        policy_modules, POLICIES[0], [0, 11, 2, 3, 4], {3: [(1, 2)]}, [4]
    )
    b = compile_policy(
        policy_modules, POLICIES[0], [0, 99, 2, 3, 4], {3: [(1, 2)], 5: [(1, 2)]}, [4]
    )
    assert torch.equal(a.layout.records, b.layout.records)
    assert torch.equal(a.layout.birth_positions, b.layout.birth_positions)


def test_frozen_death_repeated_events_and_simultaneous_drop(policy_modules):
    p = compile_policy(
        policy_modules,
        POLICIES[1],
        list(range(8)),
        {4: [(1, 2), (3, 4)], 7: [(0, 2), (5, 6)]},
    )
    ref = simulate(
        list(range(8)), {4: [(1, 2), (3, 4)], 7: [(0, 2), (5, 6)]}, policy=POLICIES[1]
    )
    assert p.layout.positions.tolist() == ref.positions
    assert ref.positions[3] == 3  # not compressed by the simultaneous removal of 1


def test_reject_empty_and_reposition_for_kv(policy_modules):
    with pytest.raises(ValueError, match="active token"):
        compile_policy(policy_modules, POLICIES[0], [0, 1], {2: [(0, 2)]})
    with pytest.raises(ValueError, match="does not accept reposition"):
        compile_policy(policy_modules, POLICIES[1], [0, 1], {}, [2])
