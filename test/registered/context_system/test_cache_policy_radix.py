"""Cache identity must imply equal multilayer computation, including retrigger."""

import random

import pytest
from cache_policy_reference import hidden_fingerprints, simulate
from test_cache_policy_ir import POLICIES, compile_policy
from test_context_radix_key import make_key

pytest_plugins = ("test_cache_policy_ir",)


@pytest.mark.parametrize("policy", POLICIES)
def test_semantic_prefix_collision_audit(policy_modules, key_types, policy):
    rng = random.Random(1711)
    exact, retry = {}, {}
    for _ in range(600):
        n = rng.randrange(2, 10)
        tokens = [rng.randrange(2) for _ in range(n)]
        drops = {
            b: [(i, i + 1) for i in range(b - 1) if rng.random() < 0.3]
            for b in range(2, n + 1)
            if rng.random() < 0.4
        }
        repos = (
            [b for b in range(2, n + 1) if rng.random() < 0.25]
            if policy == POLICIES[0]
            else []
        )
        ref = simulate(tokens, drops, repos, policy=policy)
        p = compile_policy(policy_modules, policy, tokens, drops, repos)
        hidden = hidden_fingerprints(tokens, ref)
        key = make_key(key_types, p.layout)
        for length in range(1, len(p.input_ids) + 1):
            stop = key.context.token_to_record[length - 1] + 1
            fingerprint = tuple(hidden[i] for i in p.execution_to_full[:length])
            current = tuple(zip(fingerprint, p.layout.positions[:length].tolist()))
            identity = tuple(key.context.records[: 3 * stop])
            assert exact.setdefault(identity, current) == current
            if p.allow_retry:
                rotation_identity = tuple(key.context.retry_records[: 3 * stop])
                assert retry.setdefault(rotation_identity, fingerprint) == fingerprint


def test_kv_policy_cannot_retrigger_even_if_requested_directly(
    policy_modules, key_types
):
    p = compile_policy(policy_modules, POLICIES[1], list(range(8)), {7: [(2, 3)]})
    source = compile_policy(policy_modules, POLICIES[1], list(range(8)), {})
    target, old = (make_key(key_types, item.layout) for item in (p, source))
    assert target.match(old) == 3
    assert target.match_at(old, 0, context_retry=True) == 3


def test_policy_namespaces_never_alias(policy_modules, key_types):
    a, b = [
        make_key(
            key_types, compile_policy(policy_modules, policy, [1, 2, 3], {}).layout
        )
        for policy in POLICIES
    ]
    assert a.match(b) == 0
    assert a.match_at(b, 0, context_retry=True) == 0
    assert a.child_key() != b.child_key()
    assert a.hash_page(0, 1) != b.hash_page(0, 1)
