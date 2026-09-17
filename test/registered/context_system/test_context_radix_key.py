"""Exercise the production RadixKey class without initializing the CUDA runtime."""

import ast
import hashlib
import random
from array import array

import pytest
import torch
from test_ir import ROOT, args, cases, load_file

pytest_plugins = ("test_ir",)


@pytest.fixture(scope="module")
def key_types():
    path = ROOT / "python/sglang/srt/mem_cache/radix_cache.py"
    source = ast.parse(path.read_text())
    key_class = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RadixKey"
    )
    # Execute the actual class, including slices, limits and native LCP. Importing
    # the complete module would initialize unrelated hardware/model dependencies.
    module = ast.Module(body=[source.body[0], key_class], type_ignores=[])
    namespace = {"array": array, "torch": torch}
    namespace["get_hash_str"] = lambda key, prior: hashlib.sha256(
        (prior or "").encode() + array("q", key).tobytes()
    ).hexdigest()
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - local source under test
    ir = load_file("context_key_ir", ROOT / "python/sglang/srt/context_system/ir.py")
    return namespace["RadixKey"], ir.ContextKeyData


def make_key(types, layout):
    key, data = types
    return key(
        array("q", layout.records[layout.token_to_key, 1].tolist()),
        context=data.from_layout(layout),
    )


def units(layout):
    records = layout.records.tolist()
    previous = 0
    result = []
    for index in layout.token_to_key.tolist():
        result.append(records[previous : index + 1])
        previous = index + 1
    return result


def prefix(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def test_structured_native_key_tokens_and_slices(compiler, key_types):
    rng = random.Random(49238)
    for tokens, drops, reposition in cases():
        layout = compiler(*args(tokens, drops, reposition))
        key = make_key(key_types, layout)
        oracle = units(layout)
        assert len(key) == len(tokens)
        assert list(key) == tokens
        other = make_key(key_types, compiler(*args(tokens, drops, [])))
        other_units = units(compiler(*args(tokens, drops, [])))
        assert key.match(other) == prefix(oracle, other_units)
        assert key.page_aligned(1) is key
        for _ in range(16):
            start = rng.randrange(len(tokens))
            end = rng.randrange(start + 1, len(tokens) + 1)
            edge = key[start:end]
            assert edge.context is key.context
            assert edge.match_at(key, start) == end - start
            assert edge.match_at(other, start) == prefix(
                oracle[start:end], other_units[start:]
            )
            assert edge.child_key() == key.child_key_at(start)
            assert (edge.child_key() == other.child_key_at(start)) == (
                edge.match_at(other, start) >= 1
            )


def test_plain_prefix_shares_native_keys_and_namespaces(compiler, key_types):
    Key, _ = key_types
    tokens = list(range(96))
    context = make_key(key_types, compiler(*args(tokens, {32: [(2, 9)]}, [])))
    plain = Key(array("q", tokens))
    assert context.match(plain) == plain.match(context) == 32
    assert context.child_key() == plain.child_key()
    assert context[32:].child_key() != plain[32:].child_key()
    assert context[33:].match(plain[33:]) == 63
    for key in (context, plain):
        key.extra_key, key.cache_salt = "adapter", "tenant"
        assert key[1:4].extra_key == "adapter"
        assert key[1:4].cache_salt == "tenant"
    assert context.match(plain) == 32
    plain.cache_salt = "different"
    with pytest.raises(ValueError, match="cache_salt"):
        context.match(plain)


def test_event_identity_final_positions_and_tail_events(compiler, key_types):
    tokens = [7] * 10
    drop_a = make_key(key_types, compiler(*args(tokens, {4: [(0, 1)]}, [])))
    drop_b = make_key(key_types, compiler(*args(tokens, {4: [(1, 2)]}, [])))
    assert drop_a.match(drop_b) == 4  # Same text and length, different history.
    assert drop_a[4:].child_key() != drop_b[4:].child_key()
    repos = make_key(key_types, compiler(*args(tokens, {4: [(0, 1)]}, [9])))
    assert repos.match(drop_a) == 1  # The dropped first token was never rotated.
    # Tail Drop changes no already-computed KV; its event must join the next
    # real token. Tail Reposition changed positions and was distinguished above.
    tail = make_key(key_types, compiler(*args(tokens, {10: [(0, 1)]}, [])))
    plain = key_types[0](array("q", tokens))
    assert tail.match(plain) == 10
    later = make_key(key_types, compiler(*args(tokens + [7], {10: [(0, 1)]}, [])))
    assert later[:10].match(tail) == 10
    assert later[10:].child_key() != plain[:1].child_key()
    assert drop_a.hash_page(4, 8) != drop_b.hash_page(4, 8)
    with pytest.raises(ValueError, match="bigrams"):
        drop_a.maybe_to_bigram_view(True)


def test_retry_preserves_events_and_ignores_only_token_positions(compiler, key_types):
    for tokens, drops, reposition in cases():
        target = make_key(key_types, compiler(*args(tokens, drops, reposition)))
        plain_repos = make_key(key_types, compiler(*args(tokens, drops, [])))
        left = units(compiler(*args(tokens, drops, reposition)))
        right = units(compiler(*args(tokens, drops, [])))
        for sequence in (left, right):
            for unit in sequence:
                unit[-1][2:] = [0, 0]
        assert target.match_at(plain_repos, 0, context_retry=True) == prefix(
            left, right
        )
        assert (
            target.context_retry_child_key() == plain_repos.context_retry_child_key()
        ) == (prefix(left, right) > 0)


@pytest.mark.parametrize("page_size", [4, 16, 64])
def test_context_rejects_large_pages_but_native_keys_keep_them(
    compiler, key_types, page_size
):
    Key, _ = key_types
    tokens = list(range(128))
    context = make_key(key_types, compiler(*args(tokens, {64: [(0, 16)]}, [])))
    plain = Key(array("q", tokens))
    for operation in (
        lambda: context.page_aligned(page_size),
        lambda: context.child_key(page_size),
        lambda: context.match(context, page_size),
        lambda: context.match(plain, page_size),
        lambda: plain.match(context, page_size),
        lambda: context.match_at(context, 0, page_size, context_retry=True),
    ):
        with pytest.raises(ValueError, match="page_size=1"):
            operation()
    assert len(plain.page_aligned(page_size)) == len(tokens)
    assert plain.match(plain, page_size) == len(tokens)
    assert plain.child_key(page_size) == tuple(tokens[:page_size])
