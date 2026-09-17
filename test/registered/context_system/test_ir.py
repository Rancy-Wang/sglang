"""CPU semantics tests; load only the compiler, without initializing SRT/CUDA.

Run with pytest. MINI_SGLANG_REFERENCE optionally enables differential checks
against a read-only mini checkout in addition to the independent event oracle.
These tests do not claim scheduler or GPU coverage.
"""

import dataclasses
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compiler():
    kernel = load_file(
        "context_test_kernel",
        ROOT / "python/sglang/kernels/ops/attention/context_plan.py",
    )
    ir = load_file("context_test_ir", ROOT / "python/sglang/srt/context_system/ir.py")
    ir._load_module = kernel.load_context_plan
    ir.prewarm_context_layout()
    return ir.compile_context_layout


def args(tokens, drops, reposition):
    offsets, ranges = [0], []
    for spans in drops.values():
        ranges.extend(value for span in spans for value in span)
        offsets.append(len(ranges) // 2)
    values = (
        tokens,
        list(drops),
        offsets,
        ranges,
        reposition,
        [boundary + 1 for boundary in reposition],
    )
    return tuple(torch.tensor(value, dtype=torch.int32) for value in values)


def oracle(tokens, drops, reposition):
    """Deliberately simple staged simulator, independent of linked-list compiler."""
    n = len(tokens)
    active = []
    positions, birth, birth_stages, repos = [], [], [], []
    ready, effective, ignored, effective_stages = [], [], [], []
    changed, old, new, transition_offsets = [], [], [], [0]
    stage, next_position, current = 0, 0, -1
    for insertion in range(n + 1):
        for begin, end in drops.get(insertion, ()):
            active = [raw for raw in active if not begin <= raw < end]
        if insertion - 1 in reposition:
            if not active:
                raise ValueError("empty active set")
            changes = [
                (raw, rank) for rank, raw in enumerate(active) if positions[raw] != rank
            ]
            effective.append(bool(changes))
            ignored.append(not changes)
            effective_stages.append(stage + 1 if changes else -1)
            if changes:
                stage += 1
                current = insertion - 1
                for raw, rank in changes:
                    changed.append(raw)
                    old.append(positions[raw])
                    new.append(rank)
                    positions[raw], repos[raw], ready[raw] = rank, current, stage
                transition_offsets.append(len(changed))
                next_position = len(active)
        if insertion < n:
            active.append(insertion)
            positions.append(next_position)
            birth.append(next_position)
            birth_stages.append(stage)
            ready.append(stage)
            repos.append(current)
            next_position += 1
    keys, virtual, key_raw, raw_key, drop_key = [], [], [], [], []
    effective_by_boundary = dict(zip(reposition, effective))
    for insertion in range(n + 1):
        if insertion in drops:
            drop_key.append(len(keys))
        for begin, end in drops.get(insertion, ()):
            keys.append([1, -begin - 1, -end - 1, -1])
            virtual.append(True)
            key_raw.append(-1)
        if effective_by_boundary.get(insertion - 1, False):
            keys.append([2, insertion - 1, -1, -1])
            virtual.append(True)
            key_raw.append(-1)
        if insertion < n:
            raw_key.append(len(keys))
            keys.append([0, tokens[insertion], repos[insertion], positions[insertion]])
            virtual.append(False)
            key_raw.append(insertion)
    return {
        "records": keys,
        "virtual_mask": virtual,
        "key_to_token": key_raw,
        "token_to_key": raw_key,
        "positions": positions,
        "repos_info": repos,
        "keep_mask": [raw in active for raw in range(n)],
        "materialized_stage": ready,
        "birth_positions": birth,
        "birth_stages": birth_stages,
        "transition_offsets": transition_offsets,
        "transition_raw_tokens": changed,
        "transition_old_positions": old,
        "transition_new_positions": new,
        "effective_reposition_stages": effective_stages,
        "drop_event_to_key": drop_key,
        "effective_repositions": effective,
        "ignored_repositions": ignored,
        "next_position": next_position,
        "current_reposition": current,
    }


def cases():
    yield [7, 8, 9, 10], {3: [(1, 2)]}, [2]
    yield [7, 8, 9, 10], {3: [(2, 3)]}, [2]  # tail-only R is ignored
    yield [7, 7, 7, 7], {2: [(0, 1)], 3: [(0, 1)]}, [1, 2]
    yield [7, 8, 9], {3: [(0, 2)]}, [2]  # R at terminal insertion
    rng = random.Random(20260917)
    for _ in range(300):
        n = rng.randint(1, 160)
        tokens = [rng.randrange(12) for _ in range(n)]
        drops, repositions = {}, []
        for insertion in sorted(rng.sample(range(1, n + 1), min(n, 8))):
            if insertion > 1 and rng.random() < 0.7:
                begin = rng.randrange(insertion - 1)
                # Keep the newest token active, so R is always well-defined.
                end = rng.randrange(begin + 1, insertion)
                drops[insertion] = [(begin, end)]
            if rng.random() < 0.6:
                repositions.append(insertion - 1)
        yield tokens, drops, repositions


@pytest.mark.parametrize("tokens,drops,reposition", list(cases()))
def test_staged_event_oracle(compiler, tokens, drops, reposition):
    actual = compiler(*args(tokens, drops, reposition))
    for name, expected in oracle(tokens, drops, reposition).items():
        value = getattr(actual, name)
        assert (
            value.tolist() if isinstance(value, torch.Tensor) else value
        ) == expected, name


def test_empty_reposition_rejected(compiler):
    with pytest.raises(ValueError, match="no active tokens"):
        compiler(*args([1, 2], {2: [(0, 2)]}, [1]))


@pytest.mark.parametrize("field", range(6))
def test_narrowing_cannot_wrap(compiler, field):
    values = list(args([1, 2], {1: [(0, 1)]}, [1]))
    values[field] = torch.tensor([2**32], dtype=torch.int64)
    with pytest.raises(ValueError, match="int32"):
        compiler(*values)


@pytest.mark.parametrize("field", range(6))
def test_float_metadata_rejected(compiler, field):
    values = list(args([1, 2], {}, []))
    values[field] = values[field].float()
    with pytest.raises(ValueError, match="integer tensor"):
        compiler(*values)


def test_large_raw_stream_keeps_position_compact(compiler):
    n = 140_000
    layout = compiler(*args([10] * n, {100_000: [(0, 90_000)]}, [99_999]))
    assert len(layout.positions) == n
    assert layout.next_position == 50_000
    assert layout.positions[-1] == 49_999
    assert layout.keep_mask.sum() == 50_000


def test_fixed_mini_compiler_differential(compiler):
    source = os.environ.get("MINI_SGLANG_REFERENCE")
    if not source:
        pytest.skip("set MINI_SGLANG_REFERENCE to the fixed read-only checkout")
    kernel = Path(source) / "python/minisgl/kernel"
    package = types.ModuleType("mini_context_reference")
    package.__path__ = [str(kernel)]
    sys.modules[package.__name__] = package
    reference = load_file(
        package.__name__ + ".radix_reposition", kernel / "radix_reposition.py"
    )
    for tokens, drops, reposition in cases():
        values = args(tokens, drops, reposition)
        expected = reference.compile_radix_reposition_layout(*values)
        actual = compiler(*values)
        for field in dataclasses.fields(actual):
            if field.name == "compile_ns":
                continue
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            assert (
                torch.equal(left, right)
                if isinstance(left, torch.Tensor)
                else left == right
            )


def test_generated_suffix_preserves_final_state_and_trailing_events(compiler):
    extend = compiler.__globals__["append_generated_layout"]
    for drops, repos in (({}, []), ({4: [(1, 3)]}, [3]), ({8: [(0, 2)]}, [7])):
        tokens = list(range(8))
        layout = compiler(*args(tokens, drops, repos))
        before = {
            name: value.clone()
            for name, value in vars(layout).items()
            if isinstance(value, torch.Tensor)
        }
        suffix = [11, 12, 13]
        updated = extend(layout, suffix)
        assert updated.positions[-3:].tolist() == list(
            range(layout.next_position, layout.next_position + 3)
        )
        assert updated.birth_positions[-3:].tolist() == updated.positions[-3:].tolist()
        assert (
            updated.birth_stages[-3:].tolist()
            == [len(layout.transition_offsets) - 1] * 3
        )
        assert updated.keep_mask[-3:].all()
        assert updated.records[updated.token_to_key, 1].tolist() == tokens + suffix
        for name, value in before.items():
            torch.testing.assert_close(getattr(layout, name), value)
        assert extend(layout, []) is layout
        with pytest.raises(ValueError, match="int32"):
            extend(layout, [-1])
