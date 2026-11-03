"""Sequential native tree publication, beyond standalone key comparison."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_insert_state_audit import audit

pytest_plugins = ("test_rules",)
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native SRT requires Linux"
)


@pytest.mark.parametrize("swa", [False, True])
def test_long_mixed_native_frontier(context_modules, monkeypatch, swa):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import reset_context

    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    components = (
        (ComponentType.FULL, ComponentType.SWA) if swa else (ComponentType.FULL,)
    )
    cache, allocator, _ = native.build_fixture(
        native.CacheConfig(
            components=components,
            kv_size=32768,
            sliding_window_size=16 if swa else None,
        )
    )
    rules, _, planner = context_modules
    ids, inserts, drops, repos, dropped = [], {}, {}, [], set()
    negative, previous, snapshots = -1, None, []
    try:
        for turn in range(40):
            if turn >= 9 and turn % 4 == 1:
                active = [i for i in ids if i not in dropped and i != 0]
                anchor = next((i for i in active if i < 0), active[-3])
                ids.insert(ids.index(anchor), negative)
                inserts[turn] = negative
                # Insert before a previous Insert, and a same-request nested pair.
                if turn % 8 == 1:
                    ids.insert(ids.index(negative), negative - 1)
                    inserts[negative] = negative - 1
                    negative -= 1
                if turn % 12 != 1:
                    repos.append(inserts[turn])
                negative -= 1
            ids.append(turn)
            if turn > 2:
                chosen = [i for i in range(1, turn - 2) if i not in dropped][:2]
                drops[turn] = chosen + [1, 1]  # repeated drop compiles to delta
                dropped.update(drops[turn])
            if turn % 3 == 0 and turn:
                repos.append(turn)
            messages = [dict(index=i, role="user", content=str(i)) for i in ids]
            width = lambda i: 2 + abs(i) % 5
            trace = SimpleNamespace(
                input_ids=[1000 + i for i in ids for _ in range(width(i))],
                owners=[j for j, i in enumerate(ids) for _ in range(width(i))],
            )
            rule = rules.parse_drop_rule(None, messages, legacy_drop_message=drops)
            program = planner.compile_chat_program(
                messages, trace, rule, repos, inserts
            )
            key = RadixKey.from_context(program.layout)
            if previous is not None:
                before = {
                    mode: audit.brute_frontier(
                        cache.tree_core.root_node, key, retry=mode == "retry"
                    )[0]
                    for mode in ("exact", "retry")
                }
                hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
                if key.context.has_insert:
                    report = audit.audit_match(cache, key, hit, before)
                    assert report["pass"], (turn, report)
                    assert len(hit.device_indices) >= len(previous), (
                        turn,
                        len(previous),
                        report,
                    )
            slots = allocator.alloc(len(key))
            assert slots is not None
            cache.insert(InsertParams(key=key, value=slots))
            hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
            assert len(hit.device_indices) == len(key), turn
            assert not hit.context_retry, turn
            snapshots.append((key, key.context.records[:]))
            assert all(k.context.records == saved for k, saved in snapshots)
            previous = key
    finally:
        cache.evict(EvictParams(num_tokens=32768, swa_num_tokens=32768))
        reset_context()
