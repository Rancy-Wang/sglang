"""Insert records through native radix splitting, retrigger and publication."""

import sys
from pathlib import Path

import pytest
import torch
from test_insert_ir import program

pytest_plugins = ("test_rules",)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT requires Linux")


@pytest.mark.parametrize("swa", [False, True])
def test_insert_match_publication_and_branch_isolation(context_modules, monkeypatch, swa):
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
    components = (ComponentType.FULL, ComponentType.SWA) if swa else (ComponentType.FULL,)
    cache, allocator, _ = native.build_fixture(native.CacheConfig(components=components, kv_size=512))
    try:
        p = program(context_modules, nested=True, width=7)
        target = RadixKey.from_context(p.layout)
        # Only final TOKEN positions differ: this is precisely retrigger's
        # legal rotation case. Virtual history and birth-depth stay immutable.
        source = RadixKey.from_context(p.layout)
        from dataclasses import replace
        records = p.layout.records.clone()
        records[p.layout.token_to_key, 2] += 100
        source = RadixKey.from_context(replace(p.layout, records=records, positions=p.layout.positions + 100))
        saved_source = source.context.records[:]
        slots = allocator.alloc(len(source))
        cache.insert(InsertParams(key=source, value=slots))
        match = cache.match_prefix(MatchPrefixParams(key=target[:-1], context_retry=True))
        assert len(match.device_indices) == len(target)-1
        assert match.context_retry
        assert match.context_source_positions.tolist() == (p.layout.positions[:-1]+100).tolist()
        # A separate target allocation models the scheduler's mandatory COW.
        new_slots = allocator.alloc(len(target))
        cache.insert(InsertParams(key=target, value=new_slots))
        for key, expected in ((source, slots), (target, new_slots)):
            hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
            assert len(hit.device_indices) == len(key)
            assert not hit.context_retry
            assert torch.equal(hit.device_indices, expected)
        assert source.context.records == saved_source
        # End markers, including a prefix ending immediately after IE, survive
        # page-one edge splits and re-publication.
        for end in p.layout.insert_spans.reshape(-1, 3)[:, 1].tolist():
            hit = cache.match_prefix(MatchPrefixParams(key=target[:end], context_retry=True))
            assert len(hit.device_indices) == end
    finally:
        cache.evict(EvictParams(num_tokens=512, swa_num_tokens=512))
        assert allocator.available_size() == 512
        reset_context()
