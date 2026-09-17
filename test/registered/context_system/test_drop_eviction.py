"""Drop lease, persistent candidates and real native allocator pressure tests."""

import sys
from array import array
from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)


def test_candidate_leaf_priority_and_bounded_stale_entries():
    module = load_file(
        "context_drop_eviction", ROOT / "python/sglang/srt/context_system/recovery.py"
    )
    candidates = module.DropEvictionCandidates()
    leaf, internal = SimpleNamespace(id=1), SimpleNamespace(id=2)
    candidates.update(internal, 1, -1000)
    for time in range(1000):
        candidates.update(leaf, 0, time)
    assert sum(map(len, candidates.heaps)) <= 2 * len(candidates.entries) + 64
    assert candidates.pop() == (0, leaf)
    assert candidates.pop() == (1, internal)
    assert candidates.pop() is None
    candidates.update(leaf, 0, 0)
    candidates.update(leaf, None)
    assert candidates.pop() is None


@pytest.fixture(params=["full", "full_swa"])
def native_cache(request):
    if sys.platform != "linux":
        pytest.skip("native SRT runtime requires Linux")
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    components = (ComponentType.FULL,)
    if request.param == "full_swa":
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        pool = SWAKVPool(
            size=128,
            size_swa=128,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=128,
            size_swa=128,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        components += (ComponentType.SWA,)
    else:
        allocator = TokenToKVPoolAllocator(
            size=128, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
        )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=components,
            sliding_window_size=4 if request.param == "full_swa" else None,
        )
    )
    return cache, allocator


def assert_allocator(cache, allocator, used):
    full = getattr(allocator, "full_attn_allocator", allocator)
    assert full.available_size() == 128 - used
    assert len(torch.unique(full.get_all_free_pages())) == 128 - used
    if hasattr(allocator, "swa_attn_allocator"):
        swa = allocator.swa_attn_allocator
        assert len(torch.unique(swa.get_all_free_pages())) == swa.available_size()
        if used == 0:
            assert swa.available_size() == 128
            assert not allocator.full_to_swa_index_mapping[:-1].any()
    cache.tree_core.sanity_check([], [])


def test_shared_reader_leaf_first_hole_refill_and_split(compiler, native_cache):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    key = RadixKey.from_context(compiler(*args(list(range(12)), {6: [(1, 4)]}, [])))
    slots = allocator.alloc(12)
    cache.insert(InsertParams(key=key, value=slots))
    matched = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    lease = cache.inc_lock_ref(matched.last_device_node)
    # Another reader still needs the same physical pages, despite this Drop.
    ordinary = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(4)))))
    reader = cache.inc_lock_ref(ordinary.last_device_node)
    required = torch.ones(12, dtype=torch.bool)
    required[1:4] = False
    cache.configure_context_drop_lock(matched.last_device_node, lease, required)
    assert lease.context_skip_ranges == ((1, 4),)
    assert_allocator(cache, allocator, 12)
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 12)
    cache.dec_lock_ref(ordinary.last_device_node, reader.to_dec_params())
    # A newer ordinary leaf must be reclaimed before an older Drop internal edge.
    cache.insert(
        InsertParams(key=RadixKey(array("q", [90, 91, 92])), value=allocator.alloc(3))
    )
    cache.evict(EvictParams(num_tokens=3))
    assert_allocator(cache, allocator, 12)
    assert cache.match_prefix(
        MatchPrefixParams(key=key, context_retry=True)
    ).context_resident.all()
    cache.evict(EvictParams(num_tokens=3))
    assert_allocator(cache, allocator, 9)
    holed = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    assert holed.context_resident.tolist() == required.tolist()
    assert holed.device_indices[1:4].tolist() == [-1, -1, -1]
    assert torch.equal(holed.device_indices[4:], slots[4:])
    assert (
        len(
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", range(12))))
            ).device_indices
        )
        == 1
    )
    # Split a leased hole after acquiring the receipt; release must follow the
    # inherited raw intervals, not stale node IDs for every edge.
    partial = cache.match_prefix(MatchPrefixParams(key=key[:3], context_retry=True))
    assert partial.context_resident.tolist() == [True, False, False]
    assert_allocator(cache, allocator, 9)
    # Fresh computation fills only missing pages; duplicate resident pages are
    # released once by native insert, and the preserved suffix remains shared.
    fresh = allocator.alloc(12)
    cache.insert(InsertParams(key=key, value=fresh))
    restored = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    assert restored.context_resident.all()
    assert restored.device_indices[1:4].tolist() == fresh[1:4].tolist()
    assert torch.equal(restored.device_indices[4:], slots[4:])
    assert_allocator(cache, allocator, 12)
    cache.dec_lock_ref(matched.last_device_node, lease.to_dec_params())
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)
    assert len(cache.tree_core._node_arena) == 1


def test_unmatched_future_delta_cannot_release_pages(compiler, native_cache):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    key = RadixKey.from_context(compiler(*args(list(range(12)), {6: [(0, 4)]}, [])))
    cache.insert(InsertParams(key=key, value=allocator.alloc(12)))
    matched = cache.match_prefix(MatchPrefixParams(key=key[:6], context_retry=True))
    lease = cache.inc_lock_ref(matched.last_device_node)
    cache.configure_context_drop_lock(
        matched.last_device_node, lease, torch.zeros(6, dtype=torch.bool)
    )
    assert lease.context_skip_ranges == ()
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 6)
    cache.dec_lock_ref(matched.last_device_node, lease.to_dec_params())
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)


@pytest.mark.parametrize("existing", [False, True])
def test_sparse_insert_preserves_holes_and_reclaims_only_allocated_pages(
    compiler, native_cache, existing
):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    tokens = list(range(12))
    if existing:
        original = RadixKey.from_context(compiler(*args(tokens, {6: [(1, 4)]}, [])))
        cache.insert(InsertParams(key=original, value=allocator.alloc(12)))
        tokens[8] = 999
    key = RadixKey.from_context(compiler(*args(tokens, {6: [(1, 4)]}, [])))
    resident = torch.ones(12, dtype=torch.bool)
    resident[1:4] = False
    values = torch.full((12,), -1, dtype=torch.int64)
    values[resident] = allocator.alloc(9)
    result = cache.insert(
        InsertParams(
            key=key, value=values, context_resident=resident, track_adopted_ranges=True
        )
    )
    assert result.prefix_len == (8 if existing else 0)
    assert_allocator(cache, allocator, 16 if existing else 9)
    matched = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    assert matched.context_resident.tolist() == (
        [True] * 12 if existing else resident.tolist()
    )
    assert matched.device_indices[8:].tolist() == values[8:].tolist()
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)
    assert len(cache.tree_core._node_arena) == 1


def test_sparse_insert_rejects_unproven_hole_before_mutation(compiler, native_cache):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    key = RadixKey.from_context(compiler(*args(list(range(12)), {6: [(1, 4)]}, [])))
    resident = torch.ones(12, dtype=torch.bool)
    resident[5] = False
    with pytest.raises(ValueError, match="no Drop proof"):
        cache.insert(
            InsertParams(
                key=key, value=torch.full((12,), -1), context_resident=resident
            )
        )
    assert len(cache.tree_core._node_arena) == 1
    assert_allocator(cache, allocator, 0)
