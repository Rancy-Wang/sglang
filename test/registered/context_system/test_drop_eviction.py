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
    cached = cache.all_values_flatten()
    assert len(cached) == used
    assert len(torch.unique(cached)) == used
    assert bool(torch.all(cached > 0))
    walk = cache.tree_core.walk_for_kv_canary(False, False)
    assert sorted(walk.slot_indices.tolist()) == sorted(cached.tolist())
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
    if hasattr(allocator, "swa_attn_allocator"):
        # No Delta has been matched yet. Missing SWA inside its last window
        # must retain the native safety cap instead of returning a usable hit.
        assert partial.context_resident.tolist() == [True]
    else:
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


def test_swa_holes_remain_independent_across_insert_recovery_and_cow(
    compiler, native_cache
):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
        zero_match_result,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    if not hasattr(allocator, "swa_attn_allocator"):
        pytest.skip("SWA-specific ownership")
    key = RadixKey.from_context(compiler(*args(list(range(32)), {}, [])))
    expected = torch.ones(24, dtype=torch.bool)
    expected[1:4] = False
    expected[7:12] = False

    def insert(length, resident):
        slots = allocator.alloc(length)
        allocator.free_swa(slots[~resident])
        cache.insert(
            InsertParams(key=key[:length], value=slots, context_swa_resident=resident)
        )
        assert_allocator(cache, allocator, length)
        return cache.match_prefix(
            MatchPrefixParams(key=key[:length], context_retry=True)
        )

    first = insert(24, expected)
    assert first.context_resident.all()
    assert first.context_swa_resident.tolist() == expected.tolist()
    assert first.context_swa_window == 5
    lease = cache.inc_lock_ref(first.last_device_node)
    repair = torch.ones(24, dtype=torch.bool)
    repair[:2] = False
    repair[7:10] = False
    repaired = insert(24, repair)
    expected |= repair
    assert repaired.context_swa_resident.tolist() == expected.tolist()
    # The live reader keeps Full page identities while SWA-only recovery binds
    # fresh peers; unprovided SWA must not invalidate a resident tree component.
    assert repaired.device_indices.tolist() == first.device_indices.tolist()
    appended = torch.zeros(32, dtype=torch.bool)
    appended[-4:] = True
    final = insert(32, appended)
    expected = torch.cat((expected, appended[24:]))
    assert final.context_swa_resident.tolist() == expected.tolist()
    assert allocator.swa_attn_allocator.available_size() == 128 - int(expected.sum())
    forced_miss = zero_match_result(cache, final)
    assert len(forced_miss.device_indices) == 0
    assert forced_miss.context_source_positions is None
    assert forced_miss.context_resident is None
    assert forced_miss.context_swa_resident is None
    assert forced_miss.context_exact_prefix_len == 0
    cache.dec_lock_ref(first.last_device_node, lease.to_dec_params())
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)


@pytest.mark.parametrize("early_release", [False, True])
def test_exact_swa_lease_survives_split_and_reclaims_only_unread_pages(
    compiler, native_cache, early_release
):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    if not hasattr(allocator, "swa_attn_allocator"):
        pytest.skip("SWA-specific ownership")
    key = RadixKey.from_context(compiler(*args(list(range(24)), {}, [])))
    cache.insert(InsertParams(key=key, value=allocator.alloc(24)))
    matched = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    lease = cache.inc_lock_ref(matched.last_device_node)
    needed = torch.zeros(24, dtype=torch.bool)
    needed[2:5] = True
    needed[14:17] = True
    cache.configure_context_swa_lock(matched.last_device_node, lease, needed)
    assert lease.context_swa_ranges == ((2, 5), (14, 17))
    cache.tree_core.sanity_check([], [])
    # A later match splits a held historical range; the receipt follows raw
    # intervals through the new parent instead of relying on stale leaf IDs.
    cache.match_prefix(MatchPrefixParams(key=key[:4], context_retry=True))
    cache.evict(EvictParams(num_tokens=0, swa_num_tokens=128))
    assert allocator.swa_attn_allocator.available_size() == 128 - 6
    assert_allocator(cache, allocator, 24)
    extended = needed.clone()
    extended[18] = True
    with pytest.raises(ValueError, match="cannot reacquire"):
        cache.configure_context_swa_lock(matched.last_device_node, lease, extended)
    shrunk = needed.clone()
    shrunk[14:17] = False
    cache.configure_context_swa_lock(matched.last_device_node, lease, shrunk)
    cache.evict(EvictParams(num_tokens=0, swa_num_tokens=128))
    assert allocator.swa_attn_allocator.available_size() == 128 - 3
    receipt = lease.to_dec_params()
    if early_release:
        cache.dec_swa_lock_only(matched.last_device_node, receipt)
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=128))
        assert allocator.swa_attn_allocator.available_size() == 128
    cache.dec_lock_ref(matched.last_device_node, receipt, skip_swa=early_release)
    cache.evict(EvictParams(num_tokens=128, swa_num_tokens=128))
    assert_allocator(cache, allocator, 0)


def test_sparse_swa_request_release_and_completion(native_cache):
    from sglang.srt.context_system.occurrence import (
        ContextPrefillCompletion,
        OccurrenceState,
    )
    from sglang.srt.context_system.usage import ContextUsage

    cache, allocator = native_cache
    if not hasattr(allocator, "swa_attn_allocator"):
        pytest.skip("SWA-specific ownership")
    slots = allocator.alloc(24)
    swa = torch.ones(24, dtype=torch.bool)
    swa[[0, 1, 2, 5, 8, 9, 13, 19, 23]] = False
    allocator.free_swa(slots[~swa])
    # Raw 8 is a Full hole; 20:24 are generated after prefill. Raw 0:3 was
    # released by native window eviction, independently of Context SWA metadata.
    allocator.free_full(slots[8:9])
    rows = torch.arange(20)
    rows[8] = -1
    state = OccurrenceState(
        slots[:20],
        torch.ones(20, dtype=torch.bool),
        rows.clone(),
        rows.clone(),
        torch.arange(20, dtype=torch.int32),
        0,
        swa[:20].clone(),
    )
    state.swa_resident[:3] = True  # The native floor must win over stale metadata.
    cache.req_to_token_pool = SimpleNamespace(req_to_token=slots[None, :])
    req = SimpleNamespace(
        context_state=state,
        kv=SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=3),
    )
    # Leave the final four slots to an overlapped completion receipt. In native
    # decode all new peers are resident; this separate receipt tests mixed COW.
    cache._free_context_kv_row(req, [(0, 7), (7, 20)])
    assert allocator.full_attn_allocator.available_size() == 124
    assert len(torch.unique(allocator.full_attn_allocator.get_all_free_pages())) == 124
    assert allocator.swa_attn_allocator.available_size() == 125
    usage = ContextUsage(
        torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool)
    )
    receipt = ContextPrefillCompletion(
        slots[20:],
        usage,
        torch.empty(0, dtype=torch.bool),
        torch.empty(0, dtype=torch.bool),
        4,
        retired_swa_resident=swa[20:],
    )
    receipt.complete(allocator)
    receipt.complete(allocator)
    assert usage.snapshot().actual_prefill_tokens == 4
    assert_allocator(cache, allocator, 0)
    assert allocator.swa_attn_allocator.available_size() == 128
    assert allocator.full_to_swa_index_mapping[slots].count_nonzero() == 0
