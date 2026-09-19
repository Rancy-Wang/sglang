"""Drop lease, persistent candidates and real native allocator pressure tests."""

import sys
from array import array
from types import SimpleNamespace

import pytest
import torch
from test_ir import ROOT, args, load_file

pytest_plugins = ("test_ir",)


def test_candidate_drop_priority_and_bounded_stale_entries():
    module = load_file(
        "context_drop_eviction", ROOT / "python/sglang/srt/context_system/recovery.py"
    )
    candidates = module.DropEvictionCandidates()
    leaf, internal = SimpleNamespace(id=1), SimpleNamespace(id=2)
    candidates.update(internal, 1, 10000)
    for time in range(1000):
        candidates.update(leaf, 0, time)
    assert sum(map(len, candidates.heaps)) <= 2 * len(candidates.entries) + 64
    assert candidates.pop() == (1, internal)
    assert candidates.pop() == (0, leaf)
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


@pytest.mark.parametrize("native_cache", ["full"], indirect=True)
def test_retry_source_capacity_counts_shared_edges_and_drop_receipts(
    compiler, native_cache
):
    from sglang.srt.context_system.request_storage import handle_prefill_capacity_pressure
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    source = RadixKey.from_context(
        compiler(*args(list(range(70)), {32: [(0, 16)], 60: [(16, 24)]}, [49, 69]))
    )
    target = RadixKey.from_context(
        compiler(*args(list(range(50)), {32: [(0, 16)]}, [49]))
    )
    cache.insert(InsertParams(key=source, value=allocator.alloc(70)))
    source_hit = cache.match_prefix(MatchPrefixParams(key=source, context_retry=True))
    source_receipt = cache.inc_lock_ref(source_hit.last_device_node).to_dec_params()
    cache.insert(InsertParams(key=target, value=allocator.alloc(50)))
    target_hit = cache.match_prefix(MatchPrefixParams(key=target, context_retry=True))
    target_receipt = cache.inc_lock_ref(target_hit.last_device_node).to_dec_params()
    # Both branches share 24 raw owners. Counting their lengths independently
    # would incorrectly count those pages twice; the state omits source pages.
    assert allocator.available_size() == 32
    private = allocator.alloc(30)
    state = SimpleNamespace(
        slots=private, owned=torch.ones(30, dtype=torch.bool),
        canonical_rows=torch.arange(30), terminal_rows=torch.arange(30),
    )
    req = SimpleNamespace(
        context_prefill_started=True, context_state=state,
        context_source_lease=(source_hit.last_device_node, source_receipt),
        last_node=target_hit.last_device_node, lock_receipt=target_receipt,
        context_admission_error=None,
    )
    assert cache.context_leased_page_count(req) == 96
    cache.evict(EvictParams(num_tokens=128))
    assert allocator.available_size() == 2
    handle_prefill_capacity_pressure(req, 128, 3, cache)
    assert "retains 126 KV tokens" in req.context_admission_error

    # Another request can be responsible for the missing free space. Give it
    # the first 16 pages, then convert this request's two refs to path-only refs.
    other_hit = cache.match_prefix(MatchPrefixParams(key=source[:16], context_retry=True))
    other_receipt = cache.inc_lock_ref(other_hit.last_device_node).to_dec_params()
    for hit, receipt, length in (
        (source_hit, source_receipt, 70), (target_hit, target_receipt, 50)
    ):
        required = torch.ones(length, dtype=torch.bool)
        required[:16] = False
        cache.configure_context_drop_lock(hit.last_device_node, receipt, required)
    assert cache.context_leased_page_count(req) == 80
    req.context_admission_error = None
    handle_prefill_capacity_pressure(req, 128, 3, cache)
    assert req.context_admission_error is None
    cache.evict(EvictParams(num_tokens=128))
    assert allocator.available_size() == 2  # external reader can eventually finish
    cache.dec_lock_ref(other_hit.last_device_node, other_receipt)
    cache.evict(EvictParams(num_tokens=16))
    assert allocator.available_size() == 18
    assert cache.context_leased_page_count(req) == 80  # evicted holes do not count
    cache.dec_lock_ref(target_hit.last_device_node, target_receipt)
    cache.dec_lock_ref(source_hit.last_device_node, source_receipt)
    allocator.free(private)
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)


def test_shared_reader_drop_first_hole_refill_and_split(compiler, native_cache):
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
    # Drop pages must be reclaimed while the ordinary leaf is still resident.
    cache.insert(
        InsertParams(key=RadixKey(array("q", [90, 91, 92])), value=allocator.alloc(3))
    )
    cache.evict(EvictParams(num_tokens=3))
    assert_allocator(cache, allocator, 12)
    holed = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    assert holed.context_resident.tolist() == required.tolist()
    assert holed.device_indices[1:4].tolist() == [-1, -1, -1]
    assert torch.equal(holed.device_indices[4:], slots[4:])
    assert len(cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [90, 91, 92])))
    ).device_indices) == 3
    cache.evict(EvictParams(num_tokens=3))
    assert_allocator(cache, allocator, 9)
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


def test_exhausted_pool_reclaims_all_drop_pages_before_any_leaf(
    compiler, native_cache, monkeypatch
):
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams, InsertParams, MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    leases, dropped, protected = [], set(), set()
    required = torch.ones(12, dtype=torch.bool)
    required[1:4] = False
    for offset in (0, 100):
        key = RadixKey.from_context(
            compiler(*args(list(range(offset, offset + 12)), {6: [(1, 4)]}, []))
        )
        slots = allocator.alloc(12)
        cache.insert(InsertParams(key=key, value=slots))
        hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
        lease = cache.inc_lock_ref(hit.last_device_node)
        cache.configure_context_drop_lock(hit.last_device_node, lease, required)
        leases.append((key, hit.last_device_node, lease, slots.clone()))
        dropped.update(slots[~required].tolist())
        protected.update(slots[required].tolist())
    cold = RadixKey(array("q", range(1000, 1104)))
    cold_slots = allocator.alloc(104)
    cache.insert(InsertParams(key=cold, value=cold_slots))
    assert_allocator(cache, allocator, 128)
    assert allocator.alloc(1) is None

    tree = cache.tree_core
    events = []
    evict_drop, evict_leaf = tree.evict_context_pages, tree.evict_device_leaf

    def observe_drop(node, *arguments):
        events.append(("drop", node.full_page_count))
        return evict_drop(node, *arguments)

    def observe_leaf(*arguments, **kwargs):
        # An independent physical-count check at the ordinary-leaf boundary.
        assert sum(n for kind, n in events if kind == "drop") == 6
        events.append(("leaf", 104))
        return evict_leaf(*arguments, **kwargs)

    monkeypatch.setattr(tree, "evict_context_pages", observe_drop)
    monkeypatch.setattr(tree, "evict_device_leaf", observe_leaf)
    cache.evict(EvictParams(num_tokens=110))
    assert [kind for kind, _ in events] == ["drop", "drop", "leaf"]
    assert_allocator(cache, allocator, 18)
    reused = allocator.alloc(110)
    assert reused is not None and len(torch.unique(reused)) == 110
    assert bool((reused > 0).all())
    assert set(reused.tolist()) == dropped | set(cold_slots.tolist())
    assert not set(reused.tolist()) & protected
    for key, node_id, lease, slots in leases:
        hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
        assert hit.context_resident.tolist() == required.tolist()
        assert hit.device_indices[~required].tolist() == [-1, -1, -1]
        assert torch.equal(hit.device_indices[required], slots[required])
        assert bool((hit.device_indices[required] > 0).all())
        cache.dec_lock_ref(node_id, lease.to_dec_params())
    monkeypatch.setattr(tree, "evict_device_leaf", evict_leaf)
    allocator.free(reused)
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)
    assert len(tree._node_arena) == 1


def test_drop_leaf_priority_and_empty_path_pruning(native_cache):
    from sglang.srt.context_system.recovery import DropEvictionCandidates
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams, InsertParams, MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache, allocator = native_cache
    cold = RadixKey(array("q", [90]))
    cache.insert(InsertParams(key=cold, value=allocator.alloc(1)))
    key = RadixKey(array("q", [10, 11]))
    cache.insert(InsertParams(key=key, value=allocator.alloc(2)))
    # Split a parent so removing the Drop leaf must expose it for later eviction.
    cache.match_prefix(MatchPrefixParams(key=key[:1]))
    leaf_id = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
    tree = cache.tree_core
    tree.context_eviction_candidates = DropEvictionCandidates()
    # Unit-test selection/lifecycle after an already-proven edge becomes a leaf.
    leaf = tree.node_by_id(leaf_id)
    leaf.context_drop_eligible = True
    for node in tuple(tree.evictable_device_leaves):
        tree._update_context_candidate(node)
    cache.evict(EvictParams(num_tokens=1))
    assert leaf_id not in tree._node_arena
    assert len(cache.match_prefix(MatchPrefixParams(key=cold)).device_indices) == 1
    assert len(cache.match_prefix(MatchPrefixParams(key=key)).device_indices) == 1
    assert_allocator(cache, allocator, 2)
    cache.evict(EvictParams(num_tokens=128))
    assert_allocator(cache, allocator, 0)
    assert len(tree._node_arena) == 1


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
        kv=SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=3, cache_protected_len=0),
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


@pytest.mark.parametrize("source_cache", [False, True])
@pytest.mark.parametrize("drop_aware", [False, True])
def test_swa_req_recovery_publication_and_pressure(
    compiler, native_cache, source_cache, drop_aware
):
    from array import array

    from sglang.srt.context_system.occurrence import free_context_slots
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs
    from test_occurrence_ownership import expiry_for

    cache, allocator = native_cache
    if not hasattr(allocator, "swa_attn_allocator"):
        pytest.skip("SWA-specific ownership")
    reset_context()
    publish(
        ServerArgs(
            model_path="dummy", page_size=1, context_drop_aware_eviction=drop_aware
        ),
        role="scheduler",
    )
    try:
        tokens = list(range(48))
        drops, repos = {32: [(8, 16)]}, [31, 47]
        layout = compiler(*args(tokens, drops, repos))
        program = ContextProgram(layout, expiry_for(48, drops))
        if source_cache:
            source_tokens = tokens.copy()
            source_tokens[24] = 999
            source = compiler(*args(source_tokens, drops, repos))
            full = program.visible_until > 48
            swa = torch.zeros(48, dtype=torch.bool)
            swa[20:24] = True
            swa[44:] = True
            slots = torch.full((48,), -1, dtype=torch.int64)
            slots[full] = allocator.alloc(int(full.sum()))
            allocator.free_swa(slots[full & ~swa])
            cache.insert(
                InsertParams(
                    key=RadixKey.from_context(source),
                    value=slots,
                    context_resident=full,
                    context_swa_resident=swa,
                )
            )
        req = Req(
            "swa",
            "",
            array("q", tokens),
            SamplingParams(max_new_tokens=1),
            context_program=program.to_wire(),
        )
        pool = ReqToTokenPool(2, 64, "cpu", False)
        cache.req_to_token_pool = pool
        pool.alloc([req])
        req.init_next_round_input(cache)
        assert req.context_swa_window == 5
        assert req.context_swa_resident is not None
        if source_cache:
            assert req.context_recovery_plan.matched_length == 24
            assert req.context_recovery_plan.start < 8
        PrefillAdder._req_inc_lock_ref(SimpleNamespace(tree_cache=cache), req)
        for start, end in req.context_recovery_plan.intervals:
            for cursor in range(start, end, 7):
                req.advance_context_recovery_gap()
                req.commit_context_recovery_gap()
                stop = min(cursor + 7, end)
                extra = req.plan_context_prefill(stop)
                window, plan = req.context_window_plan
                advance = req.context_state.advance(
                    window,
                    plan,
                    allocator.alloc(stop - cursor),
                    allocator.alloc(extra),
                    program.visible_until,
                )
                allocator.free_swa(advance.unused_swa_slots)
                req.context_state = advance.state
                req.set_extend_range(cursor, stop)
                req.kv.kv_committed_len = stop
                free_context_slots(
                    allocator, advance.retired_slots, advance.retired_swa_resident
                )
                cache.cache_unfinished_req(req, chunked=stop < 48)
                # Every future SWA copy/read remains pinned while cold SWA is
                # reclaimed between native chunks. Full ownership stays intact.
                cache.evict(
                    EvictParams(num_tokens=128 if drop_aware else 0, swa_num_tokens=128)
                )
                state = req.context_state
                valid = state.swa_resident
                mapping = allocator.full_to_swa_index_mapping[state.slots]
                assert torch.all(mapping[valid] > 0)
                assert (
                    len(torch.unique(allocator.swa_attn_allocator.get_all_free_pages()))
                    == allocator.swa_attn_allocator.available_size()
                )
                cache.sanity_check()
                if drop_aware and req.kv.cache_protected_len > 32:
                    assert req.lock_receipt.context_skip_ranges
                    assert torch.all(state.terminal_rows[8:16] < 0)
        if drop_aware:
            assert req.kv.cache_protected_len == 48
            assert req.lock_receipt.context_skip_ranges
        cache.cache_finished_req(req, kv_len_to_handle=48)
        assert req.context_source_lease is None
        pool.free(req)
        cache.evict(EvictParams(num_tokens=128, swa_num_tokens=128))
        assert_allocator(cache, allocator, 0)
        assert allocator.swa_attn_allocator.available_size() == 128
    finally:
        reset_context()


def test_context_decode_window_releases_sparse_prompt_then_generated_peers(
    compiler, native_cache
):
    from sglang.srt.context_system.occurrence import (
        ContextDecodeLayout,
        OccurrenceState,
    )
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    cache, allocator = native_cache
    if not hasattr(allocator, "swa_attn_allocator"):
        pytest.skip("SWA-specific ownership")
    layout = compiler(*args(list(range(24)), {16: [(8, 9)]}, [15]))
    slots = allocator.alloc(40)
    valid = torch.ones(40, dtype=torch.bool)
    valid[[1, 3, 8, 12, 18]] = False
    allocator.free_swa(slots[~valid])
    allocator.free_full(slots[8:9])
    rows = torch.arange(24)
    rows[8] = -1
    state = OccurrenceState(
        slots[:24],
        torch.ones(24, dtype=torch.bool),
        rows.clone(),
        rows.clone(),
        layout.positions,
        0,
        valid[:24].clone(),
    )
    decode = ContextDecodeLayout.from_layout(layout, "cpu")
    cache.req_to_token_pool = SimpleNamespace(req_to_token=slots[None])
    req = SimpleNamespace(
        context_program=True,
        context_state=state,
        context_decode_layout=decode,
        kv=SimpleNamespace(
            req_pool_idx=0, swa_evicted_seqlen=0, holds_kv=True, swa_dead_lo=lambda _: 0,
            cache_protected_len=0,
        ),
    )
    for computed in (24, 25, 30, 40):
        cache.components[ComponentType.SWA]._free_out_of_window_slots(req, computed)
        floor = decode.swa_raw_floor(computed, 4)
        assert req.kv.swa_evicted_seqlen == floor
        valid[:floor] = False
        assert torch.equal(allocator.full_to_swa_index_mapping[slots] > 0, valid)
    cache._free_context_kv_row(req, [(0, 40)])
    assert_allocator(cache, allocator, 0)
    assert allocator.swa_attn_allocator.available_size() == 128
