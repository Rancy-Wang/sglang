"""Structured keys through the real UnifiedRadixCache and native page allocator."""

import sys
from array import array

import pytest
import torch
from test_ir import args

pytest_plugins = ("test_ir",)
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native SRT runtime requires Linux"
)


def test_native_insert_split_lock_and_page_reclaim(compiler):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    allocator = TokenToKVPoolAllocator(
        size=1024,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    tokens = list(range(256))
    layout = compiler(*args(tokens, {128: [(0, 64)]}, []))
    key = RadixKey.from_context(layout)
    values = allocator.alloc(256)
    assert cache.insert(InsertParams(key=key, value=values)).prefix_len == 0
    # A normal request reuses the initial native prefix, but cannot consume KV
    # calculated after a different Drop history. This also forces a tree split.
    ordinary = RadixKey(array("q", tokens))
    shared = cache.match_prefix(MatchPrefixParams(key=ordinary))
    assert shared.device_indices.tolist() == values[:128].tolist()
    assert (
        cache.match_prefix(MatchPrefixParams(key=key)).device_indices.tolist()
        == values.tolist()
    )
    # Keep a partial feature edge pinned while ordinary leaf eviction runs.
    partial = cache.match_prefix(MatchPrefixParams(key=key[:192]))
    receipt = cache.inc_lock_ref(partial.last_device_node)
    cache.evict(EvictParams(num_tokens=1024))
    assert allocator.available_size() == 1024 - 192
    assert (
        cache.match_prefix(MatchPrefixParams(key=key)).device_indices.tolist()
        == values[:192].tolist()
    )
    cache.dec_lock_ref(partial.last_device_node, receipt.to_dec_params())
    cache.evict(EvictParams(num_tokens=1024))
    assert allocator.available_size() == 1024
    assert len(torch.unique(allocator.get_all_free_pages())) == 1024
    assert len(cache.match_prefix(MatchPrefixParams(key=key)).device_indices) == 0


def test_context_export_rejected_before_tree_mutation(compiler):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            tree_components=(ComponentType.FULL,),
            enable_kv_cache_events=True,
        )
    )
    key = RadixKey.from_context(compiler(*args([1, 2, 3, 4], {2: [(0, 1)]}, [])))
    before = len(cache.tree_core._node_arena)
    with pytest.raises(ValueError, match="event export"):
        cache.insert(InsertParams(key=key, value=torch.arange(4)))
    assert len(cache.tree_core._node_arena) == before


def test_longest_retry_and_one_sided_reposition(compiler):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    allocator = TokenToKVPoolAllocator(
        size=2048,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    tokens = list(range(320))
    target_layout = compiler(*args(tokens[:256], {64: [(0, 32)]}, [127]))
    target = RadixKey.from_context(target_layout)
    # The earlier/higher-bound branch disagrees at raw 160. Greedy selection
    # loses the later branch that supplies all 256 compatible tokens.
    bad_tokens = tokens.copy()
    bad_tokens[160] = 9999
    source_drops = {64: [(0, 32)], 256: [(32, 48)]}
    bad = RadixKey.from_context(compiler(*args(bad_tokens, source_drops, [127, 319])))
    good_layout = compiler(*args(tokens, source_drops, [127, 287]))
    good = RadixKey.from_context(good_layout)
    for key in (bad, good):
        slots = allocator.alloc(320)
        inserted = cache.insert(InsertParams(key=key, value=slots))
        # The native cache adopts only the new suffix; deduplicated incoming
        # prefix pages are still the inserting request's responsibility.
        allocator.free(slots[: inserted.prefix_len])
    good_slots = cache.match_prefix(MatchPrefixParams(key=good)).device_indices
    before = len(cache.tree_core._node_arena)
    selected = cache.match_prefix(MatchPrefixParams(key=target, context_retry=True))
    assert len(cache.tree_core._node_arena) <= before + 1
    assert selected.device_indices.tolist() == good_slots[:256].tolist()
    assert (
        selected.context_source_positions.tolist()
        == good_layout.positions[:256].tolist()
    )
    assert selected.context_retry
    # A same-length exact match is preferred and needs no position conversion.
    same = cache.match_prefix(MatchPrefixParams(key=good[:256], context_retry=True))
    assert not same.context_retry
    assert same.device_indices.tolist() == good_slots[:256].tolist()
    # An event in only one key must stop matching before the following query.
    no_r = RadixKey.from_context(compiler(*args(tokens[:256], {64: [(0, 32)]}, [])))
    stopped = cache.match_prefix(MatchPrefixParams(key=no_r, context_retry=True))
    assert len(stopped.device_indices) == 128
    assert stopped.context_source_positions.numel() == 128
    # Eviction must remove the lazy Retry child entries, as well as native keys.
    cache.evict(EvictParams(num_tokens=2048))
    assert allocator.available_size() == 2048
    assert not len(
        cache.match_prefix(
            MatchPrefixParams(key=target, context_retry=True)
        ).device_indices
    )
    assert not cache.tree_core.root_node.context_retry_index.signatures


def test_large_page_context_rejected_without_changing_native_cache(compiler):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=16,
            tree_components=(ComponentType.FULL,),
        )
    )
    tokens = list(range(64))
    plain = RadixKey(array("q", tokens))
    values = torch.arange(64)
    cache.insert(InsertParams(key=plain, value=values))
    before = len(cache.tree_core._node_arena)
    context = RadixKey.from_context(compiler(*args(tokens, {32: [(0, 16)]}, [])))
    for operation in (
        lambda: cache.insert(InsertParams(key=context, value=values)),
        lambda: cache.match_prefix(MatchPrefixParams(key=context)),
        lambda: cache.match_prefix(MatchPrefixParams(key=context, context_retry=True)),
    ):
        with pytest.raises(ValueError, match="page_size=1"):
            operation()
        assert len(cache.tree_core._node_arena) == before
    assert torch.equal(
        cache.match_prefix(MatchPrefixParams(key=plain)).device_indices, values
    )
