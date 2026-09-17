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


@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
def test_native_insert_split_lock_and_page_reclaim(compiler, page_size):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    allocator = PagedTokenToKVPoolAllocator(
        size=1024,
        page_size=page_size,
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
            page_size=page_size,
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
    cache.dec_lock_ref(
        partial.last_device_node, receipt.to_dec_params()
    )
    cache.evict(EvictParams(num_tokens=1024))
    assert allocator.available_size() == 1024
    assert len(torch.unique(allocator.get_all_free_pages())) == 1024 // page_size
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
