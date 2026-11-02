"""Structured keys through the real UnifiedRadixCache and native page allocator."""

import sys
from array import array
from pathlib import Path

import pytest
import torch
from test_ir import args

pytest_plugins = ("test_ir",)
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native SRT runtime requires Linux"
)


def test_initial_drop_lease_keeps_checkpoint_and_only_unlocks_proven_pages(compiler, monkeypatch):
    from types import SimpleNamespace
    from test_occurrence_ownership import expiry_for
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import reset_context
    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    cache, allocator, pool = native.build_fixture(native.CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA), kv_size=512,
        num_layers=4, full_attention_layer_ids=(3,),
    ))
    try:
        drops = {128: [(0, 64)], 256: [(64, 96)]}
        key = RadixKey.from_context(compiler(*args(list(range(300)), drops, [])))
        cache.insert(InsertParams(key=key[:256], value=allocator.alloc(256),
                                  mamba_value=pool.mamba_allocator.alloc(1)))
        match = cache.match_prefix(MatchPrefixParams(key=key[:256], context_retry=True))
        receipt = cache.inc_lock_ref(match.last_device_node).to_dec_params()
        req = SimpleNamespace(
            context_recompute_program=None,
            context_program=SimpleNamespace(visible_until=expiry_for(300, drops)),
            context_prefill_started=False, prefix_indices=match.device_indices,
            context_recovery_plan=None, last_node=match.last_device_node,
            lock_receipt=receipt, context_source_lease=None,
        )
        cache.configure_context_prefill_drop_lock(req, receipt)
        # Drop at 256 is absent from the 256-token source key, so 64:96
        # remains pinned even though this request no longer needs to read it.
        assert receipt.context_skip_ranges == ((0, 64),)
        assert cache.context_leased_page_count(req) == 192
        assert allocator.available_size() == 256  # eligibility is not eviction
        cache.evict(EvictParams(num_tokens=64))
        assert allocator.available_size() == 320
        node = cache.tree_core.node_by_id(req.last_node)
        assert node.component_data[ComponentType.MAMBA].value is not None
        cache.dec_lock_ref(req.last_node, receipt)
        cache.evict(EvictParams(num_tokens=512, mamba_num=20))
        assert allocator.available_size() == 512
    finally:
        reset_context()


def test_hybrid_retry_boundary_and_drop_lease(compiler, monkeypatch):
    """A Full split cannot invent a recurrent state; Drop keeps its native lock."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    cache, allocator, pool = native.build_fixture(native.CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA), kv_size=512,
        num_layers=4, full_attention_layer_ids=(3,),
    ))
    tokens = list(range(256))
    key = RadixKey.from_context(compiler(*args(tokens, {128: [(32, 64)]}, [])))
    resident = torch.ones(256, dtype=torch.bool)
    resident[32:64] = False
    values = torch.full((256,), -1, dtype=torch.int64)
    values[resident] = allocator.alloc(224)
    state = pool.mamba_allocator.alloc(1)
    cache.insert(InsertParams(key=key, value=values, mamba_value=state,
                              context_resident=resident))
    assert cache.tree_core.component_evictable_size_[ComponentType.MAMBA] == 1
    exact = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
    assert len(exact.device_indices) == 256
    # There is only a native state at 256, never at the partial boundary 192.
    partial = cache.match_prefix(MatchPrefixParams(key=key[:192], context_retry=True))
    assert len(partial.device_indices) == 0
    receipt = cache.inc_lock_ref(exact.last_device_node)
    required = torch.ones(256, dtype=torch.bool)
    required[32:64] = False
    cache.configure_context_drop_lock(exact.last_device_node, receipt, required)
    node = cache.tree_core.node_by_id(exact.last_device_node)
    assert node.component_data[ComponentType.MAMBA].lock_ref == 1
    assert torch.equal(node.component_data[ComponentType.MAMBA].value, state)
    assert receipt.context_skip_ranges == ((32, 64),)
    cache.dec_lock_ref(exact.last_device_node, receipt.to_dec_params())
    cache.evict(EvictParams(num_tokens=512, mamba_num=20))
    assert allocator.available_size() == 512
    assert pool.mamba_allocator.available_size() == 20


def test_drop_pressure_preserves_all_raw_checkpoints(compiler, monkeypatch):
    """Full pressure must not delete S_64/S_128 when their KV becomes holes."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import reset_context

    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    cache, allocator, pool = native.build_fixture(native.CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA), kv_size=512,
        num_layers=4, full_attention_layer_ids=(3,),
    ))
    try:
        key = RadixKey.from_context(compiler(*args(list(range(256)), {192: [(0,128)]}, [])))
        saved = {}
        for end in (64, 128, 256):
            slot = pool.mamba_allocator.alloc(1)
            pool.mamba_pool.mamba_cache.temporal[:, slot] = end
            cache.insert(InsertParams(key=key[:end], value=allocator.alloc(end), mamba_value=slot))
            hit = cache.match_prefix(MatchPrefixParams(key=key[:end], context_retry=True))
            assert len(hit.device_indices) == end
            saved[end] = (hit.last_device_node, slot.clone())
        anchor = saved[256][0]
        receipt = cache.inc_lock_ref(anchor).to_dec_params()
        needed = torch.ones(256, dtype=torch.bool)
        needed[:128] = False
        cache.configure_context_drop_lock(anchor, receipt, needed)
        assert allocator.available_size() == 256  # eligibility did not free KV
        cache.evict(EvictParams(num_tokens=128, mamba_num=0))
        assert allocator.available_size() == 384
        for end, (handle, slot) in saved.items():
            node = cache.tree_core.node_by_id(handle)
            assert torch.equal(node.component_data[ComponentType.MAMBA].value, slot)
            assert torch.all(pool.mamba_pool.mamba_cache.temporal[:, slot] == end)
        # A later key containing the Drop can reuse the complete hybrid state.
        hit = cache.match_prefix(MatchPrefixParams(key=key, context_retry=True))
        assert len(hit.device_indices) == 256
        assert not hit.context_resident[:128].any()
        cache.sanity_check()
        cache.dec_lock_ref(anchor, receipt)
        cache.evict(EvictParams(num_tokens=512, mamba_num=20))
        assert allocator.available_size() == 512
        assert pool.mamba_allocator.available_size() == 20
        cache.sanity_check()
    finally:
        reset_context()


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
        # Native UnifiedRadixCache already releases duplicate incoming slots
        # through its insert actions. The caller must not free them again.
        expected_resident = 320 if key is bad else 640 - inserted.prefix_len
        assert allocator.available_size() == 2048 - expected_resident
        assert (
            len(torch.unique(allocator.get_all_free_pages()))
            == allocator.available_size()
        )
    good_slots = cache.match_prefix(MatchPrefixParams(key=good)).device_indices
    before = len(cache.tree_core._node_arena)
    selected = cache.match_prefix(MatchPrefixParams(key=target, context_retry=True))
    assert len(cache.tree_core._node_arena) <= before + 1
    assert selected.device_indices.tolist() == good_slots[:256].tolist()
    assert selected.context_exact_prefix_len == good.match(target)
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
    assert len(torch.unique(allocator.get_all_free_pages())) == 2048
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


@pytest.mark.parametrize(
    "mode",
    [
        "cold",
        "tail_drop",
        "duplicate",
        "retry",
        "retry_abort",
        "disabled",
        "holes",
        "holes_abort",
        "recovery",
        "recovery_deferred_abort",
        "recovery_gap_abort",
    ],
)
def test_context_occurrence_native_publication_and_release(compiler, mode):
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs
    from test_occurrence_ownership import expiry_for

    reset_context()
    publish(
        ServerArgs(model_path="dummy", page_size=1),
        role="scheduler",
    )
    try:
        tokens = list(range(128))
        drops, repos = {24: [(4, 12)], 56: [(16, 36)]}, [23, 55]
        if mode == "tail_drop":
            drops[128] = [(80, 96)]
        layout = compiler(*args(tokens, drops, repos))
        program = ContextProgram(layout, expiry_for(len(tokens), drops))
        req = Req(
            mode,
            "",
            array("q", tokens),
            SamplingParams(max_new_tokens=1),
            context_program=program.to_wire(),
        )
        req._refresh_fill_ids()
        allocator = TokenToKVPoolAllocator(
            size=2048, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
        )
        pool = ReqToTokenPool(4, 256, "cpu", False)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=mode == "disabled",
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
        pool.alloc([req])
        req.last_node = cache.root_node_handle()
        if mode == "duplicate":
            cache.insert(
                InsertParams(
                    key=RadixKey.from_context(layout), value=allocator.alloc(128)
                )
            )
        if mode.startswith("retry"):
            source_tokens = tokens.copy()
            source_tokens[64] = 999
            source_drops = {**drops, 96: [(36, 48)]}
            source = compiler(*args(source_tokens, source_drops, [*repos, 95]))
            cache.insert(
                InsertParams(
                    key=RadixKey.from_context(source), value=allocator.alloc(128)
                )
            )
            matched = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey.from_context(layout)[:64],
                    context_retry=True,
                )
            )
            assert len(matched.device_indices) == 64
            assert matched.context_exact_prefix_len < 64
            req.prefix_indices = matched.device_indices
            req.context_source_positions = matched.context_source_positions
            req.context_exact_prefix_len = matched.context_exact_prefix_len
            req.kv.cache_protected_len = 64
            req.last_node = matched.last_device_node
            req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        if mode.startswith("holes"):
            resident = program.visible_until[:64] > 64
            source = torch.full((64,), -1, dtype=torch.int64)
            source[resident] = allocator.alloc(int(resident.sum()))
            cache.insert(
                InsertParams(
                    key=RadixKey.from_context(layout)[:64],
                    value=source,
                    context_resident=resident,
                )
            )
            matched = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey.from_context(layout)[:64], context_retry=True
                )
            )
            req.prefix_indices = matched.device_indices
            req.context_source_positions = matched.context_source_positions
            req.context_resident = matched.context_resident
            req.context_exact_prefix_len = matched.context_exact_prefix_len
            req.kv.cache_protected_len = 64
            req.last_node = matched.last_device_node
            req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        ends = (
            [80, 128] if mode.startswith(("retry", "holes")) else [17, 57, 128]
        )
        if mode.startswith("recovery"):
            source_tokens = tokens.copy()
            source_tokens[40] = 999
            source_layout = compiler(*args(source_tokens, drops, repos))
            resident = program.visible_until[:64] > 64
            slots = torch.full((64,), -1, dtype=torch.int64)
            slots[resident] = allocator.alloc(int(resident.sum()))
            cache.insert(
                InsertParams(
                    key=RadixKey.from_context(source_layout)[:64],
                    value=slots,
                    context_resident=resident,
                )
            )
            req.init_next_round_input(cache)
            assert req.context_recovery_plan.matched_length == 40
            assert req.context_recovery_plan.start < 40
            req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
            ends = [
                min(a + 7, end)
                for start, end in req.context_recovery_plan.intervals
                for a in range(start, end, 7)
            ]
        queried = []
        for end in ends:
            req.advance_context_recovery_gap()
            if mode == "recovery_gap_abort" and req.context_gap_prefix is not None:
                before = allocator.available_size()
                cache.cache_unfinished_req(req, chunked=True)
                assert allocator.available_size() == before
                break
            req.commit_context_recovery_gap()
            start = len(req.prefix_indices)
            queried.extend(range(start, end))
            extra = req.plan_context_prefill(end)
            window, plan = req.context_window_plan
            advanced = req.context_state.advance(
                window,
                plan,
                allocator.alloc(end - start),
                allocator.alloc(extra),
                program.visible_until,
            )
            req.context_state = advanced.state
            req.set_extend_range(start, end)
            req.kv.kv_committed_len = end
            # Simulate the native completion event before any release. Actual
            # layer writes/COW and graph replay are covered by model tests.
            allocator.free(advanced.retired_slots)
            cache.cache_unfinished_req(req, chunked=end < 128)
            if mode in {"cold", "tail_drop"} and end == 17:
                # Every computed KV is published, even before a future Drop.
                assert req.kv.cache_protected_len == end
                assert len(req.prefix_indices) == end
            if mode == "tail_drop" and end == 128:
                assert req.kv.cache_protected_len == 128
            remaining = req.context_state.terminal_rows >= 0
            assert torch.equal(
                req.context_state.terminal_slots()[remaining],
                req.prefix_indices[remaining],
            )
            cache.evict(EvictParams(num_tokens=2048))
            assert (
                len(torch.unique(allocator.get_all_free_pages()))
                == allocator.available_size()
            )
            cache.sanity_check()
            if mode in {"retry_abort", "recovery_deferred_abort"}:
                break
        if mode in {"retry_abort", "recovery_deferred_abort"}:
            assert end < 128
        if mode == "recovery":
            assert queried == [
                q for a, b in req.context_recovery_plan.intervals for q in range(a, b)
            ]
            assert len(queried) < 128
        cache.cache_finished_req(
            req,
            kv_len_to_handle=req.kv.kv_committed_len,
            is_insert=not mode.endswith("abort"),
        )
        assert req.context_state is None
        assert req.context_source_lease is None
        pool.free(req)
        cache.evict(EvictParams(num_tokens=2048))
        assert allocator.available_size() == 2048
        assert len(torch.unique(allocator.get_all_free_pages())) == 2048
        cache.sanity_check()
    finally:
        reset_context()


@pytest.mark.parametrize("finish_after_defer", [False, True])
def test_hybrid_deferred_publication_keeps_source_lease(compiler, monkeypatch, finish_after_defer):
    """A checkpoint before the final Drop must not commit a shorter target lease."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "unit/mem_cache"))
    import test_unified_radix_cache_unittest as native
    from sglang.srt.context_system.planner import ContextProgram
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from test_occurrence_ownership import expiry_for

    monkeypatch.setattr(native, "get_device", lambda: "cpu")
    cache, allocator, pool = native.build_fixture(native.CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA), kv_size=2048,
        num_layers=2, full_attention_layer_ids=(1,), enable_mamba_extra_buffer=True,
    ))
    try:
        tokens = list(range(200))
        drops = {192: [(0, 48)]}
        source = compiler(*args(tokens[:64], {}, []))
        layout = compiler(*args(tokens, drops, [191]))
        program = ContextProgram(layout, expiry_for(200, drops))
        cache.insert(InsertParams(key=RadixKey.from_context(source),
                                  value=allocator.alloc(64),
                                  mamba_value=pool.mamba_allocator.alloc(1)))
        req = Req("hybrid-defer", "", array("q", tokens), SamplingParams(max_new_tokens=8),
                  context_program=program.to_wire())
        req._refresh_fill_ids()
        pool.alloc([req])
        req.init_next_round_input(cache)
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        old_node, old_receipt = req.last_node, req.lock_receipt
        assert len(req.prefix_indices) == 64
        assert req.context_exact_prefix_len == 48
        extra = req.plan_context_prefill(200)
        window, plan = req.context_window_plan
        advanced = req.context_state.advance(
            window, plan, allocator.alloc(136), allocator.alloc(extra), program.visible_until
        )
        req.context_state = advanced.state
        req.set_extend_range(64, 200)
        req.kv.kv_committed_len = 200
        allocator.free(advanced.retired_slots)
        req.kv.mamba_last_track_seqlen = 192
        # Inject a missing target version before the Drop is on the candidate key.
        # Normal chunk materialization now publishes these KV, including Drop KV.
        saved_rows = req.context_state.terminal_rows.clone()
        req.context_state.terminal_rows[:48] = -1
        cache.cache_unfinished_req(req)
        assert not req.context_cache_published
        assert req.last_node == old_node and req.lock_receipt is old_receipt
        assert req.kv.cache_protected_len == 64
        assert len(req.prefix_indices) == 200
        assert req.context_state.owned.any()
        assert req.kv.mamba_last_track_seqlen is None
        req.context_state.terminal_rows.copy_(saved_rows)
        cache.sanity_check()
        if not finish_after_defer:
            # Model execution would have captured the real S_200 in this slot.
            # Publishing it must preserve raw depth 200 despite 48 absent KV rows.
            req.kv.mamba_last_track_seqlen = 200
            slot = req.kv.mamba_ping_pong_track_buffer[req.kv.mamba_last_track_idx].item()
            pool.mamba_pool.mamba_cache.temporal[:, slot].fill_(200)
            cache.cache_unfinished_req(req)
            assert req.context_cache_published
            assert req.kv.cache_protected_len == 200
            node = cache.tree_core.node_by_id(req.last_node)
            saved = node.component_data[ComponentType.MAMBA].value
            assert torch.all(pool.mamba_pool.mamba_cache.temporal[:, saved] == 200)
            assert (req.context_state.terminal_rows[:48] < 0).all()
            cache.sanity_check()
        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=200)
        pool.free(req)
        cache.evict(EvictParams(num_tokens=2048, mamba_num=20))
        assert allocator.available_size() == 2048
        assert pool.mamba_allocator.available_size() == 20
        assert req.context_state is None and req.context_source_lease is None
        cache.sanity_check()
    finally:
        reset_context()
