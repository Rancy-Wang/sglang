"""Bound page-1 transfer descriptors and check real TCP GPU payloads on demand."""

import concurrent.futures
import multiprocessing
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native SRT runtime")


def test_sparse_reuse_wire_is_optional_and_validates_destination_coordinates():
    import numpy as np
    from sglang.srt.disaggregation.mooncake.conn import TransferInfo

    msg = [b"1", b"127.0.0.1", b"1234", b"peer", np.arange(4, dtype=np.int32).tobytes(),
           b"0", b"", b"1", b"0", b""]
    assert TransferInfo.from_zmq(msg).context_reuse_mask is None
    parsed = TransferInfo.from_zmq(msg + [bytes([1, 0, 1, 0])])
    assert parsed.context_reuse_mask.tolist() == [True, False, True, False]
    for invalid in (bytes([1, 0]), bytes([0, 1, 2, 0])):
        with pytest.raises(ValueError, match="destination pages"):
            TransferInfo.from_zmq(msg + [invalid])


def manager(engine):
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    obj = SimpleNamespace(engine=engine)
    obj._transfer_data = lambda session, blocks: MooncakeKVManager._transfer_data(
        obj, session, blocks
    )
    return obj


@pytest.mark.parametrize("count", [0, 72, 1024, 1025, 72 * 42])
def test_descriptor_batch_bound_preserves_every_address(count):
    blocks = [(1000 + i * 37, 100000 + i * 53, 7 + i % 13) for i in range(count)]
    engine = Mock()
    engine.batch_transfer_sync.return_value = 0
    assert manager(engine)._transfer_data("peer", blocks) == 0
    observed = []
    for call in engine.batch_transfer_sync.call_args_list:
        peer, src, dst, lengths = call.args
        assert peer == "peer" and 0 < len(src) <= 1024
        observed.extend(zip(src, dst, lengths))
    assert observed == blocks
    assert engine.batch_transfer_sync.call_count == (count + 1023) // 1024


@pytest.mark.parametrize("failure", [-1, 5])
def test_failed_batch_is_not_retried_or_followed_by_later_writes(failure):
    engine = Mock()
    engine.batch_transfer_sync.side_effect = [0, failure]
    assert manager(engine)._transfer_data("peer", [(1, 2, 3)] * 3000) == failure
    assert engine.batch_transfer_sync.call_count == 2


def send_layers(engine, src_ptr, dst_ptr):
    import numpy as np
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    # 42 non-contiguous source runs expand to 3024 descriptors over K/V of
    # 36 layers. Token-index batching alone does not bound this expansion.
    layers, pages, width = 72, 42 * 64, 512
    src = np.concatenate([np.arange(i * 64, i * 64 + 32) for i in range(42)])
    dst = np.arange(src.size, dtype=np.int32)
    ptrs = [src_ptr + i * pages * width for i in range(layers)]
    targets = [dst_ptr + i * pages * width for i in range(layers)]
    obj = manager(engine)
    obj.is_mla_backend = False
    obj.is_hybrid_mla_backend = False
    obj.pp_size = 1
    obj.enable_custom_mem_pool = False
    obj.custom_mem_pool_type = None
    obj.max_transfer_batch_indices = 0
    obj.get_mha_kv_ptrs_with_pp = lambda *_: (
        ptrs[:36],
        ptrs[36:],
        targets[:36],
        targets[36:],
        36,
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ret = MooncakeKVManager._send_kvcache_generic(
            obj, "peer", ptrs, targets, [width] * layers, src, dst, executor
        )
    return ret, src


def test_all_layer_fragmentation_is_bounded_after_grouping():
    engine = Mock()
    engine.batch_transfer_sync.return_value = 0
    ret, src = send_layers(engine, 1000, 100000000)
    assert ret == 0 and len(src) == 1344
    calls = engine.batch_transfer_sync.call_args_list
    assert [len(call.args[1]) for call in calls] == [1024, 1024, 976]
    expected = [
        (
            1000 + (layer * 2688 + run * 64) * 512,
            100000000 + (layer * 2688 + run * 32) * 512,
            32 * 512,
        )
        for layer in range(72)
        for run in range(42)
    ]
    assert [entry for call in calls for entry in zip(*call.args[1:])] == expected


def tcp_target(pipe):
    import torch
    from mooncake.engine import TransferEngine

    torch.cuda.set_device(1)
    target = torch.full((72, 2688, 512), 255, dtype=torch.uint8, device="cuda:1")
    torch.cuda.synchronize()
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
    assert engine.register_memory(target.data_ptr(), target.numel()) == 0
    pipe.send((f"127.0.0.1:{engine.get_rpc_port()}", target.data_ptr()))
    assert pipe.recv() == "complete"
    torch.cuda.synchronize()
    expected = torch.arange(72 * 2688 * 512, dtype=torch.int32).reshape(72, 2688, 512)
    expected = (expected % 251).to(torch.uint8)
    src_indices = torch.cat([torch.arange(i * 64, i * 64 + 32) for i in range(42)])
    actual = target.cpu()
    assert torch.equal(actual[:, :1344], expected[:, src_indices])
    assert torch.all(actual[:, 1344:] == 255)
    pipe.send({"bytes": 72 * 1344 * 512, "payload_equal": True, "guards_equal": True})
    assert pipe.recv() == "exit"
    assert engine.unregister_memory(target.data_ptr()) == 0


@pytest.mark.skipif(
    os.environ.get("CONTEXT_TEST_MOONCAKE_TCP_GPU") != "1",
    reason="opt-in independent P/D GPU TCP transfer; no model initialization",
)
def test_real_tcp_fragmented_gpu_payload():
    import torch
    from mooncake.engine import TransferEngine

    assert torch.cuda.device_count() == 2, "set CUDA_VISIBLE_DEVICES to two free GPUs"
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    receiver = ctx.Process(target=tcp_target, args=(child,))
    receiver.start()
    try:
        torch.cuda.set_device(0)
        source = torch.arange(72 * 2688 * 512, dtype=torch.int32)
        source = (source % 251).to(dtype=torch.uint8, device="cuda:0")
        torch.cuda.synchronize()
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        assert engine.register_memory(source.data_ptr(), source.numel()) == 0
        assert parent.poll(90), "receiver initialization timed out"
        peer, dst_ptr = parent.recv()
        batches = []

        def transfer(_, src, dst, lengths):
            batches.append(len(src))
            return engine.batch_transfer_sync_write(peer, src, dst, lengths)

        wrapper = SimpleNamespace(batch_transfer_sync=transfer)
        ret, _ = send_layers(wrapper, source.data_ptr(), dst_ptr)
        assert ret == 0 and batches == [1024, 1024, 976]
        parent.send("complete")
        assert parent.poll(90), "payload verification timed out"
        report = parent.recv()
        assert report["payload_equal"] and report["guards_equal"]
        print({"tcp_gpu_transfer": report, "descriptor_batches": batches}, flush=True)
        parent.send("exit")
        receiver.join(30)
        assert receiver.exitcode == 0
        assert engine.unregister_memory(source.data_ptr()) == 0
    finally:
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(10)
        parent.close()
        child.close()
