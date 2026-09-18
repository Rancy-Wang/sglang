"""Benchmark-only CPU forward counters and optional NVTX/transfer observation.

Import this entrypoint with the frozen native repository on PYTHONPATH. No
inference implementation is replaced. TP ranks count independently; the offline
join uses the native TP0 log, rather than summing tensor-parallel replicas.
"""

import json
import os
import sys
import time

from launch_pd_observed import install as install_timestamps


def batch_work(batch):
    if batch.forward_mode.is_prebuilt() or batch.forward_mode.is_idle():
        return []
    if batch.forward_mode.is_decode():
        return [(r.time_stats, 0, 1) for r in batch.reqs]
    if batch.forward_mode.is_extend():
        if len(batch.extend_lens) != len(batch.reqs):
            raise ValueError("Missing per-request CPU extend lengths")
        return [(r.time_stats, int(n), 0) for r, n in zip(batch.reqs, batch.extend_lens)]
    raise ValueError(f"Unsupported benchmark forward mode: {batch.forward_mode}")


def install():
    install_timestamps()
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

    profile = os.environ.get("PD_MATRIX_PROFILE") == "1"
    original = Scheduler.run_batch
    duration = SchedulerReqTimeStats.convert_to_duration
    if profile:
        import torch

    def counted(self, batch, *args, **kwargs):
        work = batch_work(batch)
        if profile:
            rooms = [str(r.bootstrap_room) for r in batch.reqs]
            label = "pd_forward:" + json.dumps(
                {"mode": str(batch.forward_mode), "rooms": rooms,
                 "pf": sum(p for _, p, _ in work), "d": sum(d for _, _, d in work)},
                separators=(",", ":"),
            )
            torch.cuda.nvtx.range_push(label)
        try:
            result = original(self, batch, *args, **kwargs)
        finally:
            if profile:
                torch.cuda.nvtx.range_pop()
        for stats, pf, dec in work:
            stats.pd_matrix_pf = getattr(stats, "pd_matrix_pf", 0) + pf
            stats.pd_matrix_dec = getattr(stats, "pd_matrix_dec", 0) + dec
        return result

    def observed(stats):
        # Place the counts before the timestamp JSON, whose existing reader
        # deliberately consumes everything after r2_pd_timing=.
        prefix, timing = duration(stats).split(", r2_pd_timing=", 1)
        counts = {"pf": getattr(stats, "pd_matrix_pf", 0),
                  "decode": getattr(stats, "pd_matrix_dec", 0)}
        return prefix + ", pd_matrix_compute=" + json.dumps(counts) + ", r2_pd_timing=" + timing

    Scheduler.run_batch = counted
    SchedulerReqTimeStats.convert_to_duration = observed

    if profile:
        import threading
        from sglang.srt.disaggregation.mooncake import conn
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine

        context = threading.local()
        base_queue = conn.FastQueue

        class ObservedQueue(base_queue):
            def put(self, item):
                item._pd_matrix_enqueued = time.monotonic()
                return super().put(item)

            def get(self):
                item = super().get()
                now = time.monotonic()
                context.chunk = {
                    "room": str(item.room), "pid": os.getpid(),
                    "tid": threading.get_native_id(), "pages": len(item.prefill_kv_indices),
                    "slice_start": item.index_slice.start, "slice_stop": item.index_slice.stop,
                    "last": item.is_last_chunk, "enqueued": item._pd_matrix_enqueued,
                    "dequeued": now, "queue_seconds": now - item._pd_matrix_enqueued,
                }
                print("pd_matrix_queue=" + json.dumps(context.chunk), flush=True)
                torch.cuda.nvtx.mark("pd_dequeue:" + json.dumps(context.chunk))
                return item

        conn.FastQueue = ObservedQueue
        transfer = MooncakeTransferEngine.batch_transfer_sync

        def timed(engine, peer, src, dst, lengths):
            start = time.monotonic()
            data = {"pid": os.getpid(), "peer": peer, "start": start,
                    "tid": threading.get_native_id(), "blocks": len(lengths),
                    "bytes": sum(lengths), "chunk": getattr(context, "chunk", None)}
            torch.cuda.nvtx.range_push("pd_transfer:" + json.dumps(data))
            try:
                ret = transfer(engine, peer, src, dst, lengths)
            finally:
                torch.cuda.nvtx.range_pop()
            data.update(seconds=time.monotonic() - start, ret=ret)
            print("pd_matrix_transfer=" + json.dumps(data), flush=True)
            return ret

        MooncakeTransferEngine.batch_transfer_sync = timed


if __name__ in ("__main__", "__mp_main__"):
    install()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
