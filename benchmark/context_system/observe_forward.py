"""Opt-in TP0 forward timing; query completed CUDA events without synchronizing."""

import json
import os
import time
from collections import deque

_DRAINS = []

def drain_completed(wait=False):
    for drain in _DRAINS:
        drain(wait)



def install():
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.forward
    pending = deque()
    log = None

    def drain(wait=False):
        while pending:
            begin, end, row = pending[0]
            if wait:
                end.synchronize()  # explicit idle flush only
            elif not end.query():
                break
            pending.popleft()
            row = dict(row, gpu_ms=begin.elapsed_time(end), observed_perf=time.perf_counter(), record="gpu_completed")
            if os.environ.get("ABLATION_PROFILE_LOG"):
                from observe_cache_ablation import emit
                emit(row)
            else:
                log.write(json.dumps(row) + "\n")
    _DRAINS.append(drain)

    def measured(self, forward_batch, *args, **kwargs):
        nonlocal log
        if os.environ.get("ABLATION_PROFILE_LOG"):
            from observe_cache_ablation import enabled
            if not enabled():
                return original(self, forward_batch, *args, **kwargs)
        # Do not count TP replicas or embed instrumentation in a captured graph.
        if self.ps.tp_rank != 0 or torch.cuda.is_current_stream_capturing():
            return original(self, forward_batch, *args, **kwargs)
        if log is None:
            log = open(os.environ["PD_MATRIX_FORWARD_LOG"], "a", buffering=1)
        drain()
        mode = forward_batch.forward_mode
        if not (mode.is_extend() or mode.is_decode()):
            return original(self, forward_batch, *args, **kwargs)
        row = dict(start_perf=time.perf_counter(), start_wall=time.time(),
                   mode=str(mode), batch_size=int(forward_batch.batch_size),
                   prefill_tokens=int(forward_batch.extend_num_tokens or 0) if mode.is_extend() else 0,
                   decode_tokens=int(forward_batch.batch_size) if mode.is_decode() else 0,
                   pid=os.getpid())
        if os.environ.get("ABLATION_PROFILE_LOG"):
            from observe_cache_ablation import current_batch
            row["batch"] = dict(current_batch())
            row["timing_definition"] = "CUDA event forward span including gaps, not pure kernel sum"
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original(self, forward_batch, *args, **kwargs)
        end.record()
        row["end_perf"] = time.perf_counter()
        row["cpu_ms"] = 1000 * (row["end_perf"] - row["start_perf"])
        row["record"] = "dispatched"
        # Preserve the final dispatch even if the server becomes idle or the
        # fixed measurement window stops before a subsequent event query.
        if os.environ.get("ABLATION_PROFILE_LOG"):
            from observe_cache_ablation import emit
            emit(dict(row))
        else:
            log.write(json.dumps(row) + "\n")
        pending.append((begin, end, row))
        return result

    ModelRunner.forward = measured
