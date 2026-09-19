"""Opt-in TP0 forward timing; query completed CUDA events without synchronizing."""

import json
import os
import time
from collections import deque


def install():
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.forward
    pending = deque()
    log = None

    def measured(self, forward_batch, *args, **kwargs):
        nonlocal log
        # Do not count TP replicas or embed instrumentation in a captured graph.
        if self.ps.tp_rank != 0 or torch.cuda.is_current_stream_capturing():
            return original(self, forward_batch, *args, **kwargs)
        if log is None:
            log = open(os.environ["PD_MATRIX_FORWARD_LOG"], "a", buffering=1)
        while pending and pending[0][1].query():
            begin, end, row = pending.popleft()
            row["gpu_ms"] = begin.elapsed_time(end)
            row["observed_perf"] = time.perf_counter()
            log.write(json.dumps(row) + "\n")
        mode = forward_batch.forward_mode
        if not (mode.is_extend() or mode.is_decode()):
            return original(self, forward_batch, *args, **kwargs)
        row = dict(start_perf=time.perf_counter(), start_wall=time.time(),
                   mode=str(mode), batch_size=int(forward_batch.batch_size),
                   prefill_tokens=int(forward_batch.extend_num_tokens or 0) if mode.is_extend() else 0,
                   decode_tokens=int(forward_batch.batch_size) if mode.is_decode() else 0,
                   pid=os.getpid())
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original(self, forward_batch, *args, **kwargs)
        end.record()
        row["end_perf"] = time.perf_counter()
        row["cpu_ms"] = 1000 * (row["end_perf"] - row["start_perf"])
        pending.append((begin, end, row))
        return result

    ModelRunner.forward = measured
