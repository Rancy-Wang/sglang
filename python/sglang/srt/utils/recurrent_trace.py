"""Opt-in diagnostic spans; no wrapper remains when tracing is disabled.

Enabled with SGLANG_RECURRENT_TRACE=1 and the existing profile endpoints.
CUDA spans are stream intervals, not exclusive kernel busy time. No hot-path
synchronization or disk writes; export happens at stop_profile only.
"""

import contextvars
import functools
import json
import os
import time
from pathlib import Path

import torch

_enabled = os.environ.get("SGLANG_RECURRENT_TRACE") == "1"
_active = False
_records = []
_parent = contextvars.ContextVar("recurrent_trace_parent", default=None)


def start():
    global _active
    if _enabled:
        _records.clear()
        _active = True


def stop(directory, profile_id):
    global _active
    if not _active:
        return
    _active = False
    if any(row.get("events") for row in _records):
        torch.cuda.synchronize()
    path = Path(directory) / f"{profile_id}-recurrent-{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in _records:
            events = row.pop("events", None)
            if events is not None:
                row["gpu_stream_ms"] = events[0].elapsed_time(events[1])
            stream.write(json.dumps(row) + "\n")
    _records.clear()


def batch_metadata(_self, batch, *args, **kwargs):
    return {
        "rids": [req.rid for req in batch.reqs],
        "output_lengths": [len(req.output_ids) for req in batch.reqs],
        "mode": batch.forward_mode.name,
    }


def span(name, before=None, after=None, gpu=True):
    def decorate(function):
        if not _enabled:
            return function

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not _active:
                return function(*args, **kwargs)
            has_cuda = torch.cuda.is_initialized()
            if has_cuda and torch.cuda.is_current_stream_capturing():
                return function(*args, **kwargs)
            row = {
                "id": len(_records), "parent": _parent.get(), "stage": name,
                "start_ns": time.perf_counter_ns(),
            }
            if before is not None:
                row["input"] = before(*args, **kwargs)
            _records.append(row)
            token = _parent.set(row["id"])
            events = None
            if gpu and has_cuda:
                events = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
                events[0].record()
            try:
                with torch.profiler.record_function(f"recurrent::{name}"):
                    result = function(*args, **kwargs)
                if after is not None:
                    row["output"] = after(result, *args, **kwargs)
                return result
            finally:
                if events is not None:
                    events[1].record()
                    row["events"] = events
                row["end_ns"] = time.perf_counter_ns()
                row["cpu_ms"] = (row["end_ns"] - row["start_ns"]) / 1e6
                _parent.reset(token)

        return wrapped

    return decorate
