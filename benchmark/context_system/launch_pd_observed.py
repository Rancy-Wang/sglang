"""Launch unchanged inference with request-end PD timestamps for this benchmark.

Use the same wrapper for frozen native and modified servers. This adds no
per-token hooks or device synchronization. The timestamps are CPU observations,
not CUDA execution durations. In particular P's forward span includes gaps
between chunks, and its transfer tail excludes transfers overlapped with prefill.
"""

import json
import os
import sys


def stage_timestamps(stats):
    """Keep unset timestamps null; Linux monotonic clocks can join local P/D."""
    names = (
        "prefill_bootstrap_queue_entry_time",
        "bootstrap_done_time",
        "decode_prealloc_queue_entry_time",
        "decode_transfer_queue_entry_time",
        "wait_queue_entry_time",
        "forward_entry_time",
        "prefill_finished_time",
        "prefill_transfer_queue_entry_time",
        "completion_time",
    )
    return {
        name: value if (value := getattr(stats, name, 0.0)) > 0 else None
        for name in names
    }


def install():
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

    original = SchedulerReqTimeStats.convert_to_duration

    def observed(stats):
        return original(stats) + ", r2_pd_timing=" + json.dumps(
            stage_timestamps(stats), separators=(",", ":"), allow_nan=False
        )

    SchedulerReqTimeStats.convert_to_duration = observed


# Multiprocessing spawn executes the entrypoint as __mp_main__. Install in both
# the parent and spawned scheduler processes, before any requests are created.
if __name__ in ("__main__", "__mp_main__"):
    install()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
