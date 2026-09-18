"""Read Linux per-thread scheduler counters for a profiler process tree.

No perf privileges or CUDA context needed. Sampling occurs only in the separate
Nsight replay, never in the throughput measurement. Runnable wait is distinct
from sleeping inside a CUDA API or a condition variable.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def snapshot(parent):
    processes = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            fields = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            processes[int(p.name)] = int(fields[1])
        except (OSError, IndexError):
            continue
    selected = {parent}
    while True:
        children = {pid for pid, ppid in processes.items() if ppid in selected}
        if children.issubset(selected):
            break
        selected |= children
    rows = []
    for pid in sorted(selected - {os.getpid()}):
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except OSError:
            continue
        for thread in tasks:
            try:
                run, wait, slices = map(int, (thread / "schedstat").read_text().split())
                name = (thread / "comm").read_text().strip()
                rows.append([pid, int(thread.name), name, run, wait, slices])
            except (OSError, ValueError):
                continue
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    running = True

    def stop(*_):
        global running
        running = False

    signal.signal(signal.SIGTERM, stop)
    with args.output.open("x", buffering=1) as out:
        while running and Path(f"/proc/{args.parent}").exists():
            start = time.monotonic()
            row = dict(time=start, threads=snapshot(args.parent))
            row["sampling_seconds"] = time.monotonic() - start
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            time.sleep(max(0.01, 1.0 - row["sampling_seconds"]))
