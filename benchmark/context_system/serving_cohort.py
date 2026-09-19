"""Bounded unique SWE cohorts and durable off-loop benchmark journals."""
import asyncio
import copy
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

COMPLETED = {"all_turns_completed", "active_context_limit_reached", "model_context_budget_reached"}


def context_stop(full, output, state, limit):
    if limit is None:
        return None
    active = state.get("active_tokens", full)
    position = state.get("position_tokens", full)
    if active > limit:
        return "active_context_limit_reached"
    if position >= limit or position + output > limit:
        return "model_context_budget_reached"
    return None


class Journal:
    """Capture timestamps on arrival, serialize on one bounded writer thread."""
    def __init__(self, root):
        self.root = Path(root)
        self.queue = queue.Queue(maxsize=65536)
        self.error = None
        self.count = 0
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, event):
        if self.error:
            raise RuntimeError("Journal writer failed") from self.error
        # No futures retained per SSE; failure is explicit, never silent loss.
        self.queue.put_nowait(copy.deepcopy(event))
        self.count += 1

    def _write(self):
        try:
            with (self.root / "events.jsonl").open("x") as events, (self.root / "sse.jsonl").open("x") as sse:
                last_flush = time.monotonic()
                while True:
                    try:
                        row = self.queue.get(timeout=1)
                    except queue.Empty:
                        events.flush(); sse.flush()
                        continue
                    if row is None:
                        break
                    target = sse if row.get("kind") == "raw_sse" else events
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if time.monotonic() - last_flush >= 1 or row.get("kind") in ("task_end", "turn_end", "measurement_cutoff"):
                        events.flush(); sse.flush(); last_flush = time.monotonic()
                events.flush(); sse.flush()
        except BaseException as exc:
            self.error = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=1)
                break
            except queue.Full:
                if self.error:
                    break
        self.thread.join()
        if self.error:
            raise RuntimeError("Journal writer failed") from self.error


class UniqueCohort:
    """Finish the fixed primary cohort; fill free slots from unused source tasks."""
    def __init__(self, cases, concurrency, target, execute, emit):
        if not 1 <= concurrency <= target <= len(cases):
            raise ValueError("Require concurrency <= primary target <= source count")
        if len({x['case_id'] for x in cases}) != len(cases):
            raise ValueError("Duplicate source task")
        self.cases = cases[:target]
        self.pending = deque((case, i >= target) for i, case in enumerate(cases))
        self.concurrency, self.target, self.execute, self.emit = concurrency, target, execute, emit
        self.active, self.completed, self.instances, self.round_ends, self.workers = set(), [], [], [], []
        self.cutoff = None
        self.failure = None

    def stop(self, reason):
        if self.cutoff is None:
            self.cutoff = time.perf_counter()
            self.emit(dict(kind="measurement_cutoff", time=self.cutoff, reason=reason))
            for task in self.workers:
                if task is not asyncio.current_task():
                    task.cancel()

    async def worker(self, slot):
        while self.cutoff is None and self.pending:
            case, filler = self.pending.popleft()
            key = case['case_id']
            assert key not in self.active
            self.active.add(key)
            inst = dict(instance=len(self.instances), case_id=key, trial=case.get('trial', 0),
                        filler=filler, slot=slot, start_time=time.perf_counter())
            self.instances.append(inst)
            self.emit(dict(kind="task_start", **inst))
            try:
                status = await self.execute(case, inst)
            except asyncio.CancelledError:
                status = "cutoff_cancelled"
            except Exception as exc:
                status = "client_error:" + repr(exc)
            inst.update(status=status, end_time=time.perf_counter())
            self.active.remove(key)
            if status in COMPLETED and not filler:
                self.completed.append(dict(case=case, instance=inst))
                if len(self.completed) % self.concurrency == 0:
                    self.round_ends.append(inst['end_time'])
                    self.emit(dict(kind="round_end", round=len(self.round_ends), time=inst['end_time']))
                if len(self.completed) == self.target:
                    self.stop("primary_cohort_completed")
            elif status not in COMPLETED and self.cutoff is None:
                self.failure = dict(case_id=key, status=status, filler=filler)
                self.stop("request_failure")
            self.emit(dict(kind="task_end", **inst))
            await asyncio.sleep(0)

    async def run(self):
        self.start = time.perf_counter()
        self.workers = [asyncio.create_task(self.worker(i)) for i in range(self.concurrency)]
        try:
            outcomes = await asyncio.gather(*self.workers, return_exceptions=True)
            for value in outcomes:
                if isinstance(value, BaseException) and not isinstance(value, asyncio.CancelledError):
                    raise value
        finally:
            for task in self.workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
        return self
