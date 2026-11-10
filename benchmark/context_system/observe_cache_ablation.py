"""Opt-in benchmark observers. No cache policy or scheduling behavior changes."""
import atexit
import contextlib
import functools
import itertools
import json
import os
import queue
import threading
import time
from pathlib import Path

_local = threading.local()
_ids = itertools.count()
_sink = None
_enabled = True

def enabled():
    return _enabled



class Sink:
    def __init__(self, prefix):
        self.q = queue.Queue(maxsize=262144)
        self.error = None
        self.path = Path(f'{prefix}.{os.getpid()}.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            with self.path.open('x', buffering=1024*1024) as f:
                last = time.monotonic()
                while True:
                    try: row = self.q.get(timeout=1)
                    except queue.Empty:
                        f.flush(); continue
                    if row is None: break
                    f.write(json.dumps(row, separators=(',', ':')) + '\n')
                    if time.monotonic()-last > 1:
                        f.flush(); last = time.monotonic()
        except BaseException as e:
            self.error = repr(e)

    def emit(self, row):
        if self.error: raise RuntimeError(self.error)
        self.q.put_nowait(row)  # overflow fails visibly, never silently drops

    def close(self):
        if self.thread.is_alive():
            self.q.put(None); self.thread.join(timeout=10)


def rank():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else -1


def emit(row):
    global _sink
    prefix = os.environ.get('ABLATION_PROFILE_LOG')
    if not prefix or rank() != 0: return
    if _sink is None:
        _sink = Sink(prefix); atexit.register(_sink.close)
    _sink.emit({**row, 'pid': os.getpid(), 'rank': 0, 'stage': os.environ.get('ABLATION_STAGE')})


def current_batch():
    return getattr(_local, 'batch', {})


@contextlib.contextmanager
def batch_scope(batch):
    if not enabled():
        yield {}
        return
    old = current_batch()
    info = dict(step_id=next(_ids), mode=str(batch.forward_mode),
                rooms=[str(r.bootstrap_room) for r in batch.reqs],
                request_ids=[str(r.rid) for r in batch.reqs],
                recovery=[getattr(r, 'context_recovery_plan', None) is not None for r in batch.reqs],
                extend_lens=[int(n) for n in (getattr(batch, 'extend_lens', None) or [])],
                batch_size=len(batch.reqs))
    _local.batch = info
    try: yield info
    finally: _local.batch = old


def numeric_result(value):
    fields = ('num_tokens_evicted', 'swa_num_tokens_evicted', 'mamba_num_evicted',
              'full_kv_hit_length', 'swa_host_hit_length')
    return {k: int(getattr(value, k)) for k in fields if isinstance(getattr(value, k, None), int)}


def observe(cls, name, category, extract=None):
    original = getattr(cls, name)
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if not enabled() or rank() != 0: return original(self, *args, **kwargs)
        stack = getattr(_local, 'stack', [])
        if not stack: _local.stack = stack
        node = {'id': next(_ids), 'children': 0.0}
        parent = stack[-1]['id'] if stack else None
        before = extract(self, args, None, True) if extract else {}
        stack.append(node); start = time.perf_counter(); result = None; error = None
        try:
            result = original(self, *args, **kwargs)
            return result
        except BaseException as e:
            error = type(e).__name__; raise
        finally:
            end = time.perf_counter(); stack.pop()
            elapsed = end-start
            if stack: stack[-1]['children'] += elapsed
            after = extract(self, args, result, False) if extract else numeric_result(result)
            emit(dict(record='cpu_span', category=category, method=name,
                      span_id=node['id'], parent_id=parent, start_perf=start, end_perf=end,
                      cpu_ms=1000*elapsed, exclusive_ms=1000*max(0, elapsed-node['children']),
                      before=before, after=after, error=error, batch=current_batch()))
    setattr(cls, name, wrapped)


def lookup(self, args, result, before):
    p = args[0]
    if before:
        req = getattr(p, 'req', None)
        key = getattr(p, 'key', None)
        return dict(request_id=str(getattr(req, 'rid', '')), room=str(getattr(req, 'bootstrap_room', '')),
                    query_tokens=len(key) if key is not None else None,
                    context=getattr(key, 'context', None) is not None,
                    context_retry=bool(getattr(p, 'context_retry', False)))
    if result is None: return {}
    resident = getattr(result, 'context_resident', None)
    physical = len(result.device_indices) if resident is None else (
        int(resident.sum().item()) if resident.device.type == 'cpu' else None)
    return dict(matched_raw_tokens=len(result.device_indices), hit_tokens=physical,
                residency_count_known=physical is not None, **numeric_result(result))


def tracker(self, args, result, before):
    # evict_context_pages(node, tracker, device_frees, host_frees)
    return {str(k): int(v) for k, v in args[1].items()}


def install():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.managers.scheduler import Scheduler
    observe(UnifiedRadixCache, 'match_prefix', 'cache_lookup_inclusive', lookup)
    observe(UnifiedTreeCore, 'match_prefix', 'cache_tree_lookup')
    observe(UnifiedRadixCache, 'cache_finished_req', 'cache_back_finished')
    observe(UnifiedRadixCache, 'cache_unfinished_req', 'cache_back_unfinished')
    observe(UnifiedRadixCache, '_evict', 'eviction_pressure')
    observe(UnifiedRadixCache, 'evict_for_alloc', 'eviction_alloc_inclusive')
    observe(UnifiedRadixCache, '_evict_device_leaf', 'ordinary_leaf_eviction', tracker)
    observe(UnifiedTreeCore, 'evict_context_pages', 'dropaware_page_eviction', tracker)
    observe(UnifiedRadixCache, 'configure_context_drop_lock', 'dropaware_lock_configuration')
    observe(UnifiedRadixCache, '_configure_context_drop_eviction', 'dropaware_eligibility_update')
    original = Scheduler.flush_cache
    @functools.wraps(original)
    def flush(self, *args, **kwargs):
        global _enabled
        result = original(self, *args, **kwargs)
        if result:
            from observe_forward import drain_completed
            drain_completed(wait=True)  # idle flush, outside measurement only
            control = os.environ.get('ABLATION_PROFILE_CONTROL')
            if control:
                config = json.loads(Path(control).read_text())
                if type(config.get('enabled')) is not bool:
                    raise ValueError('Profile control enabled must be bool')
                _enabled = config['enabled']
            emit(dict(record='idle_flush', time_perf=time.perf_counter(), profile_enabled=_enabled))
        return result
    Scheduler.flush_cache = flush
