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
    if os.environ.get("ABLATION_OPERATION_CAPTURE"):
        install_operation_capture()
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.managers.scheduler import Scheduler
    observe(UnifiedRadixCache, 'match_prefix', 'cache_lookup_inclusive', lookup)
    observe(UnifiedTreeCore, 'match_prefix', 'cache_tree_lookup')
    observe(UnifiedRadixCache, '_evict', 'eviction_pressure')
    observe(UnifiedRadixCache, 'evict_for_alloc', 'eviction_alloc_inclusive')
    observe(UnifiedRadixCache, '_evict_device_leaf', 'ordinary_leaf_eviction', tracker)
    observe(UnifiedTreeCore, 'evict_context_pages', 'dropaware_page_eviction', tracker)
    observe(UnifiedRadixCache, 'configure_context_prefill_drop_lock', 'dropaware_admission_check')
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


# Deliberately intrusive, separate-run instrumentation. Never enabled in the
# formal performance matrix: tensor snapshots synchronize the current stream.
_operation_sink = None
_operation_layers = {}
_operation_ids = itertools.count()


def operation_counts(source, destination, positions, skip_unmapped):
    if not (len(source) == len(destination) == len(positions)):
        raise ValueError("Operation snapshot lengths differ")
    valid = [i for i, (a, b) in enumerate(zip(source, destination))
             if not skip_unmapped or (a > 0 and b > 0)]
    moved = [i for i in valid if positions[i][0] != positions[i][1]]
    return dict(token_events=len(source), valid_token_events=len(valid),
                skipped_token_events=len(source)-len(valid),
                moved_token_events=len(moved), same_position_events=len(valid)-len(moved),
                unique_source_slots_in_launch=len({source[i] for i in valid}))


def operation_emit(row):
    global _operation_sink
    prefix = os.environ['ABLATION_OPERATION_CAPTURE']
    if _operation_sink is None:
        _operation_sink = Sink(f"{prefix}-{os.environ.get('ABLATION_STAGE')}-rank{rank()}")
        atexit.register(_operation_sink.close)
    _operation_sink.emit(dict(row, pid=os.getpid(), rank=rank(),
                              stage=os.environ.get('ABLATION_STAGE')))


def install_operation_capture():
    import hashlib
    import numpy as np
    import torch
    from sglang.srt.layers.attention.context_backend import ContextModelBinding
    from sglang.kernels.ops.attention import context_reposition as kernel
    init = ContextModelBinding.__init__

    @functools.wraps(init)
    def bind(self, *args, **kwargs):
        init(self, *args, **kwargs)
        for layer in self.layers:
            pointer = int(layer.k_buffer.data_ptr())
            _operation_layers[pointer] = dict(layer_id=int(layer.layer_id),
                                              sliding_window=bool(layer.sliding_window))
        operation_emit(dict(record='model_binding', time_perf=time.perf_counter(),
                            layers=list(_operation_layers.values())))
    ContextModelBinding.__init__ = bind
    original = kernel.reposition_kv_layers

    @functools.wraps(original)
    def capture(kp, vp, k, v, src, dst, pos, rope, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            # A Python hook cannot count graph replay launches. Mark the gap;
            # do not silently present capture-time calls as actual executions.
            operation_emit(dict(record='graph_capture_uncovered', time_perf=time.perf_counter()))
            return original(kp, vp, k, v, src, dst, pos, rope, **kwargs)
        started = time.perf_counter()
        arrays = {name: tensor.detach().cpu().numpy().copy()
                  for name, tensor in [('source', src), ('destination', dst), ('positions', pos)]}
        pointers = kp.detach().cpu().tolist()
        # Content-addressed snapshots deduplicate repeated per-layer launches;
        # each actual launch still gets a separate event (including repeats).
        digest = hashlib.sha256()
        for name, value in arrays.items():
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        blob_id = digest.hexdigest()
        directory = Path(os.environ['ABLATION_OPERATION_CAPTURE'] + f'-blobs-{os.getpid()}')
        directory.mkdir(parents=True, exist_ok=True)
        blob = directory / (blob_id + '.npz')
        if not blob.exists():
            np.savez_compressed(blob, **arrays)
        counts = operation_counts(arrays['source'].tolist(), arrays['destination'].tolist(),
                                  arrays['positions'].tolist(), kwargs.get('skip_unmapped', False))
        row = dict(record='reposition_launch', event_id=next(_operation_ids),
                   start_perf=started, kernel_call_perf=time.perf_counter(),
                   snapshot=str(blob), snapshot_hash=blob_id,
                   counts=counts, layers=[_operation_layers.get(int(x)) for x in pointers],
                   layer_count=len(pointers), local_kv_heads=int(k.shape[1]),
                   head_dim=int(k.shape[2]), rotary_dim=kwargs.get('rotary_dim') or int(k.shape[2]),
                   kv_element_bytes=k.element_size(), rope_element_bytes=rope.element_size(),
                   kv_shape=list(k.shape), kv_stride=list(k.stride()), rope_shape=list(rope.shape),
                   is_neox_style=kwargs.get('is_neox_style', True),
                   skip_unmapped=kwargs.get('skip_unmapped', False), batch=dict(current_batch()),
                   timing_usable_for_baseline=False)
        try:
            result = original(kp, vp, k, v, src, dst, pos, rope, **kwargs)
        except BaseException as error:
            operation_emit(dict(row, error=repr(error), end_perf=time.perf_counter()))
            raise
        operation_emit(dict(row, error=None, end_perf=time.perf_counter()))
        return result
    kernel.reposition_kv_layers = capture
