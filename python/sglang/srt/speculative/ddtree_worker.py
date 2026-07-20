"""Compatibility entry point for the runnable DDTree v2 worker.

The scheduler historically imported ``DDTreeWorker`` from this module. Keep
that stable name while the implementation lives in ``ddtree_worker_v2``.
"""

from sglang.srt.speculative.ddtree_worker_v2 import DDTreeWorkerV2

DDTreeWorker = DDTreeWorkerV2

__all__ = ["DDTreeWorker", "DDTreeWorkerV2"]
