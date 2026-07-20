"""Location record for the retired spec-v1 DDTree worker.

The complete, unmodified v1 implementation is intentionally retained on the
``ddtree-v1-backup-398fc2c`` branch at commit ``398fc2c726``:

``python/sglang/srt/speculative/ddtree_worker.py``

Current main removed the spec-v1 scheduler and ``DFlashWorker`` APIs, so
importing that historical implementation into the live package would make
module discovery fail. This marker keeps its exact provenance next to the v2
implementation without pretending that v1 is runnable on the new runtime.
"""

DDTREE_V1_SOURCE_COMMIT = "398fc2c726"
DDTREE_V1_SOURCE_PATH = "python/sglang/srt/speculative/ddtree_worker.py"


class DDTreeWorkerV1:
    """Non-runnable marker for the preserved spec-v1 implementation."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "DDTreeWorkerV1 depends on the removed spec-v1 runtime. "
            f"Use {DDTREE_V1_SOURCE_COMMIT}:{DDTREE_V1_SOURCE_PATH} to inspect "
            "or run it on the ddtree-v1-backup-398fc2c branch."
        )


__all__ = [
    "DDTREE_V1_SOURCE_COMMIT",
    "DDTREE_V1_SOURCE_PATH",
    "DDTreeWorkerV1",
]
