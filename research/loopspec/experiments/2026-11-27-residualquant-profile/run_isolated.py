"""Run existing kernel-only checks without importing the model serving stack.

This supplies package namespaces and four hardware-query helpers only. Kernel,
codec, environment defaults, oracle and timing source files remain unmodified.
This entry point is explicitly unsuitable for a full-serving measurement.
"""
import json
from pathlib import Path
import runpy
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "research/loopspec/experiments/2026-11-26-vqplus-residual"


def bootstrap():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 0)
    for name in ("sglang", "sglang.kernels", "sglang.kernels.ops",
                 "sglang.kernels.ops.attention", "sglang.kernels.ops.loopvq",
                 "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.utils"):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
        sys.modules[name] = module
    utilities = sys.modules["sglang.srt.utils"]
    utilities.is_hip = lambda: torch.version.hip is not None
    utilities.is_gfx95_supported = lambda: False
    utilities.is_gfx1250_supported = lambda: False
    utilities.get_device_core_count = lambda device=0: torch.cuda.get_device_properties(device).multi_processor_count
    import triton
    print(json.dumps({"isolated_kernel_harness": True, "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda, "triton": triton.__version__,
                      "gpu": torch.cuda.get_device_name(), "root": str(ROOT),
                      "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}), flush=True)


if __name__ == "__main__":
    script = sys.argv.pop(1)
    assert script in ("diagnose_paged.py", "check_paged.py", "check_residual.py")
    bootstrap()
    sys.path.insert(0, str(OLD))
    runpy.run_path(str(OLD / script), run_name="__main__")
