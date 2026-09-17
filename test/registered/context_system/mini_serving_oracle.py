"""Run the fixed mini-sglang API/staged oracle in its own native environment.

This executable imports the existing test runner read-only. It does not replace
tokenization, scheduling, attention, sampling, or model implementations.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


async def main():
    root = Path(os.environ["CONTEXT_MINI_ROOT"])
    sys.path.insert(0, str(root / "python"))
    spec = importlib.util.spec_from_file_location(
        "mask_staged_runner", root / "tests/contextual/mask_staged_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = os.environ["CONTEXT_SERVER_MODEL"]
    runner = module.Runner(model)
    messages = [
        {"role": "user", "content": "Remember this text: " + "red blue green " * 32},
        {"role": "assistant", "content": "I have read the text."},
        {"role": "user", "content": "Reply with a short sentence about the ocean."},
    ]
    results = {
        "head": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "runs": [],
    }
    for mode, feature in (("mask", False), ("staged", True), ("mask", True)):
        runner.clear()
        request = {"messages": messages, "enable_thinking": False}
        if feature:
            request.update(drop_message={1: [0]}, reposition=[1])
        result = await runner.generate(mode, [request], max_tokens=8)
        result["request"] = request
        results["runs"].append(result)
        print("MINI_RESULT", mode, feature, result["responses"], flush=True)
    Path(os.environ["CONTEXT_ORACLE_OUTPUT"]).write_text(json.dumps(results))


if __name__ == "__main__":
    asyncio.run(main())
