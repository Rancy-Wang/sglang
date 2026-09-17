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
    # Bound only test capacity, preserving the runner's native graph path.
    original_config = module.SchedulerConfig

    def bounded_config(**kwargs):
        kwargs.update(
            max_seq_len_override=2048, num_page_override=4096, max_extend_tokens=2048
        )
        return original_config(**kwargs)

    module.SchedulerConfig = bounded_config
    original_body = module.request_body

    def draining_body(*args, **kwargs):
        kwargs["ignore_eos"] = True
        return original_body(*args, **kwargs)

    module.request_body = draining_body
    from minisgl.tokenizer.server import _build_occurrence_user_msg

    original_build = module._build_user_msg

    def native_dispatch(msg, tokenized):
        if tokenized.reposition_input_ids is not None:
            return _build_occurrence_user_msg(msg, tokenized)
        return original_build(msg, tokenized)

    module._build_user_msg = native_dispatch
    runner = module.Runner(model, reference_alignment=True)
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
    all_logits = {}
    forced = [785, 17951, 374, 264, 12767, 323, 25382, 2487]
    cases = [
        ("mask", "none"),
        ("staged", "drop"),
        ("mask", "drop"),
        ("mask", "drop_repos"),
    ]
    for mode, feature, fixed in (
        (mode, feature, fixed) for fixed in (False, True) for mode, feature in cases
    ):
        runner.clear()
        runner.forced_tokens = forced if fixed else None
        request = {"messages": messages, "enable_thinking": False}
        if feature != "none":
            request.update(drop_message={1: [0]})
        if feature == "drop_repos":
            request.update(reposition=[1])
        result = await runner.generate(mode, [request], max_tokens=8)
        result["fixed_tokens"] = forced if fixed else None
        result["request"] = request
        for record in result["records"]:
            if not record["warmup"]:
                key = f"{mode}-{feature}-{fixed}"
                all_logits[key] = runner.full_logits[record["uid"]]
                record["logits_key"] = key
        results["runs"].append(result)
        print("MINI_RESULT", mode, feature, fixed, result["responses"], flush=True)
    Path(os.environ["CONTEXT_ORACLE_OUTPUT"]).write_text(json.dumps(results))
    module.torch.save(all_logits, os.environ["CONTEXT_ORACLE_OUTPUT"] + ".pt")


if __name__ == "__main__":
    asyncio.run(main())
