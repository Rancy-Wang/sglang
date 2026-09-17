"""Capture the native mini default mask/page-occurrence BCP reference."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from bcp_numeric_fixture import load_fixture, request_for


async def main():
    root = Path(os.environ["CONTEXT_MINI_ROOT"])
    sys.path.insert(0, str(root / "python"))
    spec = importlib.util.spec_from_file_location(
        "mask_staged_runner", root / "tests/contextual/mask_staged_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_config = module.SchedulerConfig

    def bounded_config(**kwargs):
        kwargs.update(
            max_seq_len_override=24576,
            num_page_override=24576,
            max_extend_tokens=8192,
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
    model = os.environ["CONTEXT_SERVER_MODEL"]
    runner = module.Runner(model, reference_alignment=True)
    sampler = runner.scheduler.engine.sampler
    observed_sample = sampler.sample

    def raw_sample(logits, args):
        raw = logits.detach().clone()
        output = observed_sample(logits, args)
        for i, req in enumerate(runner.batch.reqs):
            record = runner.records[req.uid]
            if req.sample_is_committed:
                record["logits_gpu"][-1] = raw[i]
            else:
                record["logits_gpu"].pop()
                record["sample_rows"].pop()
        return output

    sampler.sample = raw_sample
    fixture = load_fixture()
    result = {
        "head": (
            await asyncio.to_thread(
                subprocess.check_output,
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            )
        ).strip(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": model,
        "fixture": fixture,
        "runs": {},
    }
    logits = {}
    reference_path = os.environ.get("CONTEXT_ORACLE_RETRY_REFERENCE")
    reference = json.loads(Path(reference_path).read_text()) if reference_path else None
    if reference is not None:
        assert reference["fixture"] == fixture and reference["model"] == model
        result["reference_path"] = reference_path
        result["comparison_kind"] = "matched_cache_retry_fixed_tokens"
    features = ("none", "drop_repos") if reference else ("none", "drop", "drop_repos")
    for feature in features:
        if feature == "none" or reference is None:
            runner.clear()
        if reference:
            runner.forced_tokens = reference["runs"][feature]["records"][0]["tokens"]
        run = await runner.generate(
            "mask",
            [request_for(fixture, feature)],
            max_tokens=int(os.environ.get("CONTEXT_NUMERIC_TOKENS", "64")),
        )
        (record,) = [r for r in run["records"] if not r["warmup"]]
        logits[feature] = runner.full_logits[record["uid"]]
        if reference:
            assert record["tokens"] == runner.forced_tokens
        result["runs"][feature] = run
        print("MINI_BCP", feature, run["responses"], flush=True)
    runner.clear()
    output = os.environ["CONTEXT_ORACLE_OUTPUT"]
    Path(output).write_text(json.dumps(result))
    module.torch.save(logits, output + ".pt")


if __name__ == "__main__":
    asyncio.run(main())
