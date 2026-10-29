"""Audit saved captures; distinguish common CLI settings from graph defaults."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

BASELINE = "7df281a0e656e368c90f529d8d84305db99001f5"
CANDIDATE = "16ed789a493981b61f63c5d9720564b66b4e8369"
COMMON = {
    "dtype": "bfloat16", "attention_backend": "triton", "bf16_gemm_backend": "auto",
    "max_running_requests": 16, "max_total_tokens": 12000, "context_length": 2048,
    "page_size": 1, "disable_radix_cache": True, "mem_fraction_static": .78,
    "disable_overlap_schedule": False, "tp_size": 1, "pp_size": 1, "dp_size": 1,
}


def read(path):
    return json.loads(path.read_text())


def audit(root):
    prompts, graphs, memory = {}, {}, {}
    datasets, packages, sources = set(), set(), set()
    cells = []
    for phase in ("performance", "profile"):
        folders = sorted(p for p in (root / phase).iterdir() if p.is_dir())
        expected = {f"{v}-b{b}-{p}" for v in ("ar", "old", "new")
                    for b in (1, 2, 4, 8, 16) for p in ("simultaneous", "staggered")}
        assert {p.name for p in folders} == expected
        for folder in folders:
            variant, batch, pattern = folder.name.split("-")
            bs = int(batch[1:])
            manifest, info, prompt = (read(folder / f"{name}.json")
                                      for name in ("manifest", "server_info", "prompts"))
            args = manifest["args"]
            assert manifest["sha"] == (BASELINE if variant == "old" else CANDIDATE)
            source = "/home/ruoxi/repo/test/baselines/loopspec-7df281a" if variant == "old" else "/workspace/sglang"
            assert manifest["sglang_file"] == source + "/python/sglang/__init__.py"
            assert args["warmup"] == 2 and args["repeat"] == (5 if phase == "performance" else 1)
            assert args["profile"] == (phase == "profile") and args["max_new_tokens"] == 512
            assert args["thinking"] and args["first"] == bs
            assert args["arrival_ms"] == ([0] * bs if pattern == "simultaneous" else [100 * i for i in range(bs)])
            assert args["algorithm"] == ("ar" if variant == "ar" else "loopspec")
            assert args["model"] == info["model_path"] == "/home/ruoxi/models/Ouro-2.6B-Thinking"
            assert {k: info[k] for k in COMMON} == COMMON, (phase, folder.name)
            assert info["speculative_algorithm"] == (None if variant == "ar" else "LOOPSPEC")
            if variant != "ar":
                assert info["speculative_draft_depth"] == [1]
            if bs in prompts:
                assert prompt == prompts[bs]
            else:
                prompts[bs] = prompt
            if variant in graphs:
                assert info["cuda_graph_config"] == graphs[variant]
            else:
                graphs[variant] = info["cuda_graph_config"]
            datasets.add(manifest["dataset_sha256"])
            packages.add(json.dumps(manifest["packages"], sort_keys=True))
            sources.add(manifest["sglang_file"])
            cells.append(f"{phase}/{folder.name}")
            if phase == "performance" and bs == 16 and pattern == "simultaneous":
                memory[variant] = info["internal_states"][0]["memory_usage"]
    assert len(datasets) == len(packages) == 1
    assert graphs["old"] == graphs["new"]
    assert graphs["ar"]["decode"]["bs"] == [1, 2, 4, 8, 12, 16]
    assert graphs["new"]["decode"]["bs"] == [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    normalized = copy.deepcopy(graphs)
    for graph in normalized.values():
        del graph["decode"]["bs"]
    assert normalized["ar"] == normalized["old"] == normalized["new"]
    output = dict(
        capture_cells=len(cells), cells=cells, baseline_sha=BASELINE, candidate_capture_sha=CANDIDATE,
        source_files=sorted(sources), dataset_sha256=datasets.pop(), packages=json.loads(packages.pop()),
        common_server_config=COMMON, cuda_graph_config_by_variant=graphs,
        graph_note="Same public max-bs=16; native cuda_graph_hook generates denser default sizes for speculative algorithms. Old/new LoopSpec are identical; this task did not change the hook.",
        prompt_lengths=[len(p["input_ids"]) for p in prompts[16]["prompts"]],
        prompt_sha256_by_bs={str(b): hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
                            for b, p in prompts.items()},
        startup_memory_usage_gib=memory,
        memory_note="Startup allocation census, not a continuously sampled serving peak.",
        raw_root=str(root),
    )
    (root / "capture-validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"capture_cells": len(cells), "success": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
