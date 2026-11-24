"""Distribute independent accuracy cells across two-GPU servers."""
import os
from pathlib import Path
import subprocess

from benchmark_throughput import atomic_json

HARNESS = Path.home() / "repo/LoopVQ/experiments/loopvq_reproduction/run_evaluation.py"
BENCHMARKS = ("gpqa_diamond", "evalplus_mbpp", "aime25", "aime24", "evalplus_humaneval")


def evaluate(task, root, method, manifest, url):
    name = "loopvq_0.5bit" if method == "loopvq" else "bf16"
    results = task / "results/accuracy-tp2"
    claims = results / "claims" / name
    claims.mkdir(parents=True, exist_ok=True)
    source = root / "accuracy-manifest.json"
    atomic_json(source, dict(manifest, method=name, total_ut_steps=4, sglang_git_sha=manifest["sglang_sha"]))
    python = Path.home() / "anaconda3/envs/loopvq-eval/bin/python"
    command = [str(python), "-u", str(HARNESS), "--method", name, "--model", manifest["model_path"],
               "--base-url", url, "--server-manifest", str(source), "--root", str(results/root.name),
               "--num-concurrent", "2" if method=="bf16" else "12", "--api-seed-mode", "explicit",
               "--wandb-group", "loopvq-tp2-accuracy"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_DATASETS_CACHE=str(task/"cache/huggingface/datasets"))
    for key in ("PYTHONPATH","LD_LIBRARY_PATH","VIRTUAL_ENV","HUMANEVAL_OVERRIDE_PATH",
                "MBPP_OVERRIDE_PATH","EVALPLUS_TIMEOUT_PER_TASK"):
        env.pop(key, None)
    subprocess.run(command+["--audit","--benchmarks","aime24","evalplus_humaneval","--seeds","0","1"],
                   env=env, check=True)
    for benchmark in BENCHMARKS:
        for seed in range(3):
            claim = claims / f"{benchmark}-{seed}"
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            atomic_json(claim/"owner.json",{"root":str(root),"host":manifest["host"],
                        "gpu_pair":manifest["tp_gpu_uuids"],"server_pid":manifest["server_pid"]})
            subprocess.run(command+["--benchmarks",benchmark,"--seeds",str(seed)],env=env,check=True)
            markers=list((results/root.name/name).glob(f"*/full/{benchmark}/{seed}/complete.json"))
            assert len(markers)==1
            atomic_json(claim/"done.json",{"marker":str(markers[0])})
