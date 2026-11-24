"""Run BF16 and LoopVQ on one exclusive local GPU pair in tmux."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
import urllib.request

import benchmark_throughput as bench
import decode_sweep as sweep

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TASK = sweep.safe.TASK


def query(gpu, fields, kind="gpu"):
    return subprocess.check_output(["nvidia-smi", "-i", gpu, f"--query-{kind}={fields}",
                                    "--format=csv,noheader,nounits"], text=True).strip()


def serve(args, method, root):
    cards = args.cards.split(",")
    assert len(set(cards)) == 2 and all(c.isdigit() for c in cards)
    assert all(not query(c, "pid", "compute-apps") for c in cards)
    uuids = [query(c, "uuid") for c in cards]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    model = Path(os.environ["LOOPVQ_MODEL"])
    assert json.loads((model / "config.json").read_text())["total_ut_steps"] == 4
    bundle = TASK / "artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e"
    cmd = [os.environ["LOOPVQ_PYTHON"], "-m", "sglang.launch_server", "--tp-size", "2",
           "--model-path", str(model), "--trust-remote-code", "--host", "127.0.0.1",
           "--port", str(args.port), "--dtype", "bfloat16", "--attention-backend", "triton",
           "--page-size", "8", "--max-running-requests", "16", "--max-prefill-tokens", "8192",
           "--context-length", "40960", "--mem-fraction-static", ".90", "--disable-radix-cache",
           "--chunked-prefill-size", "-1", "--cuda-graph-backend-decode", "full",
           "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "--cuda-graph-backend-prefill", "disabled",
           "--enable-metrics", "--decode-log-interval", "1", "--incremental-streaming-output",
           "--ouro-decode-profile", "optimized", "--triton-attention-num-kv-splits", "48" if method=="loopvq" else "8"]
    if method == "loopvq":
        cmd += ["--loopvq-codebook-dir", str(bundle), "--loopvq-prefix-tokens", "8",
                "--loopvq-recent-tokens", "64", "--max-total-tokens", "172032"]
    url = f"http://127.0.0.1:{args.port}"
    log = root / f"{method}.log"
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), CUDA_VISIBLE_DEVICES=args.cards,
               TORCH_CUDA_ARCH_LIST="8.0", MAX_JOBS="3")
    with log.open("x") as stream:
        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited {process.returncode}: {log}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=3):
                    break
            except OSError:
                time.sleep(4)
        else:
            raise TimeoutError(str(log))
        cfg = bench.get_server_config(url, 30)
        assert cfg['tp_size']==2 and cfg['dp_size']==1 and cfg['cuda_graph_config']['decode']['backend']=='full'
        pids = [query(c, "pid", "compute-apps").splitlines() for c in cards]
        assert all(len(ps)==1 and int(ps[0]) in sweep.safe.process_tree(process.pid) for ps in pids)
        fixture = json.loads((root / "input-2048.json").read_text())
        payload = {"input_ids":fixture["input_ids"][0],"sampling_params":{"temperature":0,"max_new_tokens":16,"ignore_eos":True}}
        req = urllib.request.Request(url + "/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=120) as response:
            probe = json.load(response)
        assert len(probe['output_ids']) == 16
        bench.atomic_json(root / f"{method}-probe.json",probe)
        manifest = dict(host=platform.node(),gpu_uuid=uuids[0],tp_gpu_uuids=uuids,rank_pids=pids,
                        server_pid=process.pid,method=method,exclusive_server=True,model_path=str(model),
                        model_sha256=sweep.safe.sha256(model/"model.safetensors"),
                        sglang_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                        environment_sha256=sweep.safe.sha256(Path(os.environ["LOOPVQ_PYTHON"]).resolve()),
                        prefix_tokens=8,recent_tokens=64,config=dict(cfg,total_ut_steps=4))
        if method == "loopvq":
            manifest["bundle_sha256"]={name:sweep.safe.sha256(bundle/name) for name in ("codebook.pt","vq_v_codebook.pt")}
        # Read the allocated BF16 capacity when ServerArgs requested automatic sizing.
        if cfg.get("max_total_tokens") is None:
            import re
            matches = re.findall(r"max_total_num_tokens[=:]\s*(\d+)", log.read_text())
            if not matches:
                raise ValueError("Missing scheduler token capacity")
            manifest["config"]["max_total_tokens"]=min(map(int,matches))
        bench.atomic_json(root / f"{method}-manifest.json",manifest)
        sweep.context.URL = url
        for output,bs in args.cells:
            sweep.measure_cell(method,output,bs,manifest,log)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=180)
        deadline = time.monotonic()+120
        while any(query(c,"pid","compute-apps") for c in cards) and time.monotonic()<deadline:
            time.sleep(2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("cards")
    parser.add_argument("port",type=int)
    parser.add_argument("run")
    parser.add_argument("cells",nargs="+",type=lambda s: tuple(map(int,s.split("/"))))
    parser.add_argument("--methods",nargs="+",choices=["bf16","loopvq"],default=["bf16","loopvq"])
    args=parser.parse_args()
    assert os.environ.get("TMUX") and all(n in sweep.OUTPUTS and b in sweep.BATCHES for n,b in args.cells)
    root=TASK/"results/tp2"/args.run
    root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(sweep.ROOT/"input-2048.json",root/"input-2048.json")
    sweep.ROOT=root
    for method in args.methods:
        serve(args,method,root)
    bench.atomic_json(root/"complete.json",{"cells":args.cells,"methods":args.methods})


if __name__=="__main__":
    main()
