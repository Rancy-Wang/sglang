"""Own isolated servers and run the approved complete-trajectory BCP method."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def replay_template(template):
    """Allow recorded, explicitly named tool returns after fixed-length output.

    Like mini's BCP method, the next tool response comes from the dataset even
    when this run generated no tool call. Keep native formatting, but resolve
    the speaker from that response's name instead of a generated call. This is
    a benchmark-only template shared by modified and frozen SGLang servers.
    """
    from sglang.srt.context_system.thinking_template import retained_template

    template, family = retained_template(template)
    if family == "gpt-oss":
        guard = "{%- if last_tool_call.name is none %}"
        speaker = '"<|start|>functions." + last_tool_call.name'
        if template.count(guard) != 1 or template.count(speaker) != 1:
            raise ValueError("Unrecognized GPT-OSS tool-result template")
        template = template.replace(
            guard, "{%- if not message.name and last_tool_call.name is none %}"
        ).replace(speaker, '"<|start|>functions." + (message.name or last_tool_call.name)')
    return template


def warmup(args, root):
    """Compile representative kernels before the measured complete trajectories.

    All engines keep this small, disjoint prefix cached; mini has no HTTP cache
    flush. No source task, generated history or measured token is warmed up.
    """
    room = time.time_ns() % (1 << 52)
    observations = []

    def post(port, body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            value = json.load(response)
        if not value.get("choices"):
            raise RuntimeError(f"Warmup did not produce a completion: {value}")
        return value.get("usage")

    for feature in ["none", "drop", "drop_repos"] if args.drop else ["none"]:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2 * args.concurrency) as executor:
            pending = []
            for index in range(args.concurrency):
                body = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"Kernel warmup only, {feature}, lane {index}.",
                        },
                        {"role": "user", "content": "red green blue " * 4096},
                        {
                            "role": "assistant",
                            "content": "I have read the warmup text.",
                        },
                        {
                            "role": "user",
                            "content": "List the colors in a short sentence.",
                        },
                    ],
                    "temperature": 0,
                    "max_tokens": 64,
                    "ignore_eos": True,
                }
                if args.engine != "mini":
                    body["chat_template_kwargs"] = {"preserve_thinking_history": True}
                if feature != "none":
                    body["drop_message"] = {"2": [1]}
                if feature == "drop_repos":
                    body["reposition"] = [2]
                if args.engine == "pd":
                    room += 1
                    body.update(
                        bootstrap_host="127.0.0.1",
                        bootstrap_port=args.port + 10,
                        bootstrap_room=room,
                    )
                    pending.append(executor.submit(post, args.port + 1, body))
                pending.append(executor.submit(post, args.port, body))
            usage = [future.result() for future in pending]
        observations.append(
            {
                "feature": feature,
                "seconds": time.perf_counter() - started,
                "usage": usage,
            }
        )
        (root / "warmup.json").write_text(json.dumps(observations, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["mini", "sglang", "pd"], required=True)
    p.add_argument("--server-python", required=True)
    p.add_argument("--server-repo", required=True)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--requests-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--decode-gpus", default="1")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=29161)
    p.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    p.add_argument("--drop", action="store_true")
    p.add_argument("--native-baseline", action="store_true")
    p.add_argument("--context-length", type=int, default=196608)
    p.add_argument("--capacity", type=int, default=262144)
    p.add_argument("--chunk", type=int, default=8192)
    args = p.parse_args()
    if args.native_baseline and (args.drop or args.engine != "sglang"):
        p.error("Native baseline is a no-Drop SGLang launch")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    processes, logs, commands = [], [], []
    repo = Path(args.server_repo).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]):
        raise ValueError("Server checkout must be clean")
    template_path = None
    template_sha256 = None
    if args.engine != "mini":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        template = replay_template(tokenizer.get_chat_template())
        template_path = root / "retained_history.jinja"
        template_path.write_text(template)
        template_sha256 = hashlib.sha256(template.encode()).hexdigest()
    try:
        modes = ["prefill", "decode"] if args.engine == "pd" else ["normal"]
        for i, mode in enumerate(modes):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpus if i == 0 else args.decode_gpus
            env["PYTHONPATH"] = str(repo / "python")
            env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
            for key in ("SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
                directory = root / mode / key.lower()
                directory.mkdir(parents=True)
                env[key] = str(directory)
            port = args.port + i
            cmd = [
                args.server_python,
                "-m",
                "minisgl" if args.engine == "mini" else "sglang.launch_server",
                "--model-path",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tp-size",
                str(args.tp),
                "--dtype",
                "bfloat16",
                "--page-size",
                "1",
                "--max-running-requests",
                "4",
            ]
            if args.engine == "mini":
                cmd += [
                    "--max-seq-len-override",
                    str(args.context_length),
                    "--num-pages",
                    str(args.capacity),
                    "--max-prefill-length",
                    str(args.chunk),
                    "--cuda-graph-max-bs",
                    "4",
                ]
                # Existing mini serving benchmark deliberately retains historical
                # Harmony reasoning so its event boundaries remain stable.
                env["MINISGL_PRESERVE_HARMONY_HISTORY"] = "1"
                if args.drop:
                    cmd += ["--drop-aware-eviction"]
            else:
                cmd += [
                    "--chat-template",
                    str(template_path),
                    "--context-length",
                    str(args.context_length),
                    "--max-total-tokens",
                    str(args.capacity),
                    "--chunked-prefill-size",
                    str(args.chunk),
                    "--cuda-graph-config",
                    json.dumps(
                        {
                            "decode": {"bs": [1, 2, 4], "max_bs": 4},
                            "prefill": {"bs": [16, 32, 64], "max_bs": 64},
                        }
                    ),
                ]
                if args.drop and mode != "decode":
                    cmd += ["--context-drop-aware-eviction"]
                if "gpt-oss" in args.model.lower():
                    cmd += [
                        "--tool-call-parser",
                        "gpt-oss",
                        "--reasoning-parser",
                        "gpt-oss",
                        "--disable-hybrid-swa-memory",
                    ]
                if args.engine == "pd":
                    cmd += [
                        "--disaggregation-mode",
                        mode,
                        "--disaggregation-bootstrap-port",
                        str(args.port + 10),
                        "--nccl-port",
                        str(args.port + 20 + i),
                    ]
                else:
                    cmd += ["--enable-mixed-chunk"]
            commands.append(
                {
                    "argv": cmd,
                    "gpu": env["CUDA_VISIBLE_DEVICES"],
                    "head": head,
                    "mode": mode,
                    "template_sha256": template_sha256,
                }
            )
            (root / "launch.json").write_text(
                json.dumps({"args": vars(args), "servers": commands}, indent=2)
            )
            log = (root / f"{mode}.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                cmd,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)
            deadline = time.monotonic() + 600
            while True:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Server startup failed; inspect {root / (mode + '.log')}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/"
                        + ("v1/models" if args.engine == "mini" else "health"),
                        timeout=5,
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
        warmup(args, root)
        common = [
            "--model",
            args.model,
            "--tokenizer",
            args.model,
            "--requests-path",
            args.requests_path,
            "--output-dir",
            str(root / "workload"),
            "--concurrency",
            str(args.concurrency),
            "--num-tasks",
            str(2 * args.concurrency),
            "--seed",
            "42",
        ]
        if args.drop:
            common += ["--drop"]
        if args.engine == "mini":
            client = [
                args.server_python,
                str(Path(args.mini_root) / "tests/benchmark/test_serving.py"),
                "run",
                "--protocol",
                "minisgl-harmony",
                "--port",
                str(args.port),
                *common,
            ]
        else:
            client = [
                sys.executable,
                str(Path(__file__).with_name("test_serving.py")),
                "--mini-root",
                args.mini_root,
                "--url",
                f"http://127.0.0.1:{args.port + len(modes) - 1}/v1/chat/completions",
                *common,
            ]
            client += [
                "--chat-template",
                str(template_path),
                "--template-kwargs",
                json.dumps({"preserve_thinking_history": True}),
            ]
            if args.engine == "pd":
                client += [
                    "--prefill-url",
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    "--bootstrap-port",
                    str(args.port + 10),
                ]
        (root / "client.json").write_text(json.dumps(client, indent=2))
        client_env = os.environ.copy()
        if args.engine == "mini":
            client_env["PYTHONPATH"] = str(Path(args.mini_root) / "python")
        with (root / "client.log").open("w") as log:
            subprocess.run(
                client, check=True, env=client_env, stdout=log, stderr=subprocess.STDOUT
            )
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
