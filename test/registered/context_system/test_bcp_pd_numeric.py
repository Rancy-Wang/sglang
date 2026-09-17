"""One real BCP task across independent native P/D processes and GPUs."""

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
torch = pytest.importorskip("torch")
from bcp_numeric_fixture import oracle_chat_template, request_for
from serving_logits_probe import serialized_probe

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_PD_BCP_ORACLE"),
    reason="explicit PD BCP reference required",
)


@pytest.fixture(scope="module")
def pd_servers():
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("CONTEXT_PD_PORT", "28961"))
    bootstrap = port + 10
    processes, logs = [], []
    bases = []
    template = oracle_chat_template(
        os.environ["CONTEXT_PD_BCP_ORACLE"],
        os.environ["CONTEXT_SERVER_MODEL"],
        directory,
    )
    try:
        for i, mode in enumerate(("prefill", "decode")):
            base = f"http://127.0.0.1:{port + i}"
            bases.append(base)
            env = os.environ.copy()
            for name, subdir in (
                ("SGLANG_CACHE_DIR", "sglang-cache"),
                ("SGLANG_JIT_CACHE_DIR", "sglang-jit-cache"),
                ("TRITON_CACHE_DIR", "triton-cache"),
                ("TMPDIR", "tmp"),
            ):
                cache_root = Path(os.environ.get("CONTEXT_PD_CACHE_ROOT", directory))
                target = cache_root / mode / subdir
                target.mkdir(parents=True, exist_ok=True)
                env[name] = str(target)
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                "CONTEXT_P_GPU" if i == 0 else "CONTEXT_D_GPU", str(i)
            )
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).parent.resolve()), env.get("PYTHONPATH", "")]
            )
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                os.environ["CONTEXT_SERVER_MODEL"],
                "--host",
                "127.0.0.1",
                "--port",
                str(port + i),
                "--nccl-port",
                str(port + 20 + i),
                "--disaggregation-mode",
                mode,
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                "--page-size",
                "1",
                "--dtype",
                "bfloat16",
                "--max-total-tokens",
                os.environ.get(
                    "CONTEXT_P_KV" if i == 0 else "CONTEXT_D_KV",
                    "24576" if i == 0 else "16384",
                ),
                "--context-length",
                os.environ.get("CONTEXT_MAX_LENGTH", "16384"),
                "--max-running-requests",
                "4",
                "--chunked-prefill-size",
                os.environ.get("CONTEXT_CHUNK_SIZE", "512"),
                "--context-drop-aware-eviction",
                "--cuda-graph-config",
                json.dumps(
                    {
                        "decode": {"bs": [1, 2, 4], "max_bs": 4},
                        "prefill": {"bs": [16, 32, 64], "max_bs": 64},
                    }
                ),
                "--enable-custom-logit-processor",
            ]
            cmd += ["--tp-size", os.environ.get("CONTEXT_TEST_TP", "1")]
            if template:
                cmd += ["--chat-template", template]
            if "gpt-oss" in os.environ["CONTEXT_SERVER_MODEL"].lower():
                cmd += [
                    "--tool-call-parser",
                    "gpt-oss",
                    "--reasoning-parser",
                    "gpt-oss",
                    "--disable-hybrid-swa-memory",
                ]
            backend = os.environ.get("CONTEXT_TEST_ATTENTION_BACKEND")
            if backend:
                cmd += ["--attention-backend", backend]
            if mode == "decode" and os.environ.get("CONTEXT_PD_RETRACT") == "1":
                # Native fault injection exercises the real host backup/resume.
                env["SGLANG_TEST_RETRACT"] = "1"
                env["SGLANG_TEST_RETRACT_INTERVAL"] = "16"
                cmd += ["--hicache-ratio", "0.5"]
            log_path = directory / f"{mode}-server.log"
            log = log_path.open("w")
            logs.append(log)
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            processes.append(proc)
            print(
                "PD_SERVER",
                mode,
                env["CUDA_VISIBLE_DEVICES"],
                json.dumps(cmd),
                flush=True,
            )
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    pytest.fail(log_path.read_text()[-16000:])
                try:
                    if requests.get(base + "/health", timeout=5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail(log_path.read_text()[-16000:])
        yield (*bases, bootstrap)
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)
        for log in logs:
            log.close()


def test_bcp_pd_terminal_handoff(pd_servers):
    p_base, d_base, bootstrap = pd_servers
    reference_path = Path(os.environ["CONTEXT_PD_BCP_ORACLE"])
    reference = json.loads(reference_path.read_text())
    reference_logits = torch.load(str(reference_path) + ".pt", weights_only=True)
    directory = Path(os.environ["CONTEXT_TRACE_DIR"])
    processor = serialized_probe()
    comparisons = {}
    room = int(time.time_ns() % (1 << 53))
    lock = threading.Lock()

    def call(feature, name, fixed=True):
        nonlocal room
        with lock:
            room += 1
            request_room = room
        tokens = reference["runs"][feature]["records"][0]["tokens"]
        payload = {
            **request_for(reference["fixture"], feature),
            "model": os.environ["CONTEXT_SERVER_MODEL"],
            "temperature": 0,
            "max_tokens": len(tokens),
            "ignore_eos": True,
            "return_meta_info": True,
            "return_input_ids_in_sglext": True,
            "return_output_ids_in_sglext": True,
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": bootstrap,
            "bootstrap_room": request_room,
        }
        if fixed:
            payload["custom_logit_processor"] = processor

        def send(mode, base, count, offset):
            body = dict(payload)
            if fixed:
                body["custom_params"] = {
                    "context_trace_path": str(directory / f"{name}-{mode}.pt"),
                    "context_trace_count": count,
                    "context_forced_tokens": tokens,
                    "context_forced_offset": offset,
                    "context_retraction_group": (
                        "bcp-retract-pair" if name.startswith("retract-") else None
                    ),
                }
            response = requests.post(
                base + "/v1/chat/completions", json=body, timeout=300
            )
            (directory / f"{name}-{mode}.json").write_text(response.text)
            assert response.status_code == 200, response.text
            return response.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            p = executor.submit(send, "prefill", p_base, 1, 0)
            d = executor.submit(send, "decode", d_base, len(tokens) - 1, 1)
            p.result()
            response = d.result()
        assert (
            response["sglext"]["input_ids"]
            == reference["runs"]["none"]["records"][0]["input"]["ids"]
        ), name
        if not fixed:
            output = response["sglext"]["output_ids"][0]
            choice = response["choices"][0]
            assert output and choice["finish_reason"] in (
                "length",
                "stop",
                "tool_calls",
            ), response
            item = {
                "comparison_kind": "native_generation_observation",
                "exact_token_match": output == tokens,
                "matching_tokens": sum(a == b for a, b in zip(output, tokens)),
                "generated_tokens": len(output),
                "reference_tokens": len(tokens),
                "same_input_tokens": response["sglext"]["input_ids"]
                == reference["runs"]["none"]["records"][0]["input"]["ids"],
                "message": choice["message"],
                "reference_message": reference["runs"][feature]["responses"][0][
                    "choices"
                ][0]["message"],
                "finish_reason": choice["finish_reason"],
            }
            comparisons[name] = item
            (directory / "comparison.json").write_text(
                json.dumps(comparisons, indent=2)
            )
            print("BCP_PD_ACTUAL", name, json.dumps(item), flush=True)
            return
        logits = torch.cat(
            [
                torch.load(directory / f"{name}-{mode}.pt", weights_only=True)
                for mode in ("prefill", "decode")
            ]
        )
        reference_key = (
            "drop_repos-retry"
            if name == "drop_repos-retry" and "drop_repos-retry" in reference_logits
            else feature
        )
        expected = reference_logits[reference_key]
        assert logits.shape == expected.shape and torch.isfinite(logits).all(), name
        assert response["sglext"]["output_ids"] == [tokens], response
        delta = (logits - expected).abs()
        item = {
            "reference_key": reference_key,
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "p99_abs": torch.quantile(delta.flatten(), 0.99).item(),
            "per_token_max": delta.max(-1).values.tolist(),
            "context_usage": response["choices"][0]
            .get("meta_info", {})
            .get("context_usage"),
            "num_retractions": response["choices"][0]
            .get("meta_info", {})
            .get("num_retractions", 0),
        }
        with lock:
            comparisons[name] = item
            (directory / "comparison.json").write_text(
                json.dumps(comparisons, indent=2)
            )
        print("BCP_PD_COMPARE", name, json.dumps(item), flush=True)

    if os.environ.get("CONTEXT_PD_RETRACT") == "1":
        # Reuse the existing no-feature calibration; this launch only tests the
        # new recovery path. Prime installs the diagnostic transfer barrier on D;
        # the pair then enters native decode together despite transport jitter.
        call("drop_repos", "prime")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            runs = [
                executor.submit(call, "drop_repos", f"retract-{i}") for i in range(2)
            ]
            for run in runs:
                run.result()
        baseline = json.loads(Path(os.environ["CONTEXT_PD_CALIBRATION"]).read_text())[
            "none"
        ]
        assert sum(comparisons[f"retract-{i}"]["num_retractions"] for i in range(2)) > 0
        for i in range(2):
            for metric, floor in (
                ("max_abs", 0.125),
                ("mean_abs", 0.02),
                ("p99_abs", 0.0625),
            ):
                assert comparisons[f"retract-{i}"][metric] <= max(
                    floor, 2 * baseline[metric]
                )
        return

    for feature in ("none", "drop", "drop_repos"):
        assert requests.post(p_base + "/flush_cache", timeout=5).status_code == 200
        call(feature, feature + "-actual", fixed=False)
        assert requests.post(p_base + "/flush_cache", timeout=5).status_code == 200
        call(feature, feature)
    call("drop_repos", "drop_repos-hot")
    assert requests.post(p_base + "/flush_cache", timeout=5).status_code == 200
    call("none", "none-retry-source")
    call("drop_repos", "drop_repos-retry")
    for name in ("drop", "drop_repos", "drop_repos-hot", "drop_repos-retry"):
        for metric, floor in (
            ("max_abs", 0.125),
            ("mean_abs", 0.02),
            ("p99_abs", 0.0625),
        ):
            assert comparisons[name][metric] <= max(
                floor, 2 * comparisons["none"][metric]
            ), (name, metric, comparisons)
