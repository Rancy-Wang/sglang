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
    not (os.environ.get("CONTEXT_PD_BCP_ORACLE") or os.environ.get("CONTEXT_PD_SMOKE")),
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
        os.environ.get("CONTEXT_PD_BCP_ORACLE"),
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
            cmd += json.loads(os.environ.get("CONTEXT_SERVER_EXTRA_ARGS", "[]"))
            if mode == "decode" and os.environ.get("CONTEXT_PD_RADIX") == "1":
                cmd += ["--disaggregation-decode-enable-radix-cache"]
            if template:
                cmd += ["--chat-template", template]
            if "gpt-oss" in os.environ["CONTEXT_SERVER_MODEL"].lower():
                cmd += [
                    "--tool-call-parser",
                    "gpt-oss",
                    "--reasoning-parser",
                    "gpt-oss",
                ]
            if os.environ.get("CONTEXT_SHARED_SWA") == "1":
                cmd += ["--disable-hybrid-swa-memory"]
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
            deadline = time.monotonic() + int(os.environ.get("CONTEXT_START_TIMEOUT", "240"))
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


@pytest.mark.skipif(not os.environ.get("CONTEXT_PD_BCP_ORACLE"), reason="numeric reference required")
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

    def flush():
        for base in (p_base, d_base) if os.environ.get("CONTEXT_PD_RADIX") == "1" else (p_base,):
            assert requests.post(base + "/flush_cache", timeout=5).status_code == 200

    def call(feature, name, fixed=True, *, warm_source=False):
        nonlocal room
        with lock:
            room += 1
            request_room = room
        tokens = reference["runs"][feature]["records"][0]["tokens"]
        if warm_source:
            tokens = tokens[:1]
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
        if warm_source:
            payload["reposition"] = reference["fixture"]["reposition"][:1]
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
        if warm_source:
            assert response["sglext"]["output_ids"] == [tokens]
            return  # Source has one R; only the final two-R request has an oracle.
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
            name
            if name in ("drop_repos-retry", "drop_repos-consecutive")
            and name in reference_logits
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

    consecutive_only = os.environ.get("CONTEXT_BCP_CONSECUTIVE_ONLY") == "1"
    for feature in (() if consecutive_only else ("none", "drop", "drop_repos")):
        flush()
        call(feature, feature + "-actual", fixed=False)
        flush()
        call(feature, feature)
    if not consecutive_only:
        call("drop_repos", "drop_repos-hot")
        flush()
        call("none", "none-retry-source")
        call("drop_repos", "drop_repos-retry")
    flush()
    call("drop_repos", "consecutive-source", warm_source=True)
    call("drop_repos", "drop_repos-consecutive")
    usage = comparisons["drop_repos-consecutive"]["context_usage"]
    assert usage["cached_tokens"] + usage["repos_tokens"] > 0
    assert usage["actual_prefill_tokens"] < len(reference["runs"]["none"]["records"][0]["input"]["ids"])
    baseline = (
        json.loads(Path(os.environ["CONTEXT_BCP_CALIBRATION"]).read_text())["none"]
        if os.environ.get("CONTEXT_BCP_CALIBRATION") else comparisons["none"]
    )
    names = ("drop_repos-consecutive",) if consecutive_only else (
        "drop", "drop_repos", "drop_repos-hot", "drop_repos-retry", "drop_repos-consecutive"
    )
    for name in names:
        for metric, floor in (
            ("max_abs", 0.125),
            ("mean_abs", 0.02),
            ("p99_abs", 0.0625),
        ):
            assert comparisons[name][metric] <= max(
                floor, 2 * baseline[metric]
            ), (name, metric, comparisons)
    if os.environ.get("CONTEXT_PD_RADIX") == "1" and not consecutive_only:
        reused = {
            name: json.loads((directory / f"{name}-decode.pt.reuse.json").read_text())
            for name in ("drop", "drop_repos", "drop_repos-hot", "drop_repos-retry", "drop_repos-consecutive")
        }
        assert reused["drop"]["reused"] == reused["drop_repos"]["reused"] == 0
        assert reused["drop_repos-hot"]["reused"] > 0
        assert reused["drop_repos-hot"]["missing"] < reused["drop_repos-hot"]["active"]
        assert reused["drop_repos-retry"]["copied"] > 0
        assert reused["drop_repos-consecutive"]["copied"] > 0


def test_pd_usage_roundtrip(pd_servers):
    """Assert P accounting survives Mooncake, independently of D cache hits."""
    p_base, d_base, bootstrap = pd_servers
    messages = [
        {"role": "user", "content": "Remember " + "red blue green " * 32},
        {"role": "assistant", "content": "I have read that." + " blue ocean" * 80},
        {"role": "user", "content": "Say hello."},
    ]
    records = []
    for stream in (False, True):
        for operation in ({"drop_message": None}, {"drop_message": {"1": [0]}, "reposition": [1]}):
            for repeat in range(2):
                body = dict(model=os.environ["CONTEXT_SERVER_MODEL"], messages=messages,
                            temperature=0, max_tokens=4, ignore_eos=True,
                            bootstrap_host="127.0.0.1", bootstrap_port=bootstrap,
                            bootstrap_room=time.time_ns() % (1 << 52), **operation)
                def send(base):
                    payload = dict(body, stream=stream if base == d_base else False)
                    response = requests.post(base + "/v1/chat/completions", json=payload, timeout=180)
                    assert response.status_code == 200, response.text
                    if payload["stream"]:
                        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                                  if line.startswith("data: ") and line != "data: [DONE]"]
                        assert all("context_usage" not in (c.get("sglext") or {}) for c in chunks)
                        reports = [c["usage"]["prompt_tokens_details"] for c in chunks
                                   if c.get("usage") is not None]
                        assert len(reports) == 1, chunks
                        return reports[0]
                    result = response.json()
                    assert "context_usage" not in (result.get("sglext") or {})
                    return result["usage"]["prompt_tokens_details"]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    p = executor.submit(send, p_base)
                    d = executor.submit(send, d_base)
                    p_usage, d_usage = p.result(), d.result()
                for key in ("cached_tokens", "repos_tokens", "drop_skipped_tokens"):
                    assert isinstance(d_usage[key], int) and d_usage[key] >= 0, d_usage
                    assert p_usage[key] == d_usage[key], (p_usage, d_usage)
                records.append(dict(stream=stream, repeat=repeat, operation=operation, prefill=p_usage, decode=d_usage))
    (Path(os.environ["CONTEXT_TRACE_DIR"]) / "usage-roundtrip.json").write_text(json.dumps(records, indent=2))
    for base in (p_base, d_base):
        assert requests.get(base + "/health", timeout=5).status_code == 200
