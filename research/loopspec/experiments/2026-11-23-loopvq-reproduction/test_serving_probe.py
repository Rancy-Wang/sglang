"""Protocol-only mock service checks; these do not validate CUDA/model behavior."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("serving_probe", Path(__file__).with_name("serving_probe.py"))
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class Tokenizer:
    chat_template = "native enable_thinking template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
        return "<user>" + messages[0]["content"] + "</user><assistant><think>"

    def encode(self, rendered, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [1] + [2] * (rendered.count(" x") + rendered.count(" a") + rendered.count(" 1") + 20) + [3]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, kind="application/json"):
        data = (json.dumps(value) if kind == "application/json" else value).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            self.reply("\n".join(f'sglang:{key}{{model="mock"}} {int(self.server.busy)}' for key in PROBE.GAUGES), "text/plain")
        elif self.path == "/server_info":
            self.reply({**self.server.config, "api_key": "must-not-persist"})
        else:
            self.reply({"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append((self.path, body))
        if self.path == "/abort_request":
            self.server.aborted.set()
            self.reply({"ok": True})
            return
        if self.path.startswith("/flush_cache"):
            self.reply({"ok": True})
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"meta_info":{"completion_tokens":16}}\n\n')
            self.wfile.flush()
            if not self.server.aborted.wait(5):
                return
            self.wfile.write(b'data: {"meta_info":{"completion_tokens":17,"finish_reason":{"type":"abort"}}}\n\ndata: [DONE]\n\n')
            self.wfile.flush()
            return
        responses = []
        for ids, sampling in zip(body["input_ids"], body["sampling_params"]):
            count = sampling["max_new_tokens"] - self.server.clamp
            responses.append({"meta_info": {"prompt_tokens": len(ids), "completion_tokens": count,
                "finish_reason": {"type": "length"}, "output_token_logprobs": [[-1.0, 5]] * count,
                "output_top_logprobs": [[[-1.0 - i, i] for i in range(20)]] * count}})
        with self.server.log.open("a") as out:
            for _ in range(max(x["max_new_tokens"] for x in body["sampling_params"]) - 1):
                out.write(f'Decode batch, #running-req: {len(responses)}, cuda graph: True\n')
        self.reply(responses)


class ServingProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log = self.root / "server.log"
        self.log.write_text("Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 16], elapsed=1s\n")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.log, self.server.clamp, self.server.busy = self.log, 0, False
        self.server.calls, self.server.aborted = [], threading.Event()
        self.server.config = {"cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 4, 8, 16]}, "prefill": {"backend": "disabled"}}}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.args = SimpleNamespace(url=f"http://127.0.0.1:{self.server.server_port}", out_dir=self.root,
                                    server_log=self.log, reference_dir=None, hf_reference=None, timeout=10, idle_timeout=0)
        self.probe = PROBE.Probe(self.args, Tokenizer(), {"config": self.server.config})

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_exact_lengths_keep_native_template_and_reject_impossible_length(self):
        for length in (63, 64, 333, 384, 4096):
            prompt = PROBE.exact_prompt(Tokenizer(), length)
            self.assertEqual(len(prompt["input_ids"]), length)
            self.assertTrue(prompt["rendered_prompt"].endswith("</user><assistant><think>"))
        with self.assertRaisesRegex(ValueError, "exactly"):
            PROBE.exact_prompt(Tokenizer(), 1)

    def test_metrics_require_all_finite_gauges(self):
        for raw in ("", "\n".join(f"sglang:{key} NaN" for key in PROBE.GAUGES)):
            with self.assertRaises(ValueError):
                PROBE.metrics_values(raw)

    def test_batch_raw_outputs_and_actual_padding_are_preserved(self):
        record = self.probe.generate("pad3", [384] * 3, [4] * 3)
        self.assertEqual(record["decode"]["expected_padding_from_actual_capture"], 1)
        rows = json.loads((self.root / "pad3-rows.json").read_text())
        self.assertEqual(len(rows), 3)
        self.assertTrue((self.root / "pad3-http.json").exists())
        self.assertGreater(record["log"]["end"], record["log"]["start"])

    def test_early_clamp_fails_but_keeps_wire_evidence(self):
        self.server.clamp = 1
        with self.assertRaisesRegex(ValueError, "clamped"):
            self.probe.generate("bs1", [384], [4])
        self.assertTrue((self.root / "bs1-http.json").exists())
        self.assertTrue((self.root / "bs1-server.log").exists())

    def test_abort_waits_for_generated_tokens_and_only_targets_owned_rid(self):
        record = self.probe.abort()
        aborts = [body for path, body in self.server.calls if path == "/abort_request"]
        self.assertEqual(aborts, [{"rid": record["rid"], "abort_all": False}])
        self.assertTrue(record["rid"].startswith("loopvq-probe-"))
        self.assertEqual(record["finish"]["type"], "abort")
        self.assertIn('"completion_tokens":16', (self.root / "abort-sse.txt").read_text())

    def test_nonidle_state_prevents_flush_from_masking_leak(self):
        self.server.busy = True
        with self.assertRaises(TimeoutError):
            self.probe.run_case("flush")
        self.assertFalse(any(path.startswith("/flush_cache") for path, _ in self.server.calls))

    def test_comparison_rejects_shapes_and_preserves_first_divergence(self):
        row = {"input_ids": [1], "sampling": {"max_new_tokens": 3}, "output_ids": [2, 3, 4],
               "top20": [[[-1.0, 2]]] * 3, "finish": "length"}
        changed = copy.deepcopy(row)
        changed["output_ids"][1] = 9
        comparison = PROBE.compare_rows([row], [changed])[0]
        self.assertEqual(comparison["first_divergence"]["generated_index_zero_based"], 1)
        self.assertEqual(len(comparison["same_prefix_top20"]), 2)
        changed["input_ids"] = [8]
        with self.assertRaisesRegex(ValueError, "shape"):
            PROBE.compare_rows([row], [changed])

    def test_config_evidence_uses_allowlist(self):
        self.probe.http("/server_info", "server-config")
        self.assertNotIn("must-not-persist", (self.root / "server-config-http.json").read_text())

    def test_vq_identity_rejects_missing_hash_and_wrong_directory(self):
        (self.root / "config.json").write_text('{"total_ut_steps":4}')
        config = {**self.server.config, "model_path": str(self.root), "dtype": "bfloat16", "tp_size": 1, "pp_size": 1,
                  "dp_size": 1, "disable_radix_cache": True, "chunked_prefill_size": -1, "speculative_algorithm": None,
                  "page_size": 8, "decode_log_interval": 1, "attention_backend": "triton", "loopvq_codebook_dir": "/bundle",
                  "loopvq_prefix_tokens": 64, "loopvq_recent_tokens": 256}
        manifest = {"config": {"total_ut_steps": 4, "loopvq_codebook_dir": "/bundle"}, "exclusive_server": True,
                    "model_sha256": "m", "sglang_sha": "s", "method": "loopvq", "bundle_sha256": "q"}
        identity = PROBE.validate_identity(manifest, config, self.root, Tokenizer.chat_template)
        self.assertEqual(identity["bundle_sha256"], "q")
        config["ouro_decode_profile"] = "optimized"
        with self.assertRaisesRegex(ValueError, "profile"):
            PROBE.validate_identity(manifest, config, self.root, Tokenizer.chat_template)
        manifest["config"]["ouro_decode_profile"] = "optimized"
        optimized = PROBE.validate_identity(manifest, config, self.root, Tokenizer.chat_template)
        self.assertNotEqual(identity, optimized)
        self.assertEqual(optimized["effective_decode"], "loopvq_official_cuda")
        manifest["config"]["loopvq_codebook_dir"] = "/other"
        with self.assertRaisesRegex(ValueError, "directory"):
            PROBE.validate_identity(manifest, config, self.root, Tokenizer.chat_template)
        manifest.pop("bundle_sha256")
        with self.assertRaisesRegex(ValueError, "hash"):
            PROBE.validate_identity(manifest, config, self.root, Tokenizer.chat_template)


if __name__ == "__main__":
    unittest.main()
