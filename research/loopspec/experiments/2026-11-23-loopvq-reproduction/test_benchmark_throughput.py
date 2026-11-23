"""Small protocol tests; these do not claim server/GPU performance validation."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("benchmark_throughput", Path(__file__).with_name("benchmark_throughput.py"))
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)

# Non-sensitive /server_info graph snapshot from the real BF16 service.
ACTUAL_GRAPH = {
    "decode": {"backend": "full", "max_bs": 16, "bs": [1, 2, 4, 8, 16],
               "tc_compiler": "eager", "max_context_size": None,
               "full_prefill_max_req": None, "full_prefill_prefix_chunk_tokens": None,
               "max_seq_len": None},
    "prefill": {"backend": "disabled", "max_bs": -1, "bs": [],
                "tc_compiler": "eager", "max_context_size": None,
                "full_prefill_max_req": None, "full_prefill_prefix_chunk_tokens": None,
                "max_seq_len": None},
}


class ThroughputProtocolTests(unittest.TestCase):
    def manifest(self, graph=None):
        actual = {"tp_size": 1, "pp_size": 1, "dp_size": 1, "dtype": "bfloat16",
                  "attention_backend": "triton", "speculative_algorithm": None,
                  "disable_radix_cache": True, "chunked_prefill_size": -1,
                  "decode_log_interval": 1, "max_running_requests": 16,
                  "cuda_graph_config": copy.deepcopy(graph or ACTUAL_GRAPH),
                  "cuda_graph_bs_decode": [1, 2, 4, 8, 16],
                  "cuda_graph_backend_decode": "full", "disable_cuda_graph": False,
                  "disable_decode_cuda_graph": False, "disable_prefill_cuda_graph": False}
        manifest = {key: key for key in BENCH.IDENTITY_KEYS}
        manifest.update(host=BENCH.platform.node(), method="bf16", protocol="common-shape",
                        prefix_tokens=64, recent_tokens=256, server_pid=1,
                        exclusive_server=True, config={**copy.deepcopy(actual), "total_ut_steps": 4})
        return manifest, actual

    def test_real_server_graph_schema_and_optional_convenience_flags(self):
        manifest, actual = self.manifest()
        BENCH.validate_manifest(manifest, actual, "bf16", "common-shape", 16, 64, 256)
        # Resolved canonical buckets remain authoritative even without input convenience flags.
        actual["cuda_graph_bs_decode"] = None
        actual["cuda_graph_backend_decode"] = None
        BENCH.validate_manifest(manifest, actual, "bf16", "common-shape", 16, 64, 256)

    def test_eager_uses_phase_backend_not_legacy_global_flag(self):
        graph = copy.deepcopy(ACTUAL_GRAPH)
        graph["decode"].update(backend="disabled", bs=[])
        manifest, actual = self.manifest(graph)
        self.assertFalse(actual["disable_cuda_graph"])
        BENCH.validate_manifest(manifest, actual, "bf16", "common-shape", 1, 64, 256)
        text = "\n".join("Decode batch, #running-req: 1, cuda graph: False" for _ in range(3))
        evidence = BENCH.decode_evidence(text, 1, 4, actual)
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["graph_padding_at_target_bs"], 0)

    def test_prefill_graph_and_missing_canonical_schema_are_rejected(self):
        graph = copy.deepcopy(ACTUAL_GRAPH)
        graph["prefill"]["backend"] = "breakable"
        manifest, actual = self.manifest(graph)
        with self.assertRaisesRegex(ValueError, "prefill"):
            BENCH.validate_manifest(manifest, actual, "bf16", "common-shape", 1, 64, 256)
        manifest, actual = self.manifest()
        actual["cuda_graph_config"] = None
        manifest["config"]["cuda_graph_config"] = None
        with self.assertRaisesRegex(ValueError, "canonical"):
            BENCH.validate_manifest(manifest, actual, "bf16", "common-shape", 1, 64, 256)

    def test_stream_counts_and_client_tpot_with_coalesced_chunks(self):
        stats = BENCH.StreamStats(2, 4, 8)
        for index in range(2):
            stats.feed({"index": index, "meta_info": {"completion_tokens": 2, "prompt_tokens": 8}}, 1.0 + index)
            stats.feed({"index": index, "meta_info": {"completion_tokens": 4, "prompt_tokens": 8,
                                                      "finish_reason": {"type": "length"}}}, 3.0 + index)
        result = stats.finalize(5.0)
        self.assertEqual(result["output_tokens"], 8)
        self.assertEqual(result["e2e_output_tokens_s"], 1.6)
        self.assertEqual(result["requests"][0]["client_tpot_seconds"], 1.0)
        self.assertIsNone(result["server_decode_seconds"])

    def test_clamped_or_early_stop_is_not_success(self):
        for count, reason in ((3, "length"), (4, "stop")):
            stats = BENCH.StreamStats(1, 4, 8)
            stats.feed({"meta_info": {"completion_tokens": count, "prompt_tokens": 8,
                                      "finish_reason": {"type": reason}}}, 1.0)
            with self.assertRaises(ValueError):
                stats.finalize(2.0)

    def test_decode_logs_require_real_sustained_batch(self):
        good = "\n".join("Decode batch, #running-req: 16, cuda graph: True" for _ in range(31))
        config = {"cuda_graph_config": copy.deepcopy(ACTUAL_GRAPH)}
        capture = BENCH.parse_capture_evidence("Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 16], elapsed=1s")
        evidence = BENCH.decode_evidence(good, 16, 32, config, capture)
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["graph_padding_at_target_bs"], 0)
        low_batch = good.replace("req: 16", "req=8")
        self.assertFalse(BENCH.decode_evidence(low_batch, 16, 32, config)["verified"])
        self.assertFalse(BENCH.decode_evidence(good.splitlines()[0], 16, 32, config)["verified"])

    def test_actual_capture_log_controls_padding_and_replay_must_be_seen(self):
        config = {"cuda_graph_config": copy.deepcopy(ACTUAL_GRAPH)}
        good = "\n".join("Decode batch, #running-req: 16, cuda graph: True" for _ in range(31))
        capture = BENCH.parse_capture_evidence("Capture target decode CUDA graph end. bs=[8, 32], elapsed=1s")
        evidence = BENCH.decode_evidence(good, 16, 32, config, capture)
        self.assertEqual(evidence["graph_padding_at_target_bs"], "unknown")
        self.assertEqual(evidence["actual_capture_evidence"]["capture_bs"], [8, 32])
        self.assertEqual(BENCH.decode_evidence(good, 16, 32, config)["graph_padding_at_target_bs"], "unknown")
        eager = good.replace("graph: True", "graph: False")
        self.assertIn("full_decode_graph_not_verified", BENCH.decode_evidence(eager, 16, 32, config)["reasons"])

    def pair(self):
        base = {"protocol": "common-shape", "bs": 1, "input_len": 2048, "output_len": 512,
                "input_ids_sha256": "fixed", "input_fixture_sha256": "master", "warmup_count": 2,
                "repeat_count": 3, "fixed_length_sampling": {"temperature": 0.0, "ignore_eos": True},
                "primary_metric": "e2e_output_tokens_s", "method": "bf16", "status": "ok",
                "e2e_output_tokens_s_median": 10.0,
                "server_manifest": {key: key for key in BENCH.IDENTITY_KEYS},
                "server_config_readback": {key: None for key in BENCH.PAIR_CONFIG_KEYS}}
        quant = copy.deepcopy(base)
        quant.update(method="loopvq", e2e_output_tokens_s_median=15.0)
        return base, quant

    def test_same_hardware_pair_and_oom_is_na(self):
        base, quant = self.pair()
        self.assertEqual(BENCH.paired_row(base, quant)["speedup"], 1.5)
        base["status"] = "oom_or_capacity_limited"
        row = BENCH.paired_row(base, quant)
        self.assertIsNone(row["speedup"])
        self.assertIsNone(row["bf16_output_tokens_s"])
        self.assertEqual(row["speedup_status"], "N/A")

    def test_different_hardware_or_workload_cannot_pair(self):
        base, quant = self.pair()
        quant["server_manifest"]["gpu_uuid"] = "another-gpu"
        with self.assertRaisesRegex(ValueError, "gpu_uuid"):
            BENCH.paired_row(base, quant)
        base, quant = self.pair()
        quant["input_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "input_ids"):
            BENCH.paired_row(base, quant)


if __name__ == "__main__":
    unittest.main()
