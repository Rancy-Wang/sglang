"""CPU checks for standalone LoopVQ dispatch boundaries and memory accounting.

Run in the target SGLang environment, without loading a model or CUDA context:
    python -m unittest discover -s research/loopspec/experiments/2026-11-23-loopvq-reproduction -p test_loopvq_wiring.py

To exercise the real resolution pipeline on A10 before starting the service:
    python research/loopspec/experiments/2026-11-23-loopvq-reproduction/test_loopvq_wiring.py \
        --model /absolute/Ouro-2.6B-Thinking --bundle /absolute/g16_lvgqe2e
This loads configuration and may probe GPU resources, but does not load model
weights or launch a server. Eager and full-decode settings use separate records.
"""

import argparse
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.arg_groups import kv_cache_hook
from sglang.srt.model_executor import pool_configurator
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.server_args import ServerArgs


class LoopVQCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name in ("codebook.pt", "vq_v_codebook.pt"):
            (Path(self.directory.name) / name).touch()
        # Construction deliberately uses the real flat ServerArgs schema.
        # No runtime Derived fields (such as attn_dcp_size) exist on this record.
        self.config = ServerArgs(
            model_path="dummy",
            loopvq_codebook_dir=self.directory.name,
            loopvq_prefix_tokens=64, loopvq_recent_tokens=256,
            device="cuda", tp_size=1, pp_size=1, dp_size=1,
            attn_dp_size=1, attn_cp_size=1, dcp_size=1,
            speculative_algorithm=None, page_size=8, max_running_requests=16,
            attention_backend="triton",
            kv_cache_dtype="auto", dtype="bfloat16", disable_radix_cache=True,
            chunked_prefill_size=-1, disaggregation_mode="null",
            enable_priority_scheduling=False, disable_priority_preemption=False,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="full"),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        self.model = NS(hf_config=NS(
            architectures=["OuroForCausalLM"], num_hidden_layers=48,
            total_ut_steps=4, num_attention_heads=16, num_key_value_heads=16,
            hidden_size=2048, rope_scaling=None,
            max_position_embeddings=40960,
        ))
        # model_config_of explicitly supports a cached fixture with no path key.
        # ResolvedView, attention selection and field access remain production code.
        self.config._model_config = self.model
        manager = patch.object(
            kv_cache_hook.envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING,
            "get", return_value=False,
        )
        manager.start()
        self.addCleanup(manager.stop)

    def test_supported_eager_and_decode_graph(self):
        self.assertFalse(hasattr(self.config, "attn_dcp_size"))
        self.assertFalse(hasattr(kv_cache_hook.resolved_view(self.config), "attn_dcp_size"))
        kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.cuda_graph_config.decode.backend = "disabled"
        kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_disabled_does_not_constrain_other_models(self):
        kv_cache_hook.handle_loopvq_compatibility(
            ServerArgs(model_path="dummy", loopvq_codebook_dir=None, device="cpu")
        )

    def test_optimized_profile_and_both_window_protocols(self):
        self.assertEqual(self.config.ouro_decode_profile, "hf_reference")
        self.config.ouro_decode_profile = "optimized"
        self.config.triton_attention_num_kv_splits = 48
        for prefix, recent in ((64, 256), (8, 64)):
            self.config.loopvq_prefix_tokens, self.config.loopvq_recent_tokens = prefix, recent
            kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.context_length = 40961
        with self.assertRaisesRegex(ValueError, "40960"):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_optimized_bf16_is_guarded_even_without_loopvq(self):
        self.config.loopvq_codebook_dir = None
        self.config.ouro_decode_profile = "optimized"
        self.config.triton_attention_num_kv_splits = 8
        kv_cache_hook.handle_loopvq_compatibility(self.config)
        for key, value in (("dtype", "float32"), ("speculative_algorithm", "LOOPSPEC"),
                           ("triton_attention_num_kv_splits", 48)):
            candidate = copy.deepcopy(self.config)
            setattr(candidate, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                kv_cache_hook.handle_loopvq_compatibility(candidate)
        self.model.hf_config.architectures = ["LlamaForCausalLM"]
        with self.assertRaisesRegex(ValueError, "Ouro model"):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_parallel_guard_reads_resolved_declarations(self):
        kv_cache_hook.declare_resolution(self.config, "loopvq_test", dcp_size=2)
        self.assertEqual(self.config.dcp_size, 1)
        with self.assertRaisesRegex(ValueError, "DCP"):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_rejects_incompatible_cache_and_execution_paths(self):
        for key, value in (
            ("tp_size", 4), ("attn_dp_size", 2), ("attn_cp_size", 2), ("dcp_size", 2),
            ("speculative_algorithm", "LOOPSPEC"),
            ("speculative_algorithm", "DFLASH"), ("page_size", 1),
            ("max_running_requests", None), ("kv_cache_dtype", "fp8_e4m3"),
            ("disable_radix_cache", False), ("chunked_prefill_size", 2048),
            ("enable_unified_memory", True), ("enable_hierarchical_cache", True),
            ("enable_torch_compile", True), ("loopvq_recent_tokens", 7),
            ("disaggregation_mode", "decode"), ("enable_priority_scheduling", True),
        ):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                setattr(config, key, value)
                with self.assertRaises(ValueError):
                    kv_cache_hook.handle_loopvq_compatibility(config)

    def test_rejects_prefill_graph_and_wrong_geometry(self):
        self.config.cuda_graph_config.prefill.backend = "full"
        with self.assertRaises(ValueError):
            kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.cuda_graph_config.prefill.backend = "disabled"
        self.model.hf_config.total_ut_steps = 3
        with self.assertRaises(ValueError):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_requires_both_codebook_files(self):
        (Path(self.directory.name) / "vq_v_codebook.pt").unlink()
        with self.assertRaises(ValueError):
            kv_cache_hook.handle_loopvq_compatibility(self.config)


class LoopVQPoolBudgetTest(unittest.TestCase):
    def test_fixed_windows_and_codebooks_are_not_token_capacity(self):
        model = NS(loopvq_prefix_tokens=64, loopvq_recent_tokens=256)
        kvc = NS(
            layer_info=NS(num_effective_layers=192),
            model_config=NS(head_dim=128, get_num_kv_heads=lambda size: 16),
        )
        with patch.object(pool_configurator, "get_schedule", return_value=NS(max_running_requests=16)), patch.object(
            pool_configurator, "get_model", return_value=model
        ), patch.object(pool_configurator, "get_parallel", return_value=NS(attn_tp_size=1)):
            sizer = pool_configurator.LoopVQPoolConfigurator(kvc)
        # 17 request slots include dummy row zero. 6672 HP rows at 1.5 MiB,
        # 768 MiB of codebooks and one reserved uint8 page are fixed costs.
        fixed_bytes = 6672 * 1572864 + 805306368 + 8 * 49152
        budget = fixed_bytes + 4096 * 49152
        self.assertEqual(sizer.calculate_pool_sizes(budget, 8).max_total_num_tokens, 4096)
        self.assertEqual(sizer.calculate_pool_sizes_from_max_tokens(1027, 8).max_total_num_tokens, 1024)
        with self.assertRaises(ValueError):
            sizer.calculate_pool_sizes(fixed_bytes, 8)
        with self.assertRaises(ValueError):
            sizer.calculate_pool_sizes(budget, 1)


def validate_real_resolution(model, bundle):
    """Resolve real model arguments; do not replace config/backend/runtime readers."""
    import sglang

    results = []
    for decode_backend in ("disabled", "full"):
        args = ServerArgs(
            model_path=str(Path(model).resolve()), trust_remote_code=True,
            device="cuda", dtype="bfloat16", kv_cache_dtype="auto",
            loopvq_codebook_dir=str(Path(bundle).resolve()),
            loopvq_prefix_tokens=64, loopvq_recent_tokens=256,
            attention_backend="triton", tp_size=1, pp_size=1, dp_size=1,
            attn_dp_size=1, attn_cp_size=1, dcp_size=1,
            speculative_algorithm=None, page_size=8, max_running_requests=16,
            context_length=40960, disable_radix_cache=True, chunked_prefill_size=-1,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=decode_backend, max_bs=16, bs=[1, 2, 4, 8, 16]),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        args.resolve_once()
        kv_cache_hook.handle_loopvq_compatibility(args)
        resolved = args.resolved_dict()
        results.append({
            "status": "resolved", "sglang_file": sglang.__file__,
            "model_path": resolved["model_path"],
            "loopvq_codebook_dir": resolved["loopvq_codebook_dir"],
            "dcp_size": resolved["dcp_size"], "page_size": resolved["page_size"],
            "cuda_graph_config": resolved["cuda_graph_config"],
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--bundle")
    options, remaining = parser.parse_known_args()
    if options.model or options.bundle:
        if not options.model or not options.bundle or remaining:
            parser.error("Real resolution requires --model and --bundle, with no unittest arguments")
        validate_real_resolution(options.model, options.bundle)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
