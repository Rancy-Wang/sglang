"""CPU checks for standalone LoopVQ dispatch boundaries and memory accounting.

Run in the target SGLang environment, without loading a model or CUDA context:
    python -m unittest discover -s research/loopspec/experiments/2026-11-23-loopvq-reproduction -p test_loopvq_wiring.py
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.arg_groups import kv_cache_hook
from sglang.srt.model_executor import pool_configurator
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig


class LoopVQCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name in ("codebook.pt", "vq_v_codebook.pt"):
            (Path(self.directory.name) / name).touch()
        self.config = NS(
            loopvq_codebook_dir=self.directory.name,
            loopvq_prefix_tokens=64, loopvq_recent_tokens=256,
            device="cuda", tp_size=1, pp_size=1, dp_size=1,
            attn_dp_size=1, attn_cp_size=1, attn_dcp_size=1,
            speculative_algorithm=None, page_size=8, max_running_requests=16,
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
        ))
        for target, replacement in (
            ("resolved_view", lambda value: value),
            ("attention_backends_of", lambda value: ("triton", "triton")),
            ("model_config_of", lambda value: self.model),
        ):
            manager = patch.object(kv_cache_hook, target, replacement)
            manager.start()
            self.addCleanup(manager.stop)
        manager = patch.object(
            kv_cache_hook.envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING,
            "get", return_value=False,
        )
        manager.start()
        self.addCleanup(manager.stop)

    def test_supported_eager_and_decode_graph(self):
        kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.cuda_graph_config.decode.backend = "disabled"
        kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_disabled_does_not_constrain_other_models(self):
        kv_cache_hook.handle_loopvq_compatibility(NS(loopvq_codebook_dir=None))

    def test_rejects_incompatible_cache_and_execution_paths(self):
        for key, value in (
            ("tp_size", 2), ("attn_dp_size", 2), ("speculative_algorithm", "LOOPSPEC"),
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
        ):
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


if __name__ == "__main__":
    unittest.main()
