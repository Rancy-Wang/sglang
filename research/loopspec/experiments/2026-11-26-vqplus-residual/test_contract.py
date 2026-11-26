"""CPU compatibility and exact storage accounting; no GPU or model weights."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '2026-11-23-loopvq-reproduction'))
from test_loopvq_wiring import LoopVQCompatibilityTest
from sglang.srt.arg_groups import kv_cache_hook
from sglang.srt.model_executor import pool_configurator


class ResidualCompatibility(LoopVQCompatibilityTest):
    def test_declared_residual_schema(self):
        self.config.loopvq_residual_bits = 4
        self.config.ouro_decode_profile = 'optimized'
        self.config.triton_attention_num_kv_splits = 48
        self.config.context_length = 32768
        path = Path(self.directory.name) / 'build_info.json'
        contract = dict(bits=4, group_size=32, scale_dtype='float16', codes='symmetric_-7_7')
        path.write_text(json.dumps({'residual': contract}))
        kv_cache_hook.handle_loopvq_compatibility(self.config)
        contract['group_size'] = 64
        path.write_text(json.dumps({'residual': contract}))
        with self.assertRaisesRegex(ValueError, 'residual schema'):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_global_requires_residual(self):
        self.config.loopvq_residual_codebook = 'global'
        with self.assertRaisesRegex(ValueError, 'requires VQ+'):
            kv_cache_hook.handle_loopvq_compatibility(self.config)

    def test_residual_contract(self):
        self.config.loopvq_residual_bits = 4
        self.config.ouro_decode_profile = 'optimized'
        self.config.triton_attention_num_kv_splits = 48
        self.config.context_length = 32768
        kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.context_length = 32769
        with self.assertRaisesRegex(ValueError, '32768'):
            kv_cache_hook.handle_loopvq_compatibility(self.config)
        self.config.context_length = 32768
        self.config.ouro_decode_profile = 'hf_reference'
        with self.assertRaisesRegex(ValueError, 'optimized'):
            kv_cache_hook.handle_loopvq_compatibility(self.config)


class Storage(unittest.TestCase):
    def test_tp2_packed_capacity(self):
        kvc = NS(layer_info=NS(num_effective_layers=192), model_config=NS(
            head_dim=128, get_num_kv_heads=lambda size: 16 // size))
        for bits, cell in ((0, 24576), (4, 245760)):
            with patch.object(pool_configurator, 'get_schedule', return_value=NS(max_running_requests=16)), patch.object(
                pool_configurator, 'get_parallel', return_value=NS(attn_tp_size=2)), patch.object(
                pool_configurator, 'get_model', return_value=NS(loopvq_prefix_tokens=8, loopvq_recent_tokens=64, loopvq_residual_bits=bits)):
                sizer = pool_configurator.LoopVQPoolConfigurator(kvc)
            self.assertEqual(sizer._cell_size, cell)
            expected_fixed = (8 + 2 * 17 * 8 + 17 * 72) * 192 * 8 * 128 * 4 + 402653184 + 8 * cell
            self.assertEqual(sizer._fixed_bytes, expected_fixed)
            self.assertEqual(sizer.calculate_pool_sizes(expected_fixed + cell * 1024, 8).max_total_num_tokens, 1024)


if __name__ == '__main__':
    unittest.main()
