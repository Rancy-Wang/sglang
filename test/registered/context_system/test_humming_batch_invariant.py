"""Configuration regression for Humming's opt-in deterministic reductions.

Real model repeatability is checked before every staged MiniMax oracle policy.
These checks exercise initialization and cached MoE tuning without model weights.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_dense_configuration_is_fixed_at_initialization(monkeypatch, enabled):
    from sglang.srt.layers.quantization import humming as module

    module._lazy_import_humming()
    key = "SGLANG_HUMMING_USE_BATCH_INVARIANT"
    monkeypatch.delenv(key, raising=False)
    if enabled is not None:
        monkeypatch.setenv(key, enabled)
    humming = Mock()
    humming.forward_layer.side_effect = lambda **kw: kw["inputs"]
    monkeypatch.setattr(module, "HummingMethod", humming)
    schema = module.HummingWeightSchema.__new__(module.HummingWeightSchema)
    method = object.__new__(module.HummingLinearMethod)
    method.weight_schema = schema
    method.input_schema = None
    method.force_weight_schema = None
    layer = SimpleNamespace(is_fallback=False, output_partition_sizes_sum=128,
                            input_size_per_partition=128, with_bias=False,
                            param_dtype=torch.bfloat16)
    method.process_weights_after_loading(layer)
    monkeypatch.setenv(key, "0" if enabled == "1" else "1")
    x = torch.randn(2, 3, 128)
    assert torch.equal(method.apply(layer, x), x)
    config = json.loads(humming.forward_layer.call_args.kwargs["compute_config"])
    assert config["use_batch_invariant"] is (enabled == "1")
    assert config["gemm_type"] == "dense"
    humming.transform_humming_layer.assert_called_once_with(layer)


@pytest.mark.parametrize("enabled", ["0", "1"])
@pytest.mark.parametrize("gemm_type", ["indexed", "grouped_contiguous", "grouped_masked"])
def test_moe_tuning_matches_compute_and_is_cached(monkeypatch, enabled, gemm_type):
    from sglang.srt.layers.moe.moe_runner import humming as module

    key = "SGLANG_HUMMING_USE_BATCH_INVARIANT"
    monkeypatch.setenv(key, enabled)
    humming = Mock()
    humming.get_default_tuning_configs.side_effect = lambda **kw: {
        "use_stream_k": not kw["use_batch_invariant"], "sublayer": kw["sublayer_name"]
    }
    monkeypatch.setattr(module, "HummingMethod", humming)
    runner = object.__new__(module.HummingRunnerCore)
    runner.humming_gemm_configs = {}
    runner.layer = object()
    kind = module.HummingGemmType(gemm_type)
    result = runner.get_humming_gemm_configs(kind)
    monkeypatch.setenv(key, "0" if enabled == "1" else "1")
    assert runner.get_humming_gemm_configs(kind) is result
    assert humming.get_default_tuning_configs.call_count == 2
    for call in humming.get_default_tuning_configs.call_args_list:
        assert call.kwargs["use_batch_invariant"] is (enabled == "1")
        assert call.kwargs["layer"] is runner.layer
    assert json.loads(result["compute_config_str"])["use_batch_invariant"] is (enabled == "1")
    for sublayer in ("w13", "w2"):
        tuning = json.loads(result[sublayer + "_tuning_config_str"])
        assert tuning["use_stream_k"] is (enabled != "1")
        assert tuning["sublayer"] == sublayer
