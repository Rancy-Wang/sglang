"""Raven config, weight-coverage, tokenizer and recurrence-schedule contracts."""

import unittest
from types import SimpleNamespace

import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel

from sglang.srt.configs.raven import RavenConfig
from sglang.srt.models.raven import RavenForCausalLM
from sglang.srt.speculative.loopspec.config import pipeline_layout
from sglang.srt.utils.hf_transformers.recurrent import patch_recurrent_tokenizer


class WeightFixture:
    map_weight = RavenForCausalLM.map_weight

    def __init__(self, post_norm):
        self.post_norm = post_norm
        self.params = {
            "core.0.self_attn.qkv_proj.weight": torch.nn.Parameter(torch.zeros(6, 2)),
            "core.0."
            + (
                "post_feedforward_layernorm"
                if post_norm
                else "post_attention_layernorm"
            )
            + ".weight": torch.nn.Parameter(torch.zeros(2)),
        }

    def named_parameters(self, **kwargs):
        return self.params.items()


class TestRaven(unittest.TestCase):
    def test_config_roundtrip_and_logical_post_depths(self):
        config = RavenConfig(
            n_embd=256,
            n_heads=2,
            n_layers=3,
            n_layers_in_prelude=1,
            n_layers_in_recurrent_block=1,
            n_layers_in_coda=1,
            mean_recurrence=4,
        )
        restored = RavenConfig.from_dict(config.to_dict())
        self.assertEqual(restored.logical_attention_layers, 9)
        self.assertEqual(restored.total_ut_steps, 4)
        self.assertEqual(restored.architectures, ["RavenForCausalLM"])
        self.assertEqual(restored.head_dim, 128)
        for noise in [True, float("nan"), 0.01]:
            with self.assertRaises(ValueError):
                RavenConfig(test_time_noise=noise)

    def test_strict_packed_mapping_both_norm_layouts(self):
        weights = [
            ("transformer.core_block.0.attn.Wqkv.weight", torch.ones(6, 2)),
            ("transformer.core_block.0.norm_2.weight", torch.ones(2)),
        ]
        for post in [False, True]:
            fixture = WeightFixture(post)
            self.assertEqual(len(RavenForCausalLM.load_weights(fixture, weights)), 2)
            for candidate in [
                weights[:1],
                weights + [weights[0]],
                weights + [("transformer.bad.weight", torch.ones(2))],
            ]:
                with self.assertRaises(ValueError):
                    RavenForCausalLM.load_weights(fixture, candidate)

    def test_r32_schedule_is_not_exponential_for_single_proposal(self):
        self.assertEqual(pipeline_layout(32, (1,))[-1][2], 32)
        self.assertEqual(pipeline_layout(4, (1, 2))[-1][2], 7)
        self.assertEqual(pipeline_layout(32, (4, 8))[-1][2], 54)
        with self.assertRaises(ValueError):
            pipeline_layout(32, (1, 2))

    def test_tokenizer_ids_and_thinking_delimiters(self):
        cfg = SimpleNamespace(model_type="huginn_raven", vocab_size=128)
        mirror = SimpleNamespace()
        model = SimpleNamespace(
            hf_config=cfg, hf_text_config=mirror, hf_eos_token_id={99}
        )
        tok = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=None)
        patch_recurrent_tokenizer(model, tok)
        self.assertEqual(model.hf_eos_token_id, {2})
        self.assertEqual(mirror.eos_token_id, 2)
        backend = Tokenizer(WordLevel({"hello": 0}, unk_token="hello"))
        backend.add_special_tokens([AddedToken("<think>", special=True)])
        token = backend.token_to_id("<think>")
        obj = SimpleNamespace(_tokenizer=backend)
        patch_recurrent_tokenizer(SimpleNamespace(model_type="ouro"), obj)
        self.assertEqual(obj._tokenizer.token_to_id("<think>"), token)
        self.assertEqual(
            obj._tokenizer.decode([token], skip_special_tokens=True), "<think>"
        )


if __name__ == "__main__":
    unittest.main()
