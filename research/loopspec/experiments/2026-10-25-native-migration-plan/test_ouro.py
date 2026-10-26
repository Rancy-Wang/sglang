"""Weight coverage and recurrent KV allocation regression tests."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
from sglang.srt.models.ouro import OuroForCausalLM


class WeightFixture:
    config = SimpleNamespace(tie_word_embeddings=False)

    def __init__(self):
        self.parameters = {
            "model.layers.0.self_attn.qkv_proj.weight": torch.nn.Parameter(
                torch.zeros(3, 2)
            ),
            "model.layers.0.mlp.gate_up_proj.weight": torch.nn.Parameter(
                torch.zeros(2, 2)
            ),
            "model.norm.weight": torch.nn.Parameter(torch.zeros(2)),
        }
        self.seen_shards = []
        for name, parameter in self.parameters.items():
            if "_proj" in name:
                parameter.weight_loader = lambda param, value, shard: (
                    self.seen_shards.append(shard)
                )

    def named_parameters(self):
        return self.parameters.items()


def weights():
    return (
        [
            (f"model.layers.0.self_attn.{projection}_proj.weight", torch.ones(1, 2))
            for projection in ("q", "k", "v")
        ]
        + [
            (f"model.layers.0.mlp.{projection}_proj.weight", torch.ones(1, 2))
            for projection in ("gate", "up")
        ]
        + [("model.norm.weight", torch.ones(2))]
    )


class TestOuro(unittest.TestCase):
    def test_every_packed_shard_is_required(self):
        for omitted in range(len(weights())):
            candidate = weights()
            candidate.pop(omitted)
            with (
                self.subTest(omitted=omitted),
                self.assertRaisesRegex(ValueError, "Missing Ouro"),
            ):
                OuroForCausalLM.load_weights(WeightFixture(), candidate)

    def test_complete_checkpoint(self):
        fixture = WeightFixture()
        loaded = OuroForCausalLM.load_weights(fixture, weights())
        self.assertEqual(loaded, set(fixture.parameters))
        self.assertEqual(fixture.seen_shards, ["q", "k", "v", 0, 1])
        torch.testing.assert_close(
            fixture.parameters["model.norm.weight"], torch.ones(2)
        )

    def test_duplicate_and_unknown_weights(self):
        for extra in [weights()[0], ("model.typo.weight", torch.ones(2))]:
            with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
                OuroForCausalLM.load_weights(WeightFixture(), weights() + [extra])

    def test_kv_depth_is_expanded_exactly_once(self):
        info = resolve_layer_indices(
            model=SimpleNamespace(start_layer=0, end_layer=48),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="ouro",
                    total_ut_steps=4,
                    architectures=["OuroForCausalLM"],
                ),
                num_hidden_layers=48,
                num_attention_layers=192,
                num_nextn_predict_layers=None,
                is_hybrid_swa=False,
                swa_attention_layer_ids=None,
            ),
            is_draft_worker=False,
        )
        self.assertEqual(info.num_effective_layers, 192)


if __name__ == "__main__":
    unittest.main()
