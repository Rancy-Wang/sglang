"""Native compatibility config for released Huginn/Raven recurrent checkpoints."""

import math

from transformers import PretrainedConfig


class RavenConfig(PretrainedConfig):
    model_type = "huginn_raven"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        n_embd=5280,
        n_heads=55,
        n_layers=8,
        vocab_size=65536,
        intermediate_size=17920,
        block_size=4096,
        mean_recurrence=32,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=4,
        n_layers_in_coda=2,
        norm_eps=1e-6,
        bias=False,
        tie_embeddings=True,
        num_key_value_heads=None,
        head_dim=None,
        rope_base=50000,
        rope_theta=None,
        rope_scaling=None,
        rope_parameters=None,
        max_position_embeddings=None,
        init_values=None,
        test_time_noise=0,
        **kwargs,
    ):
        self.n_embd, self.n_heads, self.n_layers = n_embd, n_heads, n_layers
        self.head_dim = head_dim or n_embd // n_heads
        self.num_key_value_heads = num_key_value_heads or n_heads
        self.vocab_size, self.intermediate_size = vocab_size, intermediate_size
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.n_layers_in_coda = n_layers_in_coda
        self.mean_recurrence = mean_recurrence
        self.total_ut_steps = mean_recurrence
        self.norm_eps = self.rms_norm_eps = norm_eps
        self.bias = self.attention_bias = bias
        self.hidden_act = "silu"
        self.tie_embeddings = tie_embeddings
        self.max_position_embeddings = max_position_embeddings or block_size
        rope = dict(rope_parameters or rope_scaling or {})
        rope.setdefault("rope_theta", rope_theta or rope_base)
        rope.setdefault("rope_type", "default")
        self.rope_theta = rope["rope_theta"]
        self.rope_parameters = rope
        self.rope_scaling = rope
        self.test_time_noise = 0 if test_time_noise is None else test_time_noise
        expected = (
            n_layers_in_prelude
            + mean_recurrence * n_layers_in_recurrent_block
            + n_layers_in_coda
        )
        std = math.sqrt(2 / (5 * n_embd))
        self.init_values = init_values or {
            "std": std,
            "embed_scale": math.sqrt(n_embd),
            "out_proj": std / math.sqrt(2 * expected),
        }
        kwargs.pop("tie_word_embeddings", None)
        kwargs.pop("total_ut_steps", None)
        kwargs.pop("architectures", None)
        super().__init__(
            tie_word_embeddings=tie_embeddings,
            architectures=["RavenForCausalLM"],
            **kwargs,
        )
        if type(mean_recurrence) is not int or mean_recurrence < 1:
            raise ValueError("Raven mean_recurrence must be a positive integer")
        if (
            any(
                type(n) is not int or n < 0
                for n in (n_layers_in_prelude, n_layers_in_coda)
            )
            or n_layers_in_recurrent_block < 1
        ):
            raise ValueError("Invalid Raven block counts")
        if (
            isinstance(self.test_time_noise, bool)
            or not isinstance(self.test_time_noise, (int, float))
            or self.test_time_noise != 0
        ):
            raise ValueError("Raven serving requires test_time_noise=0")

    @property
    def logical_attention_layers(self):
        # Post prefixes are depth dependent just like recurrent-core prefixes.
        return self.n_layers_in_prelude + self.mean_recurrence * (
            self.n_layers_in_recurrent_block + self.n_layers_in_coda
        )
