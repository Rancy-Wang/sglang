"""Ouro configuration for native, fixed-depth inference."""

from transformers import PretrainedConfig


class OuroConfig(PretrainedConfig):
    model_type = "ouro"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        total_ut_steps=4,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.total_ut_steps = total_ut_steps
        if not isinstance(total_ut_steps, int) or total_ut_steps < 1:
            raise ValueError("Ouro total_ut_steps must be a positive integer")
