"""Public argument validation, with no model/GPU imports."""


def validate_depths(depths, total):
    if len(depths) not in (1, 2) or any(type(d) is not int for d in depths):
        raise ValueError("LoopSpec requires one or two integer draft depths")
    first = depths[0]
    if not 0 < first < total or total % first:
        raise ValueError(
            "LoopSpec first draft depth must divide total_ut_steps and be smaller"
        )
    if len(depths) == 2 and (not first < depths[1] < total or depths[1] % first):
        raise ValueError(
            "LoopSpec second draft depth must be a multiple of first, between first and target"
        )
    return tuple(depths)


def handle_server_args(server_args):
    from sglang.srt.arg_groups.overrides import (
        declare_resolution,
        model_config_of,
        resolving_view,
    )

    cfg = resolving_view(server_args)
    model = model_config_of(server_args).hf_config
    if model.architectures != ["OuroForCausalLM"]:
        raise ValueError("LoopSpec CDB currently requires native Ouro; Raven CDB is unvalidated")
    depths = validate_depths(cfg.speculative_draft_depth or [1], model.total_ut_steps)
    if (cfg.tp_size, cfg.pp_size, cfg.dp_size, cfg.attn_cp_size) != (1, 1, 1, 1):
        raise ValueError("LoopSpec currently requires TP=PP=DP=CP=1")
    if not cfg.device.startswith("cuda") or cfg.attention_backend != "triton":
        raise ValueError(
            "LoopSpec currently requires CUDA and --attention-backend triton"
        )
    if any(
        backend not in (None, "triton")
        for backend in (cfg.prefill_attention_backend, cfg.decode_attention_backend)
    ):
        raise ValueError("LoopSpec requires Triton for both prefill and decode")
    if cfg.page_size != 1 or not cfg.disable_radix_cache:
        raise ValueError("LoopSpec requires --page-size 1 --disable-radix-cache")
    if cfg.enable_priority_preemption:
        raise ValueError("LoopSpec does not yet support priority preemption")
    if cfg.speculative_draft_model_path is not None:
        raise ValueError("LoopSpec shares target weights; do not specify a draft model")
    if (
        cfg.enable_lora
        or cfg.lora_paths
        or cfg.enable_two_batch_overlap
        or cfg.enable_mixed_chunk
    ):
        raise ValueError(
            "LoopSpec does not support LoRA, two-batch overlap or mixed chunks"
        )
    if cfg.disaggregation_mode != "null" or cfg.enable_deterministic_inference:
        raise ValueError(
            "LoopSpec does not support disaggregation or deterministic-inference mode"
        )
    if cfg.speculative_adaptive:
        raise ValueError(
            "LoopSpec depths are fixed; adaptive speculative decoding is unsupported"
        )
    if (
        any(
            getattr(cfg, name, False)
            for name in (
                "enable_hierarchical_cache",
                "enable_hisparse",
                "enable_unified_memory",
                "enable_unified_cache_external_linker",
                "enable_kv_cache_sharding",
                "enable_torch_compile",
                "enable_memory_saver",
            )
        )
        or cfg.cpu_offload_gb
    ):
        raise ValueError(
            "LoopSpec timeline KV does not support host/unified caches, sharding, memory saver or torch.compile"
        )
    if cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16", "float16"):
        raise ValueError("LoopSpec timeline KV requires unquantized KV storage")
    # Allocation reserve is an upper bound on the live binary proposal tree.
    width = (
        sum(
            end - start
            for _, start, end in pipeline_layout(model.total_ut_steps, depths)
        )
        + 2
    )
    declare_resolution(
        server_args,
        "loopspec",
        speculative_draft_depth=list(depths),
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=width,
    )


def validate_request(params, obj):
    unsupported = (
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "watermark",
        "beam_width",
    )
    if any(getattr(params, name, None) is not None for name in unsupported):
        raise ValueError(
            "LoopSpec does not yet support grammar, watermark or beam search"
        )
    if params.custom_params or getattr(obj, "custom_logit_processor", None):
        raise ValueError(
            "LoopSpec does not yet support custom logit processors or thinking budgets"
        )
    if getattr(obj, "return_sampling_mask", False) or getattr(
        obj, "return_hidden_states", False
    ):
        raise ValueError("LoopSpec does not yet return sampling masks or hidden states")
    if getattr(obj, "input_embeds", None) is not None or getattr(
        obj, "session_params", None
    ):
        raise ValueError(
            "LoopSpec currently requires token-only requests without sessions"
        )


def pipeline_layout(total, depths):
    stride = depths[0]
    steps = total // stride
    second = depths[1] // stride if len(depths) == 2 else None
    births = [1]
    for age in range(1, steps):
        births.append(
            births[age - 1]
            + (births[age - second] if second is not None and age >= second else 0)
        )
    layout, cursor = [], 0
    for stage, capacity in enumerate(reversed(births)):
        layout.append((stage * stride, cursor, cursor + capacity))
        cursor += capacity
    if cursor > 1024:
        raise ValueError(
            "LoopSpec schedule exceeds 1024 live lanes; increase draft depth"
        )
    return layout
