"""Public argument validation, with no model/GPU imports."""


def validate_depths(depths, total):
    if len(depths) not in (1, 2) or any(type(d) is not int for d in depths):
        raise ValueError("LoopSpec requires one or two integer draft depths")
    first = depths[0]
    if not 0 < first < total or total % first:
        raise ValueError("LoopSpec first draft depth must divide total_ut_steps and be smaller")
    if len(depths) == 2 and (not first < depths[1] < total or depths[1] % first):
        raise ValueError("LoopSpec second draft depth must be a multiple of first, between first and target")
    return tuple(depths)


def handle_server_args(server_args):
    from sglang.srt.arg_groups.overrides import declare_resolution, model_config_of, resolving_view

    cfg = resolving_view(server_args)
    model = model_config_of(server_args).hf_config
    if model.architectures != ["OuroForCausalLM"]:
        raise ValueError("LoopSpec currently supports native OuroForCausalLM")
    depths = validate_depths(cfg.speculative_draft_depth or [1], model.total_ut_steps)
    if (cfg.tp_size, cfg.pp_size, cfg.dp_size, cfg.attn_cp_size) != (1, 1, 1, 1):
        raise ValueError("LoopSpec currently requires TP=PP=DP=CP=1")
    if not cfg.device.startswith("cuda") or cfg.attention_backend != "triton":
        raise ValueError("LoopSpec currently requires CUDA and --attention-backend triton")
    if cfg.page_size != 1 or cfg.max_running_requests != 1:
        raise ValueError("LoopSpec currently requires --page-size 1 --max-running-requests 1")
    if not cfg.disable_overlap_schedule or not cfg.disable_radix_cache or cfg.chunked_prefill_size != -1:
        raise ValueError("LoopSpec requires --disable-overlap-schedule --disable-radix-cache --chunked-prefill-size -1")
    if not cfg.disable_cuda_graph:
        raise ValueError("LoopSpec recurrent graph support is not yet available; use --disable-cuda-graph")
    if cfg.speculative_draft_model_path is not None:
        raise ValueError("LoopSpec shares target weights; do not specify a draft model")
    if cfg.enable_lora or cfg.lora_paths or cfg.enable_two_batch_overlap or cfg.enable_mixed_chunk:
        raise ValueError("LoopSpec does not support LoRA, two-batch overlap or mixed chunks")
    if cfg.disaggregation_mode != "null" or cfg.enable_deterministic_inference:
        raise ValueError("LoopSpec does not support disaggregation or deterministic-inference mode")
    if cfg.speculative_adaptive:
        raise ValueError("LoopSpec depths are fixed; adaptive speculative decoding is unsupported")
    # Allocation reserve is an upper bound on the live binary proposal tree.
    width = 2 ** (model.total_ut_steps // depths[0])
    declare_resolution(
        server_args, "loopspec",
        speculative_draft_depth=list(depths),
        speculative_num_steps=1, speculative_eagle_topk=1,
        speculative_num_draft_tokens=width,
    )


def validate_request(params, obj):
    unsupported = ("json_schema", "regex", "ebnf", "structural_tag", "watermark", "beam_width")
    if any(getattr(params, name, None) is not None for name in unsupported):
        raise ValueError("LoopSpec does not yet support grammar, watermark or beam search")
    if params.custom_params or getattr(obj, "custom_logit_processor", None):
        raise ValueError("LoopSpec does not yet support custom logit processors or thinking budgets")
    if getattr(obj, "return_sampling_mask", False) or getattr(obj, "return_hidden_states", False):
        raise ValueError("LoopSpec does not yet return sampling masks or hidden states")
    if getattr(obj, "input_embeds", None) is not None or getattr(obj, "session_params", None):
        raise ValueError("LoopSpec currently requires token-only requests without sessions")
