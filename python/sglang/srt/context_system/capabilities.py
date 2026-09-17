"""Admission limits for the Context consumers implemented by this engine.

Run only for explicit Context requests. Native requests do not pay validation
or acquire restrictions from these staged integration limits.
"""

from __future__ import annotations


def validate_context_request(args, model_config, request):
    if args.page_size != 1:
        raise ValueError("Context Drop/Reposition requires page_size=1")
    architectures = set(model_config.hf_config.architectures or ())
    if not architectures or not architectures <= {
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "GptOssForCausalLM",
    }:
        raise ValueError("Context currently supports Qwen3/AgenticQwen and GPT-OSS")
    if model_config.is_multimodal or str(model_config.dtype) not in (
        "torch.float16",
        "torch.bfloat16",
    ):
        raise ValueError("Context requires text-only FP16/BF16 model execution")
    prefill = args.prefill_attention_backend or args.attention_backend
    decode = args.decode_attention_backend or args.attention_backend
    if prefill != "triton" or decode != "triton":
        raise ValueError(
            "Context attention integration currently requires native Triton"
        )
    if args.kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("Context requires unquantized FP16/BF16 KV")
    if args.disaggregation_mode != "null":
        raise ValueError("Context PD transfer integration is not yet enabled")
    if args.pp_size != 1 or args.attn_cp_size != 1 or args.dcp_size != 1:
        raise ValueError("Context currently supports TP without PP/CP/DCP")
    if args.speculative_algorithm or args.dllm_algorithm:
        raise ValueError("Context requires autoregressive non-speculative scheduling")
    if args.enable_deterministic_inference:
        raise ValueError("Context batch-invariant attention is not yet supported")
    for name in (
        "enable_hierarchical_cache",
        "enable_unified_cache_external_linker",
        "enable_hisparse",
        "enable_lmcache",
        "enable_flexkv",
        "enable_session_radix_cache",
        "enable_beam_search",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Context is not yet supported with {name}")
    if args.radix_cache_backend is not None:
        raise ValueError("Context requires the native unified Radix cache")
    if request.input_ids is None or request.input_embeds is not None:
        raise ValueError("A Context program requires its original input_ids")
    if request.contains_mm_input() or request.session_id or request.session_params:
        raise ValueError("Context requires text input without session KV reuse")
    if request.lora_path is not None:
        raise ValueError("Context LoRA cache compatibility is not yet supported")
    if (request.sampling_params or {}).get("beam_width", 1) > 1:
        raise ValueError("Context beam scheduling is not yet supported")
    # HTTP JSON cannot construct the internal tensor-buffer wire. Validate here
    # so a malformed /generate payload becomes a client error before IPC.
    from sglang.srt.context_system.planner import ContextProgram

    ContextProgram.from_wire(request.context_program, request.input_ids)
