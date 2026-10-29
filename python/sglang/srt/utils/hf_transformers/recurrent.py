"""Checkpoint-specific tokenizer semantics shared by native serving processes."""

import json


def patch_recurrent_tokenizer(model_config, tokenizer):
    config = getattr(model_config, "hf_config", model_config)
    family = getattr(config, "model_type", None)
    if family not in {"ouro", "huginn_raven"}:
        return
    if tokenizer is None:
        if family == "huginn_raven":
            raise ValueError("Raven requires its checkpoint tokenizer")
        return
    if family == "ouro":
        from tokenizers import Tokenizer

        backend = getattr(tokenizer, "_tokenizer", None)
        if not isinstance(backend, Tokenizer):
            raise TypeError("Ouro requires a native fast HuggingFace tokenizer")
        state = json.loads(backend.to_str())
        changed = False
        for token in state.get("added_tokens", []):
            if token["content"] in {"<think>", "</think>"} and token["special"]:
                token["special"] = False
                changed = True
        if changed:
            tokenizer._tokenizer = Tokenizer.from_str(json.dumps(state))
        return
    mirrors = [config]
    for name in ("hf_text_config", "hf_generation_config"):
        if (mirror := getattr(model_config, name, None)) is not None:
            mirrors.append(mirror)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None and (
            type(value) is not int or not 0 <= value < config.vocab_size
        ):
            raise ValueError(f"Raven tokenizer {name} is outside vocabulary")
        for mirror in mirrors:
            setattr(mirror, name, value)
    if hasattr(model_config, "hf_eos_token_id"):
        model_config.hf_eos_token_id = (
            set() if tokenizer.eos_token_id is None else {tokenizer.eos_token_id}
        )
