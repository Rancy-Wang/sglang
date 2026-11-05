"""DAS-Wave public validation without importing GPU/model code."""


def wave_widths(capacity, rounds, initial):
    if any(type(x) is not int or x <= 0 for x in (capacity, rounds, initial)):
        raise ValueError("DAS-Wave widths and rounds must be positive integers")
    if initial > capacity:
        raise ValueError("DAS-Wave initial width must not exceed capacity")
    return tuple(min(capacity, initial + j) for j in range(rounds))


def handle_server_args(server_args):
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
    from sglang.srt.speculative.loopspec.config import handle_server_args as validate_common

    # The shared recurrent storage/backend restrictions apply equally to Wave.
    # This only validates/resolves storage; it does not select a LoopSpec worker.
    validate_common(server_args)
    cfg = resolving_view(server_args)
    widths = wave_widths(cfg.speculative_das_wave_max_width,
                        cfg.speculative_das_wave_rounds,
                        cfg.speculative_das_wave_initial_width)
    if cfg.speculative_das_wave_updates not in (1, 2, 3, 4):
        raise ValueError("DAS-Wave updates must be in 1..4 for Ouro R4")
    declare_resolution(server_args, "das_wave",
                       speculative_draft_depth=None,
                       speculative_num_draft_tokens=2 * widths[-1] + 2)


def validate_request(params, obj):
    from sglang.srt.speculative.loopspec.config import validate_request as common

    common(params, obj)
    if params.temperature != 0 and params.top_k != 1:
        raise ValueError("DAS-Wave currently supports greedy decoding only")
    if (params.frequency_penalty or params.presence_penalty
            or params.repetition_penalty != 1 or params.logit_bias
            or params.min_new_tokens):
        raise ValueError("DAS-Wave greedy does not yet support logit adjustments")
    if getattr(obj, "return_logprob", False):
        raise ValueError("DAS-Wave logprob output is not yet supported")
