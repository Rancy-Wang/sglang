"""Pure wire projections and budget audits for the approved system ablation."""
import copy


def text_drop_projection(history, state):
    """RollingState currently drops complete old tool messages, by canonical ID.

    Keep canonical history immutable. This is a text-level baseline: surviving
    messages get freshly rendered contiguous positions, never Context metadata.
    Fail on unsupported partial operations instead of silently widening them.
    """
    removed = set()
    for trigger, targets in state.get('drop_message', {}).items():
        trigger = int(trigger)
        if not 0 <= trigger < len(history):
            raise ValueError('Drop trigger outside canonical history')
        if not isinstance(targets, list):
            raise ValueError('Text projection only supports rolling whole-message drops')
        for target in targets:
            if type(target) is not int or not 0 <= target < trigger:
                raise ValueError('Invalid canonical drop target')
            if history[target]['role'] not in ('tool', 'function'):
                raise ValueError('Rolling baseline must only delete old tool results')
            removed.add(target)
    ids = [i for i in range(len(history)) if i not in removed]
    return [copy.deepcopy(history[i]) for i in ids], ids, sorted(removed)


def require_complete_budget(full, output, state, limit):
    """A length guard is a failed experiment, never successful early completion."""
    if limit is None:
        raise ValueError('Strict ablation requires explicit client context limit')
    active = state.get('active_tokens', full)
    position = state.get('position_tokens', full)
    if max(active, position) + output > limit:
        raise ValueError(f'Incomplete source trajectory: full={full} active={active} '
                         f'position={position} output={output} limit={limit}')


def context_budget_audit(cases, tokenizer):
    """Planning envelope, not a proof about arbitrary model-generated strings.

    Count all incoming text plus template allowance, and twice output budgets
    for decode/re-encode expansion. Strict runtime guard remains authoritative.
    Audit all80 including possible fillers; do not silently truncate a run.
    """
    import json
    rows = []
    for case in cases:
        incoming = output = 0
        for turn in case['turns']:
            incoming += len(tokenizer.encode(json.dumps(turn['new_messages'], ensure_ascii=False), add_special_tokens=False))
            output += turn['max_new_tokens']
        allowance = 512 * len(case['turns']) + 8192
        rows.append(dict(case_id=case['case_id'], incoming_tokens=incoming,
                         output_budget=output, template_allowance=allowance,
                         planned_tokens=incoming + 2 * output + allowance))
    maximum = max(r['planned_tokens'] for r in rows)
    limit = max(262144, ((maximum + 65535) // 65536) * 65536)
    return dict(context_length=limit, cases=rows, maximum_planned_tokens=maximum,
                scope='all80 including fillers', output_expansion_factor=2,
                guarantee='Planning envelope only; strict runtime rejects overflow, never clips or marks complete')
