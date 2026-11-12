"""MLGym source replay; no trajectory command is ever executed."""
import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

LIMIT = 131072
THRESHOLD = 98304
GEN = '<|start|>assistant'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic(message):
    return message.get('action') in ('exit_context', 'exit_cost') and message['content'].startswith('Exit due to ')


def tool_delay(step):
    if step is None:
        return None
    if str(step.get('action', '')).strip().startswith('exit_'):
        return None
    value = step.get('execution_time')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Missing, non-finite or negative source tool execution_time')
    return float(value)


def source_calls(data):
    """Map exact saved history to steps; retain orphan replies as audited terminal tails."""
    history, steps = data['history'], data['trajectory']
    if [m['role'] for m in history[:2]] != ['system', 'user']:
        raise ValueError('Expected initial system/task messages')
    calls, pending, step_index = [], [], 0
    for i, msg in enumerate(history):
        if msg['role'] != 'assistant':
            if msg['role'] not in ('system', 'user') or (i > 1 and msg['role'] != 'user'):
                raise ValueError('Unexpected MLGym role')
            pending.append(dict(role=msg['role'], content=msg['content']))
            continue
        step = steps[step_index] if step_index < len(steps) else None
        if step is not None:
            if msg['content'] != step['response']:
                raise ValueError('Source assistant/step response mismatch')
            if i + 1 < len(history) and history[i + 1]['role'] == 'user':
                if not history[i + 1]['content'].startswith(step['observation']):
                    raise ValueError('Source observation mismatch')
        step_index += 1
        if synthetic(msg):
            continue
        calls.append(dict(history_index=i, new_messages=pending, source_assistant=msg['content'],
                          source_step=step_index - 1 if step is not None else None,
                          tool_seconds=tool_delay(step), orphan_tool_step=step is None,
                          exit_action=str(step.get('action', '')).strip().startswith('exit_') if step else False))
        pending = []
    if step_index < len(steps):
        raise ValueError('Unmatched source steps')
    return calls


class ObservationRollingState:
    """Original rolling semantics, identifying MLGym observations by canonical IDs."""
    def __init__(self, keep=12, threshold=THRESHOLD):
        self.keep, self.threshold = keep, threshold
        self.processed = self.removed = self.compacted = 0
        self.observations, self.drops, self.repositions = [], {}, []
        self.prefix_hash, self.old_bounds = None, {}

    def extend(self, messages, owners, full_tokens, observation_ids):
        if self.prefix_hash is not None and digest(messages[:self.processed]) != self.prefix_hash:
            raise ValueError('Previously submitted messages changed')
        counts, ends = Counter(owners), {}
        for pos, owner in enumerate(owners):
            if owner >= 0:
                ends[owner] = pos + 1
        for owner, old in self.old_bounds.items():
            if old != (counts[owner], ends.get(owner)):
                raise ValueError('Historical template token boundary changed')
        for i in range(self.processed, len(messages)):
            if i not in ends:
                raise ValueError('Message has no token ownership')
            if i in observation_ids:
                if i < 2 or messages[i]['role'] != 'user':
                    raise ValueError('Invalid observation identity')
                self.observations.append(i)
                if len(self.observations) > self.keep:
                    old = self.observations[-self.keep - 1]
                    self.drops[str(i)] = [old]
                    self.removed += counts[old]
            if ends[i] - self.compacted >= self.threshold and self.removed > self.compacted:
                self.repositions.append(i)
                self.compacted = self.removed
        self.processed, self.prefix_hash = len(messages), digest(messages)
        self.old_bounds = {i: (counts[i], ends[i]) for i in range(len(messages))}
        return dict(drop_message=copy.deepcopy(self.drops), reposition=list(self.repositions),
                    active_tokens=full_tokens-self.removed, position_tokens=full_tokens-self.compacted)


def source_sizing(data, tokenizer):
    """Exact full-template additive audit; never uses source usage counters."""
    h = data['history']
    render = lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True,
                                                     preserve_thinking_history=True)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    base = render(h[:2])
    if not base.endswith(GEN):
        raise ValueError('Unexpected generation prefix')
    pieces = [base[:-len(GEN)]]
    for m in h[2:]:
        if m['role'] == 'user':
            pieces.append('<|start|>user<|message|>'+m['content']+'<|end|>')
        elif m['role'] == 'assistant':
            pieces.append('<|start|>assistant<|channel|>final<|message|>'+m['content']+'<|end|>')
        else:
            raise ValueError('Unexpected source role')
    if ''.join(pieces)+GEN != render(h):
        raise ValueError('Source template segmentation differs from whole rendering')
    ids = [encode(s) for s in pieces]
    gen = encode(GEN)
    if [v for part in ids for v in part]+gen != encode(render(h)):
        raise ValueError('Token boundaries are not additive')
    full, removed, compacted, observations, rows = len(ids[0]), 0, 0, [], {}
    for i, msg in enumerate(h[2:], 2):
        n = len(ids[i-1])
        if msg['role'] == 'assistant':
            if not synthetic(msg):
                budget = n - len(gen)
                if budget <= 0:
                    raise ValueError('Non-positive output budget')
                rows[i] = dict(full=full+len(gen), active=full+len(gen)-removed,
                               position=full+len(gen)-compacted, output=budget)
            full += n
        else:
            observations.append(n)
            full += n
            if len(observations) > 12:
                removed += observations[-13]
        if full-compacted >= THRESHOLD and removed > compacted:
            compacted = removed
    return rows


def build_manifest(source, tokenizer, seed=42, limit=LIMIT):
    source = Path(source)
    paths = sorted(source.glob('*/*.traj'), key=lambda p: p.relative_to(source).as_posix())
    if not paths:
        raise ValueError('No MLGym trajectories')
    random.Random(seed).shuffle(paths)
    entries = []
    for rank, path in enumerate(paths, 1):
        data = json.loads(path.read_text())
        calls, sizes = source_calls(data), source_sizing(data, tokenizer)
        over = [dict(history_index=i, **v) for i, v in sizes.items() if v['active'] > limit]
        for call in calls:
            call['budget'] = sizes[call['history_index']]['output']
        entries.append(dict(rank=rank, path=path.relative_to(source).as_posix(), sha256=file_sha(path),
                            excluded_active_over128k=bool(over), exclusion_turns=over,
                            source_exit_status=data.get('info', {}).get('exit_status'),
                            calls=[{k:v for k,v in c.items() if k not in ('new_messages','source_assistant')} for c in calls],
                            source_sizing=sizes, source_no_reply=not calls,
                            tool_seconds=sum(c['tool_seconds'] or 0 for c in calls)))
        if rank % 50 == 0:
            print(json.dumps(dict(tokenized=rank, total=len(paths))), flush=True)
    return dict(schema=1, seed=seed, limit=limit, keep_observations=12, reposition_threshold=THRESHOLD,
                order_algorithm='relative path lexsort then random.Random(seed).shuffle once, then filter',
                order_sha256=digest([e['path'] for e in entries]), entries=entries,
                total=len(entries), excluded=sum(e['excluded_active_over128k'] for e in entries),
                eligible=sum(not e['excluded_active_over128k'] and not e['source_no_reply'] for e in entries),
                source_no_reply=sum(not e['excluded_active_over128k'] and e['source_no_reply'] for e in entries),
                tool_seconds=sum(e['tool_seconds'] for e in entries if not e['excluded_active_over128k']))


def load_cases(source, manifest):
    cases = []
    for entry in manifest['entries']:
        path = Path(source)/entry['path']
        if file_sha(path) != entry['sha256']:
            raise ValueError('Source changed: '+entry['path'])
        if entry['excluded_active_over128k'] or entry['source_no_reply']:
            continue
        data = json.loads(path.read_text())
        calls = source_calls(data)
        if len(calls) != len(entry['calls']):
            raise ValueError('Manifest turn count mismatch')
        for call, meta in zip(calls, entry['calls']):
            for key in ('history_index', 'tool_seconds', 'orphan_tool_step'):
                if call[key] != meta[key]:
                    raise ValueError('Manifest step mismatch')
            call['budget'] = meta['budget']
        cases.append(dict(case_id=entry['path'], rank=entry['rank'], calls=calls,
                          source_exit_status=entry['source_exit_status']))
    return cases
