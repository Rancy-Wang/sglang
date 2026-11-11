"""Replay full KV registries with recorded budgets and generated history.

Format-error responses are billed but discarded by the source agent; their
recorded user feedback stays in canonical history. Serving code is unchanged.
"""
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from summary_replay import clean, digest, wire_history
from serving_cohort import context_stop


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path, count, seed, case_ids=None):
    root = Path(path).parent
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['schema'] != 'kv-registry-500-v1':
        raise ValueError('Expected full KV registry export')
    for name in ('tasks_unique.jsonl', 'trajectories.jsonl', 'records.jsonl', 'replay_tasks.jsonl'):
        if sha(root/name) != manifest['files'][name]:
            raise ValueError('Registry checksum mismatch: ' + name)
    indices = [json.loads(x) for x in Path(path).read_text().splitlines()]
    random.Random(seed).shuffle(indices)
    if case_ids:
        indices = [x for x in indices if x['case_id'] in case_ids]
    if len(indices) < count or len({x['case_id'] for x in indices}) != len(indices):
        raise ValueError('Insufficient or duplicate registry tasks')
    cases = []
    for index in indices[:count]:
        key = index['case_id']
        names = [f'replay/{key}.json', f'trajectories/{key}.json', f'last_requests/{key}.json']
        for name in names:
            if sha(root/name) != manifest['files'][name]:
                raise ValueError('Registry component checksum mismatch: ' + name)
        replay, trajectory, request = [json.loads((root/name).read_text()) for name in names]
        if replay['case_id'] != key or trajectory['case_id'] != key:
            raise ValueError('Registry identity mismatch')
        ops = replay['operations']
        previous = -1
        for op in ops:
            budget, pos = op['observed_output_tokens'], op['source_response_message_index']
            if op['kind'] != 'agent_call' or type(budget) is not int or budget <= 0:
                raise ValueError('Unknown KV output budget or unsupported operation')
            if type(pos) is not int or not previous < pos < len(trajectory['trajectory']):
                raise ValueError('Non-monotonic response mapping')
            message = trajectory['trajectory'][pos]
            if message['role'] not in ('assistant', 'user') or message['role'] != op['source_response_role']:
                raise ValueError('Unsupported source response role')
            response_usage = message['extra']['response']['usage']
            if any(response_usage.get(k) != op['source_usage'].get(k) for k in ('prompt_tokens','completion_tokens')):
                raise ValueError('Source response usage mismatch')
            previous = pos
        last = int(replay['source_index']['last_logical_call_index'])
        matching = [op for op in ops if op['logical_call_index'] == last]
        if len(matching) != 1 or [clean(m) for m in trajectory['trajectory'][:matching[0]['source_response_message_index']]] != [clean(m) for m in request['messages']]:
            raise ValueError('Final source request reconstruction mismatch')
        cases.append(dict(case_id=key, trial=index['trial'], trajectory=trajectory['trajectory'],
                          operations=ops, tools=request.get('tools') or []))
    return cases, manifest


class History:
    def __init__(self, case):
        self.original = case['trajectory']
        self.replacements = {}
        self.names = {c['id']:c['function']['name'] for m in self.original for c in m.get('tool_calls', [])}

    def prepare(self, op):
        end = op['source_response_message_index']
        messages, adaptations = wire_history([clean(self.replacements.get(i, m)) for i,m in enumerate(self.original[:end])])
        for m in messages:
            if m['role'] not in ('system','developer','user','assistant','tool'):
                raise ValueError('Unsupported registry history role')
            if m['role'] == 'tool' and not m.get('name'):
                m['name'] = self.names[m['tool_call_id']]
        return messages, adaptations

    def commit(self, op, assistant):
        index = op['source_response_message_index']
        if self.original[index]['role'] == 'assistant':
            self.replacements[index] = clean(assistant)
            return True
        return False


async def execute_case(case, instance, *, args, renderer, rendering, transport, emit, rows, user_turns, method):
    history = History(case)
    rolling = method.RollingState(keep=12, threshold=96*1024)
    loop = asyncio.get_running_loop()
    for turn, op in enumerate(case['operations']):
        prepare_start = time.perf_counter()
        messages, adaptations = history.prepare(op)
        full, owners = await loop.run_in_executor(rendering, renderer.render, messages, case['tools'])
        state = rolling.extend(messages, owners, full)
        budget = op['observed_output_tokens']
        reason = context_stop(full, budget, state, args.model_context_limit)
        identity = dict(case_id=case['case_id'], instance=instance['instance'], filler=instance['filler'],
                        turn=turn, logical_turn=op['logical_call_index'], purpose='agent')
        if reason:
            emit(dict(kind='context_limit', **identity, reason=reason, full_tokens=full, **state))
            return 'registry_context_limit'
        if adaptations:
            emit(dict(kind='history_arguments_wrapped', **identity, adaptations=adaptations))
        payload = dict(model=args.model, messages=messages, tools=case['tools'], max_tokens=budget,
                       ignore_eos=True, temperature=0, stream=True,
                       stream_options={'include_usage':True, 'continuous_usage_stats':True},
                       chat_template_kwargs=json.loads(args.template_kwargs),
                       drop_message=state['drop_message'], reposition=state['reposition'])
        emit(dict(kind='turn_start', **identity, time=time.perf_counter(), full_tokens=full,
                  active_tokens=state['active_tokens'], position_tokens=state['position_tokens'], max_new_tokens=budget))
        row = await transport.request(args.url, payload, identity)
        if row['success'] and (row['prompt_len'] != full or row['output_len'] != budget):
            row.update(success=False, status='registry_budget_or_template_mismatch')
        row.update(**identity, trial=case['trial'], requested_max_tokens=budget, expected_prompt_tokens=full,
                   source_usage=op['source_usage'], source_response_role=op['source_response_role'],
                   history_sha256=digest(messages), drop_state=state)
        rows.append(row); emit(dict(kind='turn_end', **row))
        if not row['success']:
            return row['status']
        kept = history.commit(op, row['assistant'])
        emit(dict(kind='registry_response_commit', **identity, generated_response_retained=kept,
                  source_feedback_retained=not kept))
        parent = dict(identity, start_time=row['start_time'], prepare_start=prepare_start,
                      prepare_time_s=row['start_time']-prepare_start, raw_done_time=row['raw_done_time'],
                      latency=row['latency'], ttft=row['ttft'], tpot_s=row['tpot_s'], output_len=row['output_len'],
                      cleanup_time_s=row['cleanup_time_s'], success=True)
        user_turns.append(parent); emit(dict(kind='logical_turn_end', **parent))
    return 'all_turns_completed'
