"""Replay recorded compaction events using generated agent AND summary history.

Source pi helper files are an explicitly hashed dataset dependency, not serving
code. Original prompt hashes must validate before any measured model call.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def clean(message):
    fields = ('role', 'content', 'reasoning_content', 'reasoning', 'tool_calls', 'tool_call_id', 'name')
    return {k: copy.deepcopy(v) for k, v in message.items() if k in fields}


def helpers(directory):
    directory = Path(directory).resolve()
    checks = json.loads((directory / 'sha256.json').read_text())
    package = '_summary_source_' + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    if package in sys.modules:
        return sys.modules[package + '.pi_compaction'], sys.modules[package + '.pi_summary_prompts']
    module = types.ModuleType(package)
    module.__path__ = [str(directory)]
    sys.modules[package] = module
    for name in ('pi_summary_prompts', 'pi_compaction'):
        path = directory / (name + '.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != checks[path.name]:
            raise ValueError('Summary source helper changed: ' + str(path))
        spec = importlib.util.spec_from_file_location(package + '.' + name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[package + '.pi_compaction'], sys.modules[package + '.pi_summary_prompts']


class Replay:
    def __init__(self, case, policy_dir):
        self.case = case
        self.pi, self.prompts = helpers(policy_dir)
        self.original = case['trajectory']['trajectory']
        self.replacements = {}
        self.first_kept = 0
        self.summary = None
        self.file_details = {}

    def canonical(self, end, generated=True):
        return [clean(self.replacements.get(i, m) if generated else m)
                for i, m in enumerate(self.original[:end])]

    def prepare(self, group, generated=True):
        agent = group['agent']
        end = agent['source_response_message_index']
        if end is None:
            end = next((i for i, m in enumerate(self.original) if m['role'] == 'exit'), len(self.original))
        canonical = self.canonical(end, generated)
        if any(m['role'] not in ('system', 'developer', 'user', 'assistant', 'tool') for m in canonical):
            raise ValueError('Unsupported canonical history role')
        event = agent.get('context_policy', {}).get('summary_event')
        calls = group['summaries']
        first_kept = self.first_kept
        indices = [i for i, m in enumerate(canonical) if i >= self.first_kept and m['role'] not in ('system', 'developer')]
        if calls:
            # Cuts belong to the recorded event schedule, never recalculate from
            # newly generated text lengths. Failed summaries lack a commit event.
            source = self.canonical(end, False)
            pi_source = [self.pi.to_pi_message(source[i]) for i in indices]
            cut, turn_start = self.pi.find_cut_point(pi_source, (event or {}).get('keep_recent_tokens', 20000))
            if event:
                wanted = event['first_kept_message_index']
                if indices[cut] != wanted:
                    raise ValueError('Source compaction cut does not match recorded event')
                if indices[:cut] != event['summarized_message_indices']:
                    raise ValueError('Source compaction message map changed')
            first_kept = indices[cut]
            pi_messages = [self.pi.to_pi_message(canonical[i]) for i in indices]
            partitions = {}
            if turn_start >= 0:
                if turn_start:
                    partitions['history'] = pi_messages[:turn_start]
                partitions['turn_prefix'] = pi_messages[turn_start:cut]
            else:
                partitions['history'] = pi_messages[:cut]
            requests = []
            for op in calls:
                call = op['summary']; kind = call['kind']
                if kind not in partitions:
                    raise ValueError('Missing summary partition: ' + kind)
                instruction = (self.prompts.TURN_PREFIX_SUMMARIZATION_PROMPT if kind == 'turn_prefix'
                               else self.prompts.UPDATE_SUMMARIZATION_PROMPT if self.summary
                               else self.prompts.SUMMARIZATION_PROMPT)
                prompt = '<conversation>\n' + self.pi.serialize_conversation(partitions[kind]) + '\n</conversation>\n\n'
                if kind == 'history' and self.summary:
                    prompt += '<previous-summary>\n' + self.summary + '\n</previous-summary>\n\n'
                prompt += instruction
                requests.append(dict(operation=op, prompt=prompt, messages=[
                    dict(role='system', content=self.prompts.SUMMARIZATION_SYSTEM_PROMPT),
                    dict(role='user', content=prompt)]))
            details = self.pi.file_operations(pi_messages[:cut], self.file_details)
        else:
            requests, details, turn_start = [], self.file_details, -1
        return dict(canonical=canonical, requests=requests, first_kept=first_kept,
                    split=turn_start >= 0, event=event, details=details)

    def commit_summary(self, prepared, texts):
        if not prepared['event']:
            return  # A recorded failed summary does not replace history.
        if prepared['split']:
            combined = texts.get('history', 'No prior history.') + '\n\n---\n\n**Turn Context (split turn):**\n\n' + texts['turn_prefix']
        else:
            combined = texts['history']
        self.summary = combined + self.pi.format_file_operations(prepared['details'])
        self.first_kept, self.file_details = prepared['first_kept'], prepared['details']

    def active(self, prepared):
        canonical = prepared['canonical']
        if self.summary is None:
            return canonical
        return ([m for m in canonical if m['role'] in ('system', 'developer')]
                + [self.pi.summary_message(self.summary)]
                + [m for i, m in enumerate(canonical) if i >= self.first_kept and m['role'] not in ('system', 'developer')])

    def commit_agent(self, op, assistant):
        index = op['source_response_message_index']
        if index is not None and self.original[index]['role'] == 'assistant':
            self.replacements[index] = clean(assistant)
        # Format-error generations are billed but source feedback remains a user
        # message. Their discarded response must not enter canonical history.


def load_cases(path, count, seed, case_ids, policy_dir):
    root = Path(path).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['schema'] != 'summary-registry-top80-v1':
        raise ValueError('Expected full summary registry archive')
    for name in ('tasks_unique.jsonl', 'replay_tasks.jsonl', 'trajectories.jsonl', 'records.jsonl'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('Dataset checksum mismatch: ' + name)
    indices = [json.loads(l) for l in Path(path).read_text().splitlines()]
    random.Random(seed).shuffle(indices)
    if case_ids:
        indices = [x for x in indices if x['case_id'] in case_ids]
    cases = []
    for index in indices[:count]:
        key = index['case_id']
        replay = json.loads((root/'replay'/f'{key}.json').read_text())
        trajectory = json.loads((root/replay['trajectory_file']).read_text())
        request = json.loads((root/replay['last_request_file']).read_text())
        groups, pending = [], []
        for op in replay['operations']:
            if op['kind'] == 'summary_call':
                pending.append(op)
            else:
                groups.append(dict(agent=op, summaries=pending)); pending = []
        if pending:
            raise ValueError('Unattached summary calls')
        case = dict(case_id=key, trial=0, trajectory=trajectory, groups=groups,
                    tools=request.get('tools') or [], source_last_request=request)
        # Validate original prompt bytes independently of generated history.
        state = Replay(case, policy_dir)
        hashes = 0
        for group in groups:
            prepared = state.prepare(group, generated=False)
            texts = {}
            for req in prepared['requests']:
                source = req['operation']['summary']
                if digest(req['prompt']) != source['prompt_sha256']:
                    raise ValueError(f'Summary prompt mismatch: {key} call {group["agent"]["logical_call_index"]}')
                hashes += 1
                if source.get('summary'):
                    texts[source['kind']] = source['summary']
            state.commit_summary(prepared, texts)
            if group['agent']['logical_call_index'] == int(replay['source_index']['logical_call_index']):
                # Naming is a wire adapter concern; compare canonical payload.
                active = state.active(prepared)
                expected = [clean(m) for m in request['messages']]
                if active != expected:
                    raise ValueError('Final source request reconstruction mismatch: ' + key)
        case['validated_summary_prompts'] = hashes
        cases.append(case)
    if len(cases) != count:
        raise ValueError('Insufficient source tasks')
    return cases, manifest


async def execute_case(case, instance, *, args, renderer, rendering, transport, emit, rows, user_turns):
    import asyncio
    import time
    state = Replay(case, args.summary_policy_dir)
    loop = asyncio.get_running_loop()
    physical = 0
    groups = case['groups']
    if args.summary_smoke:
        groups = [next(g for g in groups if g['summaries'] and g['agent'].get('summary_event_before_agent'))]
        # Smoke only: source prefix before the first summary, never a formal run.
    for group in groups:
        logical_start = time.perf_counter()
        op = group['agent']
        identity = dict(case_id=case['case_id'], instance=instance['instance'], filler=instance['filler'],
                        logical_turn=op['logical_call_index'])
        emit(dict(kind='logical_turn_start', **identity, time=logical_start))
        prepared = await loop.run_in_executor(rendering, state.prepare, group)
        texts = {}
        first_dispatch = None

        async def send(messages, budget, purpose, source_usage):
            nonlocal physical, first_dispatch
            if type(budget) is not int or budget <= 0:
                raise ValueError('Unknown physical output budget')
            call_id = dict(identity, turn=physical, purpose=purpose)
            physical += 1
            # Tool result names are required by Harmony. Preserve source names
            # when new model output uses different tool IDs, as in legacy replay.
            messages = copy.deepcopy(messages)
            names = {c['id']: c['function']['name'] for m in state.original for c in m.get('tool_calls', [])}
            for message in messages:
                if message['role'] == 'tool' and not message.get('name'):
                    message['name'] = names.get(message.get('tool_call_id'))
            tools = case['tools'] if purpose == 'agent' else []
            full, _ = await loop.run_in_executor(rendering, renderer.render, messages, tools)
            if args.model_context_limit and full + budget > args.model_context_limit:
                emit(dict(kind='context_limit', **call_id, full_tokens=full, max_new_tokens=budget))
                return None
            payload = dict(model=args.model, messages=messages, tools=tools, max_tokens=budget,
                           ignore_eos=True, temperature=0, stream=True,
                           stream_options={'include_usage': True, 'continuous_usage_stats': True},
                           chat_template_kwargs=json.loads(args.template_kwargs))
            emit(dict(kind='turn_start', **call_id, time=time.perf_counter(), full_tokens=full,
                      active_tokens=full, position_tokens=full, max_new_tokens=budget))
            row = await transport.request(args.url, payload, call_id)
            if first_dispatch is None:
                first_dispatch = row["start_time"]
            if row['success'] and row['prompt_len'] != full:
                row.update(success=False, status='template_mismatch', error='Summary native template mismatch')
            row.update(**call_id, trial=case['trial'], requested_max_tokens=budget,
                       expected_prompt_tokens=full, history_sha256=digest(messages), drop_state={},
                       source_usage=source_usage)
            rows.append(row)
            emit(dict(kind='turn_end', **row))
            return row

        for request in prepared['requests']:
            source = request['operation']
            row = await send(request['messages'], source['observed_output_tokens'],
                             'summary_' + source['summary']['kind'], source['source_usage'])
            if row is None:
                return 'summary_context_limit'
            if not row['success']:
                return row['status']
            text = row['assistant'].get('content')
            if prepared['event'] and (not isinstance(text, str) or not text.strip() or row['assistant'].get('tool_calls')):
                return 'invalid_generated_summary'
            texts[source['summary']['kind']] = text
        state.commit_summary(prepared, texts)
        if prepared['event']:
            emit(dict(kind='summary_replacement', **identity, first_kept=state.first_kept,
                      summary=state.summary, event=prepared['event']['event_id'],
                      source_first_kept=prepared['event']['first_kept_message_index']))
        if not op.get('replay_output_budget_known'):
            emit(dict(kind='source_terminal', **identity, source_status=op['status'],
                      agent_requested=op.get('agent_requested'), unknown_usage=not bool(op['source_usage'])))
            return 'source_terminal_replayed'
        row = await send(state.active(prepared), op['observed_output_tokens'], 'agent', op['source_usage'])
        if row is None:
            return 'summary_context_limit'
        if not row['success']:
            return row['status']
        state.commit_agent(op, row['assistant'])
        parent = dict(identity, start_time=first_dispatch, prepare_start=logical_start,
                      prepare_time_s=first_dispatch-logical_start, raw_done_time=row['raw_done_time'],
                      latency=row['raw_done_time']-first_dispatch,
                      ttft=(row['start_time']+row['ttft']-first_dispatch) if row['ttft'] is not None else None,
                      tpot_s=row['tpot_s'], output_len=row['output_len'],
                      cleanup_time_s=row['cleanup_time_s'], success=True)
        user_turns.append(parent)
        emit(dict(kind='logical_turn_end', **parent))
    return 'all_turns_completed'
