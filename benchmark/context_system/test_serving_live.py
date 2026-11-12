"""S5 MLGym replay with fixed user concurrency including real source tool sleeps."""
import argparse
import asyncio
import copy
import json
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from mlgym_replay import ObservationRollingState, load_cases, file_sha, digest


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return value


class LiveTaskScheduler:
    ENDPOINTS = {'all_recorded_steps_replayed', 'source_terminal_replayed'}

    def __init__(self, cases, concurrency, execute, emit, completion_target=None, measurement_seconds=None):
        if concurrency < 1 or not cases or len({c['case_id'] for c in cases}) != len(cases):
            raise ValueError('Invalid concurrency or task pool')
        self.pending = deque(cases)
        self.concurrency, self.execute, self.emit = concurrency, execute, emit
        self.active, self.completed = {}, []
        self.start = self.cutoff = None
        self.failure = None
        if completion_target is not None and not 0 < completion_target <= len(cases):
            raise ValueError('Completion target outside unique task pool')
        if measurement_seconds is not None and measurement_seconds <= 0:
            raise ValueError('Measurement duration must be positive')
        self.completion_target, self.measurement_seconds = completion_target, measurement_seconds
        self.finished_endpoints = 0
        self.stop_reason = None
        self.stopped = asyncio.Event()

    def stop(self, reason, at=None):
        if not self.stopped.is_set():
            self.cutoff = time.perf_counter() if at is None else at
            self.stop_reason = reason
            self.stopped.set()

    async def deadline(self):
        at = self.start + self.measurement_seconds
        await asyncio.sleep(max(0, at-time.perf_counter()))
        self.stop('measurement_time_limit', at)

    async def worker(self, slot):
        while self.pending and not self.stopped.is_set():
            if self.measurement_seconds is not None and time.perf_counter() >= self.start+self.measurement_seconds:
                self.stop('measurement_time_limit', self.start+self.measurement_seconds)
                return
            case = self.pending.popleft()
            identity = dict(case_id=case['case_id'], instance=case['rank'], slot=slot)
            start = time.perf_counter()
            self.active[slot] = identity
            self.emit(dict(kind='task_start', **identity, time=start, wall=time.time(), active_tasks=len(self.active)))
            status = 'cancelled'
            try:
                status = await self.execute(case, identity)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                status = 'runtime_failed'
                self.failure = repr(error)
                self.stop('interrupted_or_failed')
                raise
            finally:
                end = time.perf_counter()
                row = dict(**identity, status=status, start=start, end=end, latency_s=end-start)
                self.completed.append(row)
                self.active.pop(slot)
                self.emit(dict(kind='task_end', **row, wall=time.time(), active_tasks=len(self.active)))
                if self.measurement_seconds is not None and end >= self.start+self.measurement_seconds:
                    self.stop('measurement_time_limit', self.start+self.measurement_seconds)
                if status in self.ENDPOINTS and (self.cutoff is None or end <= self.cutoff):
                    self.finished_endpoints += 1
                    if self.completion_target is not None and self.finished_endpoints >= self.completion_target:
                        self.stop('task_completion_target', end)

    async def run(self):
        self.start = time.perf_counter()
        self.emit(dict(kind='measurement_start', time=self.start, wall=time.time(), user_concurrency=self.concurrency))
        tasks = [asyncio.create_task(self.worker(i)) for i in range(min(self.concurrency, len(self.pending)))]
        group = asyncio.gather(*tasks)
        stopper = asyncio.create_task(self.stopped.wait())
        timer = asyncio.create_task(self.deadline()) if self.measurement_seconds is not None else None
        try:
            await asyncio.wait([group, stopper], return_when=asyncio.FIRST_COMPLETED)
            if group.done():
                await group
                if not self.stopped.is_set():
                    if self.completion_target is not None and self.finished_endpoints < self.completion_target:
                        raise RuntimeError('Unique source pool exhausted before completion target')
                    self.stop('source_pool_exhausted')
        except BaseException:
            self.stop('interrupted_or_failed')
            raise
        finally:
            if self.cutoff is None:
                self.stop('interrupted_or_failed')
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Retrieve failures even when the stop event and a failed worker race.
            await asyncio.gather(group, return_exceptions=True)
            stopper.cancel()
            if timer is not None:
                timer.cancel()
            await asyncio.gather(stopper, *([timer] if timer else []), return_exceptions=True)
            self.emit(dict(kind='measurement_cutoff', time=self.cutoff, wall=time.time(), failure=self.failure,
                           stop_reason=self.stop_reason, completed_endpoints=self.finished_endpoints))
            if self.failure:
                raise RuntimeError(self.failure)


async def tool_wait(seconds, identity, emit, sleep=asyncio.sleep, clock=time.perf_counter):
    if seconds is None:
        return
    start = clock()
    emit(dict(kind='tool_wait_start', **identity, time=start, wall=time.time(), source_seconds=seconds,
              wake_at_wall=time.time()+seconds))
    await sleep(seconds)
    elapsed = clock()-start
    if elapsed+1e-6 < seconds:
        raise RuntimeError('Tool wait returned early')
    emit(dict(kind='tool_wait_end', **identity, time=clock(), wall=time.time(), source_seconds=seconds,
              actual_seconds=elapsed, wake_delay_seconds=elapsed-seconds))


def make_payload(model, history, state, budget):
    return dict(model=model, messages=history, tools=[], max_tokens=budget, ignore_eos=True,
                temperature=0, stream=True, stream_options={'include_usage':True},
                chat_template_kwargs={'preserve_thinking_history':True},
                drop_message=state['drop_message'], reposition=state['reposition'])


def validate_budget(state, budget, limit):
    if max(state['active_tokens'], state['position_tokens'])+budget > limit:
        raise ValueError('Actual generated history exceeds context budget; no truncation permitted')


async def run(args):
    import aiohttp
    from test_serving import NativeTemplateAdapter, Transport, load_method
    from serving_cohort import Journal
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest['seed'] != 42 or manifest['limit'] != args.model_context_limit:
        raise ValueError('Unexpected frozen selection')
    for name, expected in manifest['tokenizer_files'].items():
        if file_sha(Path(args.tokenizer)/name) != expected:
            raise ValueError('Tokenizer changed')
    if file_sha(args.chat_template) != manifest['template_sha256']:
        raise ValueError('Template changed')
    cases = load_cases(args.source, manifest)
    method = load_method(args.mini_root)
    renderer = NativeTemplateAdapter(args.tokenizer, {'preserve_thinking_history':True}, args.chat_template)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root/'selection.json').write_text(json.dumps(manifest, indent=2))
    rows, error = [], None
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, current.cancel)
    with Journal(root) as journal, ThreadPoolExecutor(max_workers=1) as rendering:
        emit = journal.emit
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout),
                connector=aiohttp.TCPConnector(limit=args.concurrency*2)) as session:
            transport = Transport(session, method, args.prefill_url, args.bootstrap_port, emit=emit)
            async def execute(case, task_id):
                history, observation_ids = [], set()
                rolling = ObservationRollingState()
                orphan = False
                for turn, call in enumerate(case['calls']):
                    identity = dict(**task_id, turn=turn, filler=False)
                    prep = time.perf_counter()
                    for message in call['new_messages']:
                        if len(history) >= 2 and message['role'] == 'user':
                            observation_ids.add(len(history))
                        history.append(copy.deepcopy(message))
                    full, owners = await loop.run_in_executor(rendering, renderer.render, history, [])
                    state = rolling.extend(history, owners, full, observation_ids)
                    validate_budget(state, call['budget'], args.model_context_limit)
                    payload = make_payload(args.model, history, state, call['budget'])
                    emit(dict(kind='client_prepare', **identity, start=prep, end=time.perf_counter(), wall=time.time()))
                    emit(dict(kind='turn_start', **identity, time=time.perf_counter(), wall=time.time(),
                              full_tokens=full, **state, max_new_tokens=call['budget']))
                    row = await transport.request(args.url, payload, identity)
                    if row['success'] and (row['prompt_len'] != full or row['output_len'] != call['budget']):
                        row.update(success=False, status='budget_or_template_mismatch', error='Native prompt/output differs from frozen budget')
                    row.update(**identity, requested_max_tokens=call['budget'], expected_prompt_tokens=full,
                               history_sha256=digest(history), drop_state=state, source_tool_seconds=call['tool_seconds'])
                    rows.append(row)
                    emit(dict(kind='turn_end', **row))
                    if not row['success']:
                        raise RuntimeError(str(row.get('error') or row['status']))
                    history.append(copy.deepcopy(row['assistant']))
                    orphan |= call['orphan_tool_step']
                    if call['orphan_tool_step']:
                        emit(dict(kind='source_tool_time_unknown', **identity, reason='orphan final assistant without trajectory step'))
                    await tool_wait(call['tool_seconds'], identity, emit)
                if orphan:
                    return 'source_incomplete_tail'
                return 'source_terminal_replayed' if str(case['source_exit_status']).startswith('exit_') else 'all_recorded_steps_replayed'
            scheduler = LiveTaskScheduler(cases, args.concurrency, execute, emit,
                                          args.completion_target, args.measurement_seconds)
            try:
                await scheduler.run()
            except BaseException as exc:
                error = exc
    result = dict(valid=error is None, stop_reason=scheduler.stop_reason if error is None else 'interrupted_or_failed',
                  error=repr(error) if error else None, start=scheduler.start, cutoff=scheduler.cutoff,
                  duration_s=scheduler.cutoff-scheduler.start, args=vars(args), tasks=scheduler.completed,
                  turns=rows, user_concurrency=args.concurrency, manifest_sha256=file_sha(args.manifest),
                  completion_target=args.completion_target, completed_endpoints=scheduler.finished_endpoints,
                  source_no_reply=manifest['source_no_reply'], excluded_tasks=manifest['excluded'],
                  note='Wall throughput includes source tool sleeps. Source terminal/incomplete states are not task-solving success.')
    (root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    from analyze_serving_live import summarize
    (root/'metrics.json').write_text(json.dumps(summarize(result), indent=2)+'\n')
    if error is not None:
        raise error


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','manifest','mini-root','output-dir','model','tokenizer','chat-template','url','prefill-url'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--concurrency', type=positive, required=True)
    p.add_argument('--completion-target', type=positive)
    p.add_argument('--measurement-seconds', type=positive)
    p.add_argument('--bootstrap-port', type=positive, required=True)
    p.add_argument('--model-context-limit', type=positive, default=131072)
    p.add_argument('--timeout', type=positive, default=21600)
    return p


if __name__ == '__main__':
    asyncio.run(run(parser().parse_args()))
