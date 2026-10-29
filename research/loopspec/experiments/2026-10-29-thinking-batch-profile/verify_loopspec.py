"""Teacher-force independent native prefixes through real LoopSpec branch execution.

Only the final token choice is forced. Draft proposals, carried states, mixed
core execution, ownership transfer and pruning use the production worker.
This is a numerical/logic diagnostic, never a performance measurement.
"""
import argparse
from array import array
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'2026-10-29-cdb-v2'))

from batch_numerics import error
from verify_recurrent import check_envelope


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=63)
    p.add_argument('--first', type=int, default=16)
    p.add_argument('--depths', type=int, nargs='+', default=[1])
    p.add_argument('--graph', action='store_true')
    p.add_argument('--model', default='/home/ruoxi/models/Ouro-2.6B-Thinking')
    p.add_argument('--dataset', type=Path, default=Path('/home/ruoxi/repo/test/datasets/aime.jsonl'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs, _TorchBenchRunner
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.recurrent_runner import RecurrentCoreGraph, RecurrentState
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.speculative.loopspec.config import pipeline_layout
    from sglang.srt.speculative.loopspec.pipeline import Pipeline
    from sglang.srt.speculative.loopspec.utils import RequestContext
    from sglang.srt.speculative.loopspec.worker import LoopSpecWorker

    options = ServerArgs(model_path=args.model, dtype='bfloat16', attention_backend='triton',
        bf16_gemm_backend='auto', max_running_requests=args.first, page_size=1,
        disable_overlap_schedule=True, disable_radix_cache=True, chunked_prefill_size=-1,
        disable_cuda_graph=True, context_length=2048, mem_fraction_static=.85)
    ports = PortArgs.init_new(options)
    publish(options, role='scheduler', ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs()
    bench, _ = load_model(options, ports, 0, 0)
    runner = bench.torch_runner
    recurrent = runner.recurrent_runner
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    questions = json.loads((args.reference/'prompts.json').read_text())['prompts'][:args.first]
    tokens = json.loads((args.tokens/'cohort-0.json').read_text())['requests']
    reference = torch.load(args.reference/'tensors.pt', weights_only=True)
    limits = json.loads((args.reference/'numerical_envelope.json').read_text())
    assert args.steps < min(len(r['output_ids']) for r in tokens)
    records, histories, contexts = [], {}, []

    def record(q, step, hidden, logits):
        source = reference['single'][q][step]
        a, b = source['logits'][0], logits.float().cpu()
        records.append({'question': q+1, 'step': step,
            'hidden': [error(source['hidden'][d,0],hidden[d]) for d in range(4)],
            'logits': error(a,b), 'argmax_reference': int(a.argmax()),
            'argmax_candidate': int(b.argmax()),
            'reference_margin': float(a.topk(2).values.diff().abs()[0])})

    @torch.inference_mode()
    def prefill(batch):
        assert batch.forward_mode.is_extend()
        token_rows = torch.repeat_interleave(torch.arange(args.first,device=runner.device),
            batch.extend_seq_lens,output_size=len(batch.input_ids))
        state = recurrent.prepare(batch, token_rows=token_rows)
        graph = RecurrentCoreGraph(recurrent,args.first,tokens=1<<(len(batch.input_ids)-1).bit_length()) if args.graph else None
        states = []
        selection = batch.extend_seq_lens.cumsum(0).long()-1
        for d in range(4):
            depths = torch.full_like(batch.req_pool_indices,d)
            state = graph.replay(batch,state,depths) if graph else recurrent.advance(batch,state,depths,token_rows=token_rows)
            state = RecurrentState(state.hidden.clone())
            states.append(state.hidden[selection].cpu())
        logits = recurrent.finish(batch,state,depths,token_rows=token_rows)
        for q in range(args.first): record(q,0,[s[q] for s in states],logits.next_token_logits[q])
        return SimpleNamespace(logits_output=logits,can_run_graph=False)

    runner.forward = prefill
    requests = []
    for q, row in enumerate(questions):
        ids = row['input_ids']
        req = Req(rid=f'q{q+1}',origin_input_text='',origin_input_ids=array('q',ids),
            sampling_params=SamplingParams(temperature=0,max_new_tokens=args.steps+1))
        req.sampling_params.normalize(tokenizer)
        req.tokenizer = tokenizer
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = len(ids)
        requests.append(req)
    with torch.inference_mode():
        _, _, sb = _TorchBenchRunner(runner).extend(requests)
        # Construct only the worker shell here; native HTTP tests cover its
        # constructor/config/overlap wiring. No substitute compute algorithm.
        worker = object.__new__(LoopSpecWorker)
        worker.model_runner = runner
        worker._target_worker = SimpleNamespace(get_memory_pool=lambda:(runner.req_to_token_pool,runner.token_to_kv_pool_allocator))
        worker.depths, worker.total = tuple(args.depths), 4
        worker.width = pipeline_layout(4,worker.depths)[-1][2]
        worker.max_rows = worker.width*args.first
        worker.contexts = {}
        worker.role_replays = dict.fromkeys(('pre','core','post','rejection'),0)
        worker.alloc_memory_pool()
        worker.init_attention_backends()
        if args.graph:
            worker.executor.graph_pool = torch.cuda.graph_pool_handle()
            for n in sorted({1,2,4,8,worker.width,worker.width*2,worker.width*4,16,worker.width*8,worker.width*args.first}):
                worker.executor.graphs[n] = RecurrentCoreGraph(worker.executor,n)
                worker.executor.finish_graphs[n] = RecurrentCoreGraph(worker.executor,n,role='finish')
        for q, req in enumerate(requests):
            row = int(sb.req_pool_indices[q]); length = int(sb.seq_lens[q])
            prefix = runner.req_to_token_pool.req_to_token[row,:length].tolist()
            context = RequestContext(worker,req,row,prefix)
            context.pipeline = Pipeline(context,tokens[q]['output_ids'][0],prefix,args.steps+1,4,worker.depths)
            contexts.append(context); worker.contexts[req.rid] = context
        turns = 0
        while any(c.pipeline.lanes for c in contexts):
            rows, plans = [], []
            # Reordering owners every loop tests scatter-back independently of
            # the stable native request row. Stagger first admission as well.
            order = [(turns+i)%args.first for i in range(args.first)]
            for q in order:
                c = contexts[q]
                if not c.pipeline.lanes or turns < q%4:
                    continue
                lanes = c.pipeline.plan(); plans.append((q,c,lanes))
                rows.extend((c,l) for l in lanes)
            worker._advance_rows(rows)
            for c,l in rows:
                histories.setdefault((c.req.rid,l.uid),{})[l.depth-1] = l.hidden[0].cpu().clone()
            for q,c,lanes in plans:
                for lane in lanes:
                    if lane.depth != 4: continue
                    step = c.pipeline.committed
                    h = histories[(c.req.rid,lane.uid)]
                    assert set(h)=={0,1,2,3}
                    record(q,step,h,c.logits[lane.uid])
                    forced = tokens[q]['output_ids'][step]
                    accepted = next((i for i,d in enumerate(c.depths)
                        if d in lane.proposals and lane.proposals[d][0]==forced),-1)
                    c.decisions[lane.uid] = ([forced,accepted],None)
                c.pipeline.collect(lanes)
            live = {(c.req.rid,l.uid) for c in contexts for l in c.pipeline.lanes}
            histories = {key:value for key,value in histories.items() if key in live}
            worker.extra_owned_slots()
            turns += 1
            assert turns <= 4*(args.steps+1)+args.first
        assert not worker.extra_owned_slots()
    failures = check_envelope(records,limits)
    result = {'success':not failures,'graph':args.graph,'depths':args.depths,'turns':turns,
        'records':records,'failures':failures,'calibration':limits,
        'requests':[{'rid':c.req.rid,'reused':c.pipeline.reused,'restarts':c.pipeline.restarts,
            'accepted':c.accepted,'rejected':c.rejected} for c in contexts]}
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'success':not failures,'positions':len(records),'failures':failures}),flush=True)
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
