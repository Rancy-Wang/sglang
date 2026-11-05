"""Isolate verifier shape/KV effects on an identical materialized prefix.

Re-prefilling a recorded prefix is an isolation experiment, not a replay of
the original serving KV trajectory. Strict argmax differences remain failures.
"""
import argparse
from array import array
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def error(x, y):
    d = x.float() - y.float()
    return {'max_abs': float(d.abs().max()), 'rmse': float(d.square().mean().sqrt()),
            'exact': bool(torch.equal(x, y))}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1,4])
    p.add_argument('--widths', type=int, nargs='+', default=[1,5,16])
    p.add_argument('--graph', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.recurrent_runner import RecurrentState
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs
    from sglang.srt.speculative.das_wave.worker import DASWaveWorker, WaveContext
    from sglang.srt.speculative.das_wave.graph import WaveGraph

    model = '/home/ruoxi/models/Ouro-2.6B-Thinking'
    opts = ServerArgs(model_path=model, dtype='bfloat16', attention_backend='triton',
        max_running_requests=max(a.batches), page_size=1, disable_overlap_schedule=True,
        disable_radix_cache=True, chunked_prefill_size=-1, disable_cuda_graph=True,
        context_length=2048, max_total_tokens=4096, mem_fraction_static=.70)
    ports = PortArgs.init_new(opts)
    publish(opts, role='scheduler', ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs(); bench, _ = load_model(opts, ports, 0, 0)
    runner = bench.torch_runner; recurrent = runner.recurrent_runner
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    assert recurrent is not None and runner.model.config.total_ut_steps == 4

    @torch.inference_mode()
    def prefill(batch):
        token_rows = torch.repeat_interleave(torch.arange(batch.batch_size, device=runner.device),
            batch.extend_seq_lens, output_size=len(batch.input_ids))
        state = recurrent.prepare(batch, token_rows=token_rows)
        for d in range(4):
            depths = torch.full_like(batch.req_pool_indices, d)
            state = recurrent.advance(batch, state, depths, token_rows=token_rows)
        logits = recurrent.finish(batch, state, depths, token_rows=token_rows)
        return SimpleNamespace(logits_output=logits, can_run_graph=False)

    runner.forward = prefill
    worker = object.__new__(DASWaveWorker)
    worker.model_runner, worker.model, worker.device = runner, runner.model, runner.device
    worker._target_worker = SimpleNamespace(get_memory_pool=lambda:
        (runner.req_to_token_pool, runner.token_to_kv_pool_allocator))
    worker.width, worker.widths, worker.updates, worker.total = max(a.widths), (2,3,4,5), 1, 4
    worker.widths = tuple(min(w, worker.width) for w in worker.widths)
    worker.verify_graphs, worker.draft_graphs, worker.graph_replays = {}, {}, 0
    worker.alloc_memory_pool(); worker.init_attention_backends()
    worker.executor.graph_pool = torch.cuda.graph_pool_handle()
    real_readout = worker._readout
    def capture(batch, hidden):
        value = real_readout(batch, hidden)
        worker.probe_logits = value  # no host operation inside CUDA capture
        return value
    worker._readout = capture
    graphs = {}
    if a.graph:
        for rows in sorted({b*w for b in a.batches for w in a.widths}):
            graph = WaveGraph(worker, rows, draft=False)
            graph.probe_logits = worker.probe_logits
            graphs[rows] = graph

    source = json.loads((a.reference/'ar.json').read_text())
    cases = {}
    for row in source:
        key = tuple(row['payload']['input_ids'])
        if key not in cases: cases[key] = row
    records = []
    pool, allocator = runner.token_to_kv_pool, runner.token_to_kv_pool_allocator
    mapping = pool.timeline.maps['core']

    def kv(slots):
        physical = mapping[:, slots.long()].flatten()
        return torch.stack([buffer[physical].clone() for layer in (0,23,47)
                            for buffer in (pool.k_buffer[layer], pool.v_buffer[layer])])

    with torch.inference_mode():
        for case_index, (prompt, case) in enumerate(cases.items()):
            output = case['response']['output_ids']
            frontier = min(27, len(output) - worker.width - 1)
            ids = list(prompt) + output[:frontier]
            bench.clear()
            req = Req(rid=f'probe-{case_index}', origin_input_text='', origin_input_ids=array('q', ids),
                      sampling_params=SamplingParams(temperature=0, max_new_tokens=64))
            req.sampling_params.normalize(tokenizer); req.tokenizer = tokenizer
            req.full_untruncated_fill_ids = req.origin_input_ids; req.logprob_start_len = -1; req.extend_end = len(ids)
            _, _, sb = bench.extend([req])
            row = int(sb.req_pool_indices[0]); length = len(ids)
            context = WaveContext(req, row, length, torch.tensor(output[frontier:frontier+1], device=runner.device), 1)
            inputs = torch.tensor(output[frontier:frontier+worker.width+1], device=runner.device)[None, :]
            reference_slots = allocator.alloc(worker.width)
            reference = []
            for step in range(worker.width):
                context.length = length + step
                worker._verify([context], [1], reference_slots[step:step+1].reshape(1,1), inputs[:,step:])
                reference.append(worker.probe_logits[0].clone())
                runner.req_to_token_pool.req_to_token[row, length+step] = reference_slots[step].to(torch.int32)
            reference = torch.stack(reference)
            refkv = kv(reference_slots)
            context.length = length
            prefix_slots = runner.req_to_token_pool.req_to_token[row,:length].clone()
            before = kv(prefix_slots)
            draft_slots = allocator.alloc(worker.width).reshape(1,-1)
            worker._draft([context], [worker.width], draft_slots)
            assert torch.equal(before, kv(prefix_slots)), 'Draft changed mature prefix KV'
            allocator.free(draft_slots.flatten())
            for bs in a.batches:
                contexts = [WaveContext(req, row, length, context.anchor, 1) for _ in range(bs)]
                for width in a.widths:
                    slots = allocator.alloc(bs*worker.width).reshape(bs,-1)
                    guesses = inputs.expand(bs,-1).clone()
                    pred, owners, columns = worker._verify(contexts, [width]*bs, slots, guesses)
                    logits = worker.probe_logits.clone().reshape(bs,width,-1)
                    expected = reference[:width].expand(bs,-1,-1)
                    record = {'case':case_index,'frontier':frontier,'bs':bs,'width':width,
                              'logits':error(expected,logits),
                              'argmax_flips':int((expected.argmax(-1)!=logits.argmax(-1)).sum()),
                              'reference_margins':reference[:width].topk(2).values.diff().abs().flatten().tolist(),
                              'kv':[]}
                    for owner in range(bs):
                        # refkv contains depth-major slots, select the same width per depth.
                        reference_kv = refkv.reshape(6,4,worker.width,*refkv.shape[2:])[:,:,:width].flatten(1,2)
                        record['kv'].append(error(reference_kv,kv(slots[owner,:width])))
                    if a.graph:
                        batch, prefix, _, _ = worker._prepare(contexts,[width]*bs,slots,guesses)
                        hidden = runner.model.get_input_embeddings()(batch.input_ids)
                        graph = graphs[bs*width]
                        _, graph_pred = graph.replay_wave(batch,hidden,prefix)
                        record['graph_ids_equal'] = bool(torch.equal(pred,graph_pred))
                        record['graph_logits'] = error(logits.flatten(0,1),graph.probe_logits)
                        assert record['graph_ids_equal'], record
                    records.append(record)
                    print(json.dumps(record),flush=True)
                    allocator.free(slots.flatten())
            allocator.free(reference_slots)
    result = {'isolation':'same re-prefilled prefix; not original serving KV trajectory',
              'strict_equal':all(r['argmax_flips']==0 for r in records),'records':records}
    (a.output/'report.json').write_text(json.dumps(result,indent=2))
    if not result['strict_equal']: raise SystemExit(1)


if __name__ == '__main__': main()
