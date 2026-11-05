"""Compare native Ouro capture with training-style final-invocation HF hooks."""
import argparse
from array import array
import json
from pathlib import Path

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
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoConfig, AutoModelForCausalLM
    from sglang.benchmark.one_batch import load_model, _init_process_global_configs
    from sglang.srt.models.ouro import OuroForCausalLM
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.runtime_context import publish, SpawnRanks
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs, PortArgs

    model_path = '/home/ruoxi/models/Ouro-2.6B-Thinking'
    layers = [1,12,23,34,45]
    OuroForCausalLM.supports_recurrent_execution = False
    opts = ServerArgs(model_path=model_path, dtype='bfloat16', attention_backend='triton',
        max_running_requests=1, page_size=1, disable_overlap_schedule=True,
        disable_radix_cache=True, chunked_prefill_size=-1, disable_cuda_graph=True,
        context_length=2048, max_total_tokens=2048, mem_fraction_static=.65)
    ports = PortArgs.init_new(opts)
    publish(opts, role='scheduler', ranks=SpawnRanks(world_rank=0, gpu_id=0))
    _init_process_global_configs(); bench, _ = load_model(opts, ports, 0, 0)
    runner = bench.torch_runner
    assert runner.recurrent_runner is None
    rows = json.loads((a.reference/'ar.json').read_text())
    cases = {}
    for r in rows:
        key = tuple(r['payload']['input_ids'])
        if key not in cases: cases[key] = r
    native = []; captured = {}; calls = {}
    def hook(index):
        def capture(module, args, output):
            if isinstance(output, tuple): output = output[0]
            captured[index] = output.clone()
            calls[index] = calls.get(index, 0) + 1
        return capture
    handles = [runner.model.model.layers[i].register_forward_hook(hook(i)) for i in layers]
    for case_index, (prompt, row) in enumerate(cases.items()):
        ids = list(prompt) + row['response']['output_ids'][:32]
        def run():
            bench.clear()
            req = Req(rid=f'feature-{case_index}', origin_input_text='', origin_input_ids=array('q',ids),
                      sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
            req.full_untruncated_fill_ids = req.origin_input_ids; req.logprob_start_len = -1; req.extend_end = len(ids)
            return bench.extend([req])[1].clone()
        runner.model.set_dflash_layers_to_capture(layers)
        captured.clear(); calls.clear()
        logits = run()
        assert calls == dict.fromkeys(layers,4), calls
        features = [runner.model.model.dflash_features[i].clone() for i in layers]
        assert all(torch.equal(features[j],captured[i]) for j,i in enumerate(layers))
        runner.model.model.dflash_layer_ids = ()
        without = run()
        assert torch.equal(logits,without), 'Aux capture changed target logits'
        native.append({'ids':ids,'features':[f.cpu() for f in features],'logits':logits.cpu()[0]})
    for handle in handles: handle.remove()
    # Transformers 4 supplied this default; Transformers 5 no longer does.
    # No checkpoint file or mathematical target parameter is changed.
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(hf_config, 'pad_token_id'): hf_config.pad_token_id = None
    hf = AutoModelForCausalLM.from_pretrained(model_path, config=hf_config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
    assert hf.config.total_ut_steps == 4
    captured = {}; calls = {}
    handles = [hf.model.layers[i].register_forward_hook(hook(i)) for i in layers]
    records = []
    for index, row in enumerate(native):
        captured.clear(); calls.clear()
        count = len(row['ids'])
        causal = torch.triu(torch.full((1,1,count,count), torch.finfo(torch.bfloat16).min,
                                      device='cuda',dtype=torch.bfloat16), diagonal=1)
        _, depths, _ = hf.model(input_ids=torch.tensor([row['ids']],device='cuda'),use_cache=False,
                               attention_mask={'full_attention':causal})
        assert len(depths)==4 and calls == dict.fromkeys(layers,4), calls
        logits = hf.lm_head(depths[-1][:,-1]).float()[0].cpu()
        record = {'case':index,'tokens':len(row['ids']),
                  'feature_errors':[error(row['features'][j],captured[i].squeeze(0).cpu()) for j,i in enumerate(layers)],
                  'logits':error(row['logits'],logits),
                  'native_argmax':int(row['logits'].argmax()),'hf_argmax':int(logits.argmax()),
                  'hf_margin':float(logits.topk(2).values.diff().abs()[0])}
        records.append(record); print(json.dumps(record),flush=True)
    for handle in handles: handle.remove()
    report = {'capture_hook_contract':True,'target_logits_unchanged':True,
              'layers':layers,'recurrence':4,'before_loop_norm':True,
              'hf_compatibility':{'missing_pad_token_id':None,'mask':'explicit full causal additive BF16'},
              'hf_argmax_equal':all(r['native_argmax']==r['hf_argmax'] for r in records), 'records':records}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    torch.save(native,a.output/'native-features.pt')
    if not report['hf_argmax_equal']: raise SystemExit(1)


if __name__ == '__main__': main()
