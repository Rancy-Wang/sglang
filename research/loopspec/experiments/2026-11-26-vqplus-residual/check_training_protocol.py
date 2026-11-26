"""Real KV numerical audit, not benchmark evaluation or generation accuracy.

Compare assignment algorithms on identical inputs, then separately measure the
pre-RoPE vs inverse-post-RoPE input definition used by the existing training code.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(a, b):
    delta = a.float() - b.float()
    return dict(max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item(),
                relative_rms=(delta.square().mean() / b.float().square().mean().clamp_min(1e-30)).sqrt().item())


def rotate(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--book', type=Path, required=True)
    parser.add_argument('--loopvq-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261126)
    codec_path = args.loopvq_root / 'calibration/residual_codec.py'
    codec = load('training_residual_codec', codec_path)
    pack = load('vq_layout', ROOT / 'python/sglang/srt/mem_cache/loopvq_pack.py')
    encoder = load('vq_encoder', ROOT / 'python/sglang/kernels/ops/loopvq/encode.py')
    sys.path.insert(0, str(HERE.parent / '2026-11-23-loopvq-reproduction'))
    from serving_probe import exact_prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fixture = exact_prompt(tokenizer, 256)
    (args.out / 'input.json').write_text(json.dumps(fixture))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    assert model.config.total_ut_steps == 4 and model.config.num_hidden_layers == 48
    layers, heads, capture = (0, 23, 47), [0, 7, 8, 15], {}
    handles = []
    for layer in layers:
        capture[layer] = []
        handles.append(model.model.layers[layer].self_attn.k_proj.register_forward_hook(
            lambda module, inputs, output, layer=layer: capture[layer].append(output.detach().clone())))
    ids = torch.tensor([fixture['input_ids']], device='cuda')
    result = model(ids, use_cache=True, logits_to_keep=1)
    for handle in handles: handle.remove()
    assert all(len(rows) == 4 for rows in capture.values())
    cos, sin = model.model.rotary_emb(torch.empty(1, 1, 256, 1, device='cuda', dtype=torch.float32),
                                     torch.arange(256, device='cuda')[None])
    cos, sin = cos[0, :, None].float(), sin[0, :, None].float()
    cache = result.past_key_values
    report = {'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'model': str(args.model), 'book': str(args.book), 'layers': layers, 'heads': heads,
              'tokens': 256, 'full_loops': 4, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'training_codec_sha256': hashlib.sha256(codec_path.read_bytes()).hexdigest(), 'rows': []}
    for kind, filename in (('k', 'codebook.pt'), ('v', 'vq_v_codebook.pt')):
        blob = torch.load(args.book / filename, map_location='cpu', weights_only=True)
        coupled = pack.build_coupled_codebook(blob['codebooks'], 48, 4, 4, 4, 16, 128)
        for layer in layers:
            table = coupled[layer, heads].cuda()
            if kind == 'k':
                raw = [x[0].view(256, 16, 128)[:, heads].contiguous() for x in capture[layer]]
                post = [cache.key_cache[u * 48 + layer][0, heads].transpose(0, 1).contiguous() for u in range(4)]
                train = [x.float() * cos - rotate(x.float()) * sin for x in post]
            else:
                raw = [cache.value_cache[u * 48 + layer][0, heads].transpose(0, 1).contiguous() for u in range(4)]
                train, post = [x.float() for x in raw], raw
            loc = torch.arange(256, device='cuda', dtype=torch.int32) + 8
            indices = torch.empty((264, len(heads), 32), device='cuda', dtype=torch.uint8)
            encoder.encode_steps(raw, table, indices, loc)
            deployed = indices[8:].long()

            def assign(steps):
                x = torch.stack(steps, -1).reshape(256, len(heads), 32, 16)
                x = x.permute(1, 2, 0, 3).reshape(len(heads) * 32, 256, 16).float()
                cb = table.reshape(len(heads) * 32, 256, 16).float()
                return torch.cdist(x, cb).argmin(-1).view(len(heads), 32, 256).permute(2, 0, 1)

            same_input, training = assign(raw), assign(train)
            h = torch.arange(len(heads), device='cuda')[None, :, None]
            g = torch.arange(32, device='cuda')[None, None, :]
            base, served, trained = [], [], []
            for u in range(4):
                center = table[..., u::4].bfloat16()[h, g, deployed].flatten(-2)
                train_center = table[..., u::4].bfloat16()[h, g, training].flatten(-2)
                corrected = center.float() + codec.decode_residual(codec.encode_residual(raw[u], center))
                train_corrected = train_center.float() + codec.decode_residual(codec.encode_residual(train[u], train_center))
                if kind == 'k':
                    base.append((center.float() * cos + rotate(center.float()) * sin).bfloat16())
                    served.append((corrected * cos + rotate(corrected) * sin).bfloat16())
                    trained.append((train_corrected * cos + rotate(train_corrected) * sin).bfloat16())
                else:
                    base.append(center); served.append(corrected.bfloat16()); trained.append(train_corrected.bfloat16())
            row = dict(kind=kind, layer=layer, code_count=deployed.numel(),
                       assignment_mismatches_same_input=(deployed != same_input).sum().item(),
                       input_definition_assignment_mismatches=(same_input != training).sum().item(),
                       deployed_vs_training_assignment_mismatches=(deployed != training).sum().item(),
                       raw_vs_training_input=metrics(torch.stack(raw), torch.stack(train)),
                       base_reconstruction_error=metrics(torch.stack(base), torch.stack(post)),
                       residual_reconstruction_error=metrics(torch.stack(served), torch.stack(post)),
                       deployed_vs_training_reconstruction=metrics(torch.stack(served), torch.stack(trained)))
            report['rows'].append(row)
            (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
            print(kind, layer, row, flush=True)
        del blob, coupled


if __name__ == '__main__':
    main()
