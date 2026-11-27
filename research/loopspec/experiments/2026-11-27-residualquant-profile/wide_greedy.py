"""Additional full-model token-ID checks for wide and uneven request batches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'research/loopspec/experiments/2026-11-26-vqplus-residual'))
from check_serving import exact_prompt
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:30664')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(args.url + path, data, {'Content-Type': 'application/json'}),
                     timeout=1800) as response:
            return json.load(response)

    info = request('/get_server_info')
    cfg = info.get('server_args', info)
    assert cfg['tp_size'] == 2 and cfg['dp_size'] == 1
    assert Path(cfg['model_path']).resolve() == args.model.resolve()
    assert cfg['max_running_requests'] >= 64 and cfg['loopvq_residual_bits'] == 4
    assert cfg['loopvq_residual_layout'] == 'page-head'
    assert json.loads((args.model / 'config.json').read_text())['total_ut_steps'] == 4
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = {n: exact_prompt(tokenizer, n) for n in (512, 1024)}
    (args.out / 'inputs.json').write_text(json.dumps(prompts))
    (args.out / 'identity.json').write_text(json.dumps(dict(
        client_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                           text=True).strip(), server=info), indent=2))
    for label, lengths in (('b32-n1024', [1024] * 32), ('b64-n512', [512] * 64),
                           ('b32-uneven', [512, 1024] * 16)):
        payload = dict(input_ids=[prompts[n]['input_ids'] for n in lengths],
                       sampling_params=dict(temperature=0, max_new_tokens=128, ignore_eos=True))
        rows = request('/generate', payload)
        (args.out / (label + '.json')).write_text(json.dumps(rows))
        assert len(rows) == len(lengths) and all(len(row['output_ids']) == 128 for row in rows)
        if args.reference:
            reference = json.loads((args.reference / (label + '.json')).read_text())
            assert [row['output_ids'] for row in rows] == [row['output_ids'] for row in reference], label
        print(label, 'passed', flush=True)
    (args.out / 'summary.json').write_text(json.dumps(dict(passed=True)))


if __name__ == '__main__':
    main()
