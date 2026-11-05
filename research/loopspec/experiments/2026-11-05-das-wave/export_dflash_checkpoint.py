"""Audited SpecForge HF export; originals remain untouched, output must be new."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from specforge.export.checkpoint_io import materialize_draft
from specforge.export.to_hf import export_to_hf


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'Refusing to overwrite an existing artifact'
    assert sha(a.config) == '74cbfab3ab3b5567cc9dd6af20efe64707e118dff789d7d3b605b1fa2ca44ee4'
    assert sha(a.checkpoint) == '9cedcf1d1635a4fe686392a8f2209afba0c9b6c0fa3fc3308640757e343df3ba'
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    expected = {'strategy': 'dflash', 'global_step': 10000,
                'dflash_draft_num_hidden_layers': 5, 'dflash_block_size': 16,
                'dflash_mask_token_id': 16, 'dflash_target_layer_ids': (1,12,23,34,45)}
    for key, value in expected.items():
        assert state[key] == value, (key, state[key], value)
    model = materialize_draft(state, str(a.config))
    model.load_state_dict(state['draft_state_dict'], strict=True)
    assert len(state['draft_state_dict']) == 58
    assert not hasattr(model, 'embed_tokens'), 'This checkpoint borrows target embeddings'
    del model
    export_to_hf(str(a.checkpoint), str(a.config), str(a.output))
    saved = {}
    for path in sorted(a.output.glob('*.safetensors')):
        tensors = load_file(str(path))
        assert not saved.keys() & tensors.keys()
        saved.update(tensors)
    assert saved.keys() == state['draft_state_dict'].keys()
    checks = []
    for key, tensor in state['draft_state_dict'].items():
        assert tensor.dtype == saved[key].dtype == torch.bfloat16
        assert torch.equal(tensor, saved[key]), key
        checks.append({'key': key, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                       'sha256': hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
    manifest = {'checkpoint': str(a.checkpoint), 'checkpoint_sha256': sha(a.checkpoint),
                'training_config_sha256': sha(a.config), 'metadata': expected,
                'specforge_sha': 'af1c3f3868e0ad1d1bc4a3bc4c577f2c9d3fa7b5',
                'feature_semantics': 'final recurrence physical layer output before loop-end norm',
                'tensors': checks, 'artifacts': {f.name: sha(f) for f in a.output.iterdir() if f.is_file()}}
    (a.output/'export-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output':str(a.output),'strict_tensor_matches':len(checks)}), flush=True)


if __name__ == '__main__': main()
