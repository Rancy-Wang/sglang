"""Record one unprofiled-by-timing eager decode on each TP rank for numerical comparison."""

from pathlib import Path

import torch
import torch.distributed as dist


_step = {}
_finished = False


def make_hook(config):
    stage = config["stage"]
    root = Path(config["root"])

    def hook(module, inputs, output):
        global _finished
        if _finished or not (root / "armed").exists():
            return
        if stage == "projection":
            # Full prompt uses 2048 rows. The first one-row projection is
            # core layer zero of the first decode. Snapshot before in-place RoPE.
            if _step or output[0].shape[0] != 1:
                return
            _step["raw_qkv"] = output[0].detach().cpu().clone()
        elif "raw_qkv" not in _step:
            return
        elif stage == "attention" and "attention_q" not in _step:
            for name, tensor in zip(("q", "k", "v"), inputs):
                _step["attention_" + name] = tensor.detach().cpu().clone()
        elif stage == "activation" and "activation" not in _step:
            _step["gate_up"] = inputs[0].detach().cpu().clone()
            _step["activation"] = output.detach().cpu().clone()
        elif stage == "logits":
            logits = output.next_token_logits.detach().float().cpu()
            assert logits.shape[0] == 1 and torch.isfinite(logits).all()
            assert {"attention_q", "attention_k", "attention_v", "activation"} <= _step.keys()
            _step["logits"] = logits
            torch.save(_step, root / f"rank-{dist.get_rank()}.pt")
            _finished = True

    return hook
