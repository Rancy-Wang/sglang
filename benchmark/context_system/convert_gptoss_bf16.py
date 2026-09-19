"""Expand the HF GPT-OSS MXFP4 checkpoint for native unquantized loading.

Uses SGLang's own dequantizer, in small expert slices. The transpose is the
HF unquantized checkpoint convention reversed by GptOss._load_normal_weights.
Never overwrites a destination. Source files are read-only.
"""

import argparse
import json
from pathlib import Path
import shutil
import time


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4

    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    readers = {}
    for filename in sorted(set(index["weight_map"].values())):
        readers[filename] = safe_open(args.source / filename, framework="pt", device="cpu")

    def read(name):
        return readers[index["weight_map"][name]].get_tensor(name)

    out_index = {"metadata": {"total_size": 0}, "weight_map": {}}
    converted = []
    torch.set_num_threads(8)
    for shard_id, filename in enumerate(sorted(readers)):
        tensors = {}
        started = time.time()
        for name in sorted(readers[filename].keys()):
            if name.endswith("_scales"):
                continue
            value = read(name)
            output_name = name
            if name.endswith("_blocks"):
                scales = read(name.removesuffix("_blocks") + "_scales")
                shape = (value.shape[0], value.shape[2] * 32, value.shape[1])
                expanded = torch.empty(shape, dtype=torch.bfloat16)
                for start in range(0, value.shape[0], 2):
                    block = value[start:start+2].cuda()
                    scale = scales[start:start+2].cuda()
                    dense = dequant_mxfp4(block, scale, torch.bfloat16)
                    expanded[start:start+2].copy_(dense.transpose(-2, -1).cpu())
                    del block, scale, dense
                output_name = name.removesuffix("_blocks")
                converted.append(dict(name=output_name, shape=list(expanded.shape), dtype="bfloat16"))
                value = expanded
                del scales
            elif value.is_floating_point():
                value = value.to(torch.bfloat16)
            tensors[output_name] = value.contiguous()
        output_file = f"model-{shard_id:05d}.safetensors"
        save_file(tensors, args.destination / output_file, metadata={"format": "pt"})
        for name, value in tensors.items():
            out_index["weight_map"][name] = output_file
            out_index["metadata"]["total_size"] += value.numel() * value.element_size()
        print(json.dumps(dict(shard=filename, output=output_file, seconds=time.time()-started)), flush=True)
        del tensors, value
    assert len(converted) == 72, len(converted)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "chat_template.jinja", "generation_config.json"):
        shutil.copy2(args.source / filename, args.destination / filename)
    config = json.loads((args.source / "config.json").read_text())
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    (args.destination / "config.json").write_text(json.dumps(config, indent=2))
    (args.destination / "model.safetensors.index.json").write_text(json.dumps(out_index, indent=2))
    (args.destination / "conversion.json").write_text(json.dumps(dict(
        source=str(args.source), method="SGLang dequant_mxfp4 to BF16; HF dense transpose",
        experts=converted, total_bytes=out_index["metadata"]["total_size"], completed=time.time()), indent=2))


if __name__ == "__main__":
    main()
