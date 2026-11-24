"""Publish verified TP2 pairs incrementally using the shared result volume."""
import json
from pathlib import Path
import time

import wandb

from benchmark_throughput import IDENTITY_KEYS, PAIR_CONFIG_KEYS, atomic_json

ROOT = Path("/homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction/results/tp2")


def rows():
    result = {}
    for f in ROOT.glob("*/timing/*/*/*/decode-summary.json"):
        x = json.loads(f.read_text())
        m = x["server_manifest"]
        if m["config"]["tp_size"] != 2:
            continue
        key = (x["output_len"], x["bs"])
        methods = result.setdefault(key, {})
        if x["method"] in methods:
            raise ValueError(f"Duplicate cell needs review: {key}, {x['method']}")
        methods[x["method"]] = x
    table = []
    for (length, bs), methods in sorted(result.items()):
        b, q = methods.get("bf16"), methods.get("loopvq")
        if b and q:
            for k in (*IDENTITY_KEYS, "tp_gpu_uuids"):
                assert b["server_manifest"][k] == q["server_manifest"][k], (length,bs,k)
            if b["status"] == q["status"] == "ok":
                for k in PAIR_CONFIG_KEYS:
                    assert b["server_config_readback"][k] == q["server_config_readback"][k], k
                for k in ("input_ids_sha256","input_fixture_sha256","repeat_count","primary_metric"):
                    assert b[k] == q[k], k
        left = b and b["decode_tokens_s_median"]
        right = q and q["decode_tokens_s_median"]
        m = (b or q)["server_manifest"]
        table.append(dict(output_tokens=length,bs=bs,host=m["host"],gpu_pair=m["tp_gpu_uuids"],
                          bf16_status=b["status"] if b else "pending",loopvq_status=q["status"] if q else "pending",
                          bf16_tokens_s=left,loopvq_tokens_s=right,speedup=right/left if left and right else None))
    return table


def main():
    run = wandb.init(entity="529513043-tsinghua-university",project="loopvq-ouro-2p6b-thinking",
                     name="tp2-decode-2k-input", config=dict(tp=2,dp=1,input_tokens=2048,
                     ignore_eos=True,prefix=8,recent=64,measured_repeats=3,prefill_included=False))
    atomic_json(ROOT/"wandb.json",{"url":run.url})
    previous=None
    while True:
        table=rows()
        complete=len(table)==20 and all(r["bf16_status"] in ("ok","capacity_limited_not_run") and r["loopvq_status"]=="ok" for r in table)
        if table != previous:
            atomic_json(ROOT/"decode-throughput.json",{"complete":complete,"rows":table,"url":run.url})
            for r in table:
                prefix=f"output{r['output_tokens']}/bs{r['bs']}"
                for k in ("bf16_tokens_s","loopvq_tokens_s","speedup"):
                    if r[k] is not None:
                        run.summary[f"{prefix}/{k}"]=r[k]
            previous=table
        if complete:
            cols=list(table[0]);run.log({"decode_throughput":wandb.Table(columns=cols,data=[[r[k] for k in cols] for r in table])})
            run.finish()
            return
        time.sleep(30)


if __name__=="__main__":
    main()
