"""Verify completed TP2 evaluation markers and publish the independent seed table."""
import csv
import json
from pathlib import Path
import statistics
import sys
import time
import wandb

sys.path.insert(0, str(Path.home()/"repo/LoopVQ/experiments/loopvq_reproduction"))
from run_evaluation import COUNTS, collect_cell
from evaluation_io import atomic_json

ROOT = Path.home()/"experiments/loopvq/2026-11-23-reproduction/results/accuracy-tp2"
METHODS = ("bf16", "loopvq_0.5bit")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    run = wandb.init(entity="529513043-tsinghua-university", project="loopvq-ouro-2p6b-thinking",
                     group="loopvq-tp2-accuracy", name="TP2 accuracy by benchmark and seed",
                     config={"tp":2,"dp":1,"model":"Ouro-2.6B-Thinking","seeds":[0,1,2],
                             "temperature":1.0,"top_p":0.7,"max_new_tokens":32768,"high_precision":[64,256]})
    atomic_json(ROOT/"wandb.json",{"url":run.url})
    completed = {}
    while True:
        for method in METHODS:
            for bench,n in COUNTS.items():
                for seed in range(3):
                    key=(method,bench,seed)
                    done=ROOT/"claims"/method/f"{bench}-{seed}"/"done.json"
                    if key in completed or not done.exists():
                        continue
                    marker=Path(json.loads(done.read_text())["marker"])
                    recipe=json.loads((marker.parents[3]/"recipe.json").read_text())
                    assert recipe["server"]["config"]["tp_size"]==2 and recipe["server"]["method"]==method
                    checked=collect_cell(marker.parent,f"{method}_{seed}_{bench}",bench,n)
                    assert checked==json.loads(marker.read_text())["scores"]
                    completed[key]=checked["accuracy_percent"]
                    run.summary[f"{method}/{bench}/seed{seed}"]=completed[key]
        rows=[]
        for bench,n in COUNTS.items():
            for method in METHODS:
                scores=[completed.get((method,bench,s)) for s in range(3)]
                full=all(s is not None for s in scores)
                rows.append(dict(benchmark=bench,method=method,n_per_seed=n,seed0=scores[0],seed1=scores[1],
                                 seed2=scores[2],mean_percent=statistics.mean(scores) if full else None,
                                 sd_percent=statistics.stdev(scores) if full else None))
        atomic_json(ROOT/"accuracy.json",{"complete":len(completed)==30,"scored_cells":len(completed),"rows":rows})
        temp=ROOT/"accuracy.csv.partial"
        with temp.open("w", newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temp.replace(ROOT/"accuracy.csv")
        if len(completed)==30:
            cols=list(rows[0])
            run.log({"TP2 accuracy":wandb.Table(columns=cols,data=[[r[c] for c in cols] for r in rows])})
            run.finish()
            return
        time.sleep(60)


if __name__=="__main__":
    main()
