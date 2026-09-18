"""Join native P/D physical counters to the unchanged test_serving window."""

import argparse
import copy
import json
import re
from pathlib import Path

from test_serving import load_method


def read_counts(root, tp):
    counts = {}
    for mode in ("prefill", "decode"):
        for line in (root / f"{mode}.log").read_text(errors="replace").splitlines():
            if "pd_matrix_compute=" not in line:
                continue
            if "ReqTimeStats(rid=HEALTH_CHECK_" in line:
                continue
            room = re.search(r"bootstrap_room=(\d+)", line)
            if room is None:
                continue
            rank = re.search(r"\bTP(\d+)\]", line)
            rank = int(rank[1]) if rank else 0
            value = json.loads(line.split("pd_matrix_compute=", 1)[1].split(", r2_pd_timing=", 1)[0])
            key = (room[1], mode)
            old = counts.setdefault(key, {})
            if rank in old and old[rank] != value:
                raise ValueError(f"Conflicting counter records: {key} TP{rank}")
            old[rank] = value
    return counts


def join_counts(result, counts, tp):
    result = copy.deepcopy(result)
    for row in result["turns"]:
        if not row["success"]:
            continue
        room = str((row.get("server_metrics") or {}).get("pd_bootstrap_room"))
        total = {"pf": 0, "decode": 0}
        for mode in ("prefill", "decode"):
            ranks = counts.get((room, mode), {})
            if 0 not in ranks or not set(ranks).issubset(set(range(tp))):
                raise ValueError(f"Incomplete TP counters: room={room} mode={mode}: {ranks}")
            if any(value != ranks[0] for value in ranks.values()):
                raise ValueError(f"TP counts disagree: {room} {mode}: {ranks}")
            for k, v in ranks[0].items():
                if type(v) is not int or v < 0:
                    raise ValueError(f"Invalid physical count: {ranks}")
                total[k] += v
        if total["pf"] <= 0:
            raise ValueError(f"Expected at least one P forward: {room}")
        row.setdefault("server_metrics", {}).update(
            prefill_compute_tokens=total["pf"], decode_compute_tokens=total["decode"],
            context_stage_count=1,
        )
    return result


def summarize(root, mini_root, tp=2):
    method = load_method(mini_root)
    result = join_counts(json.loads((root / "workload/result.json").read_text()), read_counts(root, tp), tp)
    result["overall"] = method.summary(result["turns"], result["start"], result["cutoff"])
    result["measurement"] = {
        "physical_count_source": "CPU ScheduleBatch lengths at successful run_batch dispatch, both P/D",
        "tp_policy": "native TP0 logs count once; any additional visible ranks must agree",
        "prebuilt": "zero model tokens; first sampled token is produced by P",
        "profiler": (root / "profile-command.json").exists(),
        "original_result_preserved": "workload/result.json",
        "failed_and_cutoff_work": "not in successful numerator; elapsed time remains in denominator",
    }
    target = root / "counted-result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--tp", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(summarize(a.root, a.mini_root, a.tp)["overall"], ensure_ascii=False))
