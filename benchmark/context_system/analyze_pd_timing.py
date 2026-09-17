"""Correlate same-host P/D stages and client SSE using existing benchmark logs.

Input is a run_minimal.py output directory. No model execution or inference
hooks are used. Run after the workload ends; active log files may be incomplete.
This reports CPU observations and coarse decode progress, never GPU token times.
"""

import argparse
import datetime
import json
import math
import re
import statistics
from itertools import pairwise
from pathlib import Path

STAGES = {
    "p_bootstrap": (
        "prefill",
        "prefill_bootstrap_queue_entry_time",
        "bootstrap_done_time",
    ),
    "p_ready_wait": ("prefill", "wait_queue_entry_time", "forward_entry_time"),
    "p_forward_including_chunk_gaps": (
        "prefill",
        "forward_entry_time",
        "prefill_finished_time",
    ),
    "p_final_transfer_tail": (
        "prefill",
        "prefill_transfer_queue_entry_time",
        "completion_time",
    ),
    "d_prealloc": (
        "decode",
        "decode_prealloc_queue_entry_time",
        "decode_transfer_queue_entry_time",
    ),
    "d_transfer_wait_including_p": (
        "decode",
        "decode_transfer_queue_entry_time",
        "wait_queue_entry_time",
    ),
    "d_ready_wait": ("decode", "wait_queue_entry_time", "forward_entry_time"),
    "d_forward_to_completion": ("decode", "forward_entry_time", "completion_time"),
}


def summary(values):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values:
        return {"n": 0, "mean": None, "p90": None, "max": None}
    i = 0.9 * (len(values) - 1)
    p90 = values[int(i)] + (values[math.ceil(i)] - values[int(i)]) * (i - int(i))
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p90": p90,
        "max": values[-1],
    }


def span(t, a, b):
    a, b = t.get(a), t.get(b)
    return (a, b) if a is not None and b is not None and 0 < a <= b else None


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union_overlap(window, intervals):
    clipped = sorted(
        (max(window[0], a), min(window[1], b))
        for a, b in intervals
        if overlap(window, (a, b))
    )
    total, end = 0.0, window[0]
    for a, b in clipped:
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def analyze(root, gap_threshold):
    records, duplicate_count, ignored_logs = {}, 0, 0
    transfer_by_room = {}
    decode_samples, clock_offsets = [], []
    for mode in ("prefill", "decode"):
        for number, line in enumerate(
            (root / f"{mode}.log").read_text(errors="replace").splitlines(), 1
        ):
            stamp = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            wall = (
                datetime.datetime.strptime(stamp[1], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                if stamp
                else None
            )
            if mode == "decode" and "Decode batch," in line and wall is not None:
                decode_samples.append(wall)
            if "r2_pd_timing=" not in line:
                continue
            m = re.search(r"ReqTimeStats\((.*?)\):", line)
            if not m:
                raise ValueError((mode, number, "missing request prefix"))
            fields = dict(re.findall(r"(\w+)=([^, )]+)", m[1]))
            if fields.get("bootstrap_room") in (None, "None"):
                ignored_logs += 1
                continue
            rank = re.search(r"\bTP(\d+)\]", line)
            key = (str(fields["bootstrap_room"]), fields["type"])
            row = {
                "fields": fields,
                "time": json.loads(line.split("r2_pd_timing=", 1)[1]),
                "rank": int(rank[1]) if rank else None,
                "line": number,
            }
            if (
                mode == "decode"
                and wall is not None
                and row["time"].get("completion_time")
            ):
                clock_offsets.append(wall - row["time"]["completion_time"])
            size = re.search(r"transfer_total=([0-9.]+) MB", line)
            if size and mode == "prefill":
                transfer_by_room.setdefault(key[0], {})[row["rank"]] = float(size[1])
            if key in records:
                duplicate_count += 1
                old = records[key]
                if old["rank"] == 0 or row["rank"] != 0:
                    continue
            records[key] = row
    rows = [
        json.loads(s) for s in (root / "workload/events.jsonl").read_text().splitlines()
    ]
    raw_turns = [r for r in rows if r.get("kind") == "turn_end"]
    # The benchmark cancels an unfinished filler at the measurement cutoff.
    # Its partial response has no final server metrics and is not a failure.
    cutoffs = [r for r in raw_turns if r.get("status") == "cutoff_cancelled"]
    if any(not r.get("filler") for r in cutoffs):
        raise ValueError("A first-pass task was cancelled at the measurement cutoff")
    turns = [r for r in raw_turns if r.get("status") != "cutoff_cancelled"]
    out, missing_room, unmatched = [], [], []
    for r in turns:
        identity = {
            k: r.get(k)
            for k in ("case_id", "trial", "turn", "instance", "filler", "success")
        }
        room = (r.get("server_metrics") or {}).get("pd_bootstrap_room")
        if room is None:
            missing_room.append(identity)
            continue
        room = str(room)
        pair = {mode: records.get((room, mode)) for mode in ("prefill", "decode")}
        if not all(pair.values()):
            unmatched.append(
                dict(
                    identity,
                    room=room,
                    missing=[k for k, v in pair.items() if v is None],
                )
            )
        item = dict(
            identity,
            room=room,
            transfer_reported_mib_by_rank=transfer_by_room.get(room, {}),
            stage_s={},
            stage_intervals={},
            log_lines={},
            ttft_s=r.get("ttft"),
            latency_s=r.get("latency"),
            output_tokens=r.get("output_len"),
            tbt_s=r.get("tbt_s"),
            long_sse_gaps=[],
        )
        for name, (mode, a, b) in STAGES.items():
            times = pair[mode]["time"] if pair[mode] else {}
            bounds = span(times, a, b)
            item["stage_intervals"][name] = bounds
            item["stage_s"][name] = bounds[1] - bounds[0] if bounds else None
        for mode, v in pair.items():
            if v:
                item["log_lines"][mode] = v["line"]
        if pair["prefill"] and pair["decode"]:
            p = pair["prefill"]["time"].get("prefill_finished_time")
            d = pair["decode"]["time"].get("wait_queue_entry_time")
            item["stage_s"]["p_finished_to_d_ready"] = d - p if p and d else None
        chunks = r.get("chunk_times") or []
        start = r["start_time"]
        item["client_content_span"] = (
            [start + chunks[0], start + chunks[-1]] if chunks else None
        )
        item["client_tail_after_last_content_s"] = (
            r["latency"] - chunks[-1] if chunks else None
        )
        for a, b in pairwise(chunks):
            if b - a >= gap_threshold:
                item["long_sse_gaps"].append(
                    {"start": start + a, "end": start + b, "duration_s": b - a}
                )
        out.append(item)
    all_p = [
        v["stage_intervals"]["p_forward_including_chunk_gaps"]
        for v in out
        if v["stage_intervals"]["p_forward_including_chunk_gaps"]
    ]
    for item in out:
        for gap in item["long_sse_gaps"]:
            window = (gap["start"], gap["end"])
            gap["overlap_any_formal_p_forward_s"] = union_overlap(window, all_p)
            gap["overlap_own_d_forward_s"] = (
                overlap(window, item["stage_intervals"]["d_forward_to_completion"])
                if item["stage_intervals"]["d_forward_to_completion"]
                else None
            )
    # Request completion anchors align existing second-resolution log stamps.
    # Treat naive log stamps as UTC: any fixed host timezone offset cancels
    # through the completion anchors, without depending on the analyst's TZ.
    # Logs occur every 40 decode forwards by the recorded default, not per token.
    # Only use fully observed formal D intervals, excluding time awaiting P.
    offset = statistics.median(clock_offsets) if clock_offsets else None
    samples = [v - offset for v in decode_samples] if offset is not None else []
    intervals = sorted(
        v["stage_intervals"]["d_forward_to_completion"]
        for v in out
        if v["stage_intervals"]["d_forward_to_completion"]
    )
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    active_gaps = []
    for a, b in merged:
        bounds = [a] + sorted(v for v in samples if a < v < b) + [b]
        active_gaps.extend(y - x for x, y in pairwise(bounds))
    for item in out:
        for gap in item["long_sse_gaps"]:
            gap["decode_log_samples_inside_with_1s_margin"] = sum(
                gap["start"] + 1 < t < gap["end"] - 1 for t in samples
            )
    stages = sorted({k for v in out for k in v["stage_s"]})
    return {
        "root": str(root),
        "raw_turn_end_events": len(raw_turns),
        "cutoff_cancelled_fillers": [
            {k: r.get(k) for k in ("case_id", "trial", "turn", "instance", "filler")}
            for r in cutoffs
        ],
        "turns": len(turns),
        "successful_turns": sum(bool(r.get("success")) for r in turns),
        "correlated_turns": len(out),
        "missing_room": missing_room,
        "unmatched_server_records": unmatched,
        "duplicate_rank_records": duplicate_count,
        "ignored_logs_without_room": ignored_logs,
        "stage_s": {k: summary(v["stage_s"].get(k) for v in out) for k in stages},
        "transfer_reported_mib_summed_visible_ranks": summary(
            sum(v["transfer_reported_mib_by_rank"].values())
            for v in out
            if v["transfer_reported_mib_by_rank"]
        ),
        "decode_log_observation": {
            "samples": len(samples),
            "wall_minus_monotonic_median_s": offset,
            "clock_anchor_spread_s": max(clock_offsets) - min(clock_offsets)
            if clock_offsets
            else None,
            "active_interval_log_gap_s_approximate": summary(active_gaps),
        },
        "sse_gap_threshold_s": gap_threshold,
        "long_sse_gap_s": summary(
            g["duration_s"] for v in out for g in v["long_sse_gaps"]
        ),
        "client_tail_after_last_content_s": summary(
            v["client_tail_after_last_content_s"] for v in out
        ),
        "limitations": [
            "Same Linux host monotonic clocks only. Stage timestamps are CPU observations, not GPU durations.",
            "P forward includes inter-chunk waits; P final transfer tail excludes overlapped earlier transfers.",
            "D transfer wait includes waiting for P computation and is not network-only latency.",
            "Decode log gaps are coarse observations every 40 forwards with second-resolution timestamps; not server TBT, and cannot rule out shorter stalls.",
            "SSE chunks can contain multiple tokens or buffered parser output; gaps are not server TBT.",
            "P/SSE overlap is correlation, not proof of D starvation or its cause.",
            "Per-request stage seconds must not be summed as disjoint wall-clock time.",
            "Native transfer MB uses MiB rounded to 0.01 and is rank-local for GPT-OSS MHA/GQA; sum covers visible ranks only, excluding protocol overhead. Missing rank logs are not zero transfer.",
        ],
        "requests": out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gap-threshold", default=1.0, type=float)
    args = parser.parse_args()
    if args.gap_threshold <= 0 or not math.isfinite(args.gap_threshold):
        parser.error("--gap-threshold must be positive and finite")
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    result = analyze(args.root, args.gap_threshold)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "requests"},
            ensure_ascii=False,
            indent=2,
        )
    )
