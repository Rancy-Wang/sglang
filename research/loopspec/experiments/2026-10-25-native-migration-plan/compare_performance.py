"""Compare identical standard HTTP workloads; require paired server settings."""

import argparse
import json
from pathlib import Path
import statistics


def config(command):
    options = dict.fromkeys(
        ("--disable-cuda-graph", "--disable-radix-cache", "--disable-overlap-schedule"),
        False,
    )
    i = 3
    while i < len(command):
        key = command[i]
        i += 1
        values = []
        while i < len(command) and not command[i].startswith("--"):
            values.append(command[i])
            i += 1
        options[key] = values or True
    for key in ("--port", "--speculative-algorithm", "--speculative-draft-depth"):
        options.pop(key, None)
    return options


def summarize(record):
    assert record["warmup"] >= 2 and len(record["samples"]) >= 10
    samples = record["samples"]
    rows = {
        "client_e2e_seconds": [],
        "service_ttft_seconds": [],
        "service_decode_tokens_per_second": [],
    }
    for sample in samples:
        meta = sample["response"]["meta_info"]
        rows["client_e2e_seconds"].append(sample["generation_seconds"])
        rows["service_ttft_seconds"].append(meta["first_token_latency"])
        rows["service_decode_tokens_per_second"].append(meta["decode_throughput"])
    return {
        key: {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for key, values in rows.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in args.results]
    expected = records[0]["samples"][0]
    for record in records:
        assert config(record["command"]) == config(records[0]["command"]), (
            "Unpaired server settings"
        )
        for sample in record["samples"]:
            assert sample["output_ids"] == expected["output_ids"]
            assert sample["finish_reason"] == expected["finish_reason"]
    summaries = {
        str(path): summarize(record) for path, record in zip(args.results, records)
    }
    baseline = summaries[str(args.results[0])]["client_e2e_seconds"]["median"]
    for summary in summaries.values():
        summary["e2e_ratio_to_baseline"] = (
            baseline / summary["client_e2e_seconds"]["median"]
        )
    args.output.write_text(
        json.dumps(
            {
                "tokens": len(expected["output_ids"]),
                "results": summaries,
                "limitation": "Sequential blocks on a single short request; no throughput or general-workload speedup claim. TTFT is not isolated prefill GPU time.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
