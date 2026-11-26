"""Archive actual NCU metrics and source-correlated stalls, version defensively."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    import ncu_report
    report = ncu_report.load_report(str(args.report))
    records = []
    for ri in range(report.num_ranges()):
        region = report.range_by_idx(ri)
        for ai in range(region.num_actions()):
            action = region.action_by_idx(ai)
            record = {"name": action.name(), "metrics": {}, "source_stalls": {}}
            for name in action.metric_names():
                metric = action[name]
                try:
                    record["metrics"][name] = {"value": metric.value(), "unit": metric.unit()}
                except Exception as exc:
                    record["metrics"][name] = {"error": str(exc)}
                if "pcsamp_warps_issue_stalled" not in name or not metric.has_correlation_ids():
                    continue
                totals = defaultdict(float)
                ids = metric.correlation_ids()
                for i in range(metric.num_instances()):
                    pc = ids.as_uint64(i)
                    source = action.source_info(pc)
                    key = (source.file_name(), source.line()) if source else ("unknown", 0)
                    totals[key] += metric.as_uint64(i)
                record["source_stalls"][name] = [
                    {"file": key[0], "line": key[1], "samples": value}
                    for key, value in sorted(totals.items(), key=lambda pair: -pair[1])]
            if hasattr(action, "rule_results_as_dicts"):
                record["rules"] = action.rule_results_as_dicts()
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"report": str(args.report), "actions": records}, indent=2, default=str) + "\n")
    for record in records:
        print(record["name"])
        for name in ("gpu__time_duration.sum", "launch__registers_per_thread",
                     "launch__waves_per_multiprocessor", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                     "dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum",
                     "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"):
            print(name, record["metrics"].get(name, "unavailable"))


if __name__ == "__main__":
    main()
