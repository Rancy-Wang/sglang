"""Cohort wall throughput and request acceptance, grouped by batch and repeat."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics


def ratio(a, b):
    return a / b if b else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    old = Path(__file__).parents[1]/'2026-11-05-das-wave/summarize_aime25.py'
    spec = importlib.util.spec_from_file_location('metrics', old)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    cells = []
    for run in sorted(a.directory.glob('bs*')):
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['args']['tp_size'] == 2
        prompts = json.loads((run/'prompts.json').read_text())
        for method_dir in sorted(run.glob('*-r*')):
            method, repeat = method_dir.name.rsplit('-r', 1)
            assert method in ('ar', 'dflash', 'loopspec1')
            status = json.loads((method_dir/'status.json').read_text())
            assert status['execution_complete'] and status['full30']
            rows = [json.loads(f.read_text()) for f in sorted(method_dir.glob('q*.json'))]
            assert len(rows) == len({r['id'] for r in rows}) == 30
            cohorts = json.loads((method_dir/'cohorts.json').read_text())
            assert sorted(i for c in cohorts for i in c['ids']) == sorted(r['id'] for r in rows)
            wall = sum(c['wall_ms'] for c in cohorts)
            tokens = sum(len(r['output_ids']) for r in rows)
            assert tokens == sum(c['output_tokens'] for c in cohorts)
            assert all(len(r['output_ids']) == r['meta_info']['completion_tokens'] for r in rows)
            log = (method_dir/'server.log').read_bytes()
            measured_log = '\n'.join(log[c['log_start']:c['log_end']].decode() for c in cohorts)
            peak = max(map(int, re.findall(r'#running-req: (\d+)', measured_log)), default=0)
            launch = json.loads((method_dir/'launch.json').read_text())['command']
            assert '--disable-cuda-graph' not in launch and launch[launch.index('--tp-size')+1] == '2'
            if method == 'loopspec1':
                assert launch[launch.index('--speculative-draft-depth')+1] == '1'
            cell = {'batch_size': manifest['args']['batch_size'], 'method': method, 'repeat': int(repeat),
                'sha': manifest['sha'], 'questions': len(rows), 'output_tokens': tokens,
                'wall_ms': wall, 'throughput_tokens_s': tokens * 1000 / wall,
                'mean_ttft_ms': statistics.mean(r['ttft_ms'] for r in rows),
                'weighted_request_tpot_ms': ratio(sum(r['decode_ms'] for r in rows), sum(r['decode_tokens'] for r in rows)),
                'observed_peak_running_requests': peak,
                'retractions': sum(r['meta_info']['num_retractions'] for r in rows),
                'cuda_graph_true_log_lines': measured_log.count('cuda graph: True'),
                'cuda_graph_false_log_lines': measured_log.count('cuda graph: False'),
                'accepted_over_tested': None, 'accepted_over_proposed': None, 'mean_acceptance_per_request': None,
                'accepted_per_verify': None}
            if method != 'ar':
                counts = [metrics.counters(method, row, log.decode()) for row in rows]
                accepted = sum(sum(c['accepted']) for c in counts)
                tested = sum(sum(c['tested']) for c in counts)
                proposed = sum(sum(c['proposed']) for c in counts)
                verifies = sum(c['verify_cycles'] for c in counts)
                cell.update(accepted=accepted, tested=tested, proposed=proposed, verify_cycles=verifies,
                    accepted_over_tested=ratio(accepted, tested), accepted_over_proposed=ratio(accepted, proposed),
                    mean_acceptance_per_request=statistics.mean(ratio(sum(c['accepted']), sum(c['tested'])) for c in counts),
                    accepted_per_verify=ratio(accepted, verifies))
            baseline_dir = run/f'ar-r{repeat}'
            matches = []
            for row in rows:
                baseline = json.loads((baseline_dir/f"q{row['id']}.json").read_text())
                equal = row['output_ids'] == baseline['output_ids']
                first = next((i for i, (x,y) in enumerate(zip(row['output_ids'], baseline['output_ids'])) if x != y), None)
                matches.append({'id': row['id'], 'ids_equal': equal, 'first_difference': first,
                                'stop_equal': row['meta_info']['finish_reason'] == baseline['meta_info']['finish_reason']})
            cell['ar_exact_matches'] = sum(m['ids_equal'] and m['stop_equal'] for m in matches)
            cell['ar_comparisons'] = matches
            cells.append(cell)
    for cell in cells:
        ar = next(c for c in cells if c['method'] == 'ar' and c['batch_size'] == cell['batch_size'] and c['repeat'] == cell['repeat'])
        cell['speedup_vs_ar'] = cell['throughput_tokens_s'] / ar['throughput_tokens_s']
    assert cells
    (a.directory/'matrix-summary.json').write_text(json.dumps({'cells': cells,
        'notes': ['Throughput uses concurrent cohort wall time, excluding between-cohort cache flush and warmup.',
                  'Request TPOT is aggregate per-request latency; it is not inverse multi-request throughput.',
                  'accepted/tested and accepted/proposed have different denominators; both are reported.',
                  'CUDA graph log checks cover measured cohorts only; detailed profiles verify execution separately.',
                  'Nominal batch is admission/concurrency; observed peak does not prove every step ran at that size.']}, indent=2))
    compact = [{k:v for k,v in c.items() if k != 'ar_comparisons'} for c in cells]
    fields = list(dict.fromkeys(k for c in compact for k in c))
    with (a.directory/'matrix-summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact)


if __name__ == '__main__':
    main()
