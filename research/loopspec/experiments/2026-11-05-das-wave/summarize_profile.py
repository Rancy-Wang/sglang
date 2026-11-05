"""Inclusive intervals, direct-child exclusive estimates and kernel categories."""
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys


def stats(values):
    values=sorted(values)
    return {'n':len(values),'mean_ms':statistics.mean(values),'median_ms':statistics.median(values),
        'sum_ms':sum(values)} if values else {'n':0}


def category(name):
    name=name.lower()
    if any(s in name for s in ('attn','attention','flash_fwd','decode_stage','extend_attention')):return 'attention'
    if any(s in name for s in ('gemm','gemv','matmul','cutlass','cublas','xmma')):return 'gemm'
    if 'norm' in name:return 'normalization'
    if any(s in name for s in ('copy','memcpy','scatter','gather','cache','kv_')):return 'memory_index_cache'
    if any(s in name for s in ('sample','argmax','topk','softmax','accept')):return 'sampling_acceptance'
    return 'other'


def main():
    root=Path(sys.argv[1]); combined={}
    old=Path(__file__).resolve().parents[1]/'2026-10-29-thinking-batch-profile/analyze_profile.py'
    for trace in sorted(root.glob('*/q*-p*/trace')):
        directory=trace.parent
        subprocess.run([sys.executable,str(old),str(directory)],check=True)
        summary=json.loads((directory/'summary.json').read_text())
        exclusive=collections.defaultdict(list)
        for path in trace.glob('*-recurrent-*.jsonl'):
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            children=collections.defaultdict(list)
            for row in rows:children[row['parent']].append(row)
            for row in rows:
                for field in ('cpu_ms','gpu_stream_ms'):
                    if field in row:
                        exclusive[row['stage']+'.'+field].append(row[field]-sum(c.get(field,0) for c in children[row['id']]))
        groups=collections.defaultdict(float)
        for name,value in summary['kernels'].items():groups[category(name)]+=value['sum']
        total=sum(groups.values())
        compact={k:v for k,v in summary.items() if k not in ('kernels','server_requests','requests')}
        compact['exclusive_direct_children']={k:stats(v) for k,v in exclusive.items()}
        compact['kernel_categories']={k:{'ms':v,'fraction_summed_kernel_time':v/total if total else None} for k,v in groups.items()}
        compact['request']=json.loads((directory/'request.json').read_text())
        compact['request'].pop('output_ids');compact['request'].pop('events');compact['request'].pop('text')
        combined[str(directory.relative_to(root))]=compact
    (root/'profile-summary.json').write_text(json.dumps({'windows':combined,
        'note':'Profile includes each window prefill; stage paths distinguish decode. Inclusive nested spans cannot be added. Exclusive subtraction is an interval estimate, not kernel busy; negative values retained for diagnostics.'},indent=2)+'\n')

if __name__=='__main__':main()
