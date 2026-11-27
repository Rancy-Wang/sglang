"""CPU invariants for scheduling; synthetic capacities are not GPU benchmarks."""
import ast
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[4]
    path = root / 'python/sglang/kernels/ops/loopvq/residual_attention_adaptive_cuda.py'
    module = ast.parse(path.read_text())
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('split_policy', 'geometry_for')]
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    policy, geometry = namespace['split_policy'], namespace['geometry_for']
    checked = 0
    for capacity in (84, 192, 324, 648, 1296):
        for heads in (8, 16):
            for batch in (1, 2, 3, 4, 8, 16, 32, 64, 128):
                schedule = [policy(batch, heads, capacity, n) for n in range(33)]
                assert all(max(1, n) <= value <= 32 for n, value in enumerate(schedule))
                for length in range(32769):
                    chunks = (length + 1023) // 1024
                    active = min(32, max(chunks, min(schedule[chunks], max(1, (length + 63) // 64))))
                    per = (length + active - 1) // active
                    assert 1 <= active <= 32 and 0 <= per <= 1024
                    assert per * active >= length
                    checked += 1
                capacities = {128: capacity * 2, 256: capacity}
                selected = geometry(batch, heads, capacities)
                if selected is None:
                    assert all(c > batch * heads * 32 for c in capacities.values())
                else:
                    assert capacities[selected] <= batch * heads * 32
    print(dict(passed=True, length_cases=checked, capacities='synthetic, not hardware measurements'))


if __name__ == '__main__':
    main()
