"""40K oracle/graph boundary checks and exact retained-32K reader comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

OLD = Path(__file__).resolve().parents[1] / '2026-11-26-vqplus-residual'
sys.path.insert(0, str(OLD))
import torch
import check_residual as ref
from check_paged import packed_pool


def bounded_oracle(q, pool, ptr, loc, rope, return_lse=False):
    # Keep sanitizer reference temporaries bounded without changing the tested
    # full-batch kernel, request data, oracle mathematics or assertions.
    outputs, lses = [], []
    for b in range(q.shape[0]):
        lo, hi = int(ptr[b]), int(ptr[b + 1])
        p = ptr[b:b+2] - lo
        out, lse = ref.oracle(q[b:b+1], pool, p, loc[lo:hi], rope, return_lse=True)
        outputs.append(out[0]); lses.append(lse[0])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return (torch.stack(outputs), torch.stack(lses)) if return_lse else torch.stack(outputs)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--write-reference', type=Path)
    ap.add_argument('--read-reference', type=Path)
    args = ap.parse_args()
    assert not (args.write_reference and args.read_reference)
    torch.manual_seed(20261127)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = ref.make_pool(cap=40968, direct=True)
    pool = packed_pool(original, 2)
    old = ref.ResidualAttention(pool, max_batch=16, max_context_len=32768,
                               global_table=True, layout=2)
    new = ref.ResidualAttention(pool, max_batch=16, max_context_len=40960,
                               global_table=True, layout=2)
    assert new.payload_capacity == old.payload_capacity == 324
    theta = torch.arange(40960, device='cuda').float()[:, None] / (10000 ** (
        torch.arange(64, device='cuda').float()[None] / 64))
    rope = torch.cat((theta.cos(), theta.sin()), -1).contiguous()
    def digest(*tensors):
        h = hashlib.sha256()
        for tensor in tensors:
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            h.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()
    identity = {'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pool_sha256': digest(*[t for kind in ('k', 'v') for t in (
            getattr(original, kind + '_book').cb16, getattr(original, kind + '_indices'),
            getattr(original, kind + '_residual'), getattr(original, kind + '_residual_scale'),
            getattr(original, 'hp_' + kind))], rope)}
    cache = torch.load(args.read_reference, map_location='cpu', weights_only=True) if args.read_reference else {'identity': identity, 'cases': {}}
    assert cache['identity'] == identity
    def expected_for(label, query, ptr, loc):
        fingerprint = digest(query, ptr, loc[:int(ptr[-1])])
        if args.read_reference:
            row = cache['cases'][label]; assert row['inputs_sha256'] == fingerprint, label
            return row['out'].cuda(), row['lse'].cuda()
        out, lse = bounded_oracle(query, original, ptr, loc, rope, return_lse=True)
        if args.write_reference:
            cache['cases'][label] = {'inputs_sha256': fingerprint, 'out': out.cpu(), 'lse': lse.cpu()}
        return out, lse
    rows = []
    for batch in (1, 2, 3, 4, 8, 16):
        q = torch.randn(batch, 24, 128, device='cuda').bfloat16()[:, :8]
        for n in (0, 79, 1024, 2080, 8192, 32768, 32769, 32775, 36864, 40960):
            ptr, loc = ref.locations([n if r % 2 == 0 else n // 3 for r in range(batch)],
                                     pool.hp_global_offset, 'page8')
            new.partial.fill_(float('nan')); new.lse.fill_(float('nan'))
            new.group_partial.fill_(float('nan')); new.group_lse.fill_(float('nan'))
            expected, lse = expected_for(f'oracle-b{batch}-n{n}', q, ptr, loc)
            got = new.forward(q, 0, ptr, loc, rope, 128 ** -.5)
            torch.testing.assert_close(got, expected, atol=.002, rtol=.01)
            scratch = new.group_lse if new.use_grouped(batch) else new.lse
            torch.testing.assert_close(scratch[:batch].logsumexp(-1), lse, atol=2e-5, rtol=2e-5)
            if n <= 32768:
                before = old.forward(q, 0, ptr, loc, rope, 128 ** -.5)
                torch.testing.assert_close(got, before, atol=0, rtol=0)
            rows.append({'batch': batch, 'length': n,
                         'max_abs': (got.float() - expected.float()).abs().max().item(),
                         'old_range_bit_exact': n <= 32768})
        print('oracle passed batch', batch, flush=True)
    batch = 16
    q = torch.randn(batch, 8, 128, device='cuda').bfloat16()
    ptr, loc = ref.locations([40960] * batch, pool.hp_global_offset, 'page8')
    for _ in range(3): new.forward(q, 0, ptr, loc, rope, 128 ** -.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): got = new.forward(q, 0, ptr, loc, rope, 128 ** -.5)
    for n in (32760, 32768, 32769, 40960, 1024, 0):
        p, l = ref.locations([n if r % 2 == 0 else n // 2 for r in range(batch)],
                             pool.hp_global_offset, 'page8')
        ptr.copy_(p); loc[:len(l)].copy_(l)
        new.partial.fill_(float('nan')); new.lse.fill_(float('nan'))
        graph.replay()
        expected, _ = expected_for(f'graph-n{n}', q, ptr, loc)
        torch.testing.assert_close(got, expected, atol=.002, rtol=.01)
    loc.zero_(); graph.replay(); assert not got.any()
    graph.reset()
    args.output.write_text(json.dumps({'passed': True, 'capacity': new.payload_capacity,
        'cases': rows, 'graph_boundary_and_padding': True}, indent=2) + '\n')
    if args.write_reference: torch.save(cache, args.write_reference)
    print('40K context passed', flush=True)


if __name__ == '__main__':
    main()
