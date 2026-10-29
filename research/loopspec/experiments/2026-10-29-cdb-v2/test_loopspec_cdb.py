"""Independent oracle tests of one-loop progress, ownership and saved proposals."""
import importlib.util
from pathlib import Path
import unittest

import torch

from sglang.srt.speculative.loopspec.pipeline import Pipeline
from sglang.srt.speculative.loopspec.info import LoopSpecInput
from sglang.srt.speculative.loopspec.sampling import verify_tensor

path = Path(__file__).parent.parent / '2026-10-25-native-migration-plan/test_loopspec.py'
spec = importlib.util.spec_from_file_location('oracle', path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class TestCDB(unittest.TestCase):
    def test_interleaved_independent_owners_and_partial_loops(self):
        pipelines = []
        for first, second, origin, depth in [(1, None, 1, (1,)), (2, 1, 2, (1, 2)), (2, 3, None, (1, 2))]:
            ex = oracle.FakeExecutor(first, second, origin)
            pipelines.append(Pipeline(ex, 1, (), 19, 4, depth))
        emitted = [1, 1, 1]
        partial = 0
        # Round-robin changes packing order without changing request ownership.
        for turn in range(100):
            for index in [(turn + k) % 3 for k in range(3)]:
                p = pipelines[index]
                if not p.lanes:
                    continue
                lanes = p.plan()
                for lane in lanes:
                    p.executor.advance(lane, 1)
                result = p.collect(lanes)
                if result is None:
                    partial += 1
                else:
                    self.assertEqual(result[0], 1)
                    emitted[index] += 1
            if emitted == [19, 19, 19]:
                break
        self.assertGreater(partial, 0)
        self.assertEqual(emitted, [19, 19, 19])
        for p in pipelines:
            self.assertFalse(p.executor.slots)
            self.assertEqual(len(p.executor.committed), 18)

    def test_filter_merge_relay_keeps_request_identity(self):
        a = LoopSpecInput(torch.tensor([11, 22]), torch.tensor([31, 42]))
        a.future_indices = torch.tensor([8, 3])
        a.filter_batch(torch.tensor([1, 0]))
        b = LoopSpecInput(torch.tensor([33]), torch.tensor([53]))
        b.future_indices = torch.tensor([7])
        a.merge_batch(b)
        self.assertEqual(a.bonus_tokens.tolist(), [22, 11, 33])
        self.assertEqual(a.new_seq_lens.tolist(), [42, 31, 53])
        self.assertEqual(a.future_indices.tolist(), [3, 8, 7])

    def test_batched_residual_distribution(self):
        torch.manual_seed(11907)
        n = 100000
        for p, q, d in [([.05,.15,.8],[.8,.1,.1],[.1,.8,.1]),
                         ([0,.4,.6],[.9,.1,0],[.1,.2,.7]),
                         ([.2,.3,.5],[.2,.3,.5],[.2,.3,.5])]:
            target, first, deeper = [torch.tensor(x).expand(n,-1) for x in (p,q,d)]
            x = torch.multinomial(first,1).squeeze(-1)
            delta = (deeper-first).clamp_min(0).clone()
            delta.scatter_(1,x[:,None],0)
            mass = delta.sum(-1,keepdim=True)
            enabled = (deeper.gather(1,x[:,None]) < first.gather(1,x[:,None])) & (mass>0)
            second = torch.where(enabled, delta/mass.clamp_min(1e-30), deeper)
            y = torch.multinomial(second,1).squeeze(-1)
            result = verify_tensor(target,first,x,second,y,enabled[:,0],torch.rand(n,3))
            counts = torch.bincount(result[:,0],minlength=3)/n
            reference = torch.tensor(p)
            tolerance = torch.maximum(5*(reference*(1-reference)/n).sqrt(),torch.tensor(5/n))
            self.assertTrue(torch.all((counts-reference).abs() <= tolerance), (counts,reference))

if __name__ == '__main__':
    unittest.main()
