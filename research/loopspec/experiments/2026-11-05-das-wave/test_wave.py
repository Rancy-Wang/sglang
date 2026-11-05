import importlib.util
import itertools
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]
PKG = ROOT / 'python/sglang/srt/speculative/das_wave'
def load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

verify = load('state').verify_prefix
widths = load('config').wave_widths

class WaveContract(unittest.TestCase):
    def test_all_accept_has_no_bonus(self):
        self.assertEqual(verify([3,4,5], [3,4,5], 9), ([3,4,5],3,3))

    def test_every_rejection_frontier(self):
        for w in range(1,9):
            for rejected in range(w):
                proposal = list(range(w))
                target = proposal.copy()
                target[rejected] = 99
                self.assertEqual(verify(proposal,target,100),
                                 (list(range(rejected))+[99], rejected,rejected+1))

    def test_stop_and_remaining(self):
        self.assertEqual(verify([3,4,5], [3,4,5], 2), ([3,4],2,2))
        self.assertEqual(verify([3,4,5], [3,4,5], 9,{4}), ([3,4],2,2))
        self.assertEqual(verify([3,4,5], [3,9,5], 9,{9}), ([3,9],1,2))

    def test_ar_induction_with_arbitrary_proposals(self):
        # Deterministic history-dependent target. Verify every sequence of
        # binary proposals at multiple widths, including rejected suffix KV.
        def target(prefix): return (sum(prefix) + len(prefix)) % 3
        for w in (1,2,3,4):
            for proposal in itertools.product(range(2),repeat=w):
                prefix=[2,1]
                expected=[]
                for _ in range(10): expected.append(target(prefix+expected))
                emitted=[]
                while len(emitted)<10:
                    predictions=[target(prefix+emitted+list(proposal[:j])) for j in range(w)]
                    block,_,_=verify(list(proposal),predictions,10-len(emitted))
                    emitted+=block
                self.assertEqual(emitted,expected)

    def test_width_schedule(self):
        self.assertEqual(widths(16,4,2),(2,3,4,5))
        self.assertEqual(widths(4,6,2),(2,3,4,4,4,4))
        for args in [(0,1,1),(2,0,1),(1,2,2)]:
            with self.assertRaises(ValueError): widths(*args)

if __name__=='__main__':unittest.main()
