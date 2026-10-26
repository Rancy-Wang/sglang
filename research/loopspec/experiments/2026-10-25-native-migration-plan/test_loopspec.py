import unittest

import torch

from sglang.srt.speculative.loopspec.config import validate_depths
from sglang.srt.speculative.loopspec.pipeline import Pipeline
from sglang.srt.speculative.loopspec.sampling import second_proposal, verify


class FakeExecutor:
    """AR oracle predicts token 1; shallow heads can deliberately disagree."""

    def __init__(self, first, second, accepted):
        self.first, self.second, self.accepted = first, second, accepted
        self.slots, self.committed, self.next_slot = set(), set(), 0

    def advance(self, lane, count):
        if lane.slot is None:
            lane.slot = self.next_slot
            self.next_slot += 1
            self.slots.add(lane.slot)
        for slot in lane.prefix:
            assert slot in self.slots or slot in self.committed
        lane.depth += count

    def propose(self, lane):
        token = self.first if lane.depth == 1 else self.second
        return None if token is None else (token, None)

    def verify(self, lane):
        return 1, self.accepted, None

    def commit(self, lane):
        self.slots.remove(lane.slot)
        self.committed.add(lane.slot)

    def retain(self, lanes):
        retained = set()
        for lane in lanes:
            retained.update(lane.prefix)
            if lane.slot is not None:
                retained.add(lane.slot)
        self.slots &= retained


class TestPipeline(unittest.TestCase):
    def test_accept_first_second_and_reject_all(self):
        for first, second, accepted in [(1, None, 1), (2, 1, 2), (2, 3, None)]:
            with self.subTest(first=first, second=second):
                executor = FakeExecutor(first, second, accepted)
                pipeline = Pipeline(executor, 1, (), 12, 4, (1, 2))
                tokens = [1]
                for _ in range(11):
                    tokens.append(pipeline.advance_until_commit()[0])
                self.assertEqual(tokens, [1] * 12)
                self.assertFalse(executor.slots)
                self.assertEqual(len(executor.committed), 11)
                if accepted is not None:
                    self.assertGreater(pipeline.reused, 0)

    def test_reject_invalid_depths(self):
        for depths in [[], [0], [3], [4], [1, 1], [2, 3], [1, 4], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                validate_depths(depths, 4)
        self.assertEqual(validate_depths([1, 2], 4), (1, 2))


class TestSampling(unittest.TestCase):
    def test_gate_and_actual_second_distribution(self):
        q1 = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float64)
        deeper = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        torch.testing.assert_close(
            second_proposal(q1, deeper, 0),
            torch.tensor([0.0, 1 / 3, 2 / 3], dtype=torch.float64),
        )
        self.assertIsNone(second_proposal(q1, deeper, 1))

    def test_lossless_distribution(self):
        # Predeclared 100k draws per case; 5-sigma/bin or 5/N tolerance.
        cases = [
            ([0.05, 0.15, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),
            ([0.0, 0.4, 0.6], [0.9, 0.1, 0.0], [0.1, 0.2, 0.7]),
            ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ]
        generator = torch.Generator().manual_seed(11907)
        n = 100_000
        for target, first, deeper in cases:
            p, q, d = [
                torch.tensor(x, dtype=torch.float64) for x in (target, first, deeper)
            ]
            counts = torch.zeros(3, dtype=torch.float64)
            for _ in range(n):
                x = int(torch.multinomial(q, 1, generator=generator))
                second = second_proposal(q, d, x)
                y = (
                    None
                    if second is None
                    else int(torch.multinomial(second, 1, generator=generator))
                )
                token, _ = verify(p, q, x, second, y, generator)
                counts[token] += 1
            error = (counts / n - p).abs()
            tolerance = torch.maximum(
                5 * (p * (1 - p) / n).sqrt(), torch.full_like(p, 5 / n)
            )
            self.assertTrue(bool((error <= tolerance).all()), (counts / n, p))
            self.assertTrue(bool((counts[p == 0] == 0).all()))


if __name__ == "__main__":
    unittest.main()
