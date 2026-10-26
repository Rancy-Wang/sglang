"""CPU distribution/oracle tests and CUDA replay tests without serving imports."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]


def module(name):
    path = ROOT / "python/sglang/srt/speculative/loopspec" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


sampling = module("sampling")


def oracle(p, q, x, q2, y, enabled, u):
    if q[x] > 0 and u[0] < min(1, p[x] / q[x]):
        return [x, 0]
    r = sampling.residual(p, q)
    if enabled:
        if q2[y] > 0 and u[1] < min(1, r[y] / q2[y]):
            return [y, 1]
        r = sampling.residual(r, q2)
    cdf = r.cumsum(0)
    threshold = min(u[2] * cdf[-1], torch.nextafter(cdf[-1], torch.zeros(())))
    return [int(torch.searchsorted(cdf, threshold, right=True)), -1]


class TestRejection(unittest.TestCase):
    def test_fixed_variates_against_sequential_oracle(self):
        p = torch.tensor([0, 0.2, 0.3, 0.5, 0.0])
        q = torch.tensor([0.5, 0.4, 0.1, 0.0, 0.0])
        q2 = torch.tensor([0.0, 0.6, 0.4, 0.0, 0.0])
        seen = set()
        for x in range(3):
            for y in (1, 2):
                for enabled in (False, True):
                    for uniforms in (
                        [0.0, 0.0, 0.0],
                        [0.99, 0.99, 0.99999994],
                        [0.1, 0.8, 0.4],
                    ):
                        u = torch.tensor(uniforms)
                        actual = sampling.verify_tensor(
                            p,
                            q,
                            torch.tensor(x),
                            q2,
                            torch.tensor(y),
                            torch.tensor(enabled),
                            u,
                        ).tolist()
                        self.assertEqual(actual, oracle(p, q, x, q2, y, enabled, u))
                        self.assertGreater(p[actual[0]], 0)
                        seen.add(actual[1])
        self.assertEqual(seen, {-1, 0, 1})

    def test_target_distribution_with_conditional_second(self):
        g = torch.Generator().manual_seed(11907)
        n = 100_000
        for pt, qt, dt in [
            ([0.05, 0.15, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),
            ([0.0, 0.4, 0.6], [0.9, 0.1, 0.0], [0.1, 0.2, 0.7]),
            ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ]:
            p, q, d = [torch.tensor(x) for x in (pt, qt, dt)]
            x = torch.multinomial(q, n, replacement=True, generator=g)
            q2 = torch.zeros(n, 3)
            y = torch.zeros(n, dtype=torch.int64)
            enabled = torch.zeros(n, dtype=torch.bool)
            for first_token in range(3):
                proposal = sampling.second_proposal(q, d, first_token)
                if proposal is not None:
                    mask = x == first_token
                    q2[mask] = proposal
                    y[mask] = torch.multinomial(
                        proposal, int(mask.sum()), replacement=True, generator=g
                    )
                    enabled[mask] = True
            out = sampling.verify_tensor(
                p.expand(n, -1),
                q.expand(n, -1),
                x,
                q2,
                y,
                enabled,
                torch.rand(n, 3, generator=g),
            )[:, 0]
            counts = torch.bincount(out, minlength=3)
            tolerance = torch.maximum(
                5 * (p * (1 - p) / n).sqrt(), torch.full_like(p, 5 / n)
            )
            self.assertTrue(bool(((counts / n - p).abs() <= tolerance).all()))
            self.assertTrue(bool((counts[p == 0] == 0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph requires GPU")
    def test_cuda_replay_mutated_inputs(self):
        graph = sampling.RejectionGraph(5, "cuda")
        for p, q, q2 in [
            (
                [0.0, 0.2, 0.3, 0.5, 0.0],
                [0.5, 0.4, 0.1, 0.0, 0.0],
                [0.0, 0.6, 0.4, 0.0, 0.0],
            ),
            (
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ]:
            p, q, q2 = [torch.tensor(x, device="cuda") for x in (p, q, q2)]
            for u in ([0.1, 0.1, 0.4], [0.99, 0.99, 0.99]):
                u = torch.tensor(u, device="cuda")
                for enabled in (True, False):
                    out = graph.replay(p, q, 1, q2 if enabled else None, 2, uniforms=u)
                    expected = sampling.verify_tensor(
                        p,
                        q,
                        torch.tensor(1, device="cuda"),
                        q2,
                        torch.tensor(2, device="cuda"),
                        torch.tensor(enabled, device="cuda"),
                        u,
                    ).tolist()
                    self.assertEqual(
                        out, (expected[0], None if expected[1] < 0 else expected[1])
                    )

    def test_metrics_denominators_and_roles(self):
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        metrics = module("metrics").LoopSpecMetrics("fixture", registry=registry)
        executor = SimpleNamespace(
            accepted=[7, 2], rejected=1, proposed=[30, 12], peak_slots=40
        )
        pipeline = SimpleNamespace(cycles=18, reused=9, restarts=1)
        metrics.finish(executor, pipeline, {"core": 18, "rejection": 10}, True)

        def event(name):
            return registry.get_sample_value(
                "sglang:loopspec_events_total", {"model_name": "fixture", "event": name}
            )

        self.assertEqual(event("verified_tokens"), 10)
        self.assertEqual(event("proposed_first"), 30)
        self.assertEqual(
            event("accepted_first")
            + event("accepted_second")
            + event("residual_tokens"),
            event("verified_tokens"),
        )


if __name__ == "__main__":
    unittest.main()
