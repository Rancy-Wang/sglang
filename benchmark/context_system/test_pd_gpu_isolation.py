"""Distinguish NVLink peer contexts from foreign compute processes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from run_pd_matrix import GPUIsolationGuard


class Isolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.guard = GPUIsolationGuard.__new__(GPUIsolationGuard)
        self.guard.root = Path(self.tmp.name)
        self.guard.baseline = {str(i): {9} for i in range(4)}
        self.guard.last_check = 0
        self.guard.worker_pids = None
        self.clients = {str(i): {9, 100 + i} for i in range(4)}
        self.guard.clients = Mock(side_effect=lambda: self.clients)

    def test_peer_context_is_same_worker(self):
        self.guard.check(pin=True)
        self.clients['0'].add(101)
        self.clients['2'].add(103)
        self.guard.check(force=True)
        self.assertEqual(self.guard.worker_pids, {100, 101, 102, 103})
        self.assertFalse((self.guard.root / 'comparison-invalid.json').exists())

    def test_foreign_process_after_pin_is_rejected(self):
        self.guard.check(pin=True)
        self.clients['0'].add(200)
        with self.assertRaisesRegex(RuntimeError, 'co-location'):
            self.guard.check(force=True)

    def test_foreign_replacing_exited_worker_is_rejected(self):
        self.guard.check(pin=True)
        self.clients['0'] = {9, 200}
        with self.assertRaisesRegex(RuntimeError, 'co-location'):
            self.guard.check(force=True)

    def test_incomplete_or_contaminated_startup_cannot_pin(self):
        self.clients['0'] = {9}
        with self.assertRaises(RuntimeError):
            self.guard.check(pin=True)
        self.clients['0'] = {9, 100, 200}
        with self.assertRaises(RuntimeError):
            self.guard.check(pin=True)
        self.assertIsNone(self.guard.worker_pids)


if __name__ == '__main__':
    unittest.main()
