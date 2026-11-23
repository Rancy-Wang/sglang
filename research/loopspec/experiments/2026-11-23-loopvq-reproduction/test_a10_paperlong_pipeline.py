"""CPU guard checks; never launch, signal, or query a real service/GPU."""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paperlong_pipeline", HERE / "a10_paperlong_pipeline.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
bench = p.load_bench(HERE.parents[3])


def identity():
    return dict(pid=100, ppid=90, start_ticks=12345, uid=os.getuid(),
                exe="/env/bin/python", cwd="/frozen",
                argv=["/env/bin/python", "-u", "-m", "sglang.launch_server",
                      "--model-path", "/models/Ouro", "--port", "30555"])


class ProcessGuards(unittest.TestCase):
    def test_stat_handles_spaces_and_parentheses(self):
        fields = ["S", "90"] + ["0"] * 17 + ["12345", "0"]
        self.assertEqual(p.parse_stat("100 (name with ) spaces) " + " ".join(fields)),
                         dict(state="S", ppid=90, start_ticks=12345))

    def test_parent_requires_exact_python_model_module_and_port(self):
        p.validate_parent(identity(), "/env/bin/python", "/models/Ouro", "/frozen")
        for replace_from, replace_to in (("30555", "305550"), ("/models/Ouro", "/models/Other"),
                                          ("sglang.launch_server", "other.launch_server"),
                                          ("/env/bin/python", "/other/python")):
            candidate = identity()
            candidate["argv"] = [replace_to if x == replace_from else x for x in candidate["argv"]]
            with self.subTest(replace_from=replace_from), self.assertRaises(p.Refuse):
                p.validate_parent(candidate, "/env/bin/python", "/models/Ouro", "/frozen")

    def test_unknown_gpu_or_unrelated_scheduler_is_rejected(self):
        p.check_ownership(100, {100, 101}, {101}, {101})
        for gpu, schedulers in (({101, 999}, {101}), ({101}, {999}), (set(), {101})):
            with self.subTest(gpu=gpu), self.assertRaises(p.Refuse):
                p.check_ownership(100, {100, 101}, gpu, schedulers)

    def test_reused_pid_never_receives_signal(self):
        changed = dict(identity(), start_ticks=99999)
        with patch.object(p.os, "pidfd_open", return_value=77, create=True), \
             patch.object(p.os, "close"), \
             patch.object(p.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(p.Refuse, "identity changed"):
                p.verified_sigterm(identity(), Mock(), reader=lambda _: changed)
            send.assert_not_called()

    def test_guard_failure_never_receives_signal(self):
        with patch.object(p.os, "pidfd_open", return_value=77, create=True), \
             patch.object(p.os, "close"), \
             patch.object(p.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaisesRegex(p.Refuse, "another job"):
                p.verified_sigterm(identity(), Mock(side_effect=p.Refuse("another job")), reader=lambda _: identity())
            send.assert_not_called()

    def test_only_verified_parent_receives_sigterm(self):
        guard = Mock()
        with patch.object(p.os, "pidfd_open", return_value=77, create=True) as opened, \
             patch.object(p.os, "close") as closed, \
             patch.object(p.signal, "pidfd_send_signal", create=True) as send:
            p.verified_sigterm(identity(), guard, reader=lambda _: identity())
            opened.assert_called_once_with(100)
            guard.assert_called_once_with()
            send.assert_called_once_with(77, signal.SIGTERM)
            closed.assert_called_once_with(77)

    def test_idle_metrics_require_every_finite_gauge(self):
        text = "\n".join(f'sglang:{name}{{model="ouro"}} 0' for name in p.GAUGES)
        self.assertTrue(all(x == 0 for rows in p.idle_values(text).values() for x in rows))
        for invalid in ("\n".join(text.splitlines()[:-1]), text.replace(" 0", " NaN", 1)):
            with self.assertRaises(p.Refuse):
                p.idle_values(invalid)


class EvidenceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = [[n] * 2048 for n in range(16)]
        manifest = {key: "identity" for key in bench.IDENTITY_KEYS}
        manifest.update(method="bf16", config={}, server_pid=100)
        config = {"max_total_tokens": 40960, "ouro_decode_profile": "optimized", "triton_attention_num_kv_splits": 8}
        self.summary = dict(format_version=1, method="bf16", protocol="paper-long", bs=1,
                            input_len=2048, output_len=32768, server_manifest=manifest,
                            server_config_readback=config, base_url=p.URL, server_log="server.log",
                            client_host="a10", warmup_count=2, repeat_count=3, status="ok", attempts=[],
                            primary_metric="e2e_output_tokens_s", fixed_length_sampling={"temperature": 0.0, "ignore_eos": True},
                            input_ids_sha256=bench.digest(self.inputs[:1]), input_fixture_sha256=bench.digest(self.inputs),
                            e2e_output_tokens_s_median=32.0)
        for phase, numbers in (("warmup", (1, 2)), ("measure", (1, 2, 3))):
            for number in numbers:
                path = self.root / f"{phase}-{number}.json"
                p.new_json(path, dict(status="ok", output_tokens=32768, input_tokens=2048,
                                     e2e_seconds=1024.0, e2e_output_tokens_s=32.0,
                                     batch_evidence={"verified": True}, requests=[dict(
                                         completion_tokens=32768, prompt_tokens=2048, finish_reason={"type": "length"})]))
                self.summary["attempts"].append(dict(path=str(path), phase=phase, number=number, status="ok"))

    def test_summary_success_requires_all_full_raw_attempts(self):
        p.validate_complete(self.summary, 1, "bf16", self.inputs, bench)
        partial = copy.deepcopy(self.summary)
        partial["attempts"].pop()
        with self.assertRaisesRegex(p.Refuse, "repeat schedule"):
            p.validate_complete(partial, 1, "bf16", self.inputs, bench)
        path = self.summary["attempts"][0]["path"]
        row = p.read_json(path)
        row["requests"][0]["completion_tokens"] -= 1
        p.atomic_json(path, row)
        with self.assertRaisesRegex(p.Refuse, "full fixed length"):
            p.validate_complete(self.summary, 1, "bf16", self.inputs, bench)

    def test_capacity_row_is_unmeasured_and_pairs_to_na(self):
        skipped = p.capacity_summary(self.summary, 2, self.inputs, {"source": "actual-capacity"}, bench)
        quantized = copy.deepcopy(self.summary)
        quantized.update(bs=2, method="loopvq", input_ids_sha256=bench.digest(self.inputs[:2]))
        quantized["server_config_readback"]["triton_attention_num_kv_splits"] = 48
        result = bench.paired_row(skipped, quantized)
        self.assertEqual(skipped["requests_sent"], 0)
        self.assertEqual(skipped["attempts"], [])
        self.assertNotIn("e2e_output_tokens_s_median", skipped)
        self.assertIsNone(result["speedup"])
        self.assertIsNone(result["bf16_output_tokens_s"])
        self.assertEqual(result["speedup_status"], "N/A")
        with self.assertRaisesRegex(p.Refuse, "capacity proof"):
            p.capacity_summary(self.summary, 1, self.inputs, {}, bench)

    def test_exit_marker_failure_and_absence_are_distinct(self):
        path = self.root / "job.exit"
        self.assertFalse(p.exit_ready(path))
        path.write_text("1\n")
        with self.assertRaises(p.Refuse):
            p.exit_ready(path)
        path.write_text("0\n")
        self.assertTrue(p.exit_ready(path))

    def test_inspect_has_no_mutation_or_run_side_effects(self):
        paths = p.Paths(self.root / "absent-task", self.root / "absent-repo")
        with patch.object(p, "Paths", return_value=paths), patch.object(p, "run_locked") as run, \
             patch.object(p.sys, "argv", ["pipeline", "--inspect"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(), 0)
        run.assert_not_called()
        self.assertFalse(paths.task.exists())
        self.assertFalse(paths.repo.exists())

    def test_incomplete_restart_preserves_existing_state(self):
        paths = p.Paths(self.root / "task", self.root / "repo")
        p.new_json(paths.state, {"status": "running", "phase": "measuring_vq_bs1"})
        before = paths.state.read_bytes()
        with patch.dict(os.environ, {"TMUX": "test", "TMUX_PANE": "%test"}), \
             patch.object(p.Pipeline, "execute") as execute:
            with self.assertRaisesRegex(p.Refuse, "Incomplete prior"):
                p.run_locked(paths, 20)
        execute.assert_not_called()
        self.assertEqual(paths.state.read_bytes(), before)
        self.assertEqual(len(list(paths.directory.glob("refused-restart-*.json"))), 1)

    def test_lock_prevents_duplicate_execution(self):
        paths = p.Paths(self.root / "task", self.root / "repo")
        paths.directory.mkdir(parents=True)
        with (paths.directory / "pipeline.lock").open("w") as held, \
             patch.dict(os.environ, {"TMUX": "test", "TMUX_PANE": "%test"}), \
             patch.object(p.Pipeline, "execute") as execute:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(p.Refuse, "execution lock"):
                p.run_locked(paths, 20)
            execute.assert_not_called()

    def test_completed_inspection_rejects_changed_artifacts(self):
        paths = p.Paths(self.root / "task", self.root / "repo")
        artifact = self.root / "throughput.csv"
        artifact.write_text("result\n")
        p.new_json(paths.state, {"status": "completed", "gpu_released": True,
                                "artifact_sha256": {str(artifact): p.sha256(artifact)}})
        self.assertEqual(p.inspect_plan(paths)["status"], "completed")
        artifact.write_text("changed\n")
        with self.assertRaisesRegex(p.Refuse, "artifact changed"):
            p.inspect_plan(paths)


if __name__ == "__main__":
    unittest.main()
