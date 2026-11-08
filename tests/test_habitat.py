"""Offline checks for the Guardian/native Habitat macro boundary."""

import json
import gzip
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from roboguardian import Budget, Memory
from integrations.habitat.process import HabitatProcess
from integrations.habitat.run import _drive, _prepare, parser

METRICS = {"success": 0, "oracle_success": 1, "spl": 0, "ndtw": 0.2,
           "sdtw": 0, "distance_to_goal": 4.2, "path_length": 3.5, "steps_taken": 10}


class FakePipe:
    def __init__(self, *, after_start="completed", metrics=None, post_attempt=None):
        self.after_start = after_start
        self.metrics = metrics if metrics is not None else {"11": dict(METRICS)}
        self.state = {"state": "ready", "invocation_id": None, "attempt_id": None}
        self.calls = []
        self.post_attempt = post_attempt
        self.next_status_is_post = False
        self.closed = False

    def request(self, operation, **payload):
        self.calls.append((operation, payload))
        if operation == "start":
            self.state = {"state": self.after_start, **payload, "metrics": self.metrics}
        elif operation == "resume":
            self.state = {**self.state, "state": "completed", "metrics": self.metrics}
        elif operation == "cancel":
            self.state = {**self.state, "state": "cancelled"}
        if operation == "status" and self.next_status_is_post:
            self.next_status_is_post = False
            return {**self.state, "attempt_id": self.post_attempt}
        return dict(self.state)

    def wait(self, **_kwargs):
        self.calls.append(("wait", {}))
        self.next_status_is_post = self.post_attempt is not None
        return dict(self.state)

    def close(self):
        self.closed = True


class HabitatMacroTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = {"output_dir": str(self.output), "job_id": "job", "episode_ids": ["11"]}

    def artifacts(self):
        return (json.loads((self.output / "result.json").read_text()),
                json.loads((self.output / "audit.json").read_text()))

    def test_evaluation_completion_preserves_failed_episode_result(self):
        pipe = FakePipe()
        result = _drive(self.config, pipe, "run")
        saved, audit = self.artifacts()
        self.assertTrue(result["evaluation_complete"])
        self.assertEqual(result["navigation_results"]["11"]["success"], 0)
        self.assertEqual(saved, result)
        self.assertTrue(any(event["kind"] == "receipt" for event in audit["events"]))
        memory = Memory(self.output / "guardian.sqlite", "job")
        try:
            context = memory.context(memory.latest_observation(), Budget(1, 0, 1, 65536, 0),
                                     {"evaluation_complete": True, "navigation_success": True})
            self.assertEqual(set(context.data["verified_facts"]), {"evaluation_complete"})
            self.assertTrue(context.data["verified_facts"]["evaluation_complete"]["value"])
        finally:
            memory.close()
        self.assertTrue(pipe.closed)

    def test_incomplete_metrics_do_not_commit(self):
        for missing in METRICS:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                config = {**self.config, "output_dir": directory}
                partial = {key: value for key, value in METRICS.items() if key != missing}
                pipe = FakePipe(metrics={"11": partial})
                result = _drive(config, pipe, "run")
                self.assertFalse(result["evaluation_complete"])
                self.assertEqual(result["governance"]["receipt"]["state"], "unknown")
                self.assertFalse(pipe.closed)

    def test_terminal_unknown_reconciles_the_original_attempt(self):
        pipe = FakePipe(metrics={"11": {key: value for key, value in METRICS.items() if key != "ndtw"}})
        original = _drive(self.config, pipe, "run")
        self.assertEqual(original["worker_state"], "completed")
        self.assertFalse(original["evaluation_complete"])
        self.assertFalse(pipe.closed)
        pipe.state["metrics"] = {"11": dict(METRICS)}
        result = _drive(self.config, pipe, "resume", budget_seconds=15)
        self.assertTrue(result["evaluation_complete"])
        self.assertEqual((result["invocation_id"], result["attempt_id"]),
                         (original["invocation_id"], original["attempt_id"]))
        self.assertEqual(sum(operation == "start" for operation, _ in pipe.calls), 1)
        self.assertFalse(any(operation == "resume" for operation, _ in pipe.calls))
        self.assertTrue(pipe.closed)

    def test_paused_attempt_resumes_and_reconciles_without_restarting(self):
        pipe = FakePipe(after_start="paused")
        paused = _drive(self.config, pipe, "run")
        self.assertFalse(paused["evaluation_complete"])
        self.assertFalse(pipe.closed)
        self.assertEqual(paused["governance"]["receipt"]["state"], "unknown")
        invocation, attempt = paused["invocation_id"], paused["attempt_id"]
        resumed = _drive(self.config, pipe, "resume", budget_seconds=15)
        self.assertTrue(resumed["evaluation_complete"])
        self.assertEqual((resumed["invocation_id"], resumed["attempt_id"]), (invocation, attempt))
        self.assertEqual(sum(operation == "start" for operation, _ in pipe.calls), 1)
        self.assertIn(("resume", {"budget_seconds": 15}), pipe.calls)
        self.assertTrue(pipe.closed)
        saved, audit = self.artifacts()
        self.assertEqual(saved, resumed)
        self.assertTrue(any(event["kind"] == "manual_reconciliation" for event in audit["events"]))

    def test_post_observation_with_other_attempt_cannot_commit(self):
        pipe = FakePipe(post_attempt="different-attempt")
        result = _drive(self.config, pipe, "run")
        self.assertFalse(result["evaluation_complete"])

    def test_resume_rejects_worker_identity_mismatch(self):
        pipe = FakePipe(after_start="paused")
        _drive(self.config, pipe, "run")
        pipe.state["attempt_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identity do not match"):
            _drive(self.config, pipe, "resume", budget_seconds=5)
        self.assertEqual(sum(operation == "start" for operation, _ in pipe.calls), 1)
        self.assertFalse(any(operation == "resume" for operation, _ in pipe.calls))

    def test_cancel_preserves_audit_and_does_not_commit(self):
        pipe = FakePipe(after_start="paused")
        _drive(self.config, pipe, "run")
        cancelled = _drive(self.config, pipe, "cancel")
        self.assertEqual(cancelled["worker_state"], "cancelled")
        self.assertFalse(cancelled["evaluation_complete"])
        self.assertEqual(sum(operation == "start" for operation, _ in pipe.calls), 1)
        saved, audit = self.artifacts()
        self.assertEqual(saved, cancelled)
        self.assertTrue(audit["events"])

    def test_http_timeout_preserves_started_worker_and_launches_script(self):
        process = HabitatProcess(self.output, python="/python39", config="/job.json", request_timeout=.1)
        child = Mock()
        child.poll.return_value = None
        def launch(command, **kwargs):
            (self.output / "worker.json").write_text(json.dumps({
                "url": "http://127.0.0.1:1234", "token": "test"}))
            self.assertEqual(command[:2], ["/python39", "-B"])
            self.assertEqual(Path(command[2]).name, "worker.py")
            self.assertEqual(command[3:], ["--config-json", "/job.json"])
            self.assertTrue(kwargs["start_new_session"])
            return child
        process.http.open = Mock(side_effect=TimeoutError("lost response"))
        with patch("integrations.habitat.process.subprocess.Popen", side_effect=launch) as popen:
            with self.assertRaises(TimeoutError):
                process.request("start", invocation_id="i", attempt_id="a")
            with self.assertRaises(TimeoutError):
                process.request("status")
        self.assertEqual(popen.call_count, 1)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()


class HabitatInputTests(unittest.TestCase):
    def test_native_scene_prefix_and_integer_episode_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlnce_baselines").mkdir()
            (root / "vlnce_baselines/lavira_main.py").touch()
            (root / "mp3d/scene").mkdir(parents=True)
            (root / "mp3d/scene/scene.glb").touch()
            for name in ("python", "model.env"):
                (root / name).touch()
            with gzip.open(root / "gt.json.gz", "wt") as stream:
                json.dump({"11": {"locations": []}}, stream)
            values = {"LA_MODEL_NAME": "openai/gpt-4o-2024-11-20", "VA_MODEL_NAME": "qwen/qwen3.5-27b"}
            for number, episode_id in enumerate((11, "11")):
                with gzip.open(root / "data.json.gz", "wt") as stream:
                    json.dump({"episodes": [{"episode_id": episode_id,
                              "scene_id": "data/scene_datasets/mp3d/scene/scene.glb",
                              "instruction": {"instruction_text": "Go to the door."}}]}, stream)
                args = parser().parse_args([
                    "run", "--output", str(root / ("run" + str(number))),
                    "--worker-python", str(root / "python"), "--assets-root", str(root),
                    "--dataset", str(root / "data.json.gz"), "--ground-truth", str(root / "gt.json.gz"),
                    "--scene-root", str(root), "--env-file", str(root / "model.env"),
                    "--hf-home", str(root), "--episode-id", "11"])
                runner = importlib.import_module("integrations.habitat.run")
                with patch.object(runner, "NATIVE_ROOT", root), \
                        patch.object(runner, "read_env", return_value=values):
                    if number == 0:
                        self.assertEqual(_prepare(args)["episode_ids"], ["11"])
                    else:
                        with self.assertRaisesRegex(ValueError, "integer episode_id"):
                            _prepare(args)


if __name__ == "__main__":
    unittest.main()
