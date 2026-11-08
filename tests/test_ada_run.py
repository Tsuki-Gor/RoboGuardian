"""Offline checks for Ada experiment selection and orchestration."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from roboguardian import Signal
from integrations.adacleargrasp import run as runner


class AdaRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assets = self.root / "assets"
        self.output = self.root / "output"
        self.required = ["--assets-root", str(self.assets), "--output", str(self.output),
                         "--render-device", "cpu"]

    def scene(self, relative, data=None):
        path = self.assets / "scenes" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data if data is not None else
                                   {"target": {"model_id": "apple"}, "clutter": []}))
        return path

    def test_defaults_and_only_supported_modes(self):
        parser = runner.parser()
        args = parser.parse_args(self.required)
        self.assertEqual(args.max_plan_steps, 60)
        self.assertEqual(args.modes, ["roboguardian"])
        for mode in ("base", "roboguardian-window", "roboguardian"):
            self.assertEqual(parser.parse_args(self.required + ["--modes", mode]).modes, [mode])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(self.required + ["--modes", "old-window"])

    def test_all_seven_categories_and_three_clutter_paths_can_be_selected(self):
        paths = [f"{category}/{clutter}/scene-1.json"
                 for category in ("apple", "ball", "can", "cube", "lego", "mug", "pear")
                 for clutter in (2, 4, 6)]
        for path in paths:
            self.scene(path)
        requested = [item for path in paths for item in ("--scene", path.removesuffix(".json"))]
        args = runner.parser().parse_args(self.required + requested + ["--seed", "42"])
        pairs = runner.select_pairs(args)
        self.assertEqual([Path(pair["scene"]) for pair in pairs],
                         [(self.assets / "scenes" / path).resolve() for path in paths])
        self.assertEqual(pairs, runner.select_pairs(args))

    def test_rejects_duplicate_modes_and_bad_scene(self):
        args = runner.parser().parse_args(self.required + ["--modes", "base", "base"])
        with self.assertRaises(ValueError):
            runner.select_pairs(args)
        self.scene("apple/2/bad.json", {"target": {}, "clutter": {}})
        args = runner.parser().parse_args(self.required + ["--scene", "apple/2/bad"])
        with self.assertRaises(ValueError):
            runner.select_pairs(args)
        args = runner.parser().parse_args(self.required + ["--scene", "apple/2/missing"])
        with self.assertRaises(FileNotFoundError):
            runner.select_pairs(args)

    def test_operation_semantics_abstain_but_effect_questions_reach_model(self):
        calls = []
        model = SimpleNamespace(assess=lambda question, observation: calls.append(question) or Signal.PASS)
        assess = runner.semantic_assessor(model)
        for skill in ("initarm", "inithand", "lift", "lower"):
            observation = SimpleNamespace(state={"visual_reference_before": {"skill": skill}})
            self.assertIs(assess("question", observation), Signal.INDETERMINATE)
        self.assertEqual(calls, [])
        for skill in ("push", "pull", "grasp"):
            observation = SimpleNamespace(state={"visual_reference_before": {"skill": skill}})
            self.assertIs(assess(skill, observation), Signal.PASS)
        self.assertEqual(calls, ["push", "pull", "grasp"])

    def test_each_mode_receives_the_same_pair_and_separate_output(self):
        self.scene("apple/2/sample.json")
        checkpoint = self.assets / "model.zip"
        checkpoint.write_bytes(b"fixture")
        calls = []

        def episode(args, pair, mode, output, checkpoint, secret, endpoint):
            calls.append((dict(pair), mode, output))
            self.assertEqual(json.loads(Path(pair["scene"]).read_text())["target"]["model_id"], "apple")
            Path(pair["source_scene"]).write_text('{}')  # Later modes must use the frozen copy.
            return {"pair": pair, "mode": mode, "execution_complete": True, "native_task_success": False}

        options = self.required + ["--scene", "apple/2/sample", "--seed", "42", "--fault", "response_loss",
                                   "--checkpoint", str(checkpoint), "--modes", "base", "roboguardian-window", "roboguardian"]
        dotenv = SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
        with patch.dict(sys.modules, {"dotenv": dotenv}), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "fixture", "OPENAI_BASE_URL": "https://example.invalid"}), \
             patch.object(runner, "validate_native", return_value="revision"), \
             patch.object(runner, "run_episode", side_effect=episode), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(options), 0)
        self.assertEqual([mode for _, mode, _ in calls], ["base", "roboguardian-window", "roboguardian"])
        self.assertTrue(all(pair == calls[0][0] for pair, _, _ in calls))
        self.assertEqual(calls[0][0]["seed"], 42)
        self.assertEqual(calls[0][0]["fault"], "response_loss")
        self.assertEqual(len({output for _, _, output in calls}), 3)
        self.assertTrue((self.output / "experiment.json").is_file())
