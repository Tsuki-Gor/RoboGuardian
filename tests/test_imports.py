"""Public imports and command entry points survive the package reorganization."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

import roboguardian
from roboguardian import types


class ImportCompatibilityTests(unittest.TestCase):
    def test_core_exports_are_the_same_objects(self):
        groups = {
            "models": ("Action", "Observation", "Budget", "Context", "EffectReceipt",
                       "EvidenceReport", "Signal", "action_fingerprint", "effect_signal"),
            "interfaces": ("SkillSpec", "Planner", "Runtime", "Observer", "SemanticAssessor"),
            "serialization": ("canonical_json", "json_data"),
        }
        for module_name, names in groups.items():
            module = importlib.import_module("roboguardian." + module_name)
            for name in names:
                with self.subTest(module=module_name, name=name):
                    self.assertIs(getattr(roboguardian, name), getattr(module, name))
                    self.assertIs(getattr(types, name), getattr(module, name))
        from roboguardian.context import ContextBudgetError
        from roboguardian.memory import ContextBudgetError as LegacyError
        self.assertIs(ContextBudgetError, LegacyError)
        self.assertIs(ContextBudgetError, roboguardian.ContextBudgetError)

    def test_platform_exports_are_the_same_objects(self):
        groups = {
            "adacleargrasp": {"adapter": "AdaAdapter", "planner": "AdaPlanner", "run": "main"},
            "libero": {"adapter": "LiberoAdapter", "policy": "SmolVLAController", "run": "main"},
            "habitat": {"adapter": "HabitatAdapter", "planner": "HabitatPlanner",
                        "process": "HabitatProcess", "run": "main"},
        }
        for platform, modules in groups.items():
            package = importlib.import_module("integrations." + platform)
            for module_name, name in modules.items():
                with self.subTest(platform=platform, name=name):
                    module = importlib.import_module(package.__name__ + "." + module_name)
                    self.assertIs(getattr(package, name), getattr(module, name))

    def test_legacy_worker_import_does_not_load_the_core(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c",
             "import sys; from integrations.habitat_worker import Worker, main; "
             "assert callable(main); assert Worker.__name__ == 'Worker'; "
             "assert 'roboguardian' not in sys.modules"],
            cwd=root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_help_needs_only_the_standard_library(self):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        entries = []
        for platform in ("adacleargrasp", "libero", "habitat"):
            entries.extend((["-m", "integrations." + platform],
                            ["integrations/" + platform + ".py"]))
        entries.extend((["integrations/habitat_worker.py"], ["integrations/habitat/worker.py"]))
        for entry in entries:
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, "-S", "-B", *entry, "--help"],
                                        cwd=root, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
