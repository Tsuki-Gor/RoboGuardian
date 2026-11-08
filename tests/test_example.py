"""End-to-end checks of synthetic execution, without mocked receipts."""

from pathlib import Path
import runpy
import unittest


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.run_scenario = runpy.run_path(
            str(Path(__file__).resolve().parent / "fixtures" / "closed_loop.py")
        )["run_scenario"]

    def test_normal_effect_commits_actual_changed_state(self):
        result = self.run_scenario("normal")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["run_result"]["status"], "verified_success")
        self.assertEqual(len(result["dispatches"]), 1)
        self.assertTrue(result["world"]["held"])
        self.assertEqual([r["state"] for r in result["receipts"]], ["committed"])
        self.assertEqual(result["verified_facts"], {"held": True})
        self.assertEqual([p["goal_id"] for p in result["verified_progress"]], ["hold-object"])

    def test_false_success_has_no_effect_until_safe_retry(self):
        result = self.run_scenario("false-success")
        self.assertEqual(result["run_result"]["status"], "verified_success")
        self.assertEqual(len(result["dispatches"]), 2)
        self.assertFalse(result["dispatches"][0]["after"]["held"])
        self.assertTrue(result["dispatches"][1]["after"]["held"])
        self.assertEqual([r["state"] for r in result["receipts"]], ["rejected", "committed"])
        self.assertEqual(len(result["verified_progress"]), 1)
        self.assertEqual(result["verified_facts"], {"held": True})

    def test_response_loss_reobserves_without_duplicate_dispatch(self):
        result = self.run_scenario("response-loss")
        self.assertEqual(result["run_result"]["status"], "verified_success")
        self.assertEqual(len(result["dispatches"]), 1)
        self.assertTrue(result["world"]["held"])
        self.assertEqual([r["state"] for r in result["receipts"]], ["unknown", "committed"])
        self.assertEqual(len(result["verified_progress"]), 1)
        self.assertEqual(result["receipts"][0]["attempt_id"], result["receipts"][1]["attempt_id"])

    def test_lost_prerequisite_preserves_history_but_blocks_dependent_dispatch(self):
        result = self.run_scenario("prerequisite-loss")
        self.assertNotEqual(result["run_result"]["status"], "verified_success")
        self.assertEqual([item["skill"] for item in result["dispatches"]], ["grasp"])
        self.assertFalse(result["world"]["held"])
        self.assertEqual([r["state"] for r in result["receipts"]], ["committed"])
        self.assertEqual(result["verified_facts"], {"held": True})
        self.assertEqual([p["goal_id"] for p in result["verified_progress"]], ["hold-object"])
        self.assertFalse(result["run_result"]["last_step"]["admission"]["allowed"])


if __name__ == "__main__":
    unittest.main()
