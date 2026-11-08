"""Native-shaped fixtures; these tests neither load a model nor run LIBERO."""

from contextlib import nullcontext
import base64
import importlib
import json
import os
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from roboguardian import Action, Budget, Guardian, Memory, ReceiptState, Signal, Verifier


class NativeFixture:
    task_description = "pick up the book and place it in the back compartment of the caddy"

    def __init__(self, success_after=2):
        self.steps = 0
        self.success_after = success_after
        self.success = False
        self.observation_frames = 0
        self._env = SimpleNamespace(step=self.raw_step, check_success=lambda: self.success,
                                    env=SimpleNamespace(_get_observations=self.raw_observation))

    def raw_observation(self, force_update=False):
        if not force_update:
            raise AssertionError("new observations must force an observable refresh")
        frame = self.observation_frames
        self.observation_frames += 1
        return {"robot_state": {"eef": {"pos": [self.steps, 0, 0]}, "gripper": {"qpos": [0, 0]}},
                "pixels": {"image": f"agent-{frame}", "image2": f"wrist-{frame}"}}

    def _format_raw_obs(self, raw):
        return raw

    def raw_step(self, action):
        self.steps += 1
        self.success = self.steps >= self.success_after
        return self.raw_observation(True), float(self.success), self.success, {}

    def step(self, action):
        raise AssertionError("LiberoEnv.step would auto-reset the terminal world")

    def reset(self, *args, **kwargs):
        raise AssertionError("rollout and verification must never reset the world")


class PolicyFixture:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, observation):
        self.calls += 1
        return [0.0] * 7


class LiberoTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("integrations.libero")
        self.adapter_module = importlib.import_module("integrations.libero.adapter")
        self.semantic = importlib.import_module("integrations._semantic")
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.native, self.policy = NativeFixture(), PolicyFixture()
        patcher = patch.object(self.adapter_module, "save_frame", lambda frame, path: path.write_bytes(str(frame).encode()))
        patcher.start()
        self.addCleanup(patcher.stop)
        frames = patch.object(self.adapter_module, "visible_frame",
                              lambda formatted, key="image": formatted["pixels"][key])
        frames.start()
        self.addCleanup(frames.stop)
        self.adapter = self.module.LiberoAdapter(self.native, self.policy, self.output,
                                                config_version="test-v1", max_steps=3)

    def proposal(self, before):
        return Action("rollout", "book-in-caddy", self.adapter.embodiment, "smolvla_rollout", {},
                      self.adapter.scene_id, self.adapter.entity_ids, before.observation_id, 0,
                      {"scene_ready": True}, "test-v1")

    def guardian(self, memory, semantic):
        return Guardian(memory, [self.adapter.skill()], self.adapter, self.adapter,
                        Verifier(semantic), Budget(2, 0, 2, 16000, 1))

    def test_learned_rollout_stops_at_success_and_preserves_terminal_world(self):
        memory = Memory(":memory:", "success")
        self.addCleanup(memory.close)
        with self.guardian(memory, lambda *args: Signal.PASS) as guardian:
            before = self.adapter.observe(None)
            result = guardian.execute(self.proposal(before), observation=before)
        self.assertIs(result.receipt.state, ReceiptState.COMMITTED)
        self.assertTrue(self.native.success)
        self.assertEqual((self.native.steps, self.policy.calls, self.policy.resets), (2, 2, 1))
        self.assertEqual(len(self.adapter.dispatches), 1)
        self.assertEqual(result.receipt.runtime_result.diagnostics["native_controlstep_count"], 2)
        self.assertTrue(memory.facts["task_complete"])

    def test_step_limit_is_terminal_failure_and_does_not_commit(self):
        self.native.success_after = 100
        memory = Memory(":memory:", "limit")
        self.addCleanup(memory.close)
        with self.guardian(memory, lambda *args: Signal.FAIL) as guardian:
            before = self.adapter.observe(None)
            result = guardian.execute(self.proposal(before), observation=before)
        self.assertIs(result.receipt.state, ReceiptState.REJECTED)
        self.assertTrue(result.receipt.runtime_result.terminal)
        self.assertEqual(result.receipt.runtime_result.status, "step_limit")
        self.assertEqual(self.native.steps, 3)
        self.assertEqual(memory.facts, {})

    def test_unknown_reobserves_same_rollout_and_freezes_original_before(self):
        judgments = iter((Signal.INDETERMINATE, Signal.PASS))
        memory = Memory(":memory:", "unknown")
        self.addCleanup(memory.close)
        with self.guardian(memory, lambda *args: next(judgments)) as guardian:
            before = self.adapter.observe(None)
            action = self.proposal(before)
            first = guardian.execute(action, observation=before)
            self.assertIs(first.receipt.state, ReceiptState.UNKNOWN)
            result = guardian.reconcile(action.invocation_id)
            self.assertIs(result.receipt.state, ReceiptState.COMMITTED)
            fresh_goal = self.adapter.observe(None)
        self.assertEqual(self.native.steps, 2)
        self.assertEqual(len(self.adapter.dispatches), 1)
        self.assertEqual(fresh_goal.images[2], before.images[0])
        self.assertEqual(fresh_goal.state["visual_reference_before"]["observation_id"], before.observation_id)
        self.native.success = False
        self.assertIs(self.adapter.post_signal(self.adapter.observe(None)), Signal.FAIL)
        self.assertIs(self.module.final_native_summary(self.native)["native_task_success"], False)

    def test_invalid_policy_action_never_reaches_native_step(self):
        for action in ([0] * 6, [float("nan")] * 7, [[0] * 7]):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.module.validate_action(action)
        before = self.adapter.observe(None)
        with patch.object(PolicyFixture, "__call__", return_value=[float("nan")] * 7):
            result = self.adapter.execute(self.proposal(before), "attempt")
        self.assertIsNone(result.terminal)
        self.assertEqual(self.native.steps, 0)
        self.assertIsNone(self.adapter.status(self.proposal(before), "attempt").terminal)

    def test_single_planner_proposal_and_unknown_final_native_summary(self):
        from roboguardian import Context
        planner = self.module.SingleRolloutPlanner(self.adapter)
        before = self.adapter.observe(None)
        context = Context({"observation": {"observation_id": before.observation_id}, "state_version": 0}, before.images)
        self.assertIsNotNone(planner.propose("book in caddy", context))
        self.assertIsNone(planner.propose("book in caddy", context))
        with patch.object(self.native._env, "check_success", side_effect=RuntimeError("unavailable")):
            summary = self.module.final_native_summary(self.native)
        self.assertIsNone(summary["native_task_success"])
        self.assertIn("unavailable", summary["final_native_evaluation_error"])

    def test_observations_refresh_both_current_views_and_keep_original_agent_reference(self):
        before = self.adapter.observe(None)
        action = self.proposal(before)
        self.adapter.execute(action, "attempt")
        after = self.adapter.observe(action)
        refreshed = self.adapter.observe(None)

        self.assertEqual(len(before.images), 2)
        self.assertEqual(len(after.images), 3)
        self.assertEqual(after.state["image_roles"], (
            "CURRENT agent view", "CURRENT wrist view", "ORIGINAL BEFORE reference",
        ))
        self.assertNotEqual(after.images[:2], before.images)
        self.assertNotEqual(refreshed.images[:2], after.images[:2])
        self.assertEqual(after.images[2], before.images[0])
        self.assertEqual(refreshed.images[2], before.images[0])
        self.assertNotEqual(Path(after.images[0]).read_text(), Path(before.images[0]).read_text())
        self.assertNotEqual(Path(after.images[1]).read_text(), Path(before.images[1]).read_text())
        self.assertNotEqual(Path(refreshed.images[0]).read_text(), Path(after.images[0]).read_text())
        self.assertNotEqual(Path(refreshed.images[1]).read_text(), Path(after.images[1]).read_text())
        self.assertEqual(Path(refreshed.images[2]).read_text(), Path(before.images[0]).read_text())

    def test_semantic_http_uses_named_images_and_redacts_credentials(self):
        before = self.adapter.observe(None)
        action = self.proposal(before)
        self.adapter.execute(action, "attempt")
        self.adapter.observe(action)
        after = self.adapter.observe(None)
        requests = []
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": '{"judgment":"pass","reason":"book inside caddy"}'}}]}).encode()
        def urlopen(request, timeout):
            requests.append((request, timeout))
            return Response()
        client = self.semantic.SemanticClient("https://example.invalid/v1", "test-secret", "qwen/qwen3.5-27b",
                                              self.output, timeout=12)
        with patch.object(self.semantic, "urlopen", urlopen):
            self.assertIs(client.assess("Is the book inside the back compartment?", after), Signal.PASS)
        request, timeout = requests[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual((body["model"], body["temperature"], body["max_tokens"], timeout),
                         ("qwen/qwen3.5-27b", 0.2, 800, 12))
        content = body["messages"][1]["content"]
        self.assertIn("Image 1 is the CURRENT agent view", content[0]["text"])
        self.assertIn("Image 2 is the CURRENT wrist view", content[0]["text"])
        self.assertIn("Image 3 is the ORIGINAL BEFORE reference", content[0]["text"])
        self.assertEqual(len([part for part in content if part["type"] == "image_url"]), 3)
        self.assertNotIn("native_success", content[0]["text"])
        self.assertNotIn("robot_state", content[0]["text"])
        image_parts = [part for part in content if part["type"] == "image_url"]
        self.assertEqual([base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in image_parts],
                         [Path(path).read_bytes() for path in after.images])
        for path in self.output.rglob("*.json"):
            self.assertNotIn("test-secret", path.read_text())
        with patch.object(self.semantic, "urlopen", side_effect=TimeoutError("test-secret timed out")) as failing:
            with self.assertRaisesRegex(RuntimeError, "REDACTED"):
                client.assess("Is the task complete?", after)
        self.assertEqual(failing.call_count, 1)
        for path in self.output.rglob("*.json"):
            self.assertNotIn("test-secret", path.read_text())

    def test_semantic_without_image_roles_keeps_existing_habitat_image_selection(self):
        current = self.output / "habitat-current.png"
        reference = self.output / "habitat-before.png"
        current.write_bytes(b"current")
        reference.write_bytes(b"before")
        observation = SimpleNamespace(images=(str(current), str(reference)),
                                      state={"visual_reference_before": {"observation_id": "before"}})
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {
                    "content": '{"judgment":"pass","reason":"visible change"}'}}]}).encode()

        response = Response()
        client = self.semantic.SemanticClient("https://example.invalid/v1", "test-secret", "qwen/qwen3.5-27b",
                                              self.output)
        requests = []

        def urlopen(request, timeout):
            requests.append(request)
            return response

        with patch.object(self.semantic, "urlopen", urlopen):
            self.assertIs(client.assess("Is the task complete?", observation), Signal.PASS)
        content = json.loads(requests[0].data)["messages"][1]["content"]
        self.assertIn("Image 1 is the current observation", content[0]["text"])
        self.assertIn("Image 2 is the original observation BEFORE", content[0]["text"])
        self.assertEqual(len([part for part in content if part["type"] == "image_url"]), 2)

    def test_semantic_accepts_json_code_fence_from_configured_model(self):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"choices": [{"message": {
            "content": '```json\n{"judgment":"pass","reason":"visible completed relation"}\n```'}}]}).encode()
        client = self.semantic.SemanticClient("https://example.invalid/v1", "test-secret", "qwen/qwen3.5-27b", self.output)
        with patch.object(self.semantic, "urlopen", return_value=response):
            self.assertIs(client.assess("Is the task complete?", self.adapter.observe(None)), Signal.PASS)

    def test_credentials_read_only_two_keys_with_shell_precedence(self):
        env_file = self.output / "credentials.env"
        env_file.write_text('OPENAI_API_KEY="file-key" # comment\nOPENAI_BASE_URL=https://example.invalid/v1\nUNRELATED=$(not-executed)\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.module.read_credentials(env_file), ("file-key", "https://example.invalid/v1"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key", "OPENAI_BASE_URL": "https://shell.invalid/v1"}):
            self.assertEqual(self.module.read_credentials(env_file), ("shell-key", "https://shell.invalid/v1"))

    def test_smolvla_controller_batches_every_native_robot_state_leaf(self):
        class ShapeArray:
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, index):
                if index is not None:
                    raise AssertionError("only a leading batch axis is expected")
                return ShapeArray((1, *self.shape))

        robot_state = {
            "eef": {"pos": ShapeArray((3,)), "quat": ShapeArray((4,)), "mat": ShapeArray((3, 3))},
            "gripper": {"qpos": ShapeArray((2,)), "qvel": ShapeArray((2,))},
            "joints": {"pos": ShapeArray((7,)), "vel": ShapeArray((7,))},
        }
        formatted = {"pixels": {}, "robot_state": robot_state}
        seen = {}

        def preprocess_observation(observation):
            seen["robot_state"] = observation["robot_state"]
            return {"observation.robot_state": observation["robot_state"]}

        class ActionOutput:
            shape = (1, 7)

            def to(self, device):
                return self

            def numpy(self):
                return self

            def __getitem__(self, index):
                return [0.0] * 7

        torch_module = ModuleType("torch")
        torch_module.inference_mode = nullcontext
        lerobot_module = ModuleType("lerobot")
        lerobot_module.__path__ = []
        envs_module = ModuleType("lerobot.envs")
        envs_module.__path__ = []
        utils_module = ModuleType("lerobot.envs.utils")
        utils_module.preprocess_observation = preprocess_observation
        lerobot_module.envs = envs_module
        envs_module.utils = utils_module
        controller = object.__new__(self.module.SmolVLAController)
        controller.task_description = "test task"
        controller.env_preprocessor = lambda observation: observation
        controller.preprocessor = lambda observation: observation
        controller.policy = SimpleNamespace(select_action=lambda observation: ActionOutput())
        controller.postprocessor = lambda action: action
        controller.env_postprocessor = lambda transition: transition

        with patch.dict("sys.modules", {"torch": torch_module, "lerobot": lerobot_module,
                                        "lerobot.envs": envs_module, "lerobot.envs.utils": utils_module}):
            self.assertEqual(controller(formatted), [0.0] * 7)

        for part, fields in robot_state.items():
            for name, value in fields.items():
                self.assertEqual(seen["robot_state"][part][name].shape, (1, *value.shape), f"{part}.{name}")


if __name__ == "__main__":
    unittest.main()
