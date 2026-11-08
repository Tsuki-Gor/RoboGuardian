"""Adapter contracts with small native-shaped fixtures; simulator runs are separate."""

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from roboguardian import Action, Budget, Context, Guardian, Memory, Observation, ReceiptState, Signal, Verifier


class NativeFixture:
    def __init__(self):
        self.position = [0.0, 0.0, 0.04]
        self.qpos = [0.0] * 19
        self.qvel = [0.0] * 19
        self.contacts = 0
        self.calls = []
        self.target = SimpleNamespace(name="target", pose=SimpleNamespace(p=[self.position], q=[[1.0, 0.0, 0.0, 0.0]]))
        self.target_actors = [self.target]
        self.target_object = self.target
        robot = SimpleNamespace(get_qpos=lambda: [self.qpos], get_qvel=lambda: [self.qvel])
        self.agent = SimpleNamespace(robot=robot, tcp=SimpleNamespace(pose=SimpleNamespace(p=[[0.0, 0.0, 0.2]], q=[[1.0, 0.0, 0.0, 0.0]])),
                                     count_finger_contacts=lambda target, min_force=0.5: [self.contacts])
        self.unwrapped = self
        self.init = SimpleNamespace(cfg=SimpleNamespace(arm_init_7=[0.2] * 7), HAND_INIT_12=[0.0] * 12, initarm=self.initarm)
        self.grasper = SimpleNamespace(grasp=self.grasp)

    def evaluate(self):
        return {"height": [self.position[2]], "success": [self.position[2] > 0.15], "fail": [False]}

    def get_obs(self):
        raise AssertionError("placeholder get_obs must not be used for evidence")

    def render(self):
        return "rendered-frame"

    def initarm(self, **kwargs):
        self.calls.append("initarm")
        self.qpos[:] = [0.2] * 7 + [0.0] * 12
        return SimpleNamespace(ok=True, error_code="none", message="returned", advice="")

    def grasp(self, **kwargs):
        self.calls.append("grasp")
        return SimpleNamespace(ok=True, error_code="stuck", message="failed", advice="")


class AdaClearGraspTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("integrations.adacleargrasp")
        self.adapter_module = importlib.import_module("integrations.adacleargrasp.adapter")
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.native = NativeFixture()
        self.frames = []
        def save_frame(frame, path):
            self.frames.append(frame)
            path.write_bytes(b"test-image")
        patcher = patch.object(self.adapter_module, "save_frame", save_frame)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapter = self.module.AdaAdapter(self.native, self.native.init, self.native.grasper, self.output,
                                              scene_id="scene", target_label="apple", config_version="v1",
                                              initarm_tolerance=0.05, height_threshold=0.15,
                                              relation_threshold=0.18, min_contacts=2)

    def action(self, observation, name="grasp"):
        return Action("call", name, self.adapter.embodiment, name, {}, "scene", ("target",),
                      observation.observation_id, 0, {"target_present": True}, "v1")

    def test_observer_reads_native_poses_joints_evaluation_and_writes_every_frame(self):
        first = self.adapter.observe(None)
        self.native.position[2] = 0.16
        self.native.contacts = 2
        second = self.adapter.observe(None)
        self.assertEqual(first.state["target"]["position_m"][2], 0.04)
        self.assertEqual(second.state["target"]["quaternion_wxyz"], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(second.state["robot_qvel_rad_s"], (0.0,) * 19)
        self.assertEqual(second.state["finger_contacts"], 2)
        self.assertTrue(second.state["evaluation"]["success"])
        self.assertEqual(len(self.frames), 2)
        self.assertNotEqual(first.observation_id, second.observation_id)
        self.assertTrue(Path(second.images[0]).is_file())

    def test_software_ok_does_not_commit_a_failed_native_grasp(self):
        before = self.adapter.observe(None)
        memory = Memory(":memory:", "episode")
        self.addCleanup(memory.close)
        with Guardian(memory, self.adapter.skills(), self.adapter, self.adapter,
                      Verifier(lambda *args: Signal.FAIL), Budget(1, 1, 2, 10000, 1)) as guardian:
            result = guardian.execute(self.action(before), observation=before)
        self.assertIs(result.receipt.state, ReceiptState.REJECTED)
        self.assertIs(result.receipt.runtime_result.terminal, True)
        self.assertTrue(result.receipt.runtime_result.diagnostics["native_result"]["ok"])
        self.assertEqual(memory.facts, {})
        self.assertEqual(self.native.calls, ["grasp"])

    def test_grasp_uses_two_of_three_physical_votes(self):
        self.native.position[2] = 0.15
        self.native.contacts = 2
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.PASS)
        self.native.position[2] = 0.16
        self.native.contacts = 1
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.PASS)
        self.native.agent.tcp.pose.p = [[1.0, 0.0, 0.2]]
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.FAIL)
        self.native.contacts = 2
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.PASS)
        self.native.agent.count_finger_contacts = lambda target, min_force=0.5: 1 / 0
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.FAIL)
        for registered in self.adapter.skills():
            self.assertIsNone(registered.measure_effect(None, None, None))

    def test_initarm_postcondition_uses_bound_software_completion(self):
        before = self.adapter.observe(None)
        action = self.action(before, "initarm")
        registered = next(item for item in self.adapter.skills() if item.skill == "initarm")
        self.assertIs(registered.post_check(action, before, before), Signal.INDETERMINATE)
        self.adapter.execute(action, "attempt")
        after = self.adapter.observe(action)
        self.assertIs(registered.post_check(action, before, after), Signal.PASS)
        self.native.qpos[4] += 0.2
        self.assertIs(registered.post_check(action, before, self.adapter.observe(action)), Signal.PASS)
        self.assertEqual(registered.committed_facts(action, before, after), {})

    def test_runtime_exception_remains_unknown_and_status_never_replays(self):
        def fail(**kwargs):
            self.native.calls.append("grasp")
            raise RuntimeError("native interrupted")
        self.native.grasper.grasp = fail
        before = self.adapter.observe(None)
        proposal = self.action(before)
        result = self.adapter.execute(proposal, "attempt")
        self.assertIsNone(result.terminal)
        self.assertIsNone(self.adapter.status(proposal, "attempt").terminal)
        self.assertEqual(self.native.calls, ["grasp"])

    def test_semantic_reference_is_original_before_for_reconcile_and_fresh_goal(self):
        before = self.adapter.observe(None)
        proposal = self.action(before)
        def lift(**kwargs):
            self.native.calls.append("grasp")
            self.native.position[2] = 0.161234567
            self.native.contacts = 2
            return SimpleNamespace(ok=True, error_code="none", message="returned", advice="")
        self.native.grasper.grasp = lift
        model = self.module.ChatModel(SimpleNamespace(api_key=""), "chosen/model", self.output,
                                      temperature=0.2, max_tokens=800, image_detail="low")
        calls = []
        judgments = iter(("indeterminate", "pass", "pass"))
        def request(role, prompt, images):
            calls.append((role, prompt, images))
            if role == "planner":
                return {"action": "done", "args": {}, "reason": "finished"}
            return {"judgment": next(judgments), "reason": "visual evidence"}
        model.request = request
        memory = Memory(":memory:", "reference-episode")
        self.addCleanup(memory.close)
        with Guardian(memory, self.adapter.skills(), self.adapter, self.adapter,
                      Verifier(model.assess), Budget(2, 1, 2, 16000, 2)) as guardian:
            first = guardian.execute(proposal, observation=before)
            self.assertIs(first.receipt.state, ReceiptState.UNKNOWN)
            after_first = self.adapter.last_observation
            second = guardian.reconcile(proposal.invocation_id)
            self.assertIs(second.receipt.state, ReceiptState.COMMITTED)
            final = self.adapter.observe(None)
            self.assertIs(model.assess("Is the target currently held above the table?", final), Signal.PASS)
        self.assertEqual(self.native.calls, ["grasp"])
        for observation in (after_first, self.adapter.last_observation):
            self.assertEqual(observation.state["visual_reference_before"],
                             {"observation_id": before.observation_id, "invocation_id": proposal.invocation_id,
                              "attempt_id": first.receipt.attempt_id, "skill": "grasp"})
        for _, prompt, images in calls:
            self.assertEqual(len(images), 2)
            self.assertEqual(images[1], before.images[0])
            self.assertNotEqual(images[0], before.images[0])
            self.assertIn("Image 1 is the current observation", prompt)
            self.assertIn("Image 2 is the original observation BEFORE", prompt)
            self.assertNotIn("0.161234567", prompt)
        self.assertEqual(len({entry[2][0] for entry in calls}), 3)
        context = Context({"observation": {"observation_id": final.observation_id}, "state_version": 1}, final.images)
        planner_messages = []
        def plan(messages):
            planner_messages.extend(messages)
            return {"action": "done", "args": {}}
        model.plan = plan
        self.module.AdaPlanner(model, self.adapter).propose("lift apple", context)
        images = [item["path"] for message in planner_messages if isinstance(message.get("content"), list)
                  for item in message["content"] if item.get("type") == "image"]
        self.assertEqual(images, list(final.images[:1]))
        self.assertNotIn("0.161234567", str(planner_messages))

    def test_final_native_summary_is_fresh_even_when_latest_frame_failed(self):
        self.native.position[2] = 0.16
        self.native.contacts = 2
        cached = self.adapter.observe(None)
        self.assertTrue(cached.state["evaluation"]["success"])
        self.native.position[2] = 0.04
        self.native.contacts = 0
        with patch.object(self.adapter_module, "save_frame", side_effect=OSError("frame unavailable")):
            with self.assertRaises(OSError):
                self.adapter.observe(None)
        self.assertIs(self.adapter.last_observation, cached)
        result = self.module.final_native_summary(self.native)
        self.assertIs(result["native_task_success"], False)
        self.assertEqual(result["final_native_evaluation"]["height"], 0.04)
        self.assertIsNone(result["final_native_evaluation_error"])

    def test_final_native_summary_reports_unavailable_evaluation_as_unknown(self):
        with patch.object(self.native, "evaluate", side_effect=RuntimeError("evaluation unavailable")):
            result = self.module.final_native_summary(self.native)
        self.assertIsNone(result["native_task_success"])
        self.assertIsNone(result["final_native_evaluation"])
        self.assertIn("evaluation unavailable", result["final_native_evaluation_error"])
        self.assertIsNone(self.module.final_native_summary(None)["native_task_success"])

    def test_planner_binds_proposal_and_done_never_claims_success(self):
        class Model:
            def __init__(self):
                self.answers = [{"action": "initarm", "args": {}, "reason": "reset arm"},
                                {"action": "done", "args": {}, "reason": "stop"}]
                self.calls = []
            def request(self, role, prompt, images):
                self.calls.append((role, prompt, images))
                return self.answers.pop(0)
        model = Model()
        planner = self.module.AdaPlanner(model, self.adapter)
        before = self.adapter.observe(None)
        context = Context({"observation": {"observation_id": before.observation_id}, "state_version": 7}, before.images)
        proposal = planner.propose("lift apple", context)
        self.assertEqual((proposal.based_on_observation, proposal.expected_state_version), (before.observation_id, 7))
        self.assertEqual(proposal.skill, "initarm")
        self.assertIsNone(planner.propose("lift apple", context))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.native.calls, [])

    def test_model_request_uses_selected_model_and_logs_no_api_key(self):
        requests = []
        class Response:
            choices = [SimpleNamespace(message=SimpleNamespace(content='{"judgment":"indeterminate","reason":"occluded"}'))]
            def model_dump(self, **kwargs):
                return {"choices": [{"message": {"content": self.choices[0].message.content}}]}
        def create(**kwargs):
            requests.append(kwargs)
            return Response()
        client = SimpleNamespace(api_key="test-secret-key", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        model = self.module.ChatModel(client, "chosen/model", self.output, temperature=0.2, max_tokens=800, image_detail="low")
        observation = self.adapter.observe(None)
        self.assertIs(model.assess("Is the apple held?", observation), Signal.INDETERMINATE)
        self.assertEqual(requests[0]["model"], "chosen/model")
        self.assertEqual(requests[0]["temperature"], 0.2)
        self.assertEqual(requests[0]["max_tokens"], 800)
        self.assertEqual(requests[0]["messages"][1]["content"][1]["image_url"]["detail"], "low")
        for path in self.output.rglob("*.json"):
            self.assertNotIn("test-secret-key", path.read_text())

    def test_pci_parser_patch_is_local_and_restores_on_error(self):
        calls = []
        def original(sim, render):
            calls.append((sim, render))
            return SimpleNamespace(render_device=None, render_backend=render)
        env_module = SimpleNamespace(parse_sim_and_render_backend=original)
        sapien = SimpleNamespace(Device=lambda name: "device:" + name)
        with patch.dict("sys.modules", {"mani_skill.envs.sapien_env": env_module, "sapien": sapien}):
            with self.assertRaises(ValueError):
                with self.module.pci_renderer("pci:0000:89:00.0"):
                    result = env_module.parse_sim_and_render_backend("cpu", "pci:0000:89:00.0")
                    self.assertEqual(result.render_backend, "sapien_cuda")
                    self.assertEqual(result.render_device, "device:pci:0000:89:00.0")
                    raise ValueError("gym.make failed")
        self.assertIs(env_module.parse_sim_and_render_backend, original)
        self.assertEqual(calls, [("cpu", "none")])


if __name__ == "__main__":
    unittest.main()
