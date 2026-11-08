import copy
import json
import unittest
from types import SimpleNamespace

from roboguardian import (Budget, Context, Guardian, Memory, Observation,
                          RuntimeResult, Signal, SkillSpec, Verifier)
from roboguardian.types import json_data
from integrations.adacleargrasp.planner import AdaPlanner


class Model:
    def __init__(self):
        self.requests = []

    def plan(self, messages):
        self.requests.append(copy.deepcopy(messages))
        return {"action": "move_to", "args": {"name": "target"}}


class Adapter:
    target_id = "apple-1"
    target_label = "apple"
    objects = ("apple-1", "box-1")
    entity_ids = objects
    embodiment = "ada"
    scene_id = "scene"
    config_version = "v1"
    native_feedback = {}

    def validate_parameters(self, skill, params):
        return skill == "move_to" and params == {"name": "target"}


class PlannerTests(unittest.TestCase):
    def test_rejected_proposal_is_reported_and_next_proposal_executes(self):
        invalid = [
            ({"action": "move_to", "args": {"name": "unknown"}}, "identity_mismatch"),
            ({"action": "move_to", "args": {"name": []}}, "invalid_parameters"),
            ({"action": "unknown_tool", "args": {}}, "skill_unavailable"),
        ]
        for proposal, reason in invalid:
            with self.subTest(proposal=proposal):
                class ScriptedModel(Model):
                    def plan(self, messages):
                        self.requests.append(copy.deepcopy(messages))
                        return (proposal, {"action": "move_to", "args": {"name": "target"}},
                                {"action": "done", "args": {}})[len(self.requests) - 1]

                class Runtime(Adapter):
                    def __init__(self):
                        self.sequence, self.calls = 0, []
                        self.last_execution = {"invocation_id": "old", "attempt_id": "old", "message": "stale"}

                    def observe(self, action):
                        self.sequence += 1
                        return Observation(f"o{self.sequence}", "camera", self.sequence, "scene", self.entity_ids,
                                           {}, (), {"target_present": True})

                    def execute(self, action, attempt_id):
                        self.calls.append(action)
                        self.last_execution = {"invocation_id": action.invocation_id, "attempt_id": attempt_id,
                                               "ok": True, "message": "current"}
                        return RuntimeResult("returned", True, action.invocation_id, attempt_id)

                model, adapter = ScriptedModel(), Runtime()
                memory = Memory(":memory:", "proposal-test")
                self.addCleanup(memory.close)
                planner = AdaPlanner(model, adapter, mode="roboguardian-window", memory=memory)
                spec = SkillSpec("ada", "move_to", "v1", lambda params: adapter.validate_parameters("move_to", params),
                                 lambda action: "moved?", lambda *args: Signal.PASS, lambda *args: Signal.PASS,
                                 lambda *args: None, lambda *args: {}, ("post",))
                with Guardian(memory, [spec], adapter, adapter, Verifier(lambda *args: Signal.INDETERMINATE),
                              Budget(2, 0, 4, 20000, 1)) as guardian:
                    result = guardian.run("grasp", planner, lambda *args: Signal.FAIL)
                self.assertEqual(result.status, "needs_plan")
                self.assertEqual(json_data(planner.history[0]["action"].parameters), proposal["args"])
                self.assertEqual(len(adapter.calls), 1)
                self.assertEqual(dict(adapter.calls[0].parameters), {"name": "target"})
                feedback = json.loads([m for m in model.requests[1] if m["role"] == "tool"][-1]["content"])
                self.assertEqual((feedback["effect_status"], feedback["rejection_code"]), ("rejected", reason))
                self.assertEqual(feedback["raw_return"], {})
                self.assertNotIn("stale", json.dumps(model.requests[1]))
                last = json.loads([m for m in model.requests[2] if m["role"] == "tool"][-1]["content"])
                self.assertEqual(last["raw_return"]["message"], "current")
                self.assertNotIn("rejection_code", last)

    def test_missing_tool_or_non_object_arguments_remain_protocol_errors(self):
        model = Model()
        planner = AdaPlanner(model, Adapter())
        context = Context({"observation": {"observation_id": "o"}, "state_version": 0}, ())
        for proposal in ({"action": "move_to", "args": []}, {"args": {}}):
            with self.subTest(proposal=proposal):
                model.plan = lambda messages: proposal
                with self.assertRaises(ValueError):
                    planner.propose("grasp", context)

    def test_tool_decision_binds_real_target_and_recent_images(self):
        model = Model()
        planner = AdaPlanner(model, Adapter(), mode="roboguardian-window")
        for i in range(5):
            context = Context({"observation": {"observation_id": f"o{i}"}, "state_version": i,
                               "recovery": None, "limits": {}, "diagnostics": [],
                               "verified_facts": {"private_summary": "should not appear"}}, (f"image{i}",))
            action = planner.propose("grasp target", context)
        self.assertEqual(action.entity_ids, ("apple-1",))
        self.assertEqual((action.skill, action.parameters, action.based_on_observation),
                         ("move_to", {"name": "target"}, "o4"))
        request = json.dumps(model.requests[-1])
        self.assertNotIn("private_summary", request)
        self.assertNotIn("image0", request)
        for image in ("image2", "image3", "image4"):
            self.assertIn(image, request)

    def test_full_mode_retrieves_only_receipt_linked_relevant_images(self):
        class Memory:
            def events(self):
                observations = [{"kind": "observation", "payload": {"observation_id": f"o{i}", "images": [f"old{i}"]}} for i in range(4)]
                def receipt(name, refs, state="committed"):
                    return {"kind": "receipt", "payload": {"receipt": {"receipt_id": name, "state": state,
                            "action": {"entity_ids": ["apple-1"], "skill": "push"}, "evidence_refs": refs}}}
                return observations + [receipt("r1", ["o0", "o1"]), receipt("r2", ["o2"], "rejected")]
        model = Model()
        planner = AdaPlanner(model, Adapter(), memory=Memory())
        context = Context({"observation": {"observation_id": "now"}, "state_version": 0,
                           "verified_facts": {"proved": True}}, ("current",))
        planner.propose("grasp", context)
        request = json.dumps(model.requests[-1])
        self.assertIn("Bounded structured memory", request)
        self.assertIn("old2", request)
        self.assertIn("old0", request)
        self.assertNotIn("old3", request)
        self.assertIn('rejected', request)
        self.assertNotIn('old1', request)

    def test_base_retains_full_proposals_and_raw_feedback(self):
        model, adapter = Model(), Adapter()
        planner = AdaPlanner(model, adapter, mode="base")
        for i in range(5):
            action = planner.propose("grasp", Context({"observation": {"observation_id": f"o{i}"}, "state_version": i}, (f"image{i}",)))
            adapter.last_execution = {"invocation_id": action.invocation_id, "attempt_id": f"a{i}", "ok": False, "message": f"failure{i}"}
        request = json.dumps(model.requests[-1])
        self.assertIn("image0", request)
        self.assertIn("failure0", request)
        self.assertIn("tool_call_id", request)

    def test_model_retries_transport_and_json_but_not_invalid_tool_shape(self):
        import tempfile
        from pathlib import Path
        from integrations.adacleargrasp.planner import ChatModel
        class Response:
            def __init__(self, calls):
                self.choices = [SimpleNamespace(message=SimpleNamespace(tool_calls=calls))]
            def model_dump(self, **kwargs):
                return {"result": "redacted-secret"}
        call = lambda args: SimpleNamespace(function=SimpleNamespace(name="initarm", arguments=args))
        attempts = []
        def create(**body):
            attempts.append(body)
            if len(attempts) == 1:
                raise ConnectionError("redacted-secret")
            return Response([call("broken" if len(attempts) == 2 else "{}")])
        client = SimpleNamespace(api_key="redacted-secret", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            model = ChatModel(client, "example", directory, temperature=0, max_tokens=100, image_detail="low")
            result = model.plan([{"role": "user", "content": "choose"}])
            self.assertEqual((result["action"], result["args"]), ("initarm", {}))
            self.assertEqual(model.calls, 3)
            self.assertTrue(all(body["tool_choice"] == "required" for body in attempts))
            for path in Path(directory).rglob("*.json"):
                self.assertNotIn("redacted-secret", path.read_text())
            client.chat.completions.create = lambda **kwargs: Response([])
            with self.assertRaises(ValueError):
                model.plan([{"role": "user", "content": "choose"}])
            self.assertEqual(model.calls, 4)

    def test_semantic_request_retries_bad_json(self):
        import tempfile
        from integrations.adacleargrasp.planner import ChatModel
        answers = iter(['{broken', '{"judgment":"pass","reason":"visible"}'])
        calls = []
        class Response:
            def __init__(self, content):
                self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
            def model_dump(self, **kwargs):
                return {"content": self.choices[0].message.content}
        def create(**kwargs):
            calls.append(kwargs)
            return Response(next(answers))
        client = SimpleNamespace(api_key="", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            model = ChatModel(client, "example", directory, temperature=0, max_tokens=100, image_detail="low")
            self.assertEqual(model.request("semantic", "assess", ())['judgment'], 'pass')
            self.assertEqual(len(calls), 2)

    def test_window_feedback_uses_only_receipt_for_the_attempt(self):
        class Memory:
            def events(self):
                return [{"kind": "receipt", "payload": {"receipt": {
                    "receipt_id": "right", "attempt_id": "a1", "state": "rejected",
                    "action": {"invocation_id": first.invocation_id, "entity_ids": ["apple-1"], "skill": "push"},
                    "evidence_refs": []}}}, {"kind": "receipt", "payload": {"receipt": {
                    "receipt_id": "wrong", "attempt_id": "other", "state": "committed",
                    "action": {"invocation_id": "other", "entity_ids": ["apple-1"], "skill": "push"},
                    "evidence_refs": []}}}]
        model, adapter = Model(), Adapter()
        planner = AdaPlanner(model, adapter, mode="roboguardian-window")
        context = Context({"observation": {"observation_id": "first"}, "state_version": 0}, ("first-image",))
        first = planner.propose("grasp", context)
        planner.memory = Memory()
        adapter.last_execution = {"invocation_id": first.invocation_id, "attempt_id": "a1", "ok": True}
        adapter.native_feedback = {"ok": True}
        planner.propose("grasp", context)
        messages = model.requests[-1]
        tool = [msg for msg in messages if msg["role"] == "tool"][-1]
        feedback = json.loads(tool["content"])
        self.assertEqual((feedback["effect_status"], feedback["committed"], feedback["receipt_id"]), ("rejected", False, "right"))
        self.assertEqual(feedback["raw_return"]["ok"], True)
        self.assertNotIn("wrong", tool["content"])
        self.assertNotIn("Bounded structured memory", json.dumps(messages))
        self.assertEqual(messages[messages.index(tool)-1]["tool_calls"][0]["id"], tool["tool_call_id"])

    def test_retrieval_ignores_unrelated_and_operational_receipts(self):
        class Memory:
            def events(self):
                obs = [{"kind": "observation", "payload": {"observation_id": f"o{i}", "images": [f"image{i}"]}} for i in range(3)]
                def rec(subject, skill, ref):
                    return {"kind": "receipt", "payload": {"receipt": {"receipt_id": ref, "state": "committed", "attempt_id": ref,
                            "action": {"invocation_id": ref, "entity_ids": [subject], "skill": skill}, "evidence_refs": [ref]}}}
                return obs + [rec("apple-1", "push", "o0"), rec("apple-1", "initarm", "o1"), rec("box-1", "push", "o2")]
        model = Model()
        planner = AdaPlanner(model, Adapter(), memory=Memory())
        planner.propose("grasp", Context({"observation": {"observation_id": "now"}, "state_version": 0}, ("current",)))
        request = json.dumps(model.requests[-1])
        self.assertIn("image0", request)
        self.assertNotIn("image1", request)
        self.assertNotIn("image2", request)

    def test_authentication_failure_is_not_retried_and_tool_messages_are_preserved(self):
        import sys
        import tempfile
        from unittest.mock import patch
        from integrations.adacleargrasp.planner import ChatModel
        class APIError(Exception):
            pass
        class APIConnectionError(APIError):
            pass
        class APITimeoutError(APIConnectionError):
            pass
        class RateLimitError(APIError):
            pass
        class InternalServerError(APIError):
            pass
        class AuthenticationError(APIError):
            pass
        package = SimpleNamespace(APIConnectionError=APIConnectionError, APITimeoutError=APITimeoutError,
                                  RateLimitError=RateLimitError, InternalServerError=InternalServerError)
        attempts = []
        def create(**body):
            attempts.append(body)
            raise AuthenticationError("unauthorized")
        client = SimpleNamespace(api_key="", chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "real-call", "type": "function",
                     "function": {"name": "lift", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "real-call", "content": "{\"effect_status\":\"rejected\"}"}]
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"openai": package}):
            model = ChatModel(client, "example", directory, temperature=0, max_tokens=100, image_detail="low")
            with self.assertRaisesRegex(ValueError, "AuthenticationError"):
                model.plan(messages)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["messages"], messages)

    def test_full_mode_caps_total_images_at_five(self):
        class Memory:
            def events(self):
                obs = [{"kind": "observation", "payload": {"observation_id": f"o{i}", "images": [f"old{i}"]}} for i in range(4)]
                return obs + [{"kind": "receipt", "payload": {"receipt": {
                    "receipt_id": "r1", "attempt_id": "a1", "state": "rejected",
                    "action": {"invocation_id": "i1", "entity_ids": ["apple-1"], "skill": "push"},
                    "evidence_refs": [f"o{i}" for i in range(4)]}}}]
        model = Model()
        planner = AdaPlanner(model, Adapter(), memory=Memory())
        for i in range(5):
            planner.propose("grasp", Context({"observation": {"observation_id": f"current{i}"}, "state_version": i}, (f"current{i}",)))
        messages = model.requests[-1]
        images = [item["path"] for message in messages if isinstance(message.get("content"), list)
                  for item in message["content"] if item["type"] == "image"]
        self.assertEqual(images, ["old0", "old1", "current2", "current3", "current4"])
        self.assertIn("rejected", json.dumps(messages))

    def test_previous_native_result_cannot_be_attributed_to_a_different_action(self):
        model, adapter = Model(), Adapter()
        planner = AdaPlanner(model, adapter, mode="roboguardian-window")
        context = Context({"observation": {"observation_id": "o"}, "state_version": 0}, ("current",))
        planner.propose("grasp", context)
        adapter.last_execution = {"invocation_id": "older", "attempt_id": "old", "ok": True, "message": "stale"}
        adapter.native_feedback = {"ok": True, "message": "stale"}
        planner.propose("grasp", context)
        tool = next(m for m in model.requests[-1] if m["role"] == "tool")
        self.assertNotIn("stale", tool["content"])
        self.assertEqual(json.loads(tool["content"])["effect_status"], "not_confirmed")
