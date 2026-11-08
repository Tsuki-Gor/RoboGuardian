"""Local transport and scripted adapters, with no simulator or robot."""

import unittest
from threading import Event, Thread
from types import SimpleNamespace

from roboguardian import (Action, Budget, Guardian, Memory, Observation, ReceiptState,
                          RuntimeResult, Signal, SkillSpec, Verifier)
from integrations.adacleargrasp.execution import AdaExecution, BaseController


def action(skill="grasp"):
    return Action("action-1", "goal", "arm", skill, {}, "scene", ("target",),
                  "observation-1", 0, {}, "v1")


class Adapter:
    def __init__(self):
        self.calls = []
        self.native_feedback = {}
        self.last_execution = None
        self.last_observation = None
        self.dispatches = []
        self.sequence = 0

    def bind_before(self, action, attempt):
        pass

    def validate_action(self, action):
        return True

    def execute(self, action, attempt):
        self.calls.append((action.invocation_id, attempt))
        self.dispatches.append({"invocation_id": action.invocation_id, "attempt_id": attempt, "skill": action.skill})
        feedback = {"ok": True, "error_code": "none", "message": "ok", "advice": ""}
        self.native_feedback = feedback
        self.last_execution = {"invocation_id": action.invocation_id, "attempt_id": attempt, **feedback}
        return RuntimeResult("native_returned", True, action.invocation_id, attempt,
                             {"native_result": feedback})

    def observe(self, action):
        self.sequence += 1
        self.last_observation = Observation(f"observation-{self.sequence}", "camera", self.sequence, "scene", ("target",), {}, (), {})
        return self.last_observation


class ExecutionTests(unittest.TestCase):
    def test_response_loss_times_out_then_status_returns_original_result_without_redispatch(self):
        adapter = Adapter()
        runtime = AdaExecution(adapter, fault="response_loss", timeout=.05, response_delay=.2)
        self.addCleanup(runtime.close)
        proposal = action()
        result = runtime.execute(proposal, "attempt-1")
        self.assertEqual((result.status, result.terminal), ("timeout", None))
        self.assertEqual(adapter.native_feedback, {})
        checked = runtime.status(proposal, "attempt-1")
        self.assertEqual((checked.status, checked.terminal), ("native_returned", True))
        self.assertEqual(adapter.calls, [("action-1", "attempt-1")])

    def test_false_success_and_base_complete_without_physical_dispatch_or_receipt(self):
        adapter = Adapter()
        runtime = AdaExecution(adapter, fault="false_success", timeout=.5)
        self.addCleanup(runtime.close)
        memory = Memory(":memory:", "base")
        self.addCleanup(memory.close)
        controller = BaseController(memory, Budget(2, 0, 2, 20000, 1), runtime)
        planner = SimpleNamespace(propose=lambda instruction, context: action())
        result = controller.run("grasp", planner)
        self.assertEqual(result.status, "completed")
        self.assertEqual(adapter.calls, [])
        self.assertEqual(memory.facts, {})
        self.assertEqual(memory.progress, [])
        self.assertEqual([event for event in runtime.events if event["event"] == "physical_dispatch"], [])

    def test_base_retries_timed_out_action_as_distinct_physical_attempt(self):
        adapter = Adapter()
        runtime = AdaExecution(adapter, fault="response_loss", timeout=.05, response_delay=.2)
        self.addCleanup(runtime.close)
        memory = Memory(":memory:", "base")
        self.addCleanup(memory.close)
        controller = BaseController(memory, Budget(2, 0, 2, 20000, 1), runtime)
        result = controller.run("grasp", SimpleNamespace(propose=lambda instruction, context: action()))
        self.assertEqual(result.status, "completed")
        self.assertEqual([call[0] for call in adapter.calls], ["action-1", "action-1"])
        self.assertNotEqual(adapter.calls[0][1], adapter.calls[1][1])
        self.assertEqual(len([event for event in runtime.events if event["event"] == "physical_dispatch"]), 2)
        self.assertEqual(memory.progress, [])

    def test_guardian_does_not_commit_false_software_success(self):
        adapter = Adapter()
        runtime = AdaExecution(adapter, fault="false_success", timeout=.5)
        self.addCleanup(runtime.close)
        memory = Memory(":memory:", "guardian")
        self.addCleanup(memory.close)
        spec = SkillSpec("arm", "grasp", "v1", lambda parameters: True,
                         lambda proposal: "grasped?", lambda proposal, before: Signal.PASS,
                         lambda proposal, before, after: Signal.FAIL, lambda *args: None,
                         lambda *args: {"grasped": True}, ("post",))
        before = runtime.observe(None)
        with Guardian(memory, [spec], runtime, runtime, Verifier(lambda *args: Signal.INDETERMINATE),
                      Budget(2, 0, 2, 20000, 1)) as guardian:
            result = guardian.execute(action(), observation=before)
        self.assertNotEqual(result.receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(memory.facts, {})
        self.assertEqual(adapter.calls, [])

    def test_guardian_queries_original_attempt_after_response_loss(self):
        adapter = Adapter()
        runtime = AdaExecution(adapter, fault="response_loss", timeout=.05, response_delay=.2)
        self.addCleanup(runtime.close)
        memory = Memory(":memory:", "guardian")
        self.addCleanup(memory.close)
        spec = SkillSpec("arm", "grasp", "v1", lambda parameters: True,
                         lambda proposal: "grasped?", lambda proposal, before: Signal.PASS,
                         lambda proposal, before, after: Signal.PASS, lambda *args: None,
                         lambda *args: {"grasped": True}, ("post",))
        before = runtime.observe(None)
        with Guardian(memory, [spec], runtime, runtime, Verifier(lambda *args: Signal.INDETERMINATE),
                      Budget(2, 0, 2, 20000, 1)) as guardian:
            first = guardian.execute(action(), observation=before)
            self.assertIsNone(first.receipt.runtime_result.terminal)
            final = guardian.reconcile("action-1")
        self.assertEqual(final.recovery_status, "none")
        self.assertTrue(memory.get_invocation("action-1")["runtime_terminal"])
        self.assertEqual(len(adapter.calls), 1)
        self.assertTrue(any(event["event"] == "status_query" for event in runtime.events))

    def test_observation_waits_for_physical_execution(self):
        started, release, observed = Event(), Event(), Event()

        class BlockingAdapter(Adapter):
            def execute(self, proposal, attempt):
                started.set()
                release.wait(2)
                return super().execute(proposal, attempt)

            def observe(self, proposal):
                observed.set()
                return super().observe(proposal)

        runtime = AdaExecution(BlockingAdapter(), timeout=1)
        self.addCleanup(runtime.close)
        dispatch = Thread(target=lambda: runtime.execute(action(), "attempt"))
        dispatch.start()
        self.assertTrue(started.wait(1))
        observer = Thread(target=lambda: runtime.observe(None))
        observer.start()
        try:
            self.assertFalse(observed.wait(.05))
        finally:
            release.set()
            dispatch.join(2)
            observer.join(2)
        self.assertTrue(observed.is_set())
