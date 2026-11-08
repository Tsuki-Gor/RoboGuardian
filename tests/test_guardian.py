"""Scripted adapters exercise governance; they do not represent robot validation."""

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from roboguardian.memory import EpisodeBusy, Memory
from roboguardian.types import (
    Action, AdmissionResult, Budget, EffectMeasurement, Observation, ReceiptState, RuntimeResult,
    Signal, SkillSpec,
)
from roboguardian.verification import Verifier


def observation(number, effect="yes", **changes):
    values = dict(observation_id=f"o{number}", source_id="camera", sequence=number,
                  scene_id="table", entity_ids=("box",), state={"effect": effect},
                  facts={"ready": True})
    values.update(changes)
    return Observation(**values)


def action(number=1, **changes):
    values = dict(invocation_id=f"a{number}", goal_id="placed", embodiment="arm",
                  skill="place", parameters={}, scene_id="table", entity_ids=("box",),
                  based_on_observation=f"o{number}", expected_state_version=0,
                  required_facts={"ready": True}, skill_config_version="v1")
    values.update(changes)
    return Action(**values)


def signal(value):
    return {"yes": Signal.PASS, "no": Signal.FAIL, "unknown": Signal.INDETERMINATE}[value]


def skill(name="place", safe_retry=False, **changes):
    def measure(action, before, after):
        effect = after.state["effect"]
        return EffectMeasurement({"yes": 0.01, "no": 0.5, "unknown": None}[effect],
                                 0.01, 0.1, 0.1, "box", "world", "m", "rad")
    values = dict(embodiment="arm", skill=name, config_version="v1",
                  validator=lambda parameters: parameters.get("valid", True) is True,
                  describe=lambda action: "Is the box placed?",
                  pre_check=lambda action, before: Signal.PASS,
                  post_check=lambda action, before, after: signal(after.state.get("integrity", after.state["effect"])),
                  measure_effect=measure,
                  committed_facts=lambda action, before, after: {action.goal_id: True},
                  required_channels=("effect",), safe_retry=safe_retry)
    values.update(changes)
    return SkillSpec(**values)


class Observer:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def observe(self, action):
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class Runtime:
    def __init__(self, terminal=True, error=None):
        self.terminal = terminal
        self.error = error
        self.calls = []
        self.status_calls = []
        self.on_execute = None

    def execute(self, action, attempt_id):
        self.calls.append((action, attempt_id))
        if self.on_execute:
            self.on_execute()
        if self.error:
            raise self.error
        return RuntimeResult("completed", self.terminal, action.invocation_id, attempt_id)

    def status(self, action, attempt_id):
        self.status_calls.append((action, attempt_id))
        return RuntimeResult("checked", self.terminal, action.invocation_id, attempt_id)


class Planner:
    def __init__(self, callback):
        self.callback = callback
        self.calls = 0

    def propose(self, instruction, context):
        self.calls += 1
        return self.callback(context)


class GuardianTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("roboguardian.guardian"),
                             "Guardian orchestration must be implemented")
        from roboguardian.guardian import Guardian
        self.Guardian = Guardian
        self.memory = Memory(":memory:", "episode")
        self.addCleanup(self.memory.close)
        self.budget = Budget(2, 1, 4, 20000, 0)

    def guardian(self, observer, runtime=None, skills=None, memory=None, budget=None, verifier=None):
        guardian = self.Guardian(memory or self.memory, skills or [skill()], runtime or Runtime(),
                                 observer, verifier or Verifier(lambda question, obs: signal(obs.state["effect"])),
                                 budget or self.budget)
        self.addCleanup(guardian.close)
        return guardian

    def test_admission_rejects_outdated_and_inapplicable_proposals_before_dispatch(self):
        runtime = Runtime()
        guard = self.guardian(Observer(), runtime)
        bad_actions = [action(scene_id="other"), action(entity_ids=("missing",)),
                       action(embodiment="leg"), action(skill="unknown"),
                       action(skill_config_version="v2"), action(parameters={"valid": False}),
                       action(expected_state_version=1), action(based_on_observation="old"),
                       action(required_facts={"missing": True}), action(required_facts={"ready": False})]
        for proposal in bad_actions:
            with self.subTest(proposal=proposal):
                self.assertFalse(guard.execute(proposal, observation=observation(1)).admission.allowed)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(self.memory.progress, [])

    def test_precondition_failure_and_exception_never_dispatch(self):
        for check in (lambda action, obs: Signal.FAIL,
                      lambda action, obs: Signal.INDETERMINATE,
                      lambda action, obs: 1 / 0):
            with self.subTest(check=check):
                runtime = Runtime()
                guard = self.guardian(Observer(), runtime, [skill(pre_check=check)])
                self.assertFalse(guard.execute(action(), observation=observation(1)).admission.allowed)
                self.assertEqual(runtime.calls, [])
                guard.close()

    def test_idempotent_invocation_and_conflicting_payload(self):
        runtime = Runtime()
        guard = self.guardian(Observer(observation(2)), runtime)
        first = guard.execute(action(), observation=observation(1))
        repeat = guard.execute(action())
        self.assertEqual(first.receipt.receipt_id, repeat.receipt.receipt_id)
        conflict = guard.execute(action(parameters={"x": 1}))
        self.assertFalse(conflict.admission.allowed)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(self.memory.state_version, 1)

    def test_timeout_reconciliation_uses_original_before_once(self):
        runtime = Runtime(terminal=None, error=TimeoutError("response lost"))
        observer = Observer(observation(2, "unknown"), observation(3))
        guard = self.guardian(observer, runtime)
        first = guard.execute(action(), observation=observation(1))
        self.assertIs(first.receipt.state, ReceiptState.UNKNOWN)
        runtime.terminal = True
        reconciled = guard.reconcile("a1")
        self.assertIs(reconciled.receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(self.memory.progress), 1)
        self.assertEqual(guard.reconcile("a1").receipt.receipt_id, reconciled.receipt.receipt_id)
        reports = [event["payload"]["report"] for event in self.memory.events() if event["kind"] == "receipt"]
        self.assertEqual([report["before_observation_id"] for report in reports], ["o1", "o1"])
        self.assertEqual([report["after_observation_id"] for report in reports], ["o2", "o3"])

    def test_observation_budget_survives_restart_and_can_be_explicitly_increased(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            path = str(Path(directory) / "memory.sqlite")
            memory = Memory(path, "episode")
            observer = Observer(observation(2, "unknown"), observation(3, "unknown"), observation(4, "unknown"), observation(5))
            runtime = Runtime()
            guard = self.guardian(observer, runtime, memory=memory)
            guard.execute(action(), observation=observation(1))
            guard.reconcile("a1")
            result = guard.reconcile("a1")
            self.assertEqual(result.recovery_status, "unresolved")
            guard.close()
            memory.close()
            memory = Memory(path, "episode")
            guard = self.guardian(observer, runtime, memory=memory)
            self.assertEqual(guard.reconcile("a1").recovery_status, "unresolved")
            self.assertEqual(observer.calls, 3)
            self.assertEqual(memory.get_invocation("a1")["observations"], 2)
            guard.close()
            guard = self.guardian(observer, runtime, memory=memory, budget=replace(self.budget, max_observations=3))
            self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.COMMITTED)
            self.assertEqual(len(runtime.calls), 1)
            guard.close()
            memory.close()

    def test_nonterminal_yes_and_no_block_every_new_dispatch(self):
        for effect, state in (("yes", ReceiptState.COMMITTED), ("no", ReceiptState.REJECTED)):
            with self.subTest(effect=effect):
                memory = Memory(":memory:", "episode")
                runtime = Runtime(terminal=None)
                guard = self.guardian(Observer(observation(2, effect)), runtime, memory=memory)
                result = guard.execute(action(), observation=observation(1))
                self.assertIs(result.receipt.state, state)
                self.assertEqual(result.recovery_status, "awaiting_terminal")
                candidate = action(3, goal_id="different", expected_state_version=memory.state_version)
                self.assertFalse(guard.execute(candidate, observation=observation(3)).admission.allowed)
                self.assertFalse(guard.recover("a1", candidate, observation=observation(3)).admission.allowed)
                runtime.terminal = True
                terminal = guard.reconcile("a1")
                self.assertEqual(terminal.receipt.receipt_id, result.receipt.receipt_id)
                self.assertIsNone(memory.pending_attempt())
                self.assertEqual(len(runtime.calls), 1)
                guard.close()
                memory.close()

    def test_recovery_permissions_root_budget_and_changed_ids(self):
        runtime = Runtime()
        guard = self.guardian(Observer(observation(2, "no"), observation(4, "no")), runtime,
                              [skill(), skill("push")])
        guard.execute(action(), observation=observation(1))
        same = action(3)
        self.assertFalse(guard.recover("a1", same, observation=observation(3)).admission.allowed)
        other = action(3, skill="push", goal_id="new-goal")
        self.assertFalse(guard.execute(other, observation=observation(3)).admission.allowed)
        recovered = guard.recover("a1", other, observation=observation(3))
        self.assertIs(recovered.receipt.state, ReceiptState.REJECTED)
        self.assertEqual(self.memory.get_invocation("a3")["recovery_root"], "a1")
        exhausted = guard.recover("a3", action(5, goal_id="third"), observation=observation(5))
        self.assertEqual(exhausted.recovery_status, "budget_exhausted")
        self.assertEqual(len(runtime.calls), 2)

    def test_safe_retry_has_new_attempt_and_its_own_observation_budget(self):
        runtime = Runtime()
        observer = Observer(observation(2, "unknown"), observation(3, "no"),
                            observation(5, "unknown"), observation(6))
        guard = self.guardian(observer, runtime, [skill(safe_retry=True)],
                              budget=replace(self.budget, max_observations=1))
        guard.execute(action(), observation=observation(1))
        self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.REJECTED)
        guard.recover("a1", action(4), observation=observation(4))
        self.assertIs(guard.reconcile("a4").receipt.state, ReceiptState.COMMITTED)
        self.assertNotEqual(runtime.calls[0][1], runtime.calls[1][1])
        self.assertEqual(self.memory.get_invocation("a4")["observations"], 1)

    def test_slipped_premise_blocks_dependency_and_final_goal_without_rewriting_history(self):
        runtime = Runtime()
        observer = Observer(observation(2), observation(4, facts={"ready": False}))
        guard = self.guardian(observer, runtime)
        committed = guard.execute(action(), observation=observation(1)).receipt
        denied = guard.execute(action(3, expected_state_version=1),
                               observation=observation(3, facts={"ready": False}))
        self.assertFalse(denied.admission.allowed)
        result = guard.run("place", Planner(lambda context: None),
                           lambda obs, context: Signal.PASS if obs.facts.get("ready") is True else Signal.FAIL)
        self.assertEqual(result.status, "needs_plan")
        self.assertEqual(self.memory.get_invocation("a1")["receipt"].receipt_id, committed.receipt_id)
        self.assertEqual(self.memory.state_version, 1)

    def test_goal_integrity_is_checked_before_receipt_commit(self):
        guard = self.guardian(Observer(observation(2, state={"effect": "yes", "integrity": "unknown"})))
        self.assertIs(guard.execute(action(), observation=observation(1)).receipt.state, ReceiptState.UNKNOWN)
        self.assertEqual(self.memory.facts, {})

    def test_provider_failures_remain_unknown_with_audit(self):
        runtime = Runtime(error=RuntimeError("may have moved"))
        guard = self.guardian(Observer(RuntimeError("camera lost"), None), runtime)
        first = guard.execute(action(), observation=observation(1))
        self.assertIs(first.receipt.state, ReceiptState.UNKNOWN)
        second = guard.reconcile("a1")
        self.assertIs(second.receipt.state, ReceiptState.UNKNOWN)
        self.assertEqual(len(runtime.calls), 1)
        self.assertTrue(any(event["kind"] == "provider_error" for event in self.memory.events()))

    def test_assessor_and_fact_callback_failure_cannot_commit(self):
        def fail(*args):
            raise RuntimeError("provider failed")
        for verifier, registered in ((Verifier(fail), skill()), (None, skill(committed_facts=fail))):
            with self.subTest(verifier=verifier):
                memory = Memory(":memory:", "episode")
                guard = self.guardian(Observer(observation(2)), memory=memory, verifier=verifier, skills=[registered])
                self.assertIs(guard.execute(action(), observation=observation(1)).receipt.state, ReceiptState.UNKNOWN)
                self.assertEqual(memory.progress, [])
                guard.close()
                memory.close()

    def test_restart_after_intent_reconciles_without_execute(self):
        attempt = self.memory.begin_attempt(action(), observation(1))
        runtime = Runtime()
        guard = self.guardian(Observer(observation(2)), runtime)
        result = guard.reconcile("a1")
        self.assertIs(result.receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(result.receipt.attempt_id, attempt)
        self.assertEqual(runtime.calls, [])

    def test_active_guardian_and_reentrant_callback_are_rejected(self):
        runtime = Runtime()
        guard = self.guardian(Observer(observation(2)), runtime)
        with self.assertRaises(EpisodeBusy):
            self.guardian(Observer())
        def reenter():
            with self.assertRaises(EpisodeBusy):
                guard.execute(action(3))
        runtime.on_execute = reenter
        self.assertIs(guard.execute(action(), observation=observation(1)).receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(len(runtime.calls), 1)

    def test_run_reuses_planning_snapshot_and_requires_fresh_goal_confirmation(self):
        observer = Observer(observation(1), observation(2), observation(3))
        runtime = Runtime()
        guard = self.guardian(observer, runtime)
        planner = Planner(lambda context: action(based_on_observation=context.data["observation"]["observation_id"]))
        result = guard.run("place", planner,
                           lambda obs, context: Signal.PASS if context.data["progress"]["committed_count"] else Signal.FAIL)
        self.assertEqual(result.status, "verified_success")
        self.assertEqual(result.steps, 1)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(observer.calls, 3)
        self.assertEqual(len(runtime.calls), 1)

    def test_planner_none_and_error_are_not_success_and_budget_persists(self):
        observer = Observer(observation(1), observation(2), observation(3))
        runtime = Runtime()
        guard = self.guardian(observer, runtime, budget=replace(self.budget, max_plan_steps=1))
        planner = Planner(lambda context: None)
        self.assertEqual(guard.run("place", planner, lambda obs, context: Signal.FAIL).status, "needs_plan")
        self.assertEqual(guard.run("place", planner, lambda obs, context: Signal.FAIL).status, "budget_exhausted")
        self.assertEqual(planner.calls, 1)
        guard.close()
        guard = self.guardian(observer, runtime, budget=replace(self.budget, max_plan_steps=2))
        failing = Planner(lambda context: 1 / 0)
        self.assertEqual(guard.run("place", failing, lambda obs, context: Signal.FAIL).status, "failed")
        self.assertEqual(runtime.calls, [])

    def test_replayed_after_cannot_upgrade_unknown_when_callbacks_change(self):
        current = {"signal": Signal.INDETERMINATE}
        registered = skill(measure_effect=lambda *args: None, required_channels=("semantic",),
                           post_check=lambda *args: Signal.PASS)
        after = observation(2)
        observer = Observer(after, after, observation(3))
        guard = self.guardian(observer, skills=[registered],
                              verifier=Verifier(lambda *args: current["signal"]))
        guard.execute(action(), observation=observation(1))
        current["signal"] = Signal.PASS
        self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.UNKNOWN)
        self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.COMMITTED)

    def test_attempt_measurement_configuration_cannot_drift_during_reconciliation(self):
        for field, replacement in (("translation_tolerance", 1.0), ("rotation_tolerance", 1.0),
                                   ("frame_id", "other"), ("translation_unit", "cm"), ("rotation_unit", "deg")):
            with self.subTest(field=field):
                memory = Memory(":memory:", "episode")
                config = {}
                baseline = EffectMeasurement(None, 0.01, 0.1, 0.1, "box", "world", "m", "rad")
                registered = skill(measure_effect=lambda *args: replace(baseline, **config))
                guard = self.guardian(Observer(observation(2, "unknown"), observation(3)), memory=memory, skills=[registered])
                guard.execute(action(), observation=observation(1))
                config.update(translation_error=0.01, **{field: replacement})
                self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.UNKNOWN)
                self.assertEqual(memory.progress, [])
                guard.close()
                memory.close()

    def test_before_snapshot_cannot_be_consumed_by_a_second_attempt(self):
        guard = self.guardian(Observer(RuntimeError("camera lost"), observation(2, "no")), skills=[skill(safe_retry=True)])
        guard.execute(action(), observation=observation(1))
        guard.reconcile("a1")
        retry = action(9, based_on_observation="o1")
        self.assertFalse(guard.recover("a1", retry, observation=observation(1)).admission.allowed)

    def test_fact_callback_error_preserves_original_measurement_configuration(self):
        values = {"tolerance": 0.1, "fail_facts": True}
        def facts(*args):
            if values["fail_facts"]:
                raise RuntimeError("fact adapter temporarily unavailable")
            return {"placed": True}
        registered = skill(committed_facts=facts, measure_effect=lambda *args:
                           EffectMeasurement(0.01, 0.01, values["tolerance"], 0.1, "box", "world", "m", "rad"))
        guard = self.guardian(Observer(observation(2), observation(3)), skills=[registered])
        self.assertIs(guard.execute(action(), observation=observation(1)).receipt.state, ReceiptState.UNKNOWN)
        values.update(tolerance=1.0, fail_facts=False)
        self.assertIs(guard.reconcile("a1").receipt.state, ReceiptState.UNKNOWN)
        self.assertEqual(self.memory.progress, [])

    def test_run_cannot_report_success_while_committed_runtime_is_unconfirmed(self):
        runtime = Runtime(terminal=None)
        guard = self.guardian(Observer(observation(2)), runtime)
        guard.execute(action(), observation=observation(1))
        planner = Planner(lambda context: None)
        result = guard.run("place", planner, lambda obs, context: Signal.PASS)
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(len(runtime.calls), 1)

    def test_run_bounds_repeated_outdated_plans_across_guardian_restart(self):
        runtime = Runtime()
        observer = Observer(observation(1), observation(2), observation(3), observation(4))
        budget = replace(self.budget, max_plan_steps=2)
        guard = self.guardian(observer, runtime, budget=budget)
        planner = Planner(lambda context: action(expected_state_version=99,
                          based_on_observation=context.data["observation"]["observation_id"]))
        first = guard.run("place", planner, lambda obs, context: Signal.FAIL)
        self.assertEqual((first.status, first.steps, planner.calls), ("budget_exhausted", 2, 2))
        guard.close()
        guard = self.guardian(observer, runtime, budget=budget)
        self.assertEqual(guard.run("place", planner, lambda obs, context: Signal.FAIL).status, "budget_exhausted")
        self.assertEqual(planner.calls, 2)
        self.assertEqual(runtime.calls, [])

    def test_context_budget_error_never_drops_constraints_to_dispatch(self):
        runtime = Runtime()
        planner = Planner(lambda context: action())
        guard = self.guardian(Observer(observation(1)), runtime, budget=replace(self.budget, context_bytes=1))
        self.assertEqual(guard.run("place", planner, lambda obs, context: Signal.FAIL).status, "budget_exhausted")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(runtime.calls, [])

    def test_context_manager_exception_releases_episode_lease(self):
        with self.assertRaises(ValueError):
            with self.guardian(Observer()):
                raise ValueError("caller failed")
        guard = self.guardian(Observer(observation(2)))
        self.assertIs(guard.execute(action(), observation=observation(1)).receipt.state, ReceiptState.COMMITTED)

    def test_planner_uses_admission_feedback_to_correct_the_next_proposal(self):
        contexts = []
        def propose(context):
            contexts.append(context)
            feedback = context.data.get("diagnostics", ())
            corrected = any(item["code"] == "invalid_parameters" for item in feedback)
            return action(len(contexts), parameters={"valid": corrected},
                          based_on_observation=context.data["observation"]["observation_id"])
        runtime = Runtime()
        guard = self.guardian(Observer(observation(1), observation(2), observation(3), observation(4)), runtime)
        result = guard.run("place", Planner(propose),
                           lambda obs, context: Signal.PASS if context.data["progress"]["committed_count"] else Signal.FAIL)
        self.assertEqual(result.status, "verified_success")
        self.assertEqual(len(runtime.calls), 1)
        diagnostic = contexts[1].data["diagnostics"][0]
        self.assertEqual((diagnostic["stage"], diagnostic["code"], diagnostic["invocation_id"]),
                         ("admission", "invalid_parameters", "a1"))
        self.assertEqual(self.memory.facts, {"placed": True})
        self.assertEqual(self.memory.context(observation(5), self.budget, {}).data["diagnostics"], ())

    def test_admission_feedback_rechecks_only_its_required_facts(self):
        guard = self.guardian(Observer())
        denied = guard.execute(action(), observation=observation(1, facts={"ready": False}))
        self.assertEqual(getattr(denied.admission, "code", None), "required_facts_unconfirmed")
        self.memory.record_validity(observation(2, facts={"unrelated": True}), {"unrelated": True})
        context = self.memory.context(observation(3, facts={"ready": False}), self.budget, {})
        diagnostic = context.data["diagnostics"][0]
        self.assertEqual(diagnostic["planning_observation_id"], "o1")
        self.assertEqual(diagnostic["current_validity"]["ready"]["status"], "conflict")
        self.assertEqual(diagnostic["current_validity"]["ready"]["observation_id"], "o3")
        self.assertNotIn("unrelated", diagnostic["current_validity"])
        self.assertEqual(self.memory.facts, {})

    def test_admission_codes_reject_untyped_and_unrecognized_values(self):
        for value in (None, 7, "provider secret text"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                AdmissionResult(False, "reason", code=value)

    def test_committed_relations_flow_through_guardian_commit_and_context_once(self):
        calls = []
        def relations(action, before, after):
            calls.append((before.observation_id, after.observation_id))
            return [{"subject": "box", "relation": "on", "object": "table"}]
        registered = skill()
        self.assertIn("committed_relations", registered.__dataclass_fields__)
        guard = self.guardian(Observer(observation(2)), skills=[replace(registered, committed_relations=relations)])
        receipt = guard.execute(action(), observation=observation(1)).receipt
        guard.execute(action())
        guard.reconcile("a1")
        relation = self.memory.relations["dynamic_objects"][0]
        self.assertEqual(relation["receipt_id"], receipt.receipt_id)
        self.assertEqual(relation["relation"], {"subject": "box", "relation": "on", "object": "table"})
        self.assertEqual(calls, [("o1", "o2")])
        self.assertEqual(len(self.memory.relations["dynamic_objects"]), 1)
        self.assertEqual(len(self.memory.context(observation(3), self.budget, {}).data["relations"]["dynamic_objects"]), 1)

    def test_noncommitted_effects_never_call_relation_generator(self):
        for effect in ("no", "unknown"):
            with self.subTest(effect=effect):
                registered = skill()
                self.assertIn("committed_relations", registered.__dataclass_fields__)
                calls = []
                memory = Memory(":memory:", "episode")
                guard = self.guardian(Observer(observation(2, effect)), memory=memory,
                                      skills=[replace(registered, committed_relations=lambda *args: calls.append(args))])
                guard.execute(action(), observation=observation(1))
                self.assertEqual(calls, [])
                self.assertEqual(memory.relations["dynamic_objects"], [])
                self.assertEqual(memory.facts, {})
                guard.close()
                memory.close()

    def test_relation_generator_errors_leave_unknown_without_partial_commit(self):
        def fail(*args):
            raise RuntimeError("private provider error")
        for callback in (fail, lambda *args: {"wrong": "shape"}, lambda *args: [1],
                         lambda *args: [{"bad": float("nan")}], lambda *args: [{"bad": object()}]):
            with self.subTest(callback=callback):
                registered = skill()
                self.assertIn("committed_relations", registered.__dataclass_fields__)
                memory = Memory(":memory:", "episode")
                guard = self.guardian(Observer(observation(2)), memory=memory,
                                      skills=[replace(registered, committed_relations=callback)])
                result = guard.execute(action(), observation=observation(1))
                self.assertIs(result.receipt.state, ReceiptState.UNKNOWN)
                self.assertEqual(memory.facts, {})
                self.assertEqual(memory.progress, [])
                self.assertEqual(memory.relations["dynamic_objects"], [])
                report = [item["payload"]["report"] for item in memory.events() if item["kind"] == "receipt"][-1]
                self.assertIn("measurement", report["diagnostics"])
                guard.close()
                memory.close()


if __name__ == "__main__":
    unittest.main()
