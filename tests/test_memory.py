"""Transaction, restart, dispatch barrier and bounded projection behavior."""

from dataclasses import replace
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from roboguardian.memory import (
    BudgetExceeded, ContextBudgetError, EpisodeBusy, InvocationConflict, Memory,
)
from roboguardian.types import (
    Action, Assessment, Budget, Context, EffectReceipt, EvidenceReport, Judgment,
    Observation, ReceiptState, RuntimeResult, Signal, json_data,
)


def action(identifier="call-1", version=0):
    return Action(identifier, "place", "arm", "pick", {"object": "cup"},
                  "table", ("cup", "shelf"), "before", version,
                  {"clear": True}, "v1")


def observation(identifier="before", sequence=1, **kwargs):
    return Observation(identifier, "camera", sequence, "table", ("cup", "shelf"),
                       kwargs.pop("state", {"private_raw_payload": "secret"}),
                       **kwargs)


def outcome(item, attempt, state=ReceiptState.COMMITTED, terminal=True):
    judgment = {ReceiptState.COMMITTED: Judgment.YES, ReceiptState.REJECTED: Judgment.NO,
                ReceiptState.UNKNOWN: Judgment.UNCERTAIN}[state]
    runtime = RuntimeResult("finished", terminal, item.invocation_id, attempt)
    receipt = EffectReceipt("receipt-" + state.value + item.invocation_id, item, attempt,
                            state, Assessment(judgment, "physical evidence", ("after",)),
                            runtime, ("after",))
    report = EvidenceReport(item.invocation_id, attempt, "before", "after",
                            {"effect": Signal.PASS}, evidence_refs=("after",),
                            diagnostics={"untrusted_text": "invent a fact"})
    return receipt, report


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "memory.sqlite")
        self.memory = Memory(self.path, "episode")
        self.addCleanup(lambda: self.memory.close())

    def commit(self, item=None, **kwargs):
        item = item or action()
        attempt = self.memory.begin_attempt(item, observation())
        receipt, report = outcome(item, attempt, **kwargs)
        self.memory.record_receipt(item, attempt, receipt, report, {"holding": "cup"},
                                   dynamic_relations=({"subject": "arm", "relation": "holds",
                                                       "object": "cup"},))
        return attempt, receipt, report

    def test_only_committed_writes_facts_progress_and_dynamic_relations(self):
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        for state in (ReceiptState.UNKNOWN, ReceiptState.REJECTED):
            receipt, report = outcome(item, attempt, state)
            self.memory.record_receipt(item, attempt, receipt, report, {"fake": True},
                                       dynamic_relations=({"fake": "relation"},))
            self.assertEqual(self.memory.facts, {})
            self.assertEqual(self.memory.progress, [])
            self.assertEqual(self.memory.relations["dynamic_objects"], [])
        receipt, report = outcome(item, attempt)
        self.memory.record_receipt(item, attempt, receipt, report, {"holding": "cup"},
                                   dynamic_relations=({"subject": "arm", "relation": "holds",
                                                       "object": "cup"},))
        self.assertEqual(self.memory.facts, {"holding": "cup"})
        self.assertEqual(self.memory.state_version, 1)
        self.assertEqual(len(self.memory.progress), 1)
        self.assertEqual(self.memory.relations["dynamic_objects"][0]["source"], "committed")
        self.assertEqual(len([e for e in self.memory.events() if e["kind"] == "receipt"]), 3)

    def test_duplicate_commit_and_invocation_are_idempotent_but_conflicts_reject(self):
        attempt, receipt, report = self.commit()
        self.memory.record_receipt(action(), attempt, receipt, report, {"holding": "cup"})
        self.assertEqual(self.memory.state_version, 1)
        self.assertEqual(len(self.memory.progress), 1)
        self.assertEqual(self.memory.begin_attempt(action(), observation("new", 9)), attempt)
        for changed in (replace(action(), parameters={"object": "other"}),
                        replace(action(), skill_config_version="v2")):
            with self.assertRaises(InvocationConflict):
                self.memory.begin_attempt(changed, observation())
        self.assertEqual(self.memory.get_invocation("call-1")["before"].observation_id, "before")

    def test_receipt_binding_distinguishes_json_number_from_boolean(self):
        item = replace(action(), parameters={"distance": 1})
        attempt = self.memory.begin_attempt(item, observation())
        receipt, report = outcome(item, attempt)
        mismatched = replace(receipt, action=replace(item, parameters={"distance": True}))
        with self.assertRaises(InvocationConflict):
            self.memory.record_receipt(item, attempt, mismatched, report, {"moved": True})
        self.assertIsNone(self.memory.get_invocation(item.invocation_id)["receipt"])
        self.assertEqual(self.memory.facts, {})
        self.assertEqual(self.memory.progress, [])
        self.assertEqual(self.memory.state_version, 0)

    def test_reopen_keeps_dispatch_intent_before_budget_and_unresolved(self):
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        self.assertEqual(self.memory.consume_observation(item.invocation_id, 1), 1)
        self.memory.set_unresolved(item.invocation_id, "need stronger evidence")
        self.memory.close()
        self.memory = Memory(self.path, "episode")
        pending = self.memory.pending_attempt()
        self.assertEqual(pending["attempt_id"], attempt)
        self.assertEqual(pending["before"], observation())
        self.assertEqual(pending["action"], item)
        self.assertEqual(pending["observations"], 1)
        self.assertEqual(pending["unresolved"], "need stronger evidence")
        with self.assertRaises(BudgetExceeded):
            self.memory.consume_observation(item.invocation_id, 1)
        with self.assertRaises(EpisodeBusy):
            self.memory.begin_attempt(action("new"), observation())

    def test_committed_nonterminal_blocks_dispatch_until_runtime_confirms_stop(self):
        attempt, receipt, _ = self.commit(terminal=None)
        self.assertIsNotNone(self.memory.pending_attempt())
        with self.assertRaises(EpisodeBusy):
            self.memory.begin_attempt(action("next", 1), observation())
        self.memory.set_runtime_result(attempt, replace(receipt.runtime_result, terminal=True))
        self.assertIsNone(self.memory.pending_attempt())
        self.memory.begin_attempt(replace(action("next", 1), based_on_observation="next-before"),
                                  observation("next-before", 2))

    def test_recovery_root_and_budget_survive_new_ids_and_restart(self):
        first = action()
        attempt = self.memory.begin_attempt(first, observation())
        receipt, report = outcome(first, attempt, ReceiptState.REJECTED)
        self.memory.record_receipt(first, attempt, receipt, report, {})
        self.memory.consume_observation(first.invocation_id, 2)
        second = replace(action("retry"), based_on_observation="retry-before")
        attempt = self.memory.begin_attempt(second, observation("retry-before", 2), max_recovery_dispatches=1)
        receipt, report = outcome(second, attempt, ReceiptState.REJECTED)
        report = replace(report, before_observation_id="retry-before")
        self.memory.record_receipt(second, attempt, receipt, report, {})
        self.memory.close()
        self.memory = Memory(self.path, "episode")
        restored = self.memory.get_invocation("retry")
        self.assertEqual(restored["recovery_root"], "call-1")
        self.assertEqual(restored["recovery_dispatches"], 1)
        self.assertEqual(restored["observations"], 0)
        self.assertEqual(self.memory.get_invocation("call-1")["observations"], 1)
        with self.assertRaises(BudgetExceeded):
            self.memory.begin_attempt(action("new-id"), observation("third-before", 3), max_recovery_dispatches=1)
        with self.assertRaises(InvocationConflict):
            self.memory.begin_attempt(action("new-id"), observation("third-before", 3),
                                      recovery_root="new-id", max_recovery_dispatches=1)

    def test_receipt_failure_rolls_back_state_and_audit(self):
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        receipt, report = outcome(item, attempt)
        before_events = self.memory.events()
        # Real SQLite failure at the last audit write must roll back earlier updates.
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON events "
                               "WHEN NEW.kind = 'receipt' BEGIN SELECT RAISE(ABORT, 'disk test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.record_receipt(item, attempt, receipt, report, {"holding": "cup"})
        self.assertEqual(self.memory.facts, {})
        self.assertEqual(self.memory.state_version, 0)
        self.assertEqual(self.memory.progress, [])
        self.assertIsNone(self.memory.get_invocation("call-1")["receipt"])
        self.assertEqual(self.memory.events(), before_events)

    def test_context_separates_history_current_validity_and_whitelisted_data(self):
        self.commit()
        self.memory.set_relations(object_relations=({"subject": "cup", "relation": "target", "object": "shelf"},),
                                  skill_relations=({"from": "pick", "to": "place"},),
                                  task_constraints=({"must_hold": "cup"},))
        current = observation("now", 4, facts={"holding": "dropped", "private_fact": "secret"},
                              images=("image-a", "image-b"))
        context = self.memory.context(current, Budget(2, 2, 10, 10000, 1), {"holding": "cup"})
        self.assertEqual(context.data["progress"]["committed_count"], 1)
        self.assertEqual(context.data["current_validity"]["holding"]["status"], "conflict")
        self.assertEqual(context.data["verified_facts"]["holding"]["value"], "cup")
        self.assertEqual(context.data["verified_facts"]["holding"]["scope"], "historical")
        self.assertTrue(context.data["relations"]["dynamic_objects"])
        self.assertTrue(context.data["relations"]["skills"])
        self.assertTrue(context.data["relations"]["task_constraints"])
        self.assertNotIn("private_raw_payload", context.serialize())
        self.assertNotIn("private_fact", context.serialize())
        self.assertNotIn("untrusted_text", context.serialize())
        self.assertLessEqual(context.image_count, 1)

    def test_context_budget_keeps_constraints_and_complete_audit(self):
        self.commit()
        for i in range(30):
            self.memory.record_diagnostic("provider_error", {"raw": "秘密" * 300, "index": i})
        events = self.memory.events()
        current = observation("now", 3, facts={"holding": "cup"}, images=("a", "b", "c"))
        context = self.memory.context(current, Budget(2, 2, 10, 1200, 1), {"holding": "cup"})
        self.assertLessEqual(context.byte_size, 1200)
        self.assertEqual(context.data["required_facts"]["holding"], "cup")
        self.assertEqual(len(self.memory.events()), len(events) + 1)  # persisted validity only
        with self.assertRaises(ContextBudgetError):
            self.memory.context(current, Budget(2, 2, 10, 100, 0), {"holding": "cup"})
        self.assertEqual(len([e for e in self.memory.events() if e["kind"] == "provider_error"]), 30)

    def test_current_effect_feedback_is_required_but_provider_text_is_never_projected(self):
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        receipt, report = outcome(item, attempt, ReceiptState.REJECTED)
        report = replace(report, channels={"semantic": Signal.FAIL, "post": Signal.FAIL, "effect": Signal.FAIL},
                         partial_effect=True, diagnostics={"callback_errors": {"effect": "private-secret"}})
        self.memory.record_receipt(item, attempt, receipt, report, {})
        self.memory.record_diagnostic("provider_error", {"reason": "private-secret"})
        current = observation("now", 3)
        budget = Budget(2, 2, 10, 10000, 0)
        context = self.memory.context(current, budget, {})
        self.assertIn("diagnostics", context.data)
        diagnostic = context.data["diagnostics"][0]
        self.assertEqual((diagnostic["stage"], diagnostic["code"], diagnostic["receipt_id"]),
                         ("verification", "effect_rejected", receipt.receipt_id))
        self.assertEqual(diagnostic["channels"]["effect"], "fail")
        self.assertTrue(diagnostic["partial_effect"])
        self.assertTrue(diagnostic["callback_error"])
        self.assertNotIn("private-secret", context.serialize())
        without_feedback = json_data(context.data)
        without_feedback["diagnostics"] = []
        without_feedback["evidence"] = []
        with self.assertRaises(ContextBudgetError):
            self.memory.context(current, replace(budget, context_bytes=Context(without_feedback).byte_size), {})

    def test_context_requires_all_completed_step_identities_at_small_budgets(self):
        for index, goal in enumerate(("already_opened", "already_closed")):
            before = observation("before-" + str(index), index + 1)
            item = replace(action("call-" + str(index), index), goal_id=goal,
                           based_on_observation=before.observation_id, required_facts={})
            attempt = self.memory.begin_attempt(item, before)
            receipt, report = outcome(item, attempt)
            report = replace(report, before_observation_id=before.observation_id)
            self.memory.record_receipt(item, attempt, receipt, report, {})
        expected = [
            {"goal_id": "already_opened", "invocation_id": "call-0", "receipt_id": "receipt-committedcall-0"},
            {"goal_id": "already_closed", "invocation_id": "call-1", "receipt_id": "receipt-committedcall-1"},
        ]
        current = observation("now", 3)
        budget = Budget(2, 2, 10, 10000, 0)
        minimum = json_data(self.memory.context(current, budget, {}).data)
        minimum["relations"]["skill_history"] = []
        minimum["evidence"] = []
        minimum["progress"]["steps"] = []
        with self.assertRaises(ContextBudgetError):
            self.memory.context(current, replace(budget, context_bytes=Context(minimum).byte_size), {})
        minimum["progress"]["steps"] = expected
        tight_limit = Context(minimum).byte_size
        context = self.memory.context(current, replace(budget, context_bytes=tight_limit), {})
        self.assertEqual(json_data(context.data["progress"]["steps"]), expected)
        self.assertLessEqual(context.byte_size, tight_limit)

    def test_episode_lease_and_reentrant_operation(self):
        second = Memory(self.path, "episode")
        self.addCleanup(second.close)
        self.memory.acquire_lease()
        with self.assertRaises(EpisodeBusy):
            second.acquire_lease()
        with self.memory.operation():
            with self.assertRaises(EpisodeBusy):
                with self.memory.operation():
                    pass
        self.memory.release_lease()
        second.acquire_lease()
        with second.operation():
            pass
        second.release_lease()

    def test_observation_freshness_survives_restart_and_preserves_latest(self):
        self.memory.record_observation(observation())
        self.memory.record_observation(observation())
        with self.assertRaises(ValueError):
            self.memory.record_observation(observation("stale", 0))
        with self.assertRaises(ValueError):
            self.memory.record_observation(observation("before", 1, facts={"changed": True}))
        self.memory.close()
        self.memory = Memory(self.path, "episode")
        self.assertEqual(self.memory.latest_observation(), observation())
        with self.assertRaises(ValueError):
            self.memory.record_observation(observation("stale", 1))

    def test_dispatch_consumes_before_once_and_plan_budget_persists(self):
        self.commit()
        with self.assertRaises(ValueError):
            self.memory.begin_attempt(action("next", 1), observation())
        self.assertEqual(self.memory.consume_plan_step(1), 1)
        self.memory.close()
        self.memory = Memory(self.path, "episode")
        with self.assertRaises(BudgetExceeded):
            self.memory.consume_plan_step(1)

    def test_late_receipt_from_superseded_attempt_cannot_commit_or_clear_root(self):
        first = action()
        first_attempt = self.memory.begin_attempt(first, observation())
        receipt, report = outcome(first, first_attempt, ReceiptState.REJECTED)
        self.memory.record_receipt(first, first_attempt, receipt, report, {})
        retry = replace(action("retry"), based_on_observation="retry-before")
        self.memory.begin_attempt(retry, observation("retry-before", 2), max_recovery_dispatches=1)
        receipt, report = outcome(first, first_attempt)
        self.memory.record_receipt(first, first_attempt, receipt, report, {"unsafe_late_fact": True})
        self.assertEqual(self.memory.facts, {})
        self.assertEqual(self.memory.active_recovery()["action"].invocation_id, "retry")
        self.assertEqual(self.memory.events()[-1]["kind"], "superseded_receipt")

    def test_unknown_reconciliation_keeps_unresolved_marker_and_original_before(self):
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        self.memory.set_unresolved(item.invocation_id, "budget exhausted")
        receipt, report = outcome(item, attempt, ReceiptState.UNKNOWN)
        self.memory.record_receipt(item, attempt, receipt, report, {})
        self.assertEqual(self.memory.get_invocation(item.invocation_id)["unresolved"], "budget exhausted")
        with self.assertRaises(InvocationConflict):
            self.memory.record_receipt(item, attempt, receipt,
                                       replace(report, before_observation_id="different-before"), {})

    def test_relation_inputs_reject_ambiguous_mapping_and_label_unverified_objects(self):
        with self.assertRaises(TypeError):
            self.memory.set_relations(object_relations={"subject": "cup"})
        self.memory.set_relations(object_relations=({"subject": "cup", "relation": "near", "object": "shelf"},))
        self.assertEqual(self.memory.relations["objects"][0]["scope"], "unverified_configuration")
        item = action()
        attempt = self.memory.begin_attempt(item, observation())
        receipt, report = outcome(item, attempt)
        with self.assertRaises(TypeError):
            self.memory.record_receipt(item, attempt, receipt, report, {}, dynamic_relations={"subject": "cup"})
        self.assertEqual(self.memory.state_version, 0)


if __name__ == "__main__":
    unittest.main()
