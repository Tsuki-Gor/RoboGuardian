"""Serial admission, dispatch and reconciliation for one persistent episode."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

from .memory import BudgetExceeded, EpisodeBusy, InvocationConflict
from .loop import RunResult, run_loop
from .types import (
    Action, AdmissionResult, Assessment, Budget, Context, EvidenceReport, Judgment,
    Observation, ReceiptState, RuntimeResult, Signal, SkillSpec, StepResult,
    action_fingerprint, canonical_json, json_data,
)
from .verification import make_receipt


class Guardian:
    """Own an episode lease until close(); public operations reject reentry.

    skills is an iterable of trusted, versioned SkillSpec objects. execute and
    recover accept the fresh observation used to plan the action; otherwise the
    observer must supply that exact planning baseline. run collects and reuses
    its own before snapshot. A later snapshot requires a new planner proposal.
    Callbacks run outside SQLite write transactions, under the episode guard.
    """

    def __init__(self, memory, skills, runtime, observer, verifier, budget: Budget):
        registered = {}
        for skill in skills:
            if not isinstance(skill, SkillSpec):
                raise TypeError("skills must contain SkillSpec objects")
            key = (skill.embodiment, skill.skill, skill.config_version)
            if key in registered:
                raise ValueError("duplicate skill identity and configuration")
            registered[key] = skill
        if not isinstance(budget, Budget):
            raise TypeError("budget must be Budget")
        self.skills = MappingProxyType(registered)
        self.memory, self.runtime, self.observer = memory, runtime, observer
        self.verifier, self.budget = verifier, budget
        self._closed = True
        memory.acquire_lease()
        self._closed = False

    def close(self):
        if not self._closed:
            self.memory.release_lease()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise EpisodeBusy("Guardian is closed")
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _operation(self):
        if self._closed:
            raise EpisodeBusy("Guardian is closed")
        with self.memory.operation():
            yield

    def _skill(self, action):
        return self.skills.get((action.embodiment, action.skill, action.skill_config_version))

    def _error(self, provider, exc):
        reason = f"{provider}: {type(exc).__name__}: {str(exc)!a}"
        self.memory.record_diagnostic("provider_error", {"provider": provider, "reason": reason})
        return reason

    def _deny(self, reason, status="needs_plan", receipt=None, *, action=None, code="admission_rejected"):
        admission = AdmissionResult(False, reason, code)
        details = {"reason": reason, "status": status, "code": code}
        if isinstance(action, Action):
            details.update(invocation_id=action.invocation_id, goal_id=action.goal_id,
                           planning_observation_id=action.based_on_observation, required_facts=action.required_facts)
        self.memory.record_diagnostic("admission_rejected", details)
        return StepResult(admission, receipt, status, reason)

    def _observe(self, action):
        previous = self.memory.latest_observation()
        current = self.observer.observe(action)
        if not isinstance(current, Observation):
            raise TypeError("observer must return Observation")
        if previous is not None and current.observation_id == previous.observation_id:
            raise ValueError("observer replayed the previous observation")
        self.memory.record_observation(current)
        if canonical_json(current) != canonical_json(self.memory.latest_observation()):
            raise ValueError("observer replayed an older observation")
        return current

    def _admit(self, action, observation):
        if not isinstance(action, Action) or not isinstance(observation, Observation):
            return AdmissionResult(False, "admission requires Action and Observation", "invalid_action")
        existing = self.memory.get_invocation(action.invocation_id)
        if existing and action_fingerprint(existing["action"]) != action_fingerprint(action):
            return AdmissionResult(False, "invocation ID has a different action payload", "invocation_conflict")
        skill = self._skill(action)
        if skill is None:
            return AdmissionResult(False, "unregistered embodiment, skill or configuration", "skill_unavailable")
        if action.scene_id != observation.scene_id or not set(action.entity_ids).issubset(observation.entity_ids):
            return AdmissionResult(False, "observation does not match scene or target entities", "identity_mismatch")
        if action.based_on_observation != observation.observation_id:
            return AdmissionResult(False, "outdated plan: observation changed", "observation_outdated")
        if action.expected_state_version != self.memory.state_version:
            return AdmissionResult(False, "outdated plan: verified state version changed", "state_outdated")
        try:
            self.memory.record_observation(observation)
            if canonical_json(observation) != canonical_json(self.memory.latest_observation()):
                return AdmissionResult(False, "outdated plan: observation is not latest", "observation_outdated")
            validity = self.memory.record_validity(observation, action.required_facts)
            if any(value["status"] != "match" for value in validity.values()):
                return AdmissionResult(False, "current observation cannot confirm required facts", "required_facts_unconfirmed")
            if skill.validator(action.parameters) is not True:
                return AdmissionResult(False, "skill parameters are invalid", "invalid_parameters")
            if skill.pre_check(action, observation) is not Signal.PASS:
                return AdmissionResult(False, "current skill preconditions are not confirmed", "precondition_unconfirmed")
        except Exception as exc:
            return AdmissionResult(False, self._error("admission", exc), "admission_provider_error")
        return AdmissionResult(True, "current observation and skill preconditions admit action")

    def admit(self, action, observation):
        with self._operation():
            return self._admit(action, observation)

    def _step(self, record):
        receipt = record["receipt"]
        status, reason = "none", ""
        if receipt is None or receipt.state is ReceiptState.UNKNOWN:
            if record["observations"] >= self.budget.max_observations:
                status, reason = "unresolved", "observation budget exhausted; stronger evidence or a larger budget is needed"
                if record["unresolved"] != reason:
                    self.memory.set_unresolved(record["action"].invocation_id, reason)
            else:
                status, reason = "needs_observation", "reconcile the original physical attempt"
        elif record["runtime_terminal"] is not True:
            status, reason = "awaiting_terminal", "effect receipt does not confirm runtime termination"
        elif receipt.state is ReceiptState.REJECTED:
            status, reason = "needs_plan", "rejected effect requires an admitted recovery proposal"
        return StepResult(AdmissionResult(True, "existing admitted attempt"), receipt, status, reason)

    def _existing(self, action):
        existing = self.memory.get_invocation(action.invocation_id)
        if existing is None:
            return None
        if action_fingerprint(existing["action"]) != action_fingerprint(action):
            return self._deny("invocation ID has a different action payload", action=action, code="invocation_conflict")
        return self._step(existing)

    def execute(self, action, *, observation=None):
        with self._operation():
            return self._execute(action, observation)

    def _execute(self, action, observation):
        if not isinstance(action, Action):
            return self._deny("execute requires Action", code="invalid_action")
        existing = self._existing(action)
        if existing is not None:
            return existing
        pending = self.memory.pending_attempt()
        if pending is not None:
            result = self._step(pending)
            return self._deny("an earlier attempt still blocks dispatch", result.recovery_status, result.receipt,
                              action=action, code="pending_attempt")
        if self.memory.active_recovery() is not None:
            return self._deny("use recover for the core-managed active recovery root", action=action, code="recovery_required")
        return self._dispatch(action, observation)

    def recover(self, invocation_id, replacement, *, observation=None):
        with self._operation():
            return self._recover(invocation_id, replacement, observation)

    def _recover(self, invocation_id, replacement, observation):
        original = self.memory.get_invocation(invocation_id)
        if original is None or not isinstance(replacement, Action):
            return self._deny("recovery requires an existing invocation and a replacement Action", action=replacement, code="invalid_recovery")
        if replacement.invocation_id == invocation_id:
            return self._deny("physical recovery requires a new invocation ID", action=replacement, code="invalid_recovery")
        existing = self.memory.get_invocation(replacement.invocation_id)
        if existing is not None:
            if existing["recovery_root"] != original["recovery_root"]:
                return self._deny("replacement invocation belongs to another recovery root", action=replacement, code="invalid_recovery")
            return self._existing(replacement)
        pending = self.memory.pending_attempt()
        if pending is not None:
            result = self._step(pending)
            return self._deny("an earlier attempt still blocks recovery", result.recovery_status, result.receipt,
                              action=replacement, code="pending_attempt")
        active = self.memory.active_recovery()
        if (active is None or active["action"].invocation_id != invocation_id
                or original["receipt"] is None or original["receipt"].state is not ReceiptState.REJECTED):
            return self._deny("recover the current rejected leaf of the active recovery root", action=replacement, code="invalid_recovery")
        if original["recovery_dispatches"] >= self.budget.max_recovery_dispatches:
            return self._deny("recovery dispatch budget exhausted", "budget_exhausted", action=replacement, code="recovery_budget")
        old_skill, new_skill = self._skill(original["action"]), self._skill(replacement)
        same_skill = (replacement.embodiment, replacement.skill) == (original["action"].embodiment, original["action"].skill)
        if same_skill and (old_skill is None or new_skill is None or not old_skill.safe_retry or not new_skill.safe_retry):
            return self._deny("registered skill does not permit a safe retry", action=replacement, code="retry_forbidden")
        return self._dispatch(replacement, observation, original["recovery_root"])

    def _dispatch(self, action, before, recovery_root=None):
        try:
            if before is None:
                before = self._observe(action)
        except Exception as exc:
            return self._deny(self._error("observer.before", exc), action=action, code="observation_error")
        admission = self._admit(action, before)
        if not admission.allowed:
            return self._deny(admission.reason, action=action, code=admission.code)
        try:
            attempt_id = self.memory.begin_attempt(action, before, recovery_root=recovery_root,
                                                   max_recovery_dispatches=self.budget.max_recovery_dispatches if recovery_root else None)
        except BudgetExceeded as exc:
            return self._deny(str(exc), "budget_exhausted", action=action, code="recovery_budget")
        except (EpisodeBusy, InvocationConflict, ValueError) as exc:
            return self._deny(str(exc), action=action, code="dispatch_rejected")
        # The durable intent precedes the only physical dispatch site.
        result = self._runtime_call("execute", action, attempt_id)
        self.memory.set_runtime_result(attempt_id, result)
        return self._evaluate(self.memory.get_invocation(action.invocation_id))

    def _runtime_call(self, method, action, attempt_id):
        try:
            result = getattr(self.runtime, method)(action, attempt_id)
            if (not isinstance(result, RuntimeResult) or result.invocation_id != action.invocation_id
                    or result.attempt_id != attempt_id):
                raise ValueError("runtime result does not bind the current action and attempt")
            return result
        except Exception as exc:
            return RuntimeResult("unconfirmed", None, action.invocation_id, attempt_id,
                                 {"error": self._error(f"runtime.{method}", exc)})

    def _uncertain(self, action, attempt_id, before, after, reason):
        report = EvidenceReport(action.invocation_id, attempt_id,
                                before.observation_id if before else None,
                                after.observation_id if after else None,
                                {}, diagnostics={"action_fingerprint": action_fingerprint(action),
                                                 "invalid_evidence": [reason]})
        return report, Assessment(Judgment.UNCERTAIN, reason)

    def _measurement_changed(self, record, report):
        current = report.diagnostics.get("measurement")
        if current is None:
            return False
        fields = ("entity_id", "frame_id", "translation_unit", "rotation_unit",
                  "translation_tolerance", "rotation_tolerance")
        # ponytail: linear audit scan; index the first measurement if episodes become large.
        for event in self.memory.events():
            previous = event["payload"].get("report") if event["kind"] == "receipt" else None
            if previous and previous["attempt_id"] == record["attempt_id"]:
                baseline = previous["diagnostics"].get("measurement")
                if baseline is not None:
                    return any(current.get(name) != baseline.get(name) for name in fields)
        return False

    def _evaluate(self, record):
        action, before, attempt_id = record["action"], record["before"], record["attempt_id"]
        runtime = record["runtime_result"] or RuntimeResult("unconfirmed", None, action.invocation_id, attempt_id)
        if record["runtime_terminal"] is True and runtime.terminal is not True:
            runtime = replace(runtime, terminal=True)
        after = None
        try:
            after = self._observe(action)
        except Exception as exc:
            self._error("observer.after", exc)
        skill = self._skill(action)
        try:
            if skill is None:
                raise ValueError("original skill configuration is unavailable")
            report, assessment = self.verifier.assess(action, attempt_id, skill, runtime, before, after)
            if self._measurement_changed(record, report):
                diagnostics = dict(report.diagnostics, invalid_evidence=["original measurement configuration changed"])
                report = replace(report, diagnostics=diagnostics)
                assessment = Assessment(Judgment.UNCERTAIN, "original measurement configuration changed", report.evidence_refs)
            if assessment.judgment is Judgment.YES and report.channels.get("post") is not Signal.PASS:
                assessment = Assessment(Judgment.UNCERTAIN, "goal integrity is not confirmed", report.evidence_refs)
            updates, relations = {}, ()
            if assessment.judgment is Judgment.YES:
                callback = "committed_facts"
                try:
                    updates = skill.committed_facts(action, before, after)
                    if not isinstance(updates, Mapping):
                        raise TypeError("committed_facts must return a JSON mapping")
                    updates = json_data(updates)
                    if skill.committed_relations is not None:
                        callback = "committed_relations"
                        relations = skill.committed_relations(action, before, after)
                        if not isinstance(relations, (list, tuple)) or any(not isinstance(item, Mapping) for item in relations):
                            raise TypeError("committed_relations must return a sequence of JSON mappings")
                        relations = json_data(relations)
                except Exception as exc:
                    reason = self._error(callback, exc)
                    report = replace(report, diagnostics=dict(report.diagnostics, callback_errors={callback: reason}))
                    assessment = Assessment(Judgment.UNCERTAIN, reason, report.evidence_refs)
                    updates, relations = {}, ()
            receipt = make_receipt(action, attempt_id, runtime, report, assessment)
        except Exception as exc:
            reason = self._error("verification", exc)
            report, assessment = self._uncertain(action, attempt_id, before, after, reason)
            receipt = make_receipt(action, attempt_id, runtime, report, assessment)
            updates, relations = {}, ()
        self.memory.record_receipt(action, attempt_id, receipt, report, updates, dynamic_relations=relations)
        return self._step(self.memory.get_invocation(action.invocation_id))

    def reconcile(self, invocation_id):
        with self._operation():
            return self._reconcile(invocation_id)

    def _reconcile(self, invocation_id):
        record = self.memory.get_invocation(invocation_id)
        if record is None:
            return self._deny("unknown invocation", code="unknown_invocation")
        if record["runtime_terminal"] is not True:
            runtime = self._runtime_call("status", record["action"], record["attempt_id"])
            self.memory.set_runtime_result(record["attempt_id"], runtime)
            record = self.memory.get_invocation(invocation_id)
        if record["receipt"] is not None and record["receipt"].state is not ReceiptState.UNKNOWN:
            return self._step(record)
        try:
            self.memory.consume_observation(invocation_id, self.budget.max_observations)
        except BudgetExceeded:
            return self._step(record)
        if record["unresolved"] is not None:
            self.memory.set_unresolved(invocation_id, None)
        return self._evaluate(self.memory.get_invocation(invocation_id))

    def run(self, instruction, planner, goal_check):
        """Plan within the persisted episode budget and verify the final goal.

        goal_check(fresh_observation, current_context) must return Signal and
        verify every final constraint that must still hold. Intermediate facts
        need not remain true after later legitimate actions. None from planner
        only means needs_plan. Unknown attempts are reobserved, never replayed.
        """
        guardian = self

        class Controller:
            refresh_context = False

            def pending(self):
                return guardian.memory.pending_attempt()

            def reconcile(self, pending):
                return guardian._reconcile(pending["action"].invocation_id)

            def observe(self):
                return guardian._observe(None)

            def context(self, observation):
                if not self.refresh_context:
                    self.active = guardian.memory.active_recovery()
                    self.required = self.active["action"].required_facts if self.active else {}
                self.refresh_context = False
                return self.active, guardian.memory.context(observation, guardian.budget, self.required)

            def consume_plan(self):
                guardian.memory.consume_plan_step(guardian.budget.max_plan_steps)
                self.refresh_context = True

            def execute(self, proposal, observation, active):
                return (guardian._recover(active["action"].invocation_id, proposal, observation) if active
                        else guardian._execute(proposal, observation))

            def error(self, provider, exc):
                return guardian._error(provider, exc)

            def steps(self):
                return guardian.memory.plan_steps

        with self._operation():
            return run_loop(Controller(), instruction, planner, goal_check)
