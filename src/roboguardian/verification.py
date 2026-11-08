"""Evaluate bound physical evidence; software status never supplies an effect vote."""

from __future__ import annotations

from uuid import uuid4

from .types import (
    Action, Assessment, EffectMeasurement, EffectReceipt, EvidenceReport, Judgment,
    Observation, ReceiptState, RuntimeResult, SemanticAssessor, Signal, SkillSpec,
    action_fingerprint, effect_signal, json_data,
)


def fuse(report: EvidenceReport, required_channels: tuple[str, ...]) -> Assessment:
    """Conservative three-valued fusion of already validated evidence.

    All available semantic/post/effect signals participate in conflict detection;
    preconditions never vote. A post PASS alone need not prove the whole effect,
    so post PASS with effect FAIL conservatively remains uncertain. A reported
    missing or indeterminate postcondition cannot bypass the goal-integrity gate
    by being omitted from required_channels; unused conditions explicitly PASS.
    """
    if (not required_channels or len(set(required_channels)) != len(required_channels)
            or any(name not in ("semantic", "post", "effect") for name in required_channels)):
        raise ValueError("required_channels must name distinct semantic/post/effect channels")

    def assessment(judgment, reason):
        return Assessment(judgment, reason, report.evidence_refs)

    if report.diagnostics.get("invalid_evidence") or report.diagnostics.get("callback_errors"):
        return assessment(Judgment.UNCERTAIN, "Evidence binding or callback validation failed")
    if any(report.channels.get(name, Signal.INDETERMINATE) is Signal.INDETERMINATE
           for name in required_channels):
        return assessment(Judgment.UNCERTAIN, "Required effect evidence is missing or indeterminate")
    signals = {value for name, value in report.channels.items() if name != "pre"}
    if Signal.PASS in signals and Signal.FAIL in signals:
        return assessment(Judgment.UNCERTAIN, "Valid physical evidence conflicts")
    if Signal.FAIL in signals:
        if report.partial_effect and report.diagnostics.get("runtime_terminal") is not True:
            return assessment(Judgment.UNCERTAIN, "Partial effect without confirmed termination")
        return assessment(Judgment.NO, "Complete required evidence consistently denies the effect")
    if report.partial_effect:
        return assessment(Judgment.UNCERTAIN, "Partial effect does not confirm the full goal")
    if report.channels.get("post", Signal.INDETERMINATE) is Signal.INDETERMINATE:
        return assessment(Judgment.UNCERTAIN, "Goal integrity is indeterminate")
    return assessment(Judgment.YES, "All required effect evidence passes without counterevidence")


class Verifier:
    """Run trusted skill callbacks against one immutable before/after pair.

    Reconciliation must supply the original before snapshot and registered skill
    configuration. The trusted adapter fixes thresholds, frame and units for its
    config_version; this stateless class records them but cannot reconstruct a
    lost baseline or infer the adapter's coordinate system. post_check includes
    every necessary goal-integrity condition; unused conditions return PASS.
    """

    def __init__(self, semantic_assessor: SemanticAssessor):
        if not callable(semantic_assessor):
            raise TypeError("semantic_assessor must be callable")
        self.semantic_assessor = semantic_assessor

    def assess(self, action: Action, attempt_id: str, skill: SkillSpec,
               runtime_result: RuntimeResult, before: Observation | None,
               after: Observation | None) -> tuple[EvidenceReport, Assessment]:
        channels = dict.fromkeys(("semantic", "pre", "post", "effect"), Signal.INDETERMINATE)
        diagnostics = {
            "action_fingerprint": action_fingerprint(action),
            "skill_config_version": skill.config_version,
            "runtime_status": runtime_result.status,
            "runtime_terminal": runtime_result.terminal,
        }
        invalid = []
        if (runtime_result.invocation_id != action.invocation_id
                or runtime_result.attempt_id != attempt_id):
            invalid.append("Runtime result belongs to a different action or attempt")
        if (skill.embodiment != action.embodiment or skill.skill != action.skill
                or skill.config_version != action.skill_config_version):
            invalid.append("Skill identity or configuration does not match the action")
        if before is None or after is None:
            invalid.append("Original before and fresh after observations are required")
        for name, observation in (("before", before), ("after", after)):
            if observation is not None and (observation.scene_id != action.scene_id
                    or not set(action.entity_ids).issubset(observation.entity_ids)):
                invalid.append(f"{name} observation does not cover the action scene and entities")
        if before is not None and before.observation_id != action.based_on_observation:
            invalid.append("Before observation is not the action's original planning baseline")
        if before is not None and after is not None:
            if (before.source_id != after.source_id or after.sequence <= before.sequence
                    or before.observation_id == after.observation_id):
                invalid.append("After observation is not fresh in the original observation source")
            diagnostics.update(observation_source=before.source_id,
                               before_sequence=before.sequence, after_sequence=after.sequence)

        partial = False
        refs = ()
        if invalid:
            diagnostics["invalid_evidence"] = invalid
        else:
            refs = (before.observation_id, after.observation_id)
            errors = {}

            def call(name, callback, arguments, expected_type):
                try:
                    value = callback(*arguments)
                    if value is None and expected_type is EffectMeasurement:
                        return None
                    if not isinstance(value, expected_type):
                        raise TypeError(f"Expected {expected_type.__name__}, got {type(value).__name__}")
                    if expected_type is str:
                        if not value.strip():
                            raise ValueError("Describe must return a nonempty question")
                        value.encode("utf-8")
                    if expected_type is EffectMeasurement and value.entity_id not in action.entity_ids:
                        raise ValueError("Effect measurement entity does not match the action")
                    return value
                except Exception as exc:
                    errors[name] = f"{type(exc).__name__}: {str(exc)!a}"
                    return None

            question = call("describe", skill.describe, (action,), str)
            if question is not None:
                semantic = call("semantic", self.semantic_assessor, (question, after), Signal)
                if semantic is not None:
                    channels["semantic"] = semantic
            pre = call("pre", skill.pre_check, (action, before), Signal)
            post = call("post", skill.post_check, (action, before, after), Signal)
            if pre is not None:
                channels["pre"] = pre
            if post is not None:
                channels["post"] = post
            measurement = call("effect", skill.measure_effect, (action, before, after), EffectMeasurement)
            if measurement is not None:
                diagnostics["measurement"] = json_data(measurement)
                channels["effect"] = effect_signal(measurement)
                partial = measurement.partial
            if errors:
                diagnostics["callback_errors"] = errors

        report = EvidenceReport(
            action.invocation_id, attempt_id,
            before.observation_id if before is not None else None,
            after.observation_id if after is not None else None,
            channels, partial, refs, diagnostics,
        )
        return report, fuse(report, skill.required_channels)


def make_receipt(action: Action, attempt_id: str, result: RuntimeResult,
                 report: EvidenceReport, assessment: Assessment) -> EffectReceipt:
    """Bind a trusted fusion decision to its action, attempt and evidence.

    Each evaluation gets its own audit ID. Deduplicating logical commitment is
    the persistent commit gate's responsibility, not receipt construction.
    """
    if report.invocation_id != action.invocation_id or report.attempt_id != attempt_id:
        raise ValueError("Evidence report does not belong to this action and attempt")
    fingerprint = report.diagnostics.get("action_fingerprint")
    if fingerprint != action_fingerprint(action):
        raise ValueError("Evidence report requires the matching complete action fingerprint")
    if not set(assessment.evidence_refs).issubset(report.evidence_refs):
        raise ValueError("Assessment cites evidence outside its report")
    states = {Judgment.YES: ReceiptState.COMMITTED, Judgment.NO: ReceiptState.REJECTED,
              Judgment.UNCERTAIN: ReceiptState.UNKNOWN}
    return EffectReceipt(str(uuid4()), action, attempt_id, states[assessment.judgment],
                         assessment, result, assessment.evidence_refs)
