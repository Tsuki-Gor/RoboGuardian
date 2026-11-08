"""Physical evidence, binding and callback-boundary checks for the verifier."""

from dataclasses import replace
import importlib.util
import unittest

from roboguardian.types import (
    Action, Assessment, EffectMeasurement, EvidenceReport, Judgment, Observation,
    ReceiptState, RuntimeResult, Signal, SkillSpec, action_fingerprint, json_data,
)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("roboguardian.verification"),
                             "The evidence verifier must be implemented")
        from roboguardian import verification
        self.v = verification
        self.action = Action("call", "place", "arm", "place", {}, "table", ("box",),
                             "before", 0, {}, "v1")
        self.before = Observation("before", "camera", 1, "table", ("box",), {"x": 0})
        self.after = Observation("after", "camera", 2, "table", ("box",), {"x": 1})
        self.result = RuntimeResult("success", True, "call", "attempt")
        self.measurement = EffectMeasurement(0.01, 0.02, 0.1, 0.2, "box", "world", "m", "rad")
        self.skill = SkillSpec(
            "arm", "place", "v1", lambda parameters: True,
            lambda action: "Is the box placed?",
            lambda action, before: Signal.PASS,
            lambda action, before, after: Signal.PASS,
            lambda action, before, after: self.measurement,
            lambda action, before, after: {"placed": True},
            ("semantic", "post", "effect"),
        )
        self.verifier = self.v.Verifier(lambda question, after: Signal.PASS)

    def report(self, channels, **changes):
        values = dict(invocation_id="call", attempt_id="attempt",
                      before_observation_id="before", after_observation_id="after",
                      channels=channels, evidence_refs=("before", "after"),
                      diagnostics={"action_fingerprint": action_fingerprint(self.action)})
        values.update(changes)
        return EvidenceReport(**values)

    def assess(self, **changes):
        values = dict(action=self.action, attempt_id="attempt", skill=self.skill,
                      runtime_result=self.result, before=self.before, after=self.after)
        values.update(changes)
        return self.verifier.assess(**values)

    def test_fusion_truth_table_requires_complete_consistent_effect_evidence(self):
        cases = [
            ({"semantic": Signal.PASS, "effect": Signal.PASS, "post": Signal.PASS}, Judgment.YES),
            ({"semantic": Signal.PASS}, Judgment.UNCERTAIN),
            ({"semantic": Signal.PASS, "effect": Signal.INDETERMINATE}, Judgment.UNCERTAIN),
            ({"semantic": Signal.FAIL, "effect": Signal.PASS}, Judgment.UNCERTAIN),
            ({"semantic": Signal.FAIL, "effect": Signal.FAIL}, Judgment.NO),
            ({"semantic": Signal.FAIL}, Judgment.UNCERTAIN),
        ]
        for channels, expected in cases:
            with self.subTest(channels=channels):
                report = self.report(channels)
                assessment = self.v.fuse(report, ("semantic", "effect"))
                self.assertIs(assessment.judgment, expected)
                self.assertEqual(assessment.evidence_refs, report.evidence_refs)

    def test_precondition_is_never_an_effect_vote(self):
        for pre in Signal:
            with self.subTest(pre=pre):
                report = self.report({"pre": pre, "effect": Signal.FAIL})
                self.assertIs(self.v.fuse(report, ("effect",)).judgment, Judgment.NO)
        self.assertIs(self.v.fuse(self.report({"pre": Signal.PASS}),
                                  ("effect",)).judgment, Judgment.UNCERTAIN)

    def test_optional_valid_counterevidence_and_goal_integrity_block_success(self):
        for post in (Signal.FAIL, Signal.INDETERMINATE):
            with self.subTest(post=post):
                report = self.report({"effect": Signal.PASS, "post": post})
                self.assertIs(self.v.fuse(report, ("effect",)).judgment, Judgment.UNCERTAIN)

    def test_missing_post_cannot_confirm_success_through_the_public_fusion_entry(self):
        report = self.report({"effect": Signal.PASS})
        self.assertIs(self.v.fuse(report, ("effect",)).judgment, Judgment.UNCERTAIN)
        confirmed = replace(report, channels={"effect": Signal.PASS, "post": Signal.PASS})
        self.assertIs(self.v.fuse(confirmed, ("effect",)).judgment, Judgment.YES)

    def test_partial_is_not_full_success_and_only_terminal_clear_negative_can_reject(self):
        for terminal in (True, False, None):
            for signal in (Signal.PASS, Signal.FAIL):
                with self.subTest(terminal=terminal, signal=signal):
                    report = self.report({"semantic": signal}, partial_effect=True,
                                         diagnostics={"runtime_terminal": terminal})
                    expected = Judgment.NO if signal is Signal.FAIL and terminal is True else Judgment.UNCERTAIN
                    self.assertIs(self.v.fuse(report, ("semantic",)).judgment, expected)

    def test_fusion_rejects_invalid_required_channels(self):
        for channels in ((), ("pre",), ("software",), ("effect", "effect")):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                self.v.fuse(self.report({"effect": Signal.PASS}), channels)

    def test_describe_semantic_and_skill_callbacks_receive_the_original_inputs(self):
        calls = []
        def describe(action):
            calls.append(("describe", action))
            return "Is the box placed?"
        def semantic(question, after):
            calls.append(("semantic", question, after))
            return Signal.PASS
        def pre(action, before):
            calls.append(("pre", action, before))
            return Signal.PASS
        def post(action, before, after):
            calls.append(("post", action, before, after))
            return Signal.PASS
        def effect(action, before, after):
            calls.append(("effect", action, before, after))
            return self.measurement
        skill = replace(self.skill, describe=describe, pre_check=pre,
                        post_check=post, measure_effect=effect)
        self.verifier = self.v.Verifier(semantic)
        report, assessment = self.assess(skill=skill)
        self.assertIs(assessment.judgment, Judgment.YES)
        self.assertCountEqual(calls, [("describe", self.action),
            ("semantic", "Is the box placed?", self.after), ("pre", self.action, self.before),
            ("post", self.action, self.before, self.after),
            ("effect", self.action, self.before, self.after)])
        self.assertEqual(set(report.channels), {"pre", "semantic", "post", "effect"})
        self.assertEqual(report.diagnostics["action_fingerprint"], action_fingerprint(self.action))
        self.assertEqual(report.diagnostics["skill_config_version"], "v1")
        self.assertEqual(json_data(report.diagnostics["measurement"]), json_data(self.measurement))

    def test_software_status_does_not_change_identical_physical_evidence(self):
        for status in ("success", "error", "timeout"):
            for terminal in (True, False, None):
                for signal, expected in ((Signal.PASS, Judgment.YES), (Signal.FAIL, Judgment.NO)):
                    with self.subTest(status=status, terminal=terminal, signal=signal):
                        self.verifier = self.v.Verifier(lambda question, after: signal)
                        measurement = replace(self.measurement, translation_error=0.01 if signal is Signal.PASS else 1)
                        skill = replace(self.skill, post_check=lambda *args: signal,
                                        measure_effect=lambda *args: measurement)
                        report, assessment = self.assess(
                            skill=skill, runtime_result=replace(self.result, status=status, terminal=terminal))
                        self.assertIs(assessment.judgment, expected)
                        self.assertEqual(report.diagnostics["runtime_status"], status)

    def test_strict_tolerances_and_partial_measurements_never_confirm_completion(self):
        cases = [dict(translation_error=0.1), dict(rotation_error=0.2),
                 dict(translation_error=None), dict(partial=True)]
        for changed in cases:
            with self.subTest(changed=changed):
                self.measurement = replace(EffectMeasurement(0.01, 0.02, 0.1, 0.2,
                                                            "box", "world", "m", "rad"), **changed)
                report, assessment = self.assess()
                self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
                self.assertIsNot(report.channels["effect"], Signal.PASS)
                self.assertEqual(report.partial_effect, changed.get("partial", False))

    def test_invalid_observation_identity_or_freshness_is_not_effect_evidence(self):
        invalid = [replace(self.after, observation_id="before"),
                   replace(self.after, source_id="other-camera"),
                   replace(self.after, sequence=1), replace(self.after, sequence=0),
                   replace(self.after, scene_id="other-room"),
                   replace(self.after, entity_ids=("other-box",))]
        for after in invalid:
            with self.subTest(after=after):
                report, assessment = self.assess(after=after)
                self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
                self.assertEqual(assessment.evidence_refs, ())
                self.assertTrue(report.diagnostics["invalid_evidence"])

    def test_missing_or_replaced_original_baseline_cannot_confirm_completion(self):
        for before in (None, replace(self.before, observation_id="replacement"),
                       replace(self.before, scene_id="elsewhere"),
                       replace(self.before, entity_ids=())):
            with self.subTest(before=before):
                report, assessment = self.assess(before=before)
                self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
                self.assertEqual(assessment.evidence_refs, ())
        self.assertIs(self.assess(after=None)[1].judgment, Judgment.UNCERTAIN)

    def test_reconciliation_keeps_original_before_and_replaces_only_after(self):
        baseline_ids = []
        def effect(action, before, after):
            baseline_ids.append(before.observation_id)
            return replace(self.measurement, translation_error=abs(after.state["x"] - before.state["x"] - 1))
        skill = replace(self.skill, measure_effect=effect)
        uncertain, first = self.assess(skill=skill, after=replace(self.after, state={"x": 0.5}))
        confirmed, second = self.assess(skill=skill, after=replace(self.after, observation_id="later", sequence=3))
        self.assertIs(first.judgment, Judgment.UNCERTAIN)
        self.assertIs(second.judgment, Judgment.YES)
        self.assertEqual(baseline_ids, ["before", "before"])
        self.assertEqual(uncertain.before_observation_id, confirmed.before_observation_id)
        self.assertEqual(confirmed.after_observation_id, "later")

    def test_runtime_and_skill_binding_mismatches_abstain_without_using_callbacks(self):
        calls = []
        def forbidden(*args):
            calls.append(args)
        skill = replace(self.skill, describe=forbidden, pre_check=forbidden,
                        post_check=forbidden, measure_effect=forbidden)
        invalid = [dict(runtime_result=replace(self.result, invocation_id="other")),
                   dict(runtime_result=replace(self.result, attempt_id="other")),
                   dict(skill=replace(skill, embodiment="other")),
                   dict(skill=replace(skill, skill="other")),
                   dict(skill=replace(skill, config_version="v2"))]
        for changed in invalid:
            with self.subTest(changed=changed):
                report, assessment = self.assess(**dict({"skill": skill}, **changed))
                self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
                self.assertEqual(assessment.evidence_refs, ())
                self.assertEqual(calls, [])

    def test_wrong_measurement_entity_cannot_count_as_target_effect(self):
        self.measurement = replace(self.measurement, entity_id="other-box")
        report, assessment = self.assess()
        self.assertIs(report.channels["effect"], Signal.INDETERMINATE)
        self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
        self.assertTrue(report.diagnostics["callback_errors"])

    def test_callback_exceptions_and_wrong_returns_abstain_with_diagnostics(self):
        def raises(*args):
            raise RuntimeError("sensor unavailable")
        for name in ("describe", "pre_check", "post_check", "measure_effect", "semantic"):
            for callback in (raises, lambda *args: False):
                with self.subTest(name=name, callback=callback):
                    self.verifier = self.v.Verifier(callback if name == "semantic" else lambda *args: Signal.PASS)
                    skill = self.skill if name == "semantic" else replace(self.skill, **{name: callback})
                    report, assessment = self.assess(skill=skill)
                    self.assertIs(assessment.judgment, Judgment.UNCERTAIN)
                    self.assertTrue(report.diagnostics["callback_errors"])

    def test_missing_effect_and_invalid_numeric_measurement_are_not_physical_failure(self):
        for value in (None, -0.1, float("nan"), float("inf")):
            def effect(*args):
                return None if value is None else replace(self.measurement, translation_tolerance=value)
            with self.subTest(value=value):
                report, assessment = self.assess(skill=replace(self.skill, measure_effect=effect))
                self.assertIs(report.channels["effect"], Signal.INDETERMINATE)
                self.assertIs(assessment.judgment, Judgment.UNCERTAIN)

    def test_receipts_map_only_three_judgments_and_keep_separate_audit_ids(self):
        for judgment, state in ((Judgment.YES, ReceiptState.COMMITTED),
                                (Judgment.NO, ReceiptState.REJECTED),
                                (Judgment.UNCERTAIN, ReceiptState.UNKNOWN)):
            with self.subTest(judgment=judgment):
                report = self.report({"effect": Signal.PASS, "post": Signal.PASS})
                assessment = Assessment(judgment, "trusted fusion", report.evidence_refs)
                first = self.v.make_receipt(self.action, "attempt", self.result, report, assessment)
                second = self.v.make_receipt(self.action, "attempt", self.result, report, assessment)
                self.assertIs(first.state, state)
                self.assertEqual(first.action, self.action)
                self.assertEqual(first.runtime_result, self.result)
                self.assertEqual(first.evidence_refs, report.evidence_refs)
                self.assertNotEqual(first.receipt_id, second.receipt_id)

    def test_receipt_rejects_cross_attempt_report_runtime_or_unbound_evidence(self):
        report, assessment = self.assess()
        for changed in (dict(report=replace(report, invocation_id="other")),
                        dict(report=replace(report, attempt_id="other")),
                        dict(result=replace(self.result, attempt_id="other")),
                        dict(assessment=replace(assessment, evidence_refs=("unbound",)))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.v.make_receipt(**dict(dict(action=self.action, attempt_id="attempt",
                    result=self.result, report=report, assessment=assessment), **changed))

    def test_receipt_binds_the_complete_action_not_only_its_invocation_id(self):
        report, assessment = self.assess()
        changed = replace(self.action, parameters={"target": "elsewhere"})
        with self.assertRaises(ValueError):
            self.v.make_receipt(changed, "attempt", self.result, report, assessment)

    def test_receipt_rejects_a_missing_or_empty_action_fingerprint(self):
        report, assessment = self.assess()
        for diagnostics in ({}, {"action_fingerprint": None}, {"action_fingerprint": ""}):
            with self.subTest(diagnostics=diagnostics), self.assertRaises(ValueError):
                self.v.make_receipt(self.action, "attempt", self.result,
                                    replace(report, diagnostics=diagnostics), assessment)


if __name__ == "__main__":
    unittest.main()
