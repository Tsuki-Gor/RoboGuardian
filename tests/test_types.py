"""Contract checks for stable identity, bounded inputs and physical thresholds."""

import dataclasses
import importlib.util
import json
import unittest


class TypeContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("roboguardian"),
                             "The public roboguardian contracts must be importable")
        from roboguardian import types
        self.t = types

    def action(self, **changes):
        values = dict(invocation_id="call-1", goal_id="place", embodiment="arm",
                      skill="place", parameters={"pose": [1, 2], "name": "盒"},
                      scene_id="table", entity_ids=("box",),
                      based_on_observation="obs-1", expected_state_version=0,
                      required_facts={"grasped": True}, skill_config_version="v1")
        values.update(changes)
        return self.t.Action(**values)

    def test_fingerprint_ignores_mapping_order_but_binds_every_action_field(self):
        action = self.action()
        fingerprint = self.t.action_fingerprint(action)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, self.t.action_fingerprint(
            self.action(parameters={"name": "盒", "pose": [1, 2]})))
        alternatives = dict(invocation_id="call-2", goal_id="lift", embodiment="other",
                            skill="move", parameters={"pose": [2, 1]}, scene_id="floor",
                            entity_ids=("other",), based_on_observation="obs-2",
                            expected_state_version=1, required_facts={"grasped": False},
                            skill_config_version="v2")
        for field, value in alternatives.items():
            with self.subTest(field=field):
                self.assertNotEqual(fingerprint, self.t.action_fingerprint(
                    dataclasses.replace(action, **{field: value})))

    def test_action_snapshots_nested_input_and_prevents_later_mutation(self):
        parameters = {"pose": [1, {"x": 2}]}
        facts = {"grasped": {"entity": "box"}}
        action = self.action(parameters=parameters, required_facts=facts)
        fingerprint = self.t.action_fingerprint(action)
        parameters["pose"][1]["x"] = 99
        facts["grasped"]["entity"] = "other"
        self.assertEqual(fingerprint, self.t.action_fingerprint(action))
        with self.assertRaises(TypeError):
            action.parameters["pose"][1]["x"] = 3

    def test_json_inputs_reject_nonfinite_values_and_non_json_objects(self):
        for value in (float("nan"), float("inf"), -float("inf"), object(), {1: "key"}):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.action(parameters={"value": value})
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(ValueError):
            self.action(required_facts={"cycle": cyclic})

    def test_json_snapshots_reject_keys_that_cannot_encode_as_utf8(self):
        invalid = {"nested": [json.loads('{"\\ud800":1}')]}
        for constructor in (lambda value: self.action(parameters=value), self.t.Context):
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor(invalid)
        valid = {"\U0001f4e6": {"盒": 1}}
        self.assertEqual(len(self.t.action_fingerprint(self.action(parameters=valid))), 64)
        self.assertGreater(self.t.Context(valid).byte_size, 0)

    def test_json_integers_are_serializable_before_snapshot_is_accepted(self):
        huge = 10 ** 5000
        constructors = (lambda value: self.action(parameters={"value": value}),
                        lambda value: self.t.Context({"value": value}),
                        lambda value: self.action(expected_state_version=value))
        try:
            json.dumps(huge)
        except ValueError:
            for constructor in constructors:
                with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                    constructor(huge)
        else:  # Python builds without an integer digit cap can serialize it.
            for constructor in constructors:
                self.t.canonical_json(constructor(huge)).encode("utf-8")

    def test_sequence_and_version_are_nonnegative_integers_not_bools(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.action(expected_state_version=value)
                with self.assertRaises((TypeError, ValueError)):
                    self.t.Observation("obs", "camera", value, "table", ("box",), {})

    def test_observation_and_diagnostics_are_detached_json_snapshots(self):
        state = {"pose": [1, 2]}
        observation = self.t.Observation("obs", "camera", 1, "table", ("box",), state)
        result = self.t.RuntimeResult("timeout", None, "call-1", "attempt-1", state)
        state["pose"].append(3)
        self.assertEqual(self.t.json_data(observation.state), {"pose": [1, 2]})
        self.assertEqual(self.t.json_data(result.diagnostics), {"pose": [1, 2]})
        with self.assertRaises((TypeError, ValueError)):
            self.t.RuntimeResult("running", 1, "call-1", "attempt-1")

    def test_budget_limits_are_explicit_and_have_defined_zero_boundaries(self):
        values = dict(max_observations=0, max_recovery_dispatches=0, max_plan_steps=1,
                      context_bytes=1, max_images=0)
        self.t.Budget(**values)
        for field in values:
            for invalid in (-1, True, 1.5):
                with self.subTest(field=field, invalid=invalid), self.assertRaises((TypeError, ValueError)):
                    self.t.Budget(**dict(values, **{field: invalid}))
        for field in ("max_plan_steps", "context_bytes"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.t.Budget(**dict(values, **{field: 0}))

    def test_effect_uses_strict_thresholds_and_abstains_on_missing_or_partial(self):
        base = dict(translation_error=0.01, rotation_error=0.02,
                    translation_tolerance=0.1, rotation_tolerance=0.2,
                    entity_id="box", frame_id="world", translation_unit="m", rotation_unit="rad")
        cases = [({}, self.t.Signal.PASS),
                 ({"translation_error": 0.1}, self.t.Signal.FAIL),
                 ({"rotation_error": 0.2}, self.t.Signal.FAIL),
                 ({"translation_error": None}, self.t.Signal.INDETERMINATE),
                 ({"partial": True}, self.t.Signal.INDETERMINATE)]
        for changed, expected in cases:
            with self.subTest(changed=changed):
                self.assertEqual(self.t.effect_signal(self.t.EffectMeasurement(**dict(base, **changed))), expected)
        for field in ("translation_error", "rotation_error", "translation_tolerance", "rotation_tolerance"):
            for invalid in (-0.1, float("nan"), float("inf"), True):
                with self.subTest(field=field, invalid=invalid), self.assertRaises((TypeError, ValueError)):
                    self.t.EffectMeasurement(**dict(base, **{field: invalid}))

    def test_skill_requires_a_physical_effect_channel_and_trusted_callbacks(self):
        values = dict(embodiment="arm", skill="place", config_version="v1",
                      validator=lambda parameters: True, describe=lambda action: "placed",
                      pre_check=lambda action, observation: self.t.Signal.PASS,
                      post_check=lambda action, before, after: self.t.Signal.PASS,
                      measure_effect=lambda action, before, after: None,
                      committed_facts=lambda action, before, after: {"placed": True},
                      required_channels=("effect",))
        self.assertEqual(self.t.SkillSpec(**values).required_channels, ("effect",))
        for channels in ((), ("pre",), ("software",)):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                self.t.SkillSpec(**dict(values, required_channels=channels))
        with self.assertRaises(TypeError):
            self.t.SkillSpec(**dict(values, validator=True))

    def test_context_measures_deterministic_utf8_bytes_and_reference_count(self):
        context = self.t.Context({"instruction": "盒", "progress": []}, images=("image://one",))
        expected = '{"data":{"instruction":"盒","progress":[]},"images":["image://one"]}'
        self.assertEqual(context.serialize(), expected)
        self.assertEqual(context.byte_size, len(expected.encode("utf-8")))
        self.assertEqual(context.image_count, 1)

    def test_receipt_rejects_mismatched_attempt_and_judgment(self):
        assessment = self.t.Assessment(self.t.Judgment.YES, "verified", ("obs-2",))
        result = self.t.RuntimeResult("complete", True, "call-1", "attempt-1")
        values = dict(receipt_id="receipt-1", action=self.action(), attempt_id="attempt-1",
                      state=self.t.ReceiptState.COMMITTED, assessment=assessment,
                      runtime_result=result, evidence_refs=("obs-2",))
        self.t.EffectReceipt(**values)
        with self.assertRaises(ValueError):
            self.t.EffectReceipt(**dict(values, attempt_id="other"))
        with self.assertRaises(ValueError):
            self.t.EffectReceipt(**dict(values, state=self.t.ReceiptState.REJECTED))


if __name__ == "__main__":
    unittest.main()
