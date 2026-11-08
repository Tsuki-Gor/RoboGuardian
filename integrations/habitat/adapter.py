"""One native evaluation is the governance unit, not an individual waypoint."""

from uuid import uuid4
from roboguardian import Observation, RuntimeResult, Signal, SkillSpec
from roboguardian.types import json_data
from .tasks import validate_metrics_exact

EMBODIMENT = "habitat:evaluation"
CONFIG_VERSION = "lavira-native-macro-v2"


class HabitatAdapter:
    def __init__(self, pipe, job_id, episode_ids, *, sequence=0):
        self.pipe, self.episode_ids = pipe, tuple(episode_ids)
        self.scene = "lavira-evaluation:" + job_id
        self.entities = tuple("episode:" + item for item in self.episode_ids)
        self.sequence = sequence
        self._attempts = {}

    def observe(self, _action=None):
        state = self.pipe.request("status")
        if state.get("state") not in ("ready", "running", "paused", "completed", "failed", "cancelled"):
            raise ValueError("unrecognized native worker state")
        # This is a fresh executor query, not a physical-pose measurement.
        self.sequence += 1
        return Observation(uuid4().hex, "habitat-macro-status", self.sequence,
                           self.scene, self.entities, state, (), {})

    @staticmethod
    def _runtime(value, action, attempt_id):
        if (value.get("invocation_id"), value.get("attempt_id")) != (action.invocation_id, attempt_id):
            raise ValueError("worker status does not belong to the original attempt")
        state = value.get("state", "unknown")
        return RuntimeResult(state, state in ("completed", "failed", "cancelled"),
                             action.invocation_id, attempt_id, {"worker_state": state})

    def execute(self, action, attempt_id):
        self._attempts[action.invocation_id] = attempt_id
        self.pipe.request("start", invocation_id=action.invocation_id, attempt_id=attempt_id)
        return self._runtime(self.pipe.wait(), action, attempt_id)

    def status(self, action, attempt_id):
        self._attempts[action.invocation_id] = attempt_id
        return self._runtime(self.pipe.wait(), action, attempt_id)

    def post(self, action, _before, after):
        state = after.state
        expected = self._attempts.get(action.invocation_id)
        if (not expected or state.get("invocation_id") != action.invocation_id
                or state.get("attempt_id") != expected):
            return Signal.INDETERMINATE
        if state.get("state") in ("failed", "cancelled"):
            return Signal.FAIL
        if state.get("state") != "completed":
            return Signal.INDETERMINATE
        try:
            validate_metrics_exact(json_data(state.get("metrics")), self.episode_ids)
        except (ValueError, TypeError):
            return Signal.INDETERMINATE
        return Signal.PASS


def _skill_specs(adapter):
    return [SkillSpec(
        EMBODIMENT, "run_navigation_eval", CONFIG_VERSION,
        lambda p: set(p) == {"episode_ids"} and tuple(p["episode_ids"]) == adapter.episode_ids,
        lambda a: "Verify complete native evaluation results for the requested episodes; not navigation success.",
        lambda a, o: Signal.PASS if o.state.get("state") == "ready" else Signal.FAIL,
        adapter.post, lambda a, b, c: None,
        lambda a, b, c: {"evaluation_complete": True}, ("post",), safe_retry=False)]
