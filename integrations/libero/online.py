"""Per-action Guardian control with proprioceptive, not task-success, evidence."""

from collections import deque
import math
import time
from uuid import uuid4

from roboguardian import Action, Budget, Guardian, Observation, Signal, SkillSpec, Verifier

from .adapter import plain, validate_action

EVIDENCE_SCOPE = ("valid post-terminal robot state; a full cumulative requested pose activity window "
                  "has not triggered stagnation (incomplete windows are undecided); activity may include "
                  "reverse motion or jitter; not per-action motion, SE3, object effect, or task success")


def _state(raw):
    try:
        robot = plain(raw["robot_state"])
        pos, quat, grip = robot["eef"]["pos"], robot["eef"]["quat"], robot["gripper"]["qpos"]
        if (not isinstance(pos, list) or len(pos) != 3 or not isinstance(quat, list) or len(quat) != 4
                or not isinstance(grip, list) or not grip
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pos + quat + grip)):
            return None
        if sum(v * v for v in quat) <= 1e-12:
            return None
        return {"eef": {"pos": pos, "quat": quat}, "gripper": {"qpos": grip}}
    except (KeyError, TypeError, ValueError):
        return None


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class OnlineController:
    """One action per step; policy and execution callbacks never receive fault labels.

    observe() returns a native observation. policy(native_observation) returns a
    vector or chunk. execute(vector, invocation_id, attempt_id) and status(vector,
    invocation_id, attempt_id) return bound RuntimeResult objects. An observed
    state only establishes robot motion/stagnation, not task or object success.
    """
    def __init__(self, memory, observe, policy, execute, status, *, chunk_size=1,
                 request_ttl=5.0, window=8, position_epsilon=0.001,
                 rotation_epsilon=0.01,
                 max_observations=2, max_recoveries=2, clock=time.monotonic, mode="roboguardian",
                 scene_id="libero", entity_ids=("robot",)):
        if mode not in ("base", "roboguardian"):
            raise ValueError("mode must be base or roboguardian")
        if (type(chunk_size) is not int or chunk_size < 1 or type(window) is not int or window < 1
                or not math.isfinite(request_ttl) or request_ttl <= 0
                or any(not math.isfinite(v) or v < 0 for v in (position_epsilon, rotation_epsilon))):
            raise ValueError("invalid online control limits")
        self.memory, self.observe_native, self.policy = memory, observe, policy
        self.scene_id, self.entity_ids = scene_id, tuple(entity_ids)
        self.execute_native, self.status_native = execute, status
        self.chunk_size, self.request_ttl, self.window = chunk_size, request_ttl, window
        self.eps = position_epsilon, rotation_epsilon
        self.clock, self.queue, self.history = clock, deque(), deque(maxlen=window)
        self.sequence = 0
        self.raw = None
        self.results = {}
        self.evaluated = {}
        self.events = []
        self.recoveries = 0
        self.mode, self.max_recoveries, self.stopped = mode, max_recoveries, False
        self.terminal = True
        skill = SkillSpec("libero:panda", "online_action", "online-v1",
                          validator=lambda p: _valid_parameters(p),
                          describe=lambda a: "Is the terminal robot state valid without a stagnant activity window?",
                          pre_check=lambda a, o: Signal.PASS if o.state.get("robot_state") else Signal.INDETERMINATE,
                          post_check=self._post, measure_effect=lambda *args: None,
                          committed_facts=lambda *args: {}, required_channels=("post",), safe_retry=True)
        self.guardian = Guardian(memory, [skill], self, self,
                                 Verifier(lambda *args: Signal.INDETERMINATE),
                                 Budget(max_observations, max_recoveries, 1, 16000, 0)) if mode == "roboguardian" else None

    def close(self):
        if self.guardian is not None:
            self.guardian.close()

    def observe(self, action):
        # Guardian evaluates immediately after dispatch, including unknown dispatches.
        # Do not read a queued/running simulator or treat its state as post-action evidence.
        if action is not None and not self.terminal:
            robot = None
        else:
            self.raw = self.observe_native()
            robot = _state(self.raw)
        self.sequence += 1
        return Observation(f"online-{self.sequence}", "libero-proprioception", self.sequence,
                           self.scene_id, self.entity_ids, {"robot_state": robot})

    def execute(self, action, attempt_id):
        self.terminal = False
        result = self.execute_native(list(action.parameters["action"]), action.invocation_id, attempt_id)
        self.terminal = (result.invocation_id == action.invocation_id and result.attempt_id == attempt_id
                         and result.terminal is True)
        return result

    def status(self, action, attempt_id):
        self.terminal = False
        result = self.status_native(list(action.parameters["action"]), action.invocation_id, attempt_id)
        self.terminal = (result.invocation_id == action.invocation_id and result.attempt_id == attempt_id
                         and result.terminal is True)
        return result

    def _post(self, action, before, after):
        a, b = before.state.get("robot_state"), after.state.get("robot_state")
        if a is None or b is None:
            return Signal.INDETERMINATE
        ap, bp = a["eef"]["pos"], b["eef"]["pos"]
        aq, bq = a["eef"]["quat"], b["eef"]["quat"]
        ag, bg = a["gripper"]["qpos"], b["gripper"]["qpos"]
        if len(ag) != len(bg):
            return Signal.INDETERMINATE
        an, bn = math.sqrt(sum(x*x for x in aq)), math.sqrt(sum(x*x for x in bq))
        qa, qb = [x/an for x in aq], [x/bn for x in bq]
        distances = (_distance(ap, bp), min(_distance(qa, qb), _distance(qa, [-x for x in qb])))
        requested = action.parameters["action"]
        requested_parts = (any(abs(v) > 1e-6 for v in requested[:3]),
                           any(abs(v) > 1e-6 for v in requested[3:6]))
        if action.invocation_id not in self.evaluated:
            self.history.append((*(distances[i] if requested_parts[i] else 0 for i in range(2)),
                                 *requested_parts))
            self.evaluated[action.invocation_id] = True
        stagnant = (len(self.history) == self.window and any(h[2] or h[3] for h in self.history)
                    and all(not any(h[i + 2] for h in self.history)
                            or sum(h[i] for h in self.history) < self.eps[i] for i in range(2)))
        if stagnant:
            return Signal.FAIL
        return Signal.PASS

    def _recover(self, reason):
        if self.recoveries >= self.max_recoveries:
            self.events.append({"kind": "recovery_budget_exhausted", "reason": reason})
            self.stopped = True
            return False
        self.recoveries += 1
        self.events.append({"kind": "recovery", "reason": reason, "count": self.recoveries})
        return True

    def _clear(self):
        self.queue.clear()
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def step(self):
        if self.stopped:
            return None
        pending = self.memory.pending_attempt() if self.guardian else None
        if pending:
            result = self.guardian.reconcile(pending["action"].invocation_id)
            self.events.append({"kind": "reconcile", "invocation_id": pending["action"].invocation_id,
                                "status": result.recovery_status})
            if result.recovery_status in ("unresolved", "budget_exhausted"):
                self.stopped = True
            return result
        before = self.observe(None)
        active = self.memory.active_recovery() if self.guardian else None
        if active:
            self._clear()
            self.history.clear()
        if not self.queue:
            request_id, started = uuid4().hex, self.clock()
            raw_proposal = self.policy(self.raw)
            try:
                proposal = plain(raw_proposal)
            except (TypeError, ValueError):
                self.events.append({"kind": "invalid_policy_response", "request_id": request_id,
                                    "response_repr": repr(raw_proposal), "observation_id": before.observation_id})
                raise
            elapsed = self.clock() - started
            self.events.append({"kind": "policy", "request_id": request_id,
                                "observation_id": before.observation_id, "elapsed": elapsed,
                                "observation_source": before.source_id, "observation_sequence": before.sequence,
                                "observation_state": plain(before.state), "state_version": self.memory.state_version,
                                "response": proposal, "expired": elapsed > self.request_ttl})
            if elapsed > self.request_ttl and self.guardian:
                self._clear()
                self._recover("expired_policy_response")
                return None
            if isinstance(proposal, list) and proposal and isinstance(proposal[0], list):
                chunk = proposal[:self.chunk_size]
            else:
                chunk = [proposal]
            validated = [validate_action(item) for item in chunk]
            for values in validated:
                self.queue.append((request_id, values, [max(-1.0, min(1.0, v)) for v in values]))
        request_id, original, command = self.queue.popleft()
        action = Action(uuid4().hex, "online-control", "libero:panda", "online_action",
                        {"action": command, "evidence_scope": EVIDENCE_SCOPE}, self.scene_id, self.entity_ids, before.observation_id,
                        self.memory.state_version, {}, "online-v1")
        self.events.append({"kind": "proposal", "request_id": request_id,
                            "invocation_id": action.invocation_id, "original_action": original,
                            "executed_action": command, "observation_id": before.observation_id})
        if self.guardian is None:
            attempt = uuid4().hex
            result = self.execute_native(command, action.invocation_id, attempt)
            self.events.append({"kind": "dispatch", "invocation_id": action.invocation_id,
                                "attempt_id": attempt, "status": result.status})
            if result.terminal is None:
                attempt = uuid4().hex
                result = self.execute_native(command, action.invocation_id, attempt)
                self.events.append({"kind": "retry", "invocation_id": action.invocation_id,
                                    "attempt_id": attempt, "status": result.status})
                if result.terminal is None:
                    self.stopped = True
            self.results[action.invocation_id] = result
            return result
        if active:
            if not self._recover("stagnant_activity_window"):
                return None
            result = self.guardian.recover(active["action"].invocation_id, action, observation=before)
        else:
            result = self.guardian.execute(action, observation=before)
        self.results[action.invocation_id] = result
        if result.recovery_status != "none":
            self._clear()
        if result.recovery_status in ("unresolved", "budget_exhausted"):
            self.stopped = True
        self.events.append({"kind": "result", "invocation_id": action.invocation_id,
                            "status": result.recovery_status,
                            "receipt": result.receipt.state.value if result.receipt else None,
                            "evidence_scope": EVIDENCE_SCOPE})
        return result


def _valid_parameters(parameters):
    try:
        values = validate_action(parameters["action"])
        return all(-1 <= value <= 1 for value in values)
    except (KeyError, TypeError, ValueError):
        return False
