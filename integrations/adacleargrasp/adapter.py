"""Native observations, execution, and skill evidence."""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Mapping

from roboguardian import Observation, RuntimeResult, Signal, SkillSpec
from roboguardian.types import action_fingerprint, json_data


def plain(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return json_data(value)


def scalar(value):
    value = plain(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, dict)):
        raise ValueError("expected one native scalar")
    return value


def final_native_summary(env):
    """Read the final native endpoint independently of rendering or cached frames."""
    result = {"native_task_success": None, "final_native_evaluation": None, "final_native_evaluation_error": None}
    try:
        if env is None:
            raise RuntimeError("environment unavailable")
        evaluation = {key: scalar(value) for key, value in env.unwrapped.evaluate().items()}
        if type(evaluation.get("success")) is not bool or type(evaluation.get("height")) not in (int, float):
            raise ValueError("native evaluate must return scalar success and height")
        result.update(native_task_success=evaluation["success"], final_native_evaluation=evaluation)
    except Exception as exc:
        result["final_native_evaluation_error"] = f"{type(exc).__name__}: {exc}"
    return result


def vector(value):
    value = plain(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("expected one finite native vector")
    return value


def save_frame(frame, path):
    import numpy as np
    from PIL import Image

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected RGB render, received shape {frame.shape}")
    Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(path)


class AdaAdapter:
    embodiment = "adacleargrasp:xarm7_xhand"

    def __init__(self, env, initializer, grasper, output, *, scene_id, target_label, config_version,
                 initarm_tolerance, height_threshold, relation_threshold, min_contacts, mover=None, pusher=None, puller=None):
        for value in (initarm_tolerance, height_threshold, relation_threshold):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("adapter thresholds must be finite and positive")
        if type(min_contacts) is not int or not 1 <= min_contacts <= 5:
            raise ValueError("min_contacts must be an integer from 1 to 5")
        self.env, self.initializer, self.grasper = env, initializer, grasper
        self.mover, self.pusher, self.puller = mover, pusher, puller
        self.output, self.scene_id, self.target_label = Path(output), scene_id, target_label
        self.config_version = config_version
        self.initarm_tolerance, self.height_threshold = initarm_tolerance, height_threshold
        self.relation_threshold, self.min_contacts = relation_threshold, min_contacts
        targets = env.unwrapped.target_actors
        if len(targets) != 1:
            raise ValueError("this integration supports exactly one simulated environment and target")
        self.target_id = str(targets[0].name)
        actors = [targets[0]] + [group[0] for group in getattr(env.unwrapped, "clutter_actors_per_item", []) if group]
        self._actors = {str(actor.name): actor for actor in actors}
        if len(self._actors) != len(actors) or ("target" in self._actors and self.target_id != "target"):
            raise ValueError("native object names must be unique and target alias unambiguous")
        self.objects = self.entity_ids = tuple(sorted(self._actors))
        self.bound_subject = None
        self.lowered = False
        self.transition_required = False
        self.stuck_transitions = set()
        self.native_feedback = {}
        self.last_execution = None
        self.pending_move = None
        self.init_target = tuple(initializer.cfg.arm_init_7) + tuple(initializer.HAND_INIT_12)
        self.sequence = 0
        self.last_observation = None
        self.visual_before = None
        self.results = {}
        self.attempt_fingerprints = {}
        self.dispatches = []
        (self.output / "frames").mkdir(exist_ok=True)

    def observe(self, action):
        native = self.env.unwrapped
        target = native.target_actors[0]
        target_p, target_q = vector(target.pose.p), vector(target.pose.q)
        tcp_p, tcp_q = vector(native.agent.tcp.pose.p), vector(native.agent.tcp.pose.q)
        if len(target_p) != 3 or len(tcp_p) != 3 or len(target_q) != 4 or len(tcp_q) != 4:
            raise ValueError("native target/TCP poses require xyz and wxyz")
        qpos, qvel = vector(native.agent.robot.get_qpos()), vector(native.agent.robot.get_qvel())
        if len(qpos) < len(self.init_target) or len(qpos) != len(qvel):
            raise ValueError("native robot joint dimensions do not match the configured initial pose")
        initial = self.init_target + (0.0,) * (len(qpos) - len(self.init_target))
        evaluation_error = None
        try:
            evaluation = {key: scalar(value) for key, value in native.evaluate().items()}
        except Exception as exc:
            evaluation, evaluation_error = {}, f"{type(exc).__name__}: {exc}"
        contacts = None
        contact_error = None
        try:
            contacts = scalar(native.agent.count_finger_contacts(native.target_object, min_force=0.5))
            if type(contacts) is not int or not 0 <= contacts <= 5:
                raise ValueError("contact count must be a native integer from zero to five")
        except Exception as exc:
            contacts, contact_error = None, f"{type(exc).__name__}: {exc}"
        self.sequence += 1
        frame = self.output / "frames" / f"{self.sequence:04d}.png"
        save_frame(self.env.render(), frame)
        actors = {name: vector(actor.pose.p) for name, actor in self._actors.items()}
        if self.pending_move is not None:
            subject = self.pending_move
            self.bound_subject = subject if math.dist(actors[subject][:2], tcp_p[:2]) <= .08 else None
            self.pending_move = None
        state = {"actors": actors,
                 "operation": {"bound_subject": self.bound_subject, "lowered": self.lowered,
                               "transition_required": self.transition_required,
                               "stuck_transitions": sorted(self.stuck_transitions)},
                 "last_execution": self.last_execution,
                 "target": {"name": self.target_id, "label": self.target_label, "position_m": target_p, "quaternion_wxyz": target_q},
                 "tcp": {"position_m": tcp_p, "quaternion_wxyz": tcp_q}, "robot_qpos_rad": qpos, "robot_qvel_rad_s": qvel,
                 "evaluation": evaluation, "evaluation_error": evaluation_error,
                 "finger_contacts": contacts, "contact_error": contact_error,
                 "target_tcp_distance_m": math.dist(target_p, tcp_p),
                 "initarm_max_error_rad": max(abs(actual - expected) for actual, expected in zip(qpos, initial))}
        facts = {"target_present": str(target.name) == self.target_id,
                 "arm_initialized": state["initarm_max_error_rad"] < self.initarm_tolerance}
        images = (str(frame.resolve()),)
        if self.visual_before is not None:
            invocation_id, attempt_id, before, skill = self.visual_before
            if action is None or action.invocation_id == invocation_id:
                state["visual_reference_before"] = {"observation_id": before.observation_id,
                                                    "invocation_id": invocation_id, "attempt_id": attempt_id, "skill": skill}
                images += before.images[:1]
        observation = Observation(f"ada-observation-{self.sequence}", "adacleargrasp-native", self.sequence,
                                  self.scene_id, self.entity_ids, state, images, facts)
        self.last_observation = observation
        return observation

    def resolve_object(self, name):
        return self.target_id if name == "target" else name

    def validate_parameters(self, skill, parameters):
        if not isinstance(parameters, Mapping):
            return False
        if skill == "move_to":
            return (set(parameters) == {"name"} and isinstance(parameters["name"], str)
                    and self.resolve_object(parameters["name"]) in self._actors)
        if skill in ("push", "pull"):
            distance = parameters.get("dist_m")
            return (set(parameters) == {"name", "side", "dist_m"}
                    and isinstance(parameters["name"], str)
                    and self.resolve_object(parameters["name"]) in self._actors
                    and parameters["side"] in ("left", "center", "middle", "right")
                    and type(distance) in (int, float) and math.isfinite(distance) and distance > 0)
        return skill in ("lift", "lower", "initarm", "inithand", "grasp") and not parameters

    def validate_action(self, action):
        if (not self.validate_parameters(action.skill, action.parameters)
                or action.embodiment != self.embodiment or action.skill_config_version != self.config_version
                or action.scene_id != self.scene_id or not set(action.entity_ids).issubset(self.entity_ids)):
            return False
        subject = (self.resolve_object(action.parameters["name"])
                   if action.skill in ("move_to", "push", "pull")
                   else self.target_id if action.skill == "grasp" else None)
        return subject is None or subject in action.entity_ids

    def bind_before(self, action, attempt_id):
        if self.visual_before is not None and self.visual_before[:2] == (action.invocation_id, attempt_id):
            return
        before = self.last_observation
        if (self.visual_before is not None and self.visual_before[0] == action.invocation_id
                and self.visual_before[2].observation_id == action.based_on_observation):
            before = self.visual_before[2]
        if before is None or before.observation_id != action.based_on_observation:
            raise ValueError("native dispatch requires the action's original before observation")
        self.visual_before = (action.invocation_id, attempt_id, before, action.skill)

    def _pre(self, action, observation):
        operation = observation.state.get("operation", {})
        if observation.facts.get("target_present") is not True:
            return Signal.FAIL
        if action.skill == "grasp" and self.target_id not in action.entity_ids:
            return Signal.FAIL
        if action.skill in ("move_to", "push", "pull"):
            subject = self.resolve_object(action.parameters.get("name"))
            if subject not in action.entity_ids:
                return Signal.FAIL
        if operation.get("transition_required"):
            if action.skill not in ("lift", "initarm") or action.skill in operation.get("stuck_transitions", ()):
                return Signal.FAIL
        if action.skill in ("push", "pull"):
            if operation.get("bound_subject") != subject or operation.get("lowered") is not True:
                return Signal.FAIL
        return Signal.PASS

    def execute(self, action, attempt_id):
        fingerprint = action_fingerprint(action)
        if attempt_id in self.results:
            if self.attempt_fingerprints[attempt_id] == fingerprint:
                return self.results[attempt_id]
            return RuntimeResult("attempt_payload_conflict", None, action.invocation_id, attempt_id)
        self.attempt_fingerprints[attempt_id] = fingerprint
        if not self.validate_action(action):
            feedback = {"ok": False, "error_code": "invalid_action",
                        "message": "invalid native action parameters or identity",
                        "advice": "correct the object name and parameters"}
            self.native_feedback = feedback
            self.last_execution = {"invocation_id": action.invocation_id, "attempt_id": attempt_id, **feedback}
            result = RuntimeResult("invalid_action", True, action.invocation_id, attempt_id,
                                   {"native_result": feedback})
            self.results[attempt_id] = result
            return result
        try:
            self.bind_before(action, attempt_id)
            self.dispatches.append({"invocation_id": action.invocation_id, "attempt_id": attempt_id, "skill": action.skill})
            skill = action.skill
            if skill == "move_to":
                self.bound_subject, self.lowered = None, False
                self.pending_move = self.resolve_object(action.parameters["name"])
                value = self.mover.move_to(self.pending_move, render=False, verbose=False)
            elif skill in ("lower", "lift"):
                value = getattr(self.mover, skill)(render=False, verbose=False)
            elif skill in ("push", "pull"):
                self.transition_required = True
                value = getattr(self.pusher if skill == "push" else self.puller, skill)(
                    action.parameters["side"], float(action.parameters["dist_m"]), render=False, verbose=False)
            elif skill in ("initarm", "inithand"):
                value = getattr(self.initializer, skill)(render=False, verbose=False)
            else:
                value = self.grasper.grasp(render=False)
            feedback = {key: getattr(value, key) for key in ("ok", "error_code", "message", "advice")}
            if type(feedback["ok"]) is not bool:
                raise ValueError("native result ok must be bool")
            self.native_feedback = feedback
            self.last_execution = {"invocation_id": action.invocation_id, "attempt_id": attempt_id, **feedback}
            if skill == "lower":
                self.lowered = bool(self.bound_subject and feedback["ok"])
            elif skill in ("lift", "initarm"):
                if feedback["ok"]:
                    self.lowered = False
                    self.transition_required = False
                    self.stuck_transitions.clear()
                elif self.transition_required and feedback["error_code"] == "stuck":
                    self.stuck_transitions.add(skill)
            result = RuntimeResult("native_returned", True, action.invocation_id, attempt_id,
                                   {"native_result": feedback})
        except Exception as exc:
            result = RuntimeResult("native_exception", None, action.invocation_id, attempt_id,
                                   {"error": f"{type(exc).__name__}: {exc}"})
        self.results[attempt_id] = result
        return result

    def status(self, action, attempt_id):
        result = self.results.get(attempt_id)
        if (result is not None and result.invocation_id == action.invocation_id
                and self.attempt_fingerprints.get(attempt_id) == action_fingerprint(action)):
            return result
        return RuntimeResult("native_status_unknown", None, action.invocation_id, attempt_id)

    def grasp_signal(self, observation):
        state = observation.state
        position = state.get("target", {}).get("position_m")
        distance, contacts = state.get("target_tcp_distance_m"), state.get("finger_contacts")
        height = position[2] if position is not None and len(position) >= 3 else None
        votes = [height >= self.height_threshold if type(height) in (int, float) and math.isfinite(height) else None,
                 distance <= self.relation_threshold if type(distance) in (int, float) and math.isfinite(distance) else None,
                 contacts >= self.min_contacts if type(contacts) is int else None]
        available = [vote for vote in votes if vote is not None]
        if len(available) < 2:
            return Signal.INDETERMINATE
        return Signal.PASS if sum(available) >= 2 else Signal.FAIL

    def _post(self, action, before, after):
        if action.skill == "grasp":
            return self.grasp_signal(after)
        if action.skill == "move_to":
            position = after.state.get("actors", {}).get(self.resolve_object(action.parameters["name"]))
            tcp = after.state.get("tcp", {}).get("position_m")
            if position is None or tcp is None:
                return Signal.INDETERMINATE
            return Signal.PASS if math.dist(position[:2], tcp[:2]) <= .08 else Signal.FAIL
        if action.skill in ("push", "pull"):
            name = self.resolve_object(action.parameters["name"])
            start = before.state.get("actors", {}).get(name)
            end = after.state.get("actors", {}).get(name)
            if start is None or end is None:
                return Signal.INDETERMINATE
            dx, dy = end[0] - start[0], end[1] - start[1]
            magnitude = math.hypot(dx, dy)
            side = action.parameters["side"]
            angle = {"left": math.pi / 6, "center": 0, "middle": 0, "right": -math.pi / 6}[side]
            if action.skill == "pull":
                angle = math.pi - angle
            cosine = (dx * math.cos(angle) + dy * math.sin(angle)) / magnitude if magnitude else -1
            return Signal.PASS if magnitude >= .01 and cosine >= .5 else Signal.FAIL
        result = after.state.get("last_execution")
        reference = after.state.get("visual_reference_before")
        if (not isinstance(result, Mapping) or not isinstance(reference, Mapping)
                or result.get("invocation_id") != action.invocation_id
                or reference.get("invocation_id") != action.invocation_id
                or result.get("attempt_id") != reference.get("attempt_id")):
            return Signal.INDETERMINATE
        return Signal.PASS if result.get("ok") is True else Signal.FAIL

    def skills(self):
        def describe(action):
            if action.skill == "grasp":
                return f"Is the target {self.target_label} visibly grasped by the robot hand and lifted clear of the tabletop?"
            if action.skill == "move_to":
                name = self.target_label if action.parameters["name"] == "target" else action.parameters["name"]
                return (f"Is the robot hand horizontally aligned with {name} in the table plane? "
                        "move_to only controls XY alignment; hand height or touching the tabletop is not a failure. "
                        "Abstain if the named object or alignment cannot be judged from the image.")
            if action.skill in ("push", "pull"):
                return (f"Did {action.parameters['name']} visibly move after the {action.skill} "
                        f"toward {action.parameters['side']}? Compare the before and after images.")
            return f"Did the robot complete the {action.skill} operation?"
        return [SkillSpec(
            embodiment=self.embodiment, skill=name, config_version=self.config_version,
            validator=lambda parameters, skill=name: self.validate_parameters(skill, parameters), describe=describe,
            pre_check=self._pre, post_check=self._post,
            measure_effect=lambda action, before, after: None,
            committed_facts=(lambda action, before, after: {"target_grasped": True}) if name == "grasp"
                            else lambda action, before, after: {},
            committed_relations=(lambda action, before, after:
                [{"subject": self.target_id, "relation": "held_by", "object": "robot_hand"}]) if name == "grasp" else None,
            required_channels=("post", "semantic") if name == "grasp" else ("post",), safe_retry=True,
        ) for name in ("move_to", "lift", "lower", "push", "pull", "initarm", "inithand", "grasp")]
