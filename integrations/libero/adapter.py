"""Native observations, execution, and skill evidence."""

import json
import math
from pathlib import Path

from roboguardian import Observation, RuntimeResult, Signal, SkillSpec
from roboguardian.types import json_data


def plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return json_data(value)


def validate_action(action):
    values = plain(action)
    if (not isinstance(values, list) or len(values) != 7
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ValueError("native policy action must be one finite vector of length seven")
    return values


def native_success(native):
    value = plain(native._env.check_success())
    if type(value) is not bool:
        raise ValueError("native check_success must return a boolean")
    return value


def final_native_summary(native):
    try:
        if native is None:
            raise RuntimeError("environment unavailable")
        return {"native_task_success": native_success(native), "final_native_evaluation_error": None}
    except Exception as exc:
        return {"native_task_success": None, "final_native_evaluation_error": f"{type(exc).__name__}: {exc}"}


def visible_frame(formatted, camera="image"):
    # Same camera orientation as native LiberoEnv.render and LiberoProcessorStep.
    image = formatted["pixels"][camera]
    return image[::-1, ::-1].copy()


def save_frame(frame, path):
    from PIL import Image
    Image.fromarray(frame).save(path)


class LiberoAdapter:
    embodiment = "libero:panda"
    scene_id = "libero:legacy-rollout"
    entity_ids = ("robot",)

    def __init__(self, native, policy, output, *, config_version, max_steps=520, video=None):
        if type(max_steps) is not int or not 1 <= max_steps <= 520:
            raise ValueError("rollout step limit must be between 1 and the native 520")
        self.native, self.policy, self.output = native, policy, Path(output)
        self.config_version, self.max_steps, self.video = config_version, max_steps, video
        self.sequence = self.native_controlstep_count = 0
        self.dispatches, self.results = [], {}
        self.last_observation = self.visual_before = None
        (self.output / "frames").mkdir(exist_ok=True)

    def fresh_raw(self):
        return self.native._env.env._get_observations(force_update=True)

    def observe(self, action):
        formatted = self.native._format_raw_obs(self.fresh_raw())
        success = native_success(self.native)
        self.sequence += 1
        frame = self.output / "frames" / f"{self.sequence:04d}.png"
        wrist_frame = self.output / "frames" / f"{self.sequence:04d}-wrist.png"
        save_frame(visible_frame(formatted), frame)
        save_frame(visible_frame(formatted, "image2"), wrist_frame)
        state = {"native_success": success, "robot_state": plain(formatted["robot_state"]),
                 "native_controlstep_count": self.native_controlstep_count, "task": self.native.task_description,
                 "image_roles": ["CURRENT agent view", "CURRENT wrist view"]}
        images = (str(frame.resolve()), str(wrist_frame.resolve()))
        if self.visual_before is not None:
            invocation_id, attempt_id, before = self.visual_before
            if action is None or action.invocation_id == invocation_id:
                images += before.images[:1]
                state["image_roles"].append("ORIGINAL BEFORE reference")
                state["visual_reference_before"] = {"observation_id": before.observation_id,
                                                    "invocation_id": invocation_id, "attempt_id": attempt_id}
        observation = Observation(f"libero-observation-{self.sequence}", "libero-native", self.sequence,
                                  self.scene_id, self.entity_ids, state, images, {"scene_ready": True})
        self.last_observation = observation
        return observation

    def execute(self, action, attempt_id):
        try:
            if self.dispatches:
                raise ValueError("only one native rollout is authorized; observations never replay it")
            before = self.last_observation
            if before is None or before.observation_id != action.based_on_observation:
                raise ValueError("rollout requires the proposal's original before observation")
            self.visual_before = (action.invocation_id, attempt_id, before)
            self.dispatches.append({"invocation_id": action.invocation_id, "attempt_id": attempt_id,
                                    "skill": "smolvla_rollout"})
            self.policy.reset()
            formatted = self.native._format_raw_obs(self.fresh_raw())
            if self.video is not None:
                self.video.append_data(visible_frame(formatted))
            status = "step_limit"
            with (self.output / "native_steps.jsonl").open("w", encoding="utf-8") as trace:
                for _ in range(self.max_steps):
                    command = self.policy(formatted)
                    values = validate_action(command)
                    # Deliberately bypass LiberoEnv.step, which resets a successful world.
                    raw, reward, done, _info = self.native._env.step(command)
                    self.native_controlstep_count += 1
                    success = native_success(self.native)
                    formatted = self.native._format_raw_obs(raw)
                    trace.write(json.dumps({"step": self.native_controlstep_count, "action": values,
                                            "reward": plain(reward), "done": bool(done), "native_success": success}, allow_nan=False) + "\n")
                    trace.flush()
                    if self.video is not None:
                        self.video.append_data(visible_frame(formatted))
                    if success or done:
                        status = "native_success" if success else "native_done"
                        break
            result = RuntimeResult(status, True, action.invocation_id, attempt_id,
                                   {"native_controlstep_count": self.native_controlstep_count})
        except Exception as exc:
            result = RuntimeResult("native_exception", None, action.invocation_id, attempt_id,
                                   {"native_controlstep_count": self.native_controlstep_count,
                                    "error": f"{type(exc).__name__}: {exc}"})
        self.results[attempt_id] = result
        return result

    def status(self, action, attempt_id):
        result = self.results.get(attempt_id)
        if result is not None and result.invocation_id == action.invocation_id:
            return result
        return RuntimeResult("native_status_unknown", None, action.invocation_id, attempt_id)

    def post_signal(self, observation):
        value = observation.state.get("native_success")
        return Signal.PASS if value is True else Signal.FAIL if value is False else Signal.INDETERMINATE

    def question(self):
        return f'Is this task currently complete: "{self.native.task_description}"? Judge the visible final relation.'

    def skill(self):
        return SkillSpec(self.embodiment, "smolvla_rollout", self.config_version,
                         validator=lambda parameters: not parameters, describe=lambda action: self.question(),
                         pre_check=lambda action, observation: Signal.PASS if observation.facts.get("scene_ready") else Signal.FAIL,
                         post_check=lambda action, before, after: self.post_signal(after),
                         measure_effect=lambda action, before, after: None,
                         committed_facts=lambda action, before, after: {"task_complete": True},
                         required_channels=("post", "semantic"), safe_retry=False,
                         committed_relations=None)


class NativeSession:
    """Own a terminal-preserving world; expose only proprioception to governance."""
    def __init__(self, native, output, *, scene_id, entity_ids, video=None, max_steps=520):
        import threading
        self.native, self.output, self.video = native, Path(output), video
        self.scene_id, self.entity_ids = scene_id, tuple(entity_ids)
        self.lock = threading.RLock()
        self.steps, self.sequence, self.max_steps = 0, 0, max_steps
        self.dispatches = []
        self.done, self.success = False, False
        self.formatted = native._format_raw_obs(native._env.env._get_observations(force_update=True))
        (self.output / 'frames').mkdir(exist_ok=True)
        self.trace = (self.output / 'native_steps.jsonl').open('a', encoding='utf-8')

    def observe(self):
        with self.lock:
            # Read-only refresh; never reset or advance physics during verification.
            self.formatted = self.native._format_raw_obs(
                self.native._env.env._get_observations(force_update=True))
            return {'robot_state': plain(self.formatted['robot_state'])}

    def step(self, payload):
        import numpy as np
        with self.lock:
            if self.done or self.steps >= self.max_steps:
                raise RuntimeError('episode is terminal or physical step budget exhausted')
            command = validate_action(payload['action'])
            self.trace.write(json.dumps({'event': 'dispatch_started', **payload}, allow_nan=False) + '\n')
            self.trace.flush()
            try:
                self.steps += 1
                self.dispatches.append(dict(payload))
                raw, reward, done, _ = self.native._env.step(np.asarray(command, dtype=np.float32))
                self.formatted = self.native._format_raw_obs(raw)
                self.success = native_success(self.native)
                self.done = bool(done) or self.success
                self.trace.write(json.dumps({'event': 'dispatch_completed', **payload, 'step': self.steps,
                    'official_success': self.success, 'reward': plain(reward), 'done': bool(done)}, allow_nan=False) + '\n')
                if self.video is not None:
                    self.video.append_data(visible_frame(self.formatted))
                return {'robot_state': plain(self.formatted['robot_state'])}
            except Exception as exc:
                self.trace.write(json.dumps({'event': 'dispatch_ambiguous', **payload,
                                            'error': type(exc).__name__}) + '\n')
                raise
            finally:
                self.trace.flush()

    def visual_observation(self):
        with self.lock:
            self.observe()
            self.sequence += 1
            paths = []
            for camera in ('image', 'image2'):
                path = self.output / 'frames' / f'{self.sequence:04d}-{camera}.png'
                save_frame(visible_frame(self.formatted, camera), path)
                paths.append(str(path.resolve()))
            return Observation(f'visual-{self.sequence}', 'libero-camera', self.sequence,
                self.scene_id, self.entity_ids,
                {'image_roles': ['CURRENT agent view', 'CURRENT wrist view']}, tuple(paths))

    def close(self):
        self.trace.close()
