"""Synthetic state-changing closed loops; no robot, model, network or files."""

from __future__ import annotations

import argparse
import json

from roboguardian import (
    Action, Budget, EffectMeasurement, Guardian, Memory, Observation, RuntimeResult,
    Signal, SkillSpec, Verifier, json_data,
)


SCENARIOS = ("normal", "false-success", "response-loss", "prerequisite-loss")
TARGET_X = {"grasp": 1.0, "carry": 2.0}
DESCRIPTIONS = {
    "grasp": "Is the object held after grasping?",
    "carry": "Has the held object reached the delivery position?",
}


class World:
    """A tiny environment implementing both the runtime and observation roles."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.state = {"held": False, "delivered": False, "x": 0.0, "yaw": 0.0}
        self.sequence = 0
        self.after_count = 0
        self.dispatches = []
        self.attempts = {}

    def execute(self, action, attempt_id):
        before = dict(self.state)
        if not (self.scenario == "false-success" and not self.dispatches):
            self.state.update(held=True, x=TARGET_X[action.skill])
            if action.skill == "carry":
                self.state["delivered"] = True
        lost = self.scenario == "response-loss"
        status = "timeout" if lost else "completion"
        self.dispatches.append({"invocation_id": action.invocation_id, "attempt_id": attempt_id,
                                "skill": action.skill, "software_status": status,
                                "before": before, "after": dict(self.state)})
        self.attempts[attempt_id] = action.invocation_id
        self.after_count = 0
        return RuntimeResult(status, None if lost else True, action.invocation_id, attempt_id)

    def status(self, action, attempt_id):
        if self.attempts.get(attempt_id) != action.invocation_id:
            raise ValueError("unknown synthetic runtime attempt")
        return RuntimeResult("completion", True, action.invocation_id, attempt_id)

    def observe(self, action):
        self.sequence += 1
        if self.dispatches:
            self.after_count += 1
        if self.scenario == "prerequisite-loss" and len(self.dispatches) == 1 and self.after_count > 1:
            self.state.update(held=False, x=0.0)  # Slip after the first verified grasp.
        visible = not (self.scenario == "response-loss" and self.after_count == 1)
        state = {"visible": visible, **(self.state if visible else {})}
        facts = {"held": self.state["held"]} if visible else {}
        return Observation(f"frame-{self.sequence}", "synthetic-camera", self.sequence,
                           "synthetic-table", ("object",), state, facts=facts)


def state_signal(observation, skill):
    state = observation.state
    if not state["visible"]:
        return Signal.INDETERMINATE
    complete = state["held"] and (skill == "grasp" or state["delivered"])
    return Signal.PASS if complete else Signal.FAIL


def semantic_assessor(question, observation):
    skill = next(skill for skill, description in DESCRIPTIONS.items() if description == question)
    return state_signal(observation, skill)


def pre_check(action, observation):
    if not observation.state["visible"]:
        return Signal.INDETERMINATE
    held = observation.state["held"]
    return Signal.PASS if (held if action.skill == "carry" else not held) else Signal.FAIL


def measure_effect(action, before, after):
    visible = before.state["visible"] and after.state["visible"]
    translation_error = rotation_error = None
    if visible:
        measured = after.state["x"] - before.state["x"]
        expected = TARGET_X[action.skill] - before.state["x"]
        translation_error = abs(measured - expected)
        rotation_error = abs(after.state["yaw"] - before.state["yaw"])
    # Demonstration tolerances only; real adapters must calibrate their own.
    return EffectMeasurement(translation_error, rotation_error, 0.01, 0.05,
                             "object", "synthetic-world", "m", "rad")


def skills():
    return [SkillSpec(
        embodiment="synthetic-arm", skill=name, config_version="synthetic-v1",
        validator=lambda parameters: not parameters,
        describe=lambda action: DESCRIPTIONS[action.skill],
        pre_check=pre_check,
        post_check=lambda action, before, after: state_signal(after, action.skill),
        measure_effect=measure_effect,
        committed_facts=lambda action, before, after: {"held": after.state["held"]},
        required_channels=("semantic", "post", "effect"), safe_retry=True,
    ) for name in TARGET_X]


class ScriptedPlanner:
    """Propose at most two actions using the public bounded context."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.proposals = 0

    def propose(self, instruction, context):
        if self.proposals == 2:
            return None
        completed = {step["goal_id"] for step in context.data["progress"]["steps"]}
        skill = "carry" if self.scenario == "prerequisite-loss" and "hold-object" in completed else "grasp"
        goal = "deliver-object" if skill == "carry" else "hold-object"
        if goal in completed:
            return None
        self.proposals += 1
        observation = context.data["observation"]
        return Action(
            invocation_id=f"{self.scenario}-{self.proposals}", goal_id=goal,
            embodiment="synthetic-arm", skill=skill, parameters={},
            scene_id=observation["scene_id"], entity_ids=("object",),
            based_on_observation=observation["observation_id"],
            expected_state_version=context.data["state_version"],
            required_facts={"held": True} if skill == "carry" else {},
            skill_config_version="synthetic-v1",
        )


def run_scenario(name):
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    world = World(name)
    memory = Memory(":memory:", f"synthetic-{name}")
    # Explicit example budgets, not paper parameters or robot recommendations.
    budget = Budget(max_observations=3, max_recovery_dispatches=2, max_plan_steps=8,
                    context_bytes=8192, max_images=2)
    final_skill = "carry" if name == "prerequisite-loss" else "grasp"

    def goal_check(observation, context):
        signal = state_signal(observation, final_skill)
        if signal is Signal.PASS and abs(observation.state["x"] - TARGET_X[final_skill]) >= 0.01:
            return Signal.FAIL
        return signal

    try:
        with Guardian(memory, skills(), world, world, Verifier(semantic_assessor), budget) as guardian:
            result = guardian.run("Hold the object and preserve required effects.",
                                  ScriptedPlanner(name), goal_check)
        receipts = [event["payload"]["receipt"] for event in memory.events() if event["kind"] == "receipt"]
        return {"scenario": name, "synthetic": True, "run_result": json_data(result),
                "dispatches": world.dispatches, "receipts": receipts,
                "verified_progress": memory.progress, "verified_facts": memory.facts,
                "world": dict(world.state)}
    finally:
        memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    args = parser.parse_args()
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    print(json.dumps([run_scenario(name) for name in names], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
