"""Exercise the shared loop without a simulator or physical robot."""

from types import SimpleNamespace
import unittest

from roboguardian.loop import run_loop
from roboguardian.types import Action, Signal


class AgentLoopTests(unittest.TestCase):
    def test_ungoverned_controller_uses_shared_loop_and_does_not_claim_verification(self):
        class Controller:
            count = 0
            executed = []

            def pending(self):
                return None

            def observe(self):
                return len(self.executed)

            def context(self, observation):
                return None, observation

            def consume_plan(self):
                self.count += 1

            def execute(self, proposal, observation, active):
                self.executed.append((proposal, observation))

            def steps(self):
                return self.count

        action = Action(invocation_id="a1", goal_id="goal", embodiment="arm", skill="move",
                        parameters={}, scene_id="scene", entity_ids=("box",),
                        based_on_observation="o1", expected_state_version=0,
                        required_facts={}, skill_config_version="v1")
        controller = Controller()
        planner = SimpleNamespace(propose=lambda instruction, context: action)
        result = run_loop(controller, "task", planner,
                          lambda observation, context: Signal.PASS if observation else Signal.FAIL,
                          success_status="completed", success_reason="controller finished")
        assert result.status == "completed"
        assert result.steps == 1
        assert controller.executed == [(action, 0)]
