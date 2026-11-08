"""Offline online-control behavior; no simulator or model is loaded."""
import unittest

from roboguardian import Memory, ReceiptState, RuntimeResult
from integrations.libero.online import OnlineController


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(":memory:", "online-test")
        self.addCleanup(self.memory.close)
        self.x = 0.0
        self.quat = [0, 0, 0, 1]
        self.dispatched = []

    def observation(self):
        return {"robot_state": {"eef": {"pos": [self.x, 0, 0], "quat": self.quat},
                                "gripper": {"qpos": [0, 0]}}}

    def controller(self, **kwargs):
        def execute(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            self.x += 0.0005
            return RuntimeResult("returned", True, invocation, attempt)
        ctl = OnlineController(self.memory, self.observation,
                               kwargs.pop("policy", lambda obs: [[2, 0, 0, 0, 0, 0, 0]] * 3),
                               kwargs.pop("execute", execute), kwargs.pop("status", execute), **kwargs)
        self.addCleanup(ctl.close)
        return ctl

    def test_chunk_clipping_and_window_evidence(self):
        ctl = self.controller(chunk_size=3, window=3, position_epsilon=0.001)
        for _ in range(3):
            result = ctl.step()
            self.assertIs(result.receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(len([e for e in ctl.events if e["kind"] == "policy"]), 1)
        self.assertEqual(self.dispatched[0][0][0], 1.0)
        self.assertEqual(self.memory.facts, {})

    def test_stagnation_recovers_with_new_invocation(self):
        def stationary(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            return RuntimeResult("returned", True, invocation, attempt)
        ctl = self.controller(execute=stationary, window=2, max_recoveries=1)
        ctl.step()
        self.assertIs(ctl.step().receipt.state, ReceiptState.REJECTED)
        ctl.step()
        self.assertEqual(ctl.recoveries, 1)
        self.assertEqual(len({x[1] for x in self.dispatched}), 3)

    def test_recovery_budget_bounds_repeated_stagnation(self):
        def stationary(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            return RuntimeResult("returned", True, invocation, attempt)
        ctl = self.controller(execute=stationary, window=1, max_recoveries=1)
        self.assertIs(ctl.step().receipt.state, ReceiptState.REJECTED)
        self.assertIs(ctl.step().receipt.state, ReceiptState.REJECTED)
        self.assertIsNone(ctl.step())
        self.assertTrue(ctl.stopped)
        self.assertEqual(len(self.dispatched), 2)

    def test_timeout_reconciles_without_dispatch_again(self):
        def timeout(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            self.x += .1
            return RuntimeResult("timeout", None, invocation, attempt)
        def status(command, invocation, attempt):
            return RuntimeResult("finished", True, invocation, attempt)
        ctl = self.controller(execute=timeout, status=status)
        ctl.step()
        ctl.step()
        self.assertEqual(len(self.dispatched), 1)
        self.assertFalse(ctl.stopped)

    def test_pending_attempt_waits_for_terminal_before_observing(self):
        observations = []
        states = iter([None, None, True])
        def observe():
            observations.append(self.x)
            return self.observation()
        def timeout(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            return RuntimeResult("timeout", None, invocation, attempt)
        def status(command, invocation, attempt):
            terminal = next(states)
            if terminal:
                self.x += .01
            return RuntimeResult("finished" if terminal else "running", terminal, invocation, attempt)
        ctl = self.controller(execute=timeout, status=status, max_observations=3)
        ctl.observe_native = observe
        self.assertIs(ctl.step().receipt.state, ReceiptState.UNKNOWN)
        self.assertEqual(observations, [0])
        for _ in range(2):
            self.assertIs(ctl.step().receipt.state, ReceiptState.UNKNOWN)
            self.assertFalse(ctl.stopped)
            self.assertEqual(observations, [0])
        self.assertIs(ctl.step().receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(observations, [0, .01])
        self.assertEqual(len(self.dispatched), 1)

    def test_pending_attempt_has_finite_query_budget(self):
        def timeout(command, invocation, attempt):
            return RuntimeResult("running", None, invocation, attempt)
        ctl = self.controller(execute=timeout, status=timeout, max_observations=2)
        ctl.step()
        ctl.step()
        self.assertFalse(ctl.stopped)
        self.assertEqual(ctl.step().recovery_status, "unresolved")
        self.assertTrue(ctl.stopped)

    def test_base_retries_timeout_and_rg_rejects_stale(self):
        ticks = iter([0, 10])
        ctl = self.controller(clock=lambda: next(ticks), request_ttl=1, max_recoveries=0)
        self.assertIsNone(ctl.step())
        self.assertTrue(ctl.stopped)
        self.assertEqual(self.dispatched, [])
        ctl.close()
        def timeout(command, invocation, attempt):
            self.dispatched.append((command, invocation, attempt))
            return RuntimeResult("timeout", None, invocation, attempt)
        base = self.controller(execute=timeout, mode="base")
        base.step()
        self.assertEqual(len(self.dispatched), 2)
        self.assertEqual(self.dispatched[0][1], self.dispatched[1][1])
        self.assertNotEqual(self.dispatched[0][2], self.dispatched[1][2])

    def test_saturated_gripper_and_idle_inertia_are_valid_states(self):
        def movement(command, invocation, attempt):
            self.x += .01
            return RuntimeResult("returned", True, invocation, attempt)
        ctl = self.controller(execute=movement, window=2,
                              policy=lambda _: [[1, 0, 0, 0, 0, 0, 1], [0] * 7], chunk_size=2)
        first, second = ctl.step(), ctl.step()
        self.assertIs(first.receipt.state, ReceiptState.COMMITTED)
        self.assertIs(second.receipt.state, ReceiptState.COMMITTED)
        self.assertIn("not per-action motion", first.receipt.action.parameters["evidence_scope"])
        self.assertEqual(self.memory.facts, {})

    def test_rotation_counts_for_pure_rotation_window(self):
        def rotate(command, invocation, attempt):
            self.quat = [0, 0, 0.05, .99875] if self.quat[2] == 0 else [0, 0, 0.1, .995]
            return RuntimeResult("returned", True, invocation, attempt)
        ctl = self.controller(execute=rotate, window=2, policy=lambda _: [0, 0, 0, 0, 0, 1, 1])
        ctl.step()
        self.assertIs(ctl.step().receipt.state, ReceiptState.COMMITTED)

    def test_stale_response_is_recorded_and_recovery_budget_is_global(self):
        ticks = iter([0, 2, 3, 3, 4, 6])
        ctl = self.controller(clock=lambda: next(ticks), request_ttl=1, max_recoveries=1)
        ctl.step()
        event = ctl.events[0]
        self.assertEqual(event["response"], [[2, 0, 0, 0, 0, 0, 0]] * 3)
        self.assertEqual(event["observation_id"], "online-1")
        self.assertEqual(event["observation_state"]["robot_state"]["eef"]["pos"], [0, 0, 0])
        ctl.step()
        ctl.step()
        self.assertTrue(ctl.stopped)
        self.assertEqual(ctl.recoveries, 1)
        self.assertEqual(len(self.dispatched), 1)

    def test_missing_state_is_unknown(self):
        ctl = self.controller()
        ctl.observe_native = lambda: {}
        self.assertFalse(ctl.step().admission.allowed)
        self.assertEqual(self.dispatched, [])


if __name__ == "__main__":
    unittest.main()
