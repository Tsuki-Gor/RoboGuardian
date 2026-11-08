"""Production experiment callbacks over real loopback HTTP, without GPU or VLM."""
from contextlib import closing
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from roboguardian import Memory, Observation, Signal
from integrations.libero.run import ExperimentRuntime, ExperimentPolicy, verify_terminal
from integrations.libero.online import OnlineController
from integrations.libero.transport import LoopbackTransport


class RunnerTests(unittest.TestCase):
    def test_timeout_pair_has_one_vs_two_physical_executions(self):
        for mode, count in [('base', 2), ('roboguardian', 1)]:
            with self.subTest(mode=mode), closing(Memory(':memory:', mode)) as memory:
                session = SimpleNamespace(steps=0)
                def step(payload):
                    session.steps += 1
                    return {}
                def observe():
                    return {'robot_state': {'eef': {'pos': [session.steps * .01, 0, 0], 'quat': [0,0,0,1]}, 'gripper': {'qpos':[0,0]}}}
                args = SimpleNamespace(fault='dispatch_timeout_retry', fault_step=0, step_timeout=.05)
                with LoopbackTransport(step, timeout=.05) as transport:
                    runtime = ExperimentRuntime(transport, session, args)
                    controller = OnlineController(memory, observe, lambda _: [1,0,0,0,0,0,1], runtime.execute, runtime.status, mode=mode)
                    try:
                        controller.step()
                        if mode == 'roboguardian':
                            self.assertIsNotNone(memory.pending_attempt())
                            controller.step()
                            self.assertIsNone(memory.pending_attempt())
                        self.assertEqual(session.steps, count)
                    finally:
                        controller.close()

    def test_late_real_policy_response_is_rejected_only_by_guardian(self):
        for mode, count in [('base', 1), ('roboguardian', 0)]:
            with self.subTest(mode=mode), closing(Memory(':memory:', mode)) as memory:
                session = SimpleNamespace(steps=0, formatted={}, visual_observation=lambda: SimpleNamespace(images=(), observation_id='frame'))
                policy = SimpleNamespace(inferences=0, chunk=lambda obs: [[1,0,0,0,0,0,0]], reset=lambda: None)
                args = SimpleNamespace(fault='stale_chunk', fault_step=0, policy_ttl=.01)
                proposal = ExperimentPolicy(policy, session, args)
                from roboguardian import RuntimeResult
                def execute(values, invocation, attempt):
                    session.steps += 1
                    return RuntimeResult('done', True, invocation, attempt)
                observe = lambda: {'robot_state': {'eef': {'pos':[0,0,0], 'quat':[0,0,0,1]}, 'gripper': {'qpos':[0,0]}}}
                controller = OnlineController(memory, observe, proposal, execute, execute, request_ttl=.01, mode=mode)
                try:
                    controller.step()
                    self.assertEqual(session.steps, count)
                    self.assertTrue(controller.events[0]['expired'])
                finally:
                    controller.close()

    def test_terminal_semantic_commit_has_no_official_success_dependency(self):
        class Session:
            scene_id, entity_ids, seq = 'task2', ('cup',), 0
            def visual_observation(self):
                self.seq += 1
                return Observation(str(self.seq), 'camera', self.seq, self.scene_id, self.entity_ids, {}, ('a.png','b.png'))
        for signal, expected in [(Signal.PASS, True), (Signal.FAIL, False), (Signal.INDETERMINATE, False)]:
            with self.subTest(signal=signal), closing(Memory(':memory:', signal.value)) as memory:
                result = verify_terminal(memory, Session(), SimpleNamespace(assess=lambda *a: signal), 'task?', 2)
                self.assertEqual(bool(memory.facts.get('task_complete')), expected)
                self.assertEqual(result.receipt.runtime_result.status, 'observation_only')

    def test_goal_assessment_is_independent_of_stopped_online_recovery(self):
        state = {'robot_state': {'eef': {'pos':[0,0,0], 'quat':[0,0,0,1]}, 'gripper': {'qpos':[0,0]}}}
        from roboguardian import RuntimeResult
        with closing(Memory(':memory:', 'online')) as online_memory, closing(Memory(':memory:', 'goal')) as goal_memory:
            execute = lambda v, i, a: RuntimeResult('done', True, i, a)
            controller = OnlineController(online_memory, lambda: state, lambda _: [1,0,0,0,0,0,0],
                                          execute, execute, window=1, max_recoveries=0)
            controller.step()
            controller.step()
            self.assertTrue(controller.stopped)
            self.assertIsNotNone(online_memory.active_recovery())
            controller.close()
            seq = iter(range(1,10))
            session = SimpleNamespace(scene_id='scene', entity_ids=('robot',),
                visual_observation=lambda: Observation(str(next(seq)), 'camera', next(seq), 'scene', ('robot',), {}, ('a','b')))
            result = verify_terminal(goal_memory, session, SimpleNamespace(assess=lambda *a: Signal.PASS), 'task?', 2)
            self.assertTrue(goal_memory.facts['task_complete'])
            self.assertEqual(online_memory.facts, {})

    def test_nonfinite_policy_response_has_diagnostic_without_dispatch(self):
        with closing(Memory(':memory:', 'invalid')) as memory:
            state = {'robot_state': {'eef': {'pos':[0,0,0], 'quat':[0,0,0,1]}, 'gripper': {'qpos':[0,0]}}}
            def execute(*a):
                self.fail('invalid response reached runtime')
            controller = OnlineController(memory, lambda: state, lambda _: [float('nan')] * 7, execute, execute)
            try:
                with self.assertRaises(ValueError):
                    controller.step()
                self.assertEqual(controller.events[-1]['kind'], 'invalid_policy_response')
            finally:
                controller.close()
