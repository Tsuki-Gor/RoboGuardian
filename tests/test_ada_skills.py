"""Native Ada skill contracts, exercised with a small simulator-shaped fixture."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from roboguardian import Action, Signal
from integrations.adacleargrasp.adapter import AdaAdapter


class AdaSkillsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.directory.cleanup)
        self.target = SimpleNamespace(name='apple', pose=SimpleNamespace(p=[[0., 0., .04]], q=[[1., 0., 0., 0.]]))
        self.clutter = SimpleNamespace(name='banana', pose=SimpleNamespace(p=[[.2, 0., .04]], q=[[1., 0., 0., 0.]]))
        self.tcp = SimpleNamespace(pose=SimpleNamespace(p=[[0., 0., .2]], q=[[1., 0., 0., 0.]]))
        robot = SimpleNamespace(get_qpos=lambda: [[0.] * 19], get_qvel=lambda: [[0.] * 19])
        self.env = SimpleNamespace(target_actors=[self.target], target_object=self.target,
                                   clutter_actors_per_item=[[self.clutter]],
                                   agent=SimpleNamespace(tcp=self.tcp, robot=robot,
                                                         count_finger_contacts=lambda *a, **k: [0]),
                                   evaluate=lambda: {'height': [.04], 'success': [False]},
                                   render=lambda: 'frame')
        self.env.unwrapped = self.env
        self.calls = []
        self.reply = SimpleNamespace(ok=True, error_code='none', message='ok', advice='')
        def operation(name):
            def call(*args, **kwargs):
                self.calls.append((name, args))
                return self.reply
            return call
        self.initializer = SimpleNamespace(cfg=SimpleNamespace(arm_init_7=[0.] * 7), HAND_INIT_12=[0.] * 12,
                                           initarm=operation('initarm'), inithand=operation('inithand'))
        self.mover = SimpleNamespace(move_to=operation('move_to'), lower=operation('lower'), lift=operation('lift'))
        self.pusher = SimpleNamespace(push=operation('push'))
        self.puller = SimpleNamespace(pull=operation('pull'))
        self.grasper = SimpleNamespace(grasp=operation('grasp'))
        patcher = patch('integrations.adacleargrasp.adapter.save_frame', lambda frame, path: path.write_bytes(b'image'))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapter = AdaAdapter(self.env, self.initializer, self.grasper, Path(self.directory.name),
                                  scene_id='scene', target_label='apple', config_version='v1',
                                  initarm_tolerance=.05, height_threshold=.15, relation_threshold=.18,
                                  min_contacts=2, mover=self.mover, pusher=self.pusher, puller=self.puller)

    def action(self, observation, skill, params, identifier='call'):
        name = params.get('name', 'target')
        entity = self.adapter.target_id if name == 'target' else name
        return Action(identifier, 'goal', self.adapter.embodiment, skill, params, 'scene', (entity,),
                      observation.observation_id, 0, {}, 'v1')

    def test_parameters_validate_exact_fields_and_known_objects(self):
        a = self.adapter
        self.assertEqual(a.objects, ('apple', 'banana'))
        self.assertEqual(a.entity_ids, a.objects)
        self.assertTrue(a.validate_parameters('move_to', {'name': 'target'}))
        self.assertTrue(a.validate_parameters('push', {'name': 'banana', 'side': 'middle', 'dist_m': .2}))
        for skill, params in [('move_to', {'name': 'unknown'}), ('lift', {'name': 'apple'}),
                              ('push', {'name': 'banana', 'side': 'left', 'dist_m': True}),
                              ('pull', {'name': 'banana', 'side': 'up', 'dist_m': .1}),
                              ('push', {'name': 'banana', 'side': 'left', 'dist_m': float('nan')}),
                              ('push', {'name': 'banana', 'side': 'left', 'dist_m': .1, 'extra': 1})]:
            self.assertFalse(a.validate_parameters(skill, params))

    def test_invalid_native_proposal_reports_rejection_without_moving(self):
        before = self.adapter.observe(None)
        invalid = self.action(before, 'move_to', {'name': 'missing'})
        result = self.adapter.execute(invalid, 'bad-attempt')
        self.assertTrue(result.terminal)
        self.assertFalse(result.diagnostics['native_result']['ok'])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.adapter.last_execution['invocation_id'], invalid.invocation_id)
        fresh = self.adapter.observe(None)
        valid = self.action(fresh, 'move_to', {'name': 'target'}, 'corrected')
        self.assertTrue(self.adapter.execute(valid, 'good-attempt').diagnostics['native_result']['ok'])
        self.assertEqual(self.calls, [('move_to', ('apple',))])

    def test_native_preparation_and_named_direction_evidence(self):
        a = self.adapter
        specs = {s.skill: s for s in a.skills()}
        self.assertEqual(set(specs), {'move_to', 'lift', 'lower', 'push', 'pull', 'initarm', 'inithand', 'grasp'})
        before = a.observe(None)
        push = self.action(before, 'push', {'name': 'banana', 'side': 'center', 'dist_m': .2})
        self.assertIs(specs['push'].pre_check(push, before), Signal.FAIL)
        move = self.action(before, 'move_to', {'name': 'banana'}, 'move')
        a.execute(move, 'attempt-move')
        self.tcp.pose.p = [[.2, 0., .2]]
        after_move = a.observe(move)
        self.assertIs(specs['move_to'].post_check(move, before, after_move), Signal.PASS)
        lower = self.action(after_move, 'lower', {}, 'lower')
        a.execute(lower, 'attempt-lower')
        before_push = a.observe(lower)
        push = self.action(before_push, 'push', {'name': 'banana', 'side': 'center', 'dist_m': .2}, 'push')
        self.assertIs(specs['push'].pre_check(push, before_push), Signal.PASS)
        a.execute(push, 'attempt-push')
        self.clutter.pose.p[0][0] += .02
        after_push = a.observe(push)
        self.assertIs(specs['push'].post_check(push, before_push, after_push), Signal.PASS)
        wrong = self.action(after_push, 'move_to', {'name': 'target'}, 'wrong')
        self.assertIs(specs['move_to'].pre_check(wrong, after_push), Signal.FAIL)
        lift = self.action(after_push, 'lift', {}, 'lift')
        self.reply = SimpleNamespace(ok=False, error_code='stuck', message='stuck', advice='')
        a.execute(lift, 'attempt-lift')
        stuck = a.observe(lift)
        self.assertIs(specs['lift'].pre_check(self.action(stuck, 'lift', {}, 'again'), stuck), Signal.FAIL)
        self.assertIs(specs['initarm'].pre_check(self.action(stuck, 'initarm', {}, 'reset'), stuck), Signal.PASS)
        self.assertEqual(a.native_feedback['error_code'], 'stuck')
        self.assertEqual(a.status(lift, 'attempt-lift').diagnostics['native_result']['error_code'], 'stuck')
        self.assertEqual([name for name, _ in self.calls], ['move_to', 'lower', 'push', 'lift'])

    def test_grasp_two_of_three_ignores_official_success_and_needs_semantics(self):
        a = self.adapter
        self.target.pose.p[0][2] = .16
        self.env.agent.count_finger_contacts = lambda *args, **kwargs: [0]
        observation = a.observe(None)
        self.assertFalse(observation.state['evaluation']['success'])
        self.assertIs(a.grasp_signal(observation), Signal.PASS)
        grasp = next(s for s in a.skills() if s.skill == 'grasp')
        self.assertEqual(grasp.required_channels, ('post', 'semantic'))
        self.target.pose.p[0][2] = .04
        self.tcp.pose.p = [[1., 0., .2]]
        self.env.agent.count_finger_contacts = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('missing'))
        self.assertIs(a.grasp_signal(a.observe(None)), Signal.FAIL)
        from dataclasses import replace
        state = dict(observation.state)
        state['finger_contacts'] = None
        state['target_tcp_distance_m'] = None
        self.assertIs(a.grasp_signal(replace(observation, state=state)), Signal.INDETERMINATE)

    def test_official_evaluation_unavailable_does_not_block_physical_evidence(self):
        self.target.pose.p[0][2] = .16
        self.env.evaluate = lambda: (_ for _ in ()).throw(RuntimeError('unavailable'))
        self.assertIs(self.adapter.grasp_signal(self.adapter.observe(None)), Signal.PASS)

    def test_guardian_fuses_grasp_with_independent_semantics(self):
        from roboguardian import Budget, Guardian, Memory, ReceiptState, Verifier
        a = self.adapter
        self.target.pose.p[0][2] = .16
        before = a.observe(None)
        action = self.action(before, 'grasp', {}, 'grasp')
        memory = Memory(':memory:', 'episode')
        self.addCleanup(memory.close)
        with Guardian(memory, a.skills(), a, a, Verifier(lambda *args: Signal.FAIL),
                      Budget(1, 1, 2, 20000, 2)) as guardian:
            result = guardian.execute(action, observation=before)
            guardian.execute(action, observation=before)
        self.assertIs(result.receipt.state, ReceiptState.UNKNOWN)
        self.assertEqual(memory.facts, {})
        attempt = result.receipt.attempt_id
        a.execute(action, attempt)
        self.assertEqual([name for name, _ in self.calls], ['grasp'])
        self.assertEqual(a.status(action, attempt).terminal, True)
        self.assertEqual(a.last_observation.state['visual_reference_before']['skill'], 'grasp')

    def test_wrong_subject_and_wrong_displacement_do_not_pass(self):
        a = self.adapter
        specs = {s.skill: s for s in a.skills()}
        before = a.observe(None)
        proposal = self.action(before, 'grasp', {'name': 'banana'})
        self.assertIs(specs['grasp'].pre_check(proposal, before), Signal.FAIL)
        push = self.action(before, 'push', {'name': 'banana', 'side': 'center', 'dist_m': .2})
        self.target.pose.p[0][0] += .2
        self.clutter.pose.p[0][0] -= .02
        after = a.observe(None)
        self.assertIs(specs['push'].post_check(push, before, after), Signal.FAIL)

    def test_target_alias_dispatches_actual_native_actor_name(self):
        before = self.adapter.observe(None)
        action = self.action(before, 'move_to', {'name': 'target'})
        self.adapter.execute(action, 'attempt')
        self.assertEqual(self.calls, [('move_to', ('apple',))])

    def test_new_attempt_can_redispatch_but_status_never_dispatches(self):
        a = self.adapter
        before = a.observe(None)
        action = self.action(before, 'grasp', {}, 'grasp')
        a.execute(action, 'first')
        a.observe(action)
        a.status(action, 'first')
        a.execute(action, 'second')
        self.assertEqual([name for name, _ in self.calls], ['grasp', 'grasp'])
        after = a.observe(action)
        reference = after.state['visual_reference_before']
        self.assertEqual(reference['attempt_id'], 'second')
        self.assertEqual(reference['observation_id'], before.observation_id)

    def test_operational_success_records_no_object_fact(self):
        from roboguardian import Budget, Guardian, Memory, ReceiptState, Verifier
        a = self.adapter
        before = a.observe(None)
        action = self.action(before, 'initarm', {}, 'init')
        memory = Memory(':memory:', 'operational')
        self.addCleanup(memory.close)
        with Guardian(memory, a.skills(), a, a, Verifier(lambda *args: Signal.INDETERMINATE),
                      Budget(1, 1, 2, 20000, 2)) as guardian:
            result = guardian.execute(action, observation=before)
        self.assertIs(result.receipt.state, ReceiptState.COMMITTED)
        self.assertEqual(memory.facts, {})
        self.assertEqual(memory.relations["dynamic_objects"], [])

    def test_unguarded_native_dispatch_does_not_enforce_operation_phase(self):
        a = self.adapter
        before = a.observe(None)
        push = self.action(before, 'push', {'name': 'banana', 'side': 'center', 'dist_m': .2})
        self.assertIs(next(s for s in a.skills() if s.skill == 'push').pre_check(push, before), Signal.FAIL)
        result = a.execute(push, 'attempt')
        self.assertTrue(result.terminal)
        self.assertEqual([name for name, _ in self.calls], ['push'])

    def test_recovery_requires_new_action_and_explicit_budget(self):
        from roboguardian import Budget, Guardian, Memory, ReceiptState, Verifier
        a = self.adapter
        self.tcp.pose.p = [[1., 0., .2]]
        before = a.observe(None)
        first = self.action(before, 'grasp', {}, 'first')
        memory = Memory(':memory:', 'recovery')
        self.addCleanup(memory.close)
        with Guardian(memory, a.skills(), a, a, Verifier(lambda *args: Signal.FAIL),
                      Budget(1, 1, 3, 20000, 2)) as guardian:
            failed = guardian.execute(first, observation=before)
            self.assertIs(failed.receipt.state, ReceiptState.REJECTED)
            fresh = a.observe(None)
            second = self.action(fresh, 'grasp', {}, 'second')
            retried = guardian.recover('first', second, observation=fresh)
            self.assertTrue(retried.admission.allowed)
        self.assertEqual([name for name, _ in self.calls], ['grasp', 'grasp'])
