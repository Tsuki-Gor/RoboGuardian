"""Paired LIBERO-10 execution with per-action governance and isolated faults."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from uuid import uuid4

from roboguardian import Action, Budget, Guardian, Memory, RuntimeResult, Signal, SkillSpec, Verifier
from roboguardian.types import canonical_json, json_data
from integrations._semantic import SemanticClient, write_json
from .adapter import NativeSession
from .policy import SmolVLAController
from .tasks import choose_pair, entity_ids_from_bddl, generate_question
from .transport import LoopbackTransport, delayed_policy_result
from .online import OnlineController


def read_credentials(path):
    keys = ('OPENAI_API_KEY', 'OPENAI_BASE_URL')
    values = {key: os.environ[key] for key in keys if key in os.environ}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        key, separator, value = line.strip().removeprefix('export ').partition('=')
        key = key.strip()
        if separator and key in keys and key not in values:
            parts = shlex.split(value, comments=True)
            if len(parts) != 1:
                raise ValueError(f'{key} requires one simple quoted or unquoted value')
            values[key] = parts[0]
    return tuple(values.get(key) for key in keys)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--env-file', type=Path, required=True)
    p.add_argument('--model', default='qwen/qwen3.5-27b')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/smolvla_libero'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--suite', choices=('libero_10',), default='libero_10')
    p.add_argument('--task-id', type=int, choices=range(10))
    p.add_argument('--all-tasks', action='store_true')
    p.add_argument('--pairs', type=int, default=1, help='pairs per selected task; otherwise randomly select a task per pair')
    p.add_argument('--seed', type=int)
    p.add_argument('--initial-state', type=int)
    p.add_argument('--mode', choices=('paired', 'base', 'roboguardian'), default='paired')
    p.add_argument('--fault', choices=('none', 'stale_chunk', 'dispatch_timeout_retry', 'action_dropout'), default='none')
    p.add_argument('--fault-step', type=int, default=20)
    p.add_argument('--dropout-steps', type=int, default=25)
    p.add_argument('--k', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=520)
    p.add_argument('--max-observations', type=int, default=2)
    p.add_argument('--max-recoveries', type=int, default=2)
    p.add_argument('--stagnation-window', type=int, default=20)
    p.add_argument('--position-epsilon', type=float, default=.005, help='cumulative translation threshold in metres')
    p.add_argument('--policy-ttl', type=float, default=10.0)
    p.add_argument('--step-timeout', type=float, default=5.0)
    p.add_argument('--request-timeout', type=float, default=120.0)
    p.add_argument('--record-video', action='store_true')
    return p


class ExperimentPolicy:
    """Only the experiment knows why response delivery was delayed."""
    def __init__(self, policy, session, args):
        self.policy, self.session, self.args = policy, session, args
        self.injected = False
        self.events = []

    def reset(self):
        self.policy.reset()

    def __call__(self, observation):
        started = time.monotonic()
        observation_refs = self.session.visual_observation()
        self.events.append({'event': 'policy_input', 'inference': self.policy.inferences + 1,
                            'images': observation_refs.images, 'observation_id': observation_refs.observation_id})
        chunk = self.policy.chunk(self.session.formatted)
        if self.args.fault == 'stale_chunk' and not self.injected and self.session.steps >= self.args.fault_step:
            self.injected = True
            deadline = started + self.args.policy_ttl
            chunk, _ = delayed_policy_result(chunk, max(0, deadline - time.monotonic()) + .05, deadline)
            self.events.append({'fault': 'stale_chunk', 'physical_step': self.session.steps, 'delivery_at': time.monotonic()})
        return chunk


class ExperimentRuntime:
    def __init__(self, transport, session, args):
        self.transport, self.session, self.args = transport, session, args
        self.injected = False
        self.events = []

    @staticmethod
    def result(record, invocation, attempt):
        terminal = True if record and record['status'] == 'completed' else None
        return RuntimeResult(record['status'] if record else 'unknown', terminal, invocation, attempt)

    def execute(self, values, invocation, attempt):
        step, delay = self.session.steps, 0
        executed = list(values)
        if self.args.fault == 'dispatch_timeout_retry' and not self.injected and step >= self.args.fault_step:
            self.injected = True
            delay = self.args.step_timeout + .2
            self.events.append({'fault': self.args.fault, 'physical_step': step, 'invocation_id': invocation})
        if self.args.fault == 'action_dropout' and self.args.fault_step <= step < self.args.fault_step + self.args.dropout_steps:
            executed[:6] = [0.0] * 6
            self.events.append({'fault': self.args.fault, 'physical_step': step, 'invocation_id': invocation})
        try:
            record = self.transport.execute(invocation, attempt,
                {'action': executed, 'proposed_action': values, 'invocation_id': invocation, 'attempt_id': attempt}, delay)
            return self.result(record, invocation, attempt)
        except TimeoutError:
            return RuntimeResult('timeout', None, invocation, attempt)

    def status(self, values, invocation, attempt):
        try:
            return self.result(self.transport.query(invocation, attempt), invocation, attempt)
        except TimeoutError:
            return RuntimeResult('timeout', None, invocation, attempt)



def verify_terminal(memory, session, client, question, observations):
    """Observation-only assessment; never dispatch or reset a robot command."""
    class AssessmentRuntime:
        def execute(self, action, attempt):
            return RuntimeResult('observation_only', True, action.invocation_id, attempt)
        status = execute
        def observe(self, action):
            return session.visual_observation()
    runtime = AssessmentRuntime()
    skill = SkillSpec('libero:panda', 'assess_task', 'visual-task-v1',
        validator=lambda p: not p, describe=lambda a: question,
        pre_check=lambda a, o: Signal.PASS,
        post_check=lambda a, before, after: Signal.PASS if len(after.images) == 2 else Signal.INDETERMINATE,
        measure_effect=lambda *a: None, committed_facts=lambda *a: {'task_complete': True},
        required_channels=('post', 'semantic'), safe_retry=False)
    with Guardian(memory, [skill], runtime, runtime, Verifier(client.assess),
                  Budget(observations, 0, 2, 24000, 1)) as guardian:
        before = runtime.observe(None)
        action = Action(uuid4().hex, 'task-goal', 'libero:panda', 'assess_task', {},
                        session.scene_id, session.entity_ids, before.observation_id,
                        memory.state_version, {}, 'visual-task-v1')
        result = guardian.execute(action, observation=before)
        for _ in range(observations - 1):
            if result.receipt is None or result.receipt.state.value != 'unknown':
                break
            result = guardian.reconcile(action.invocation_id)
        return result


def run_episode(args, suite, pair, mode, output, checkpoint, question, entities, secret, endpoint):
    from lerobot.envs.libero import LiberoEnv
    from lerobot.utils.random_utils import set_seed
    output.mkdir()
    native = session = policy = controller = memory = video = transport = runtime = proposal = None
    client = SemanticClient(endpoint, secret, args.model, output, timeout=args.request_timeout)
    summary = {'pair': asdict(pair), 'mode': mode, 'official_success': None,
               'semantic_judgment': 'indeterminate', 'execution_complete': False}
    started = time.monotonic()
    try:
        with (output / 'native.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            set_seed(pair.seed)
            native = LiberoEnv(suite, task_id=pair.task_id, task_suite_name='libero_10', obs_type='pixels_agent_pos',
                observation_width=360, observation_height=360, init_states=True, episode_index=pair.initial_state,
                n_envs=1, num_steps_wait=10, control_mode='relative')
            if pair.initial_state >= len(native._init_states):
                raise ValueError('initial state out of range')
            native.reset(seed=pair.seed)
            policy = SmolVLAController(checkpoint, native.task_description, 'cuda', task_id=pair.task_id, chunk_size=args.k)
            set_seed(pair.seed)
            if args.record_video:
                import imageio.v2 as imageio
                video = imageio.get_writer(output / 'rollout.mp4', fps=30, macro_block_size=1)
            scene = f'libero_10:task{pair.task_id}:init{pair.initial_state}'
            session = NativeSession(native, output, scene_id=scene, entity_ids=entities, video=video, max_steps=args.max_steps)
            transport = LoopbackTransport(session.step, timeout=args.step_timeout)
            runtime = ExperimentRuntime(transport, session, args)
            proposal = ExperimentPolicy(policy, session, args)
            memory = Memory(output / 'memory.sqlite', uuid4().hex)
            controller = OnlineController(memory, session.observe, proposal, runtime.execute, runtime.status,
                chunk_size=args.k, request_ttl=args.policy_ttl, window=args.stagnation_window, position_epsilon=args.position_epsilon,
                max_observations=args.max_observations, max_recoveries=args.max_recoveries,
                mode=mode, scene_id=scene, entity_ids=entities)
            # Control attempts are bounded independently of actual physics steps.
            for _ in range(args.max_steps + (args.max_recoveries + 1) * (args.max_observations + 2)):
                if session.done or session.steps >= args.max_steps or controller.stopped:
                    break
                controller.step()
            # Reconcile an outstanding final response even when evaluator ended the episode.
            pending = memory.pending_attempt()
            for _ in range(args.max_observations):
                if pending is None:
                    break
                controller.step()
                pending = memory.pending_attempt()
            # Join every physical callback before any terminal image or resource close.
            # Reconcile RG first while the status endpoint is still open.
            transport.close()
            session.visual_observation()
            controller.close()
            # A stopped online recovery must not suppress independent goal evidence.
            goal_memory = Memory(output / 'goal.sqlite', memory.episode_id + '-goal')
            try:
                assessment = verify_terminal(goal_memory, session, client, question, args.max_observations)
                summary['task_receipt'] = json_data(assessment.receipt)
                summary['goal_judgment'] = assessment.receipt.assessment.judgment.value if assessment.receipt else 'uncertain'
                reports = [e for e in goal_memory.events() if e['kind'] == 'receipt']
                summary['semantic_judgment'] = reports[-1]['payload']['report']['channels'].get('semantic', 'indeterminate') if reports else 'indeterminate'
                summary['semantic_verified_success'] = bool(goal_memory.facts.get('task_complete'))
                summary['guardian_verified_success'] = bool(mode == 'roboguardian' and goal_memory.facts.get('task_complete')
                    and memory.pending_attempt() is None and memory.active_recovery() is None)
                write_json(output / 'goal-audit.json', goal_memory.events(), secret)
            finally:
                goal_memory.close()
            summary.update(official_success=session.success, physical_steps=session.steps,
                policy_inferences=policy.inferences, recovery_triggered=any(e['kind'] in ('recovery', 'recovery_budget_exhausted', 'reconcile') for e in controller.events),
                recovery_success=bool(session.success) if any(e['kind'] in ('recovery', 'recovery_budget_exhausted', 'reconcile') for e in controller.events) else None,
                online_stopped=controller.stopped, unresolved=memory.pending_attempt() is not None,
                fault_triggered=bool(runtime.events or any('fault' in e for e in proposal.events)))
            write_json(output / 'governance.json', controller.events, secret)
            write_json(output / 'faults.json', runtime.events + proposal.events, secret)
            summary['execution_complete'] = True
    except Exception as exc:
        summary['execution_complete'] = False
        summary['error'] = f'{type(exc).__name__}: {exc}'.replace(secret, '[REDACTED]')
    finally:
        if transport is not None:
            transport.close()
        def persist(name, value):
            try:
                write_json(output / name, value, secret)
            except Exception as exc:
                summary['execution_complete'] = False
                summary.setdefault('artifact_errors', {})[name] = type(exc).__name__
        if controller is not None:
            controller.close()
            inputs = [e for e in proposal.events if e.get('event') == 'policy_input'] if proposal else []
            requests = [e for e in controller.events if e.get('kind') == 'policy']
            for event, image_input in zip(requests, inputs):
                event['input_images'] = image_input['images']
            expired = {e['request_id'] for e in requests if e.get('expired')}
            summary['expired_responses'] = len(expired)
            summary['expired_actions_dispatched'] = sum(e.get('kind') == 'proposal' and e.get('request_id') in expired for e in controller.events)
            summary['blocked_expired_responses'] = len(expired) if mode == 'roboguardian' else 0
            summary['recovery_dispatches'] = controller.recoveries
            persist('governance.json', controller.events)
        if runtime is not None and proposal is not None:
            persist('faults.json', runtime.events + proposal.events)
        if transport is not None:
            persist('transport.json', transport.events)
            starts = session.dispatches if session is not None else []
            counts = {}
            for e in starts:
                counts[e['invocation_id']] = counts.get(e['invocation_id'], 0) + 1
            summary['duplicate_dispatches'] = sum(n - 1 for n in counts.values())
        if session is not None:
            summary.update(physical_steps=session.steps, official_success=session.success)
        if policy is not None:
            summary['policy_inferences'] = policy.inferences
        if memory is not None:
            persist('audit.json', memory.events())
            memory.close()
        for name, resource in (('session', session), ('video', video), ('environment', native)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    summary[name + '_close_error'] = type(exc).__name__
        summary.update(semantic_calls=client.calls, elapsed_seconds=time.monotonic() - started)
        write_json(output / 'summary.json', summary, secret)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    import math
    if (args.gpu < 0 or not 1 <= args.max_steps <= 520 or args.pairs < 1 or args.k < 1
            or args.fault_step < 0 or args.dropout_steps < 1 or args.max_observations < 1 or args.max_recoveries < 0
            or args.stagnation_window < 1 or not math.isfinite(args.position_epsilon) or args.position_epsilon < 0 or (args.all_tasks and args.task_id is not None)
            or any(not math.isfinite(v) or v <= 0 for v in (args.policy_ttl, args.step_timeout, args.request_timeout))):
        raise ValueError('invalid experiment limits')
    secret, endpoint = read_credentials(args.env_file.expanduser())
    if not secret or not endpoint or not args.model:
        raise ValueError('explicit model credentials are required')
    root = args.native_root.expanduser().resolve()
    checkpoint = (root / args.checkpoint).resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_EGL_DEVICE_ID=str(args.gpu), MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', HF_HOME=str(root / 'cache/huggingface'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', LIBERO_CONFIG_PATH=str(root / 'cache/home/.libero'))
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()['libero_10']()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'env_file'}
    config['config_version'] = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {'configuration': config, 'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'packages': {n: version(n) for n in ('lerobot', 'hf-libero', 'robosuite', 'mujoco', 'torch')}, 'pairs': []}
    write_json(output / 'manifest.json', manifest, secret)
    modes = ('base', 'roboguardian') if args.mode == 'paired' else (args.mode,)
    selected = list(range(10)) if args.all_tasks else [args.task_id]
    rows = []
    for task_id in selected:
        for _ in range(args.pairs):
            ids = [task_id] if task_id is not None else list(range(10))
            counts = {i: len(suite.get_task_init_states(i)) for i in ids}
            pair = choose_pair(ids, counts, args.seed, args.initial_state, args.fault)
            pair_dir = output / f'pair-{len(manifest["pairs"]):04d}'
            pair_dir.mkdir()
            task = suite.get_task(pair.task_id)
            client = SemanticClient(endpoint, secret, args.model, pair_dir, timeout=args.request_timeout)
            record = {'pair_id': pair_dir.name, **asdict(pair)}
            manifest['pairs'].append(record)
            write_json(output / 'manifest.json', manifest, secret)
            try:
                entities = entity_ids_from_bddl(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file)
                record['entities'] = entities
                question = generate_question(client, task.language, entities)
            except Exception as exc:
                record['preparation_error'] = str(exc).replace(secret, '[REDACTED]')
                for mode in modes:
                    rows.append({'pair_id': pair_dir.name, 'pair': asdict(pair), 'mode': mode,
                                 'execution_complete': False, 'error': record['preparation_error']})
                write_json(output / 'manifest.json', manifest, secret)
                write_json(output / 'results.json', rows, secret)
                continue
            record['question'] = question
            record['question_model_calls'] = client.calls
            record['task_description'] = task.language
            write_json(output / 'manifest.json', manifest, secret)
            for mode in modes:
                try:
                    row = run_episode(args, suite, pair, mode, pair_dir / mode, checkpoint, question, entities, secret, endpoint)
                except Exception as exc:
                    row = {'pair': asdict(pair), 'mode': mode, 'execution_complete': False,
                           'error': f'{type(exc).__name__}: {exc}'.replace(secret, '[REDACTED]')}
                row['pair_id'] = pair_dir.name
                rows.append(row)
                write_json(output / 'results.json', rows, secret)
    expected = len(selected) * args.pairs * len(modes)
    complete = len(rows) == expected and all(r['execution_complete'] for r in rows)
    write_json(output / 'summary.json', {'expected_episodes': expected, 'recorded_episodes': len(rows),
               'execution_complete': complete, 'episodes': rows}, secret)
    print(json.dumps({'output': str(output), 'execution_complete': complete, 'episodes': len(rows)}))
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
