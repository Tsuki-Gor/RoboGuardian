"""Attach budget, model recording and video to the original LaViRA eval loop."""
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys


def read_env(path):
    keys = ('LA_API_KEY', 'LA_BASE_URL', 'LA_MODEL_NAME', 'VA_API_KEY', 'VA_BASE_URL', 'VA_MODEL_NAME')
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.removeprefix('export ').partition('=')
        key = key.strip()
        if sep and key in keys:
            parts = shlex.split(value, comments=True)
            if len(parts) != 1 or key in values:
                raise ValueError('invalid or repeated model configuration: ' + key)
            values[key] = parts[0]
    if any(not values.get(key) for key in keys):
        raise ValueError('model env file must define both endpoints, keys and model names')
    return values


class RetryingGenerate:
    def __init__(self, native, control, output, timeout, interval, width, height):
        self.native, self.control = native, control
        self.output = Path(output) if output is not None else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
        self.timeout, self.interval = timeout, interval
        self.width, self.height = width, height
        self.count = 0
        self.waypoints = set()
        self.secrets = [getattr(native.la_client, 'api_key', ''), getattr(native.va_client, 'api_key', '')]

    def write(self, path, value):
        if self.output is None:
            return
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, '[REDACTED]')
        (self.output / path).write_text(text, encoding='utf-8')

    def validate(self, text, use_la, model):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError('response has no JSON object')
        obj = json.loads(match.group())
        if not isinstance(obj, dict):
            raise ValueError('response is not an object')
        if use_la:
            action = obj.get('action')
            if not isinstance(action, str):
                raise ValueError('missing LA action')
            nav = re.fullmatch(r'navigate to (left|right|forward|behind)', action.strip(), re.I)
            back = re.fullmatch(r'backtrack to (?:waypoint\s*)?(\d+)', action.strip(), re.I)
            if not nav and not (back and int(back.group(1)) in self.waypoints):
                raise ValueError('invalid LA action or waypoint')
            if not all(isinstance(obj.get(k), str) for k in ('reasoning', 'progress_analysis')):
                raise ValueError('missing LA reasoning or progress')
        else:
            action, box = obj.get('action'), obj.get('bbox_2d')
            if not isinstance(action, str) or action.upper() not in ('NAVIGATE', 'STOP'):
                raise ValueError('invalid VA action')
            if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in ('target', 'reasoning', 'progress')):
                raise ValueError('missing VA target or reasoning')
            if (not isinstance(box, list) or len(box) != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in box)):
                raise ValueError('invalid VA box')
            w, h = (1000, 1000) if 'qwen3' in model.lower() else (self.width, self.height)
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError('VA box outside image or empty')
            # Native parsing casts after denormalization; subpixel boxes cannot be used.
            if int(box[2] * self.width / w) <= int(box[0] * self.width / w) or int(box[3] * self.height / h) <= int(box[1] * self.height / h):
                raise ValueError('VA box is smaller than one pixel')
        obj['action'] = obj['action'].strip()
        if not use_la and 'qwen3' in model.lower():
            obj['bbox_2d'] = [box[0] * self.width / 1000, box[1] * self.height / 1000,
                              box[2] * self.width / 1000, box[3] * self.height / 1000]
        return json.dumps(obj, ensure_ascii=False)

    def __call__(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_la=False, **kwargs):
        native = self.native
        client = native.la_client if use_la else native.va_client
        model = native.la_model_name if use_la else native.va_model_name
        if not use_la and 'qwen3.5' in model.lower():
            kwargs['extra_body'] = ({'reasoning': {'effort': 'none'}} if 'openrouter.ai' in getattr(native, 'va_base_url', os.environ.get('VA_BASE_URL', '')).lower()
                                    else {'enable_thinking': False})
        request = dict(model=model, messages=messages, max_tokens=max_new_tokens, temperature=temperature, **kwargs)
        while True:
            self.control.checkpoint()
            self.count += 1
            prefix = '%05d-%s' % (self.count, 'la' if use_la else 'va')
            self.write(prefix + '.request.json', request)
            try:
                remaining = self.control.checkpoint()
                response = client.with_options(timeout=min(self.timeout, remaining), max_retries=0).chat.completions.create(**request)
                content = response.choices[0].message.content
                if hasattr(response, 'model_dump'):
                    raw = response.model_dump()
                elif hasattr(response, 'dict'):
                    raw = response.dict()
                else:
                    raw = str(response)
                self.write(prefix + '.response.json', {'text': content, 'response': raw})
                if not isinstance(content, str):
                    raise ValueError('model response is not text')
                key = 'Language Action Model' if use_la else 'Vision Action Model'
                stats = native.stats[key]
                stats['calls'] += 1
                usage = getattr(response, 'usage', None)
                if usage:
                    for field, source in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'), ('total_tokens', 'total_tokens')):
                        stats[field] += getattr(usage, source, 0) or 0
                return self.validate(content, use_la, model)
            except Exception as exc:
                self.write(prefix + '.error.json', {'error': type(exc).__name__ + ': ' + str(exc)})
                self.control.wait(self.interval)


class EnvProxy:
    def __init__(self, envs, control, output, video):
        self.envs, self.control, self.output, self.video = envs, control, Path(output), video
        self.writer = None
        self.closed = False
        self.steps = 0
        self.episode_steps = 0
        self.latest_info = None
        self.episode = None
        self.trajectory = (self.output / 'trajectory.jsonl').open('x', encoding='utf-8')

    def __getattr__(self, name):
        return getattr(self.envs, name)

    def reset(self):
        self.control.checkpoint()
        observations = self.envs.reset()
        self.latest_info = None
        self.episode_steps = 0
        self.episode = str(self.envs.current_episodes()[0].episode_id)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.video:
            import cv2
            rgb = observations[0]['rgb']
            if not re.fullmatch(r'[A-Za-z0-9_-]+', self.episode):
                raise ValueError('invalid episode ID for video')
            path = self.output / ('episode-' + self.episode + '.mp4')
            if path.exists():
                raise FileExistsError(str(path))
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (rgb.shape[1], rgb.shape[0]))
            if not self.writer.isOpened():
                raise RuntimeError('MP4 video writer could not open')
            self.frame(rgb)
        return observations

    def frame(self, rgb):
        if self.writer is not None:
            self.writer.write(rgb[:, :, :3][:, :, ::-1].copy())

    def step(self, actions):
        self.control.checkpoint()
        result = self.envs.step(actions)
        self.steps += 1
        self.episode_steps += 1
        self.latest_info = result[0][3]
        position = self.latest_info.get('position', {})
        positions, distances = position.get('position', []), position.get('distance', [])
        record = {'episode_id': self.episode, 'step': self.steps, 'actions': actions,
                  'position': positions[-1] if len(positions) else None,
                  'distance_to_goal': distances[-1] if len(distances) else None, 'done': bool(result[0][2])}
        if self.episode_steps == 1 and len(positions):
            record['start_position'] = positions[0]
        self.trajectory.write(json.dumps(record, default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value)) + '\n')
        self.trajectory.flush()
        self.frame(result[0][0]['rgb'])
        self.control.update(native_steps=self.steps)
        return result

    def metric(self, calculate, infos):
        # Panorama can finish an episode while discarding its final infos.
        return calculate([self.latest_info] if self.latest_info is not None else infos)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.trajectory.close()
            finally:
                try:
                    if self.writer is not None:
                        self.writer.release()
                finally:
                    self.envs.close()


def run_native(options, control):
    root = Path(options['native_root']).resolve()
    output = Path(options['output_dir']).resolve()
    assets = Path(options['assets_root']).resolve()
    if any(not math.isfinite(float(options[key])) or float(options[key]) <= 0
           for key in ('model_timeout', 'retry_interval')):
        raise ValueError('model timeout and retry interval must be positive')
    values = read_env(options['env_file'])
    control.secrets = (values['LA_API_KEY'], values['VA_API_KEY'])
    for kind in ('la', 'va'):
        expected = options.get(kind + '_model')
        if expected and values[kind.upper() + '_MODEL_NAME'] != expected:
            raise ValueError(kind + ' model does not match configuration')
    os.environ.update(values)
    os.environ.update(HF_HOME=str(Path(options['hf_home']).resolve()), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(root))
    os.chdir(str(root))
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.utils import seed_everything
    from vlnce_baselines.lavira_main import LaViRA
    with gzip.open(options['dataset_path'], 'rt', encoding='utf-8') as stream:
        ids = [str(episode['episode_id']) for episode in json.load(stream)['episodes']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('selected dataset must contain unique episodes')
    config = get_config(options.get('config_path', 'vlnce_baselines/config/r2r.yaml'))
    config.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = str(Path(options['dataset_path']).resolve())
    config.TASK_CONFIG.DATASET.SCENES_DIR = str(Path(options['scene_root']).resolve())
    config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ids
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['*']
    config.TASK_CONFIG.TASK.NDTW.GT_PATH = str(Path(options['ground_truth_path']).resolve())
    config.TASK_CONFIG.TASK.SDTW.GT_PATH = str(Path(options['ground_truth_path']).resolve())
    config.TASK_CONFIG.SEED = int(options['seed'])
    config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = int(options['max_native_steps'])
    config.TORCH_GPU_ID = int(options['gpu'])
    config.TORCH_GPU_IDS = [int(options['gpu'])]
    config.SIMULATOR_GPU_IDS = [int(options['gpu'])]
    config.NUM_ENVIRONMENTS = 1
    config.EVAL.EPISODE_COUNT = len(ids)
    config.RESULTS_DIR = config.EVAL_CKPT_PATH_DIR = str(output)
    config.VIDEO_DIR = str(output / 'videos')
    config.RGB_SAVE_DIR = str(output / 'frames')
    # Native VLM map rendering also needs the initialized visualization canvas.
    config.MAP.VISUALIZE = True
    config.MAP.PRINT_IMAGES = False
    for name in ('GROUNDING_DINO_CONFIG_PATH', 'GROUNDING_DINO_CHECKPOINT_PATH', 'RepViTSAM_CHECKPOINT_PATH', 'SAM_CHECKPOINT_PATH'):
        setattr(config.MAP, name, str(assets / getattr(config.MAP, name)))
    config.local_rank = 0
    config.world_size = 1
    config.freeze()
    seed_everything(config.TASK_CONFIG.SEED)
    trainer = None
    try:
        control.checkpoint()
        trainer = LaViRA(config, r2r=True)
        trainer.visualize = False
        generate = RetryingGenerate(trainer.agent.model, control, output / 'model_requests',
                                    float(options['model_timeout']), float(options['retry_interval']), trainer.width, trainer.height)
        trainer.agent.model.generate = generate
        original_decision = trainer.agent.navigate_or_backtrack
        def decision(instruction, visited_targets):
            generate.waypoints = {i for i, target in enumerate(visited_targets[:-1]) if 'description' in target}
            return original_decision(instruction, visited_targets)
        trainer.agent.navigate_or_backtrack = decision
        original_query = trainer.agent.query_llm
        def query(*args, **kwargs):
            while True:
                result = original_query(*args, **kwargs)
                if result.get('target') != 'fallback target' and result.get('reasoning') != 'Fallback due to parsing error':
                    return result
                control.wait(float(options['retry_interval']))
        trainer.agent.query_llm = query
        init_envs = trainer._init_envs
        def wrapped_init():
            init_envs()
            trainer.envs = EnvProxy(trainer.envs, control, output, bool(options.get('record_video', True)))
        trainer._init_envs = wrapped_init
        calculate_metric = trainer._calculate_metric
        def wrapped_metric(infos):
            return trainer.envs.metric(calculate_metric, infos)
        trainer._calculate_metric = wrapped_metric
        trainer.eval()
        return {str(k): v for k, v in trainer.state_eps.items()}
    finally:
        if trainer is not None and getattr(trainer, 'envs', None) is not None:
            trainer.envs.close()
