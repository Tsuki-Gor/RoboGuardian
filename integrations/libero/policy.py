"""Native SmolVLA control and its single-rollout proposal."""

from pathlib import Path
from uuid import uuid4

from roboguardian import Action


class SmolVLAController:
    """Use the checkpoint's native observation/action processor chain unchanged."""
    def __init__(self, checkpoint, task_description, device, *, task_id=0, chunk_size=1):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if cfg.type != "smolvla":
            raise ValueError("this integration requires the native SmolVLA checkpoint")
        cfg.pretrained_path, cfg.device = Path(checkpoint), device
        if not 1 <= chunk_size <= cfg.chunk_size:
            raise ValueError("K must fit the checkpoint prediction horizon")
        cfg.n_action_steps = chunk_size
        self.chunk_size = chunk_size
        env_cfg = LiberoConfig(task="libero_10", task_ids=[task_id])
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.policy.config.device)},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.task_description = task_description
        self.inferences = 0

    def reset(self):
        self.policy.reset()

    def chunk(self, formatted):
        self.inferences += 1
        return [self(formatted).tolist() for _ in range(self.chunk_size)]

    def __call__(self, formatted):
        import torch
        from lerobot.envs.utils import preprocess_observation
        robot_state = {part: {name: value[None] for name, value in fields.items()}
                       for part, fields in formatted["robot_state"].items()}
        observation = preprocess_observation({**formatted, "robot_state": robot_state})
        observation["task"] = [self.task_description]
        observation = self.preprocessor(self.env_preprocessor(observation))
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.env_postprocessor({"action": self.postprocessor(action)})["action"]
        array = action.to("cpu").numpy()
        if array.shape != (1, 7):
            raise ValueError("native policy output must have shape (1, 7)")
        return array[0]


class SingleRolloutPlanner:
    def __init__(self, adapter):
        self.adapter, self.proposed = adapter, False

    def propose(self, instruction, context):
        if self.proposed:
            return None
        self.proposed = True
        return Action(uuid4().hex, "task-goal", self.adapter.embodiment, "smolvla_rollout", {},
                      self.adapter.scene_id, self.adapter.entity_ids, context.data["observation"]["observation_id"],
                      context.data["state_version"], {"scene_ready": True}, self.adapter.config_version)
