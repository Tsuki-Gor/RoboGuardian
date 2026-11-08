"""Propose one evaluation; LaViRA owns all navigation planning inside it."""

from uuid import uuid4
from roboguardian import Action
from .adapter import CONFIG_VERSION, EMBODIMENT


class HabitatPlanner:
    def __init__(self, adapter):
        self.adapter = adapter

    def propose(self, _instruction, context):
        observation = context.data["observation"]
        return Action(uuid4().hex, "navigation-evaluation", EMBODIMENT, "run_navigation_eval",
                      {"episode_ids": self.adapter.episode_ids}, self.adapter.scene,
                      self.adapter.entities, observation["observation_id"],
                      context.data["state_version"], {}, CONFIG_VERSION)
