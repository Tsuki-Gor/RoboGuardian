"""Skill callbacks and the planner, runtime, observer, and assessor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .models import Action, Context, EffectMeasurement, Observation, RuntimeResult, Signal, _refs, _text
from .serialization import JSONMap, JSONValue


@dataclass(frozen=True)
class SkillSpec:
    """Trusted adapter callbacks; exceptions must be handled by the caller.

    Exact callback signatures:
      validator(parameters: Mapping[str, JSONValue]) -> bool
      describe(action: Action) -> str
      pre_check(action: Action, observation: Observation) -> Signal
      post_check(action: Action, before: Observation, after: Observation) -> Signal
      measure_effect(action: Action, before: Observation, after: Observation)
          -> EffectMeasurement | None
      committed_facts(action: Action, before: Observation, after: Observation)
          -> Mapping[str, JSONValue]
      committed_relations(action: Action, before: Observation, after: Observation)
          -> list[JSONMap] | tuple[JSONMap, ...] (optional; None means no callback)

    post_check includes every goal-integrity condition that must still hold.
    required_channels explicitly names semantic/post/effect; pre cannot prove
    success. The registered adapter fixes measurement units, frame and
    tolerances for config_version. safe_retry is permission, not a retry policy.
    """
    embodiment: str
    skill: str
    config_version: str
    validator: Callable[[JSONMap], bool]
    describe: Callable[[Action], str]
    pre_check: Callable[[Action, Observation], Signal]
    post_check: Callable[[Action, Observation, Observation], Signal]
    measure_effect: Callable[[Action, Observation, Observation], EffectMeasurement | None]
    committed_facts: Callable[[Action, Observation, Observation], JSONMap]
    required_channels: tuple[str, ...]
    safe_retry: bool = False
    committed_relations: Callable[[Action, Observation, Observation], list[JSONMap] | tuple[JSONMap, ...]] | None = None

    def __post_init__(self):
        for name in ("embodiment", "skill", "config_version"):
            _text(getattr(self, name), name)
        for name in ("validator", "describe", "pre_check", "post_check", "measure_effect", "committed_facts"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        _refs(self, "required_channels")
        if (not self.required_channels or
                any(channel not in ("semantic", "post", "effect") for channel in self.required_channels)):
            raise ValueError("required_channels must include semantic, post or effect")
        if len(set(self.required_channels)) != len(self.required_channels):
            raise ValueError("required_channels must not contain duplicates")
        if type(self.safe_retry) is not bool:
            raise TypeError("safe_retry must be bool")
        if self.committed_relations is not None and not callable(self.committed_relations):
            raise TypeError("committed_relations must be callable or None")


class Planner(Protocol):
    def propose(self, instruction: str, context: Context) -> Action | None:
        """None means no new proposal, never verified task success."""
        ...


class Runtime(Protocol):
    def execute(self, action: Action, attempt_id: str) -> RuntimeResult: ...

    def status(self, action: Action, attempt_id: str) -> RuntimeResult:
        """Query an existing attempt; must never call execute again."""
        ...


class Observer(Protocol):
    def observe(self, action: Action | None) -> Observation: ...


class SemanticAssessor(Protocol):
    def __call__(self, question: str, observation: Observation) -> Signal: ...
