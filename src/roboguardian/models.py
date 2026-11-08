"""Immutable action, observation, evidence, receipt, and control records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
import math
from types import MappingProxyType

from .serialization import JSONMap, JSONValue, _freeze, _snapshot, canonical_json


class Judgment(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class Signal(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


class ReceiptState(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


def _mapping(record, name):
    value = getattr(record, name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    object.__setattr__(record, name, _snapshot(value))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    value.encode("utf-8")


def _refs(record, name):
    value = getattr(record, name)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of references")
    for item in value:
        _text(item, name)
    object.__setattr__(record, name, tuple(value))


def _integer(value, name, minimum=0):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    _freeze(value)


@dataclass(frozen=True)
class Action:
    invocation_id: str
    goal_id: str
    embodiment: str
    skill: str
    parameters: JSONMap
    scene_id: str
    entity_ids: tuple[str, ...]
    based_on_observation: str
    expected_state_version: int
    required_facts: JSONMap
    skill_config_version: str

    def __post_init__(self):
        for name in ("invocation_id", "goal_id", "embodiment", "skill", "scene_id",
                     "based_on_observation", "skill_config_version"):
            _text(getattr(self, name), name)
        _integer(self.expected_state_version, "expected_state_version")
        _refs(self, "entity_ids")
        _mapping(self, "parameters")
        _mapping(self, "required_facts")


def action_fingerprint(action: Action) -> str:
    """Bind all identity, parameters, planning premises and configuration fields."""
    return hashlib.sha256(canonical_json(action).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source_id: str
    sequence: int
    scene_id: str
    entity_ids: tuple[str, ...]
    state: JSONMap
    images: tuple[str, ...] = ()
    facts: JSONMap = field(default_factory=dict)

    def __post_init__(self):
        for name in ("observation_id", "source_id", "scene_id"):
            _text(getattr(self, name), name)
        _integer(self.sequence, "sequence")
        _refs(self, "entity_ids")
        _refs(self, "images")
        _mapping(self, "state")
        _mapping(self, "facts")


@dataclass(frozen=True)
class RuntimeResult:
    """Software status is diagnostic only; terminal=None means unknown."""
    status: str
    terminal: bool | None
    invocation_id: str
    attempt_id: str
    diagnostics: JSONMap = field(default_factory=dict)

    def __post_init__(self):
        for name in ("status", "invocation_id", "attempt_id"):
            _text(getattr(self, name), name)
        if self.terminal is not None and type(self.terminal) is not bool:
            raise TypeError("terminal must be bool or None (unknown)")
        _mapping(self, "diagnostics")


@dataclass(frozen=True)
class EvidenceReport:
    invocation_id: str
    attempt_id: str
    before_observation_id: str | None
    after_observation_id: str | None
    channels: Mapping[str, Signal]
    partial_effect: bool = False
    evidence_refs: tuple[str, ...] = ()
    diagnostics: JSONMap = field(default_factory=dict)

    def __post_init__(self):
        for name in ("invocation_id", "attempt_id"):
            _text(getattr(self, name), name)
        for name in ("before_observation_id", "after_observation_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if not isinstance(self.channels, Mapping):
            raise TypeError("channels must be a mapping")
        if any(key not in ("semantic", "pre", "post", "effect") or
               not isinstance(value, Signal) for key, value in self.channels.items()):
            raise ValueError("channels must map semantic/pre/post/effect to Signal")
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))
        if type(self.partial_effect) is not bool:
            raise TypeError("partial_effect must be bool")
        _refs(self, "evidence_refs")
        _mapping(self, "diagnostics")


@dataclass(frozen=True)
class Assessment:
    judgment: Judgment
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.judgment, Judgment):
            raise TypeError("judgment must be Judgment")
        _text(self.reason, "reason")
        _refs(self, "evidence_refs")


@dataclass(frozen=True)
class EffectReceipt:
    receipt_id: str
    action: Action
    attempt_id: str
    state: ReceiptState
    assessment: Assessment
    runtime_result: RuntimeResult
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.receipt_id, "receipt_id")
        _text(self.attempt_id, "attempt_id")
        if not isinstance(self.action, Action) or not isinstance(self.assessment, Assessment):
            raise TypeError("receipt requires Action and Assessment")
        if not isinstance(self.runtime_result, RuntimeResult) or not isinstance(self.state, ReceiptState):
            raise TypeError("receipt requires RuntimeResult and ReceiptState")
        if (self.runtime_result.invocation_id != self.action.invocation_id or
                self.runtime_result.attempt_id != self.attempt_id):
            raise ValueError("runtime result does not belong to this action and attempt")
        expected = {Judgment.YES: ReceiptState.COMMITTED, Judgment.NO: ReceiptState.REJECTED,
                    Judgment.UNCERTAIN: ReceiptState.UNKNOWN}[self.assessment.judgment]
        if self.state is not expected:
            raise ValueError("receipt state disagrees with assessment")
        _refs(self, "evidence_refs")


@dataclass(frozen=True)
class EffectMeasurement:
    """Errors and tolerances use the adapter's declared units and frame.

    Missing error values are None. Tolerances are finite and nonnegative; zero
    deliberately permits no PASS under the strict less-than rule.
    """
    translation_error: float | None
    rotation_error: float | None
    translation_tolerance: float
    rotation_tolerance: float
    entity_id: str
    frame_id: str
    translation_unit: str
    rotation_unit: str
    partial: bool = False

    def __post_init__(self):
        for name in ("translation_error", "rotation_error", "translation_tolerance", "rotation_tolerance"):
            value = getattr(self, name)
            if value is None and name.endswith("error"):
                continue
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("entity_id", "frame_id", "translation_unit", "rotation_unit"):
            _text(getattr(self, name), name)
        if type(self.partial) is not bool:
            raise TypeError("partial must be bool")


def effect_signal(measurement: EffectMeasurement) -> Signal:
    """Partial or missing measurements abstain; both errors must be < tolerance."""
    if (measurement.partial or measurement.translation_error is None or
            measurement.rotation_error is None):
        return Signal.INDETERMINATE
    if (measurement.translation_error < measurement.translation_tolerance and
            measurement.rotation_error < measurement.rotation_tolerance):
        return Signal.PASS
    return Signal.FAIL


@dataclass(frozen=True)
class Budget:
    """Re-observation/recovery/image limits allow zero; steps and bytes are positive."""
    max_observations: int
    max_recovery_dispatches: int
    max_plan_steps: int
    context_bytes: int
    max_images: int

    def __post_init__(self):
        for item in fields(self):
            minimum = 1 if item.name in ("max_plan_steps", "context_bytes") else 0
            _integer(getattr(self, item.name), item.name, minimum)


_ADMISSION_CODES = frozenset({
    "", "admission_rejected", "invalid_action", "invocation_conflict", "skill_unavailable",
    "identity_mismatch", "observation_outdated", "state_outdated", "required_facts_unconfirmed",
    "invalid_parameters", "precondition_unconfirmed", "admission_provider_error",
    "pending_attempt", "recovery_required", "invalid_recovery", "recovery_budget",
    "retry_forbidden", "observation_error", "dispatch_rejected", "unknown_invocation",
})


@dataclass(frozen=True)
class AdmissionResult:
    allowed: bool
    reason: str
    code: str = ""

    def __post_init__(self):
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be bool")
        _text(self.reason, "reason")
        if not isinstance(self.code, str) or self.code not in _ADMISSION_CODES:
            raise ValueError("code must be a recognized admission diagnostic code")


@dataclass(frozen=True)
class StepResult:
    admission: AdmissionResult
    receipt: EffectReceipt | None = None
    recovery_status: str = "none"
    recovery_reason: str = ""


@dataclass(frozen=True)
class Context:
    """Structured planner projection; the memory layer selects permitted data."""
    data: JSONMap
    images: tuple[str, ...] = ()

    def __post_init__(self):
        _mapping(self, "data")
        _refs(self, "images")

    def serialize(self) -> str:
        return canonical_json(self)

    @property
    def byte_size(self) -> int:
        return len(self.serialize().encode("utf-8"))

    @property
    def image_count(self) -> int:
        return len(self.images)
