"""Frozen JSON payloads and deterministic serialization; standard library only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import TypeAlias

JSONValue: TypeAlias = "None | bool | int | float | str | tuple[JSONValue, ...] | Mapping[str, JSONValue]"
JSONMap: TypeAlias = Mapping[str, JSONValue]


def _freeze(value):
    if value is None or type(value) in (bool, int, str):
        if isinstance(value, str):
            value.encode("utf-8")
        if type(value) is int:
            str(value)  # Respect Python's configured integer serialization limit.
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({_freeze(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"Not a JSON value: {type(value).__name__}")


def _snapshot(value):
    try:
        return _freeze(value)
    except RecursionError as exc:
        raise ValueError("JSON must be acyclic and within Python's nesting limit") from exc


def json_data(value):
    """Convert a contract or frozen JSON value into ordinary JSON containers."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: json_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_data(item) for item in value]
    return _snapshot(value)


def canonical_json(value) -> str:
    """Sorted-key, compact JSON; byte budgets count its UTF-8 encoding."""
    return json.dumps(json_data(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
