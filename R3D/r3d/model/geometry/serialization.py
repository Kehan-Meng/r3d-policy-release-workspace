"""Canonical serialization and hashes for frame profiles."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import PurePath
from typing import Any

import numpy as np
import torch


def to_primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_primitive(value.tolist())
    if isinstance(value, np.generic):
        return to_primitive(value.item())
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("Cannot serialize a tensor that requires gradients")
        return to_primitive(value.detach().cpu().tolist())
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot canonically serialize {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
