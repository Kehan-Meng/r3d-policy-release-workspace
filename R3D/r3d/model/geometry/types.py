"""Shared types for frame-aware geometric transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch

from .errors import InvalidTransformError


ArrayLike = Union[np.ndarray, torch.Tensor]

FieldKind = Literal[
    "point",
    "vector",
    "direction",
    "orientation",
    "absolute_pose",
    "relative_pose_spatial",
    "relative_pose_body",
    "twist_spatial",
    "twist_body",
    "scalar",
    "joint",
    "actuator",
    "passthrough",
]

FIELD_KINDS = frozenset(
    {
        "point",
        "vector",
        "direction",
        "orientation",
        "absolute_pose",
        "relative_pose_spatial",
        "relative_pose_body",
        "twist_spatial",
        "twist_body",
        "scalar",
        "joint",
        "actuator",
        "passthrough",
    }
)

QuaternionOrder = Literal["wxyz", "xyzw"]


@dataclass(frozen=True)
class Transform:
    """A rigid transform with unambiguous ``T_target_from_source`` semantics."""

    source_frame: str
    target_frame: str
    matrix: ArrayLike
    convention: str = "T_target_from_source"
    length_unit: str = "meter"

    def __post_init__(self) -> None:
        if not self.source_frame or not self.target_frame:
            raise InvalidTransformError("source_frame and target_frame must be non-empty")
        if self.convention != "T_target_from_source":
            raise InvalidTransformError(
                f"Unsupported transform convention: {self.convention!r}"
            )
        if not self.length_unit:
            raise InvalidTransformError("length_unit must be non-empty")

        from .validation import validate_transform_matrix

        validate_transform_matrix(self.matrix)
        if self.source_frame == self.target_frame:
            from .validation import is_identity_transform

            if not is_identity_transform(self.matrix):
                raise InvalidTransformError(
                    "A transform whose source and target frames are equal must be identity"
                )

    @property
    def rotation(self) -> ArrayLike:
        return self.matrix[..., :3, :3]

    @property
    def translation(self) -> ArrayLike:
        return self.matrix[..., :3, 3]
