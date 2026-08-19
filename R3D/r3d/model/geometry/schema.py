"""Explicit semantic schemas for tensors transformed by frame adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .errors import (
    MissingDeltaCompositionError,
    MissingQuaternionConventionError,
    SchemaCoverageError,
    SchemaDimensionError,
    SchemaOverlapError,
    UnsupportedRepresentationError,
)
from .types import FIELD_KINDS, FieldKind, QuaternionOrder


_FRAME_KINDS = frozenset(
    {
        "point",
        "vector",
        "direction",
        "orientation",
        "absolute_pose",
        "relative_pose_spatial",
        "twist_spatial",
    }
)
_BODY_KINDS = frozenset({"relative_pose_body", "twist_body"})
_PASSTHROUGH_KINDS = frozenset({"scalar", "joint", "actuator", "passthrough"})

_ORIENTATION_DIMS = {
    "quaternion": 4,
    "rotation_matrix_9d": 9,
    "axis_angle": 3,
    "rotation_6d_columns": 6,
}
_POSE_DIMS = {
    "xyz_quaternion": 7,
    "xyz_axis_angle": 6,
    "xyz_rotation_6d_columns": 9,
    "matrix_16d": 16,
}


def normalize_key_path(path: str | Tuple[str, ...]) -> Tuple[str, ...]:
    if isinstance(path, str):
        if path == "":
            return ()
        result = tuple(part for part in path.split(".") if part)
    else:
        result = tuple(path)
    if any(not isinstance(part, str) or not part for part in result):
        raise ValueError(f"Invalid key path: {path!r}")
    return result


@dataclass(frozen=True)
class FieldSpec:
    name: str
    start: int
    end: int
    kind: FieldKind
    source_frame: Optional[str] = None
    target_frame: Optional[str] = None
    representation: Optional[str] = None
    quaternion_order: Optional[QuaternionOrder] = None
    composition: Optional[str] = None
    body_frame: Optional[str] = None
    units: Optional[str] = None
    preserve_zero_sentinel: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FieldSpec.name must be non-empty")
        if self.kind not in FIELD_KINDS:
            raise UnsupportedRepresentationError(f"Unknown field kind: {self.kind!r}")
        if self.start < 0 or self.end <= self.start:
            raise SchemaDimensionError(
                f"Field {self.name!r} has invalid slice [{self.start}:{self.end}]"
            )

        width = self.end - self.start
        if self.kind in _FRAME_KINDS:
            if not self.source_frame or not self.target_frame:
                raise ValueError(
                    f"Field {self.name!r} ({self.kind}) requires source_frame and target_frame"
                )
            if self.body_frame is not None:
                raise ValueError(f"Field {self.name!r} cannot declare body_frame")
        elif self.kind in _BODY_KINDS:
            if not self.body_frame:
                raise ValueError(f"Field {self.name!r} ({self.kind}) requires body_frame")
            if self.source_frame is not None or self.target_frame is not None:
                raise UnsupportedRepresentationError(
                    f"Body field {self.name!r} is parent-frame invariant and cannot declare "
                    "source_frame/target_frame; arbitrary body-frame conversion is unsupported"
                )
        elif self.kind in _PASSTHROUGH_KINDS:
            if any(
                value is not None
                for value in (self.source_frame, self.target_frame, self.body_frame)
            ):
                raise ValueError(
                    f"Passthrough field {self.name!r} cannot declare geometric frames"
                )

        if self.kind in ("point", "vector", "direction") and width != 3:
            raise SchemaDimensionError(f"Field {self.name!r} ({self.kind}) must be 3D")
        if self.preserve_zero_sentinel and self.kind != "point":
            raise ValueError(
                f"Field {self.name!r} can preserve a zero sentinel only for point geometry"
            )
        if self.kind == "scalar" and width != 1:
            raise SchemaDimensionError(f"Scalar field {self.name!r} must have width 1")

        if self.kind == "orientation":
            expected = _ORIENTATION_DIMS.get(self.representation)
            if expected is None:
                raise UnsupportedRepresentationError(
                    f"Orientation field {self.name!r} has unsupported representation "
                    f"{self.representation!r}"
                )
            if width != expected:
                raise SchemaDimensionError(
                    f"Orientation field {self.name!r} expects width {expected}, got {width}"
                )

        if self.kind in ("absolute_pose", "relative_pose_spatial", "relative_pose_body"):
            expected = _POSE_DIMS.get(self.representation)
            if expected is None:
                raise UnsupportedRepresentationError(
                    f"Pose field {self.name!r} has unsupported representation "
                    f"{self.representation!r}"
                )
            if width != expected:
                raise SchemaDimensionError(
                    f"Pose field {self.name!r} expects width {expected}, got {width}"
                )

        if self.kind in ("twist_spatial", "twist_body"):
            if width != 6:
                raise SchemaDimensionError(f"Twist field {self.name!r} must have width 6")
            if self.representation != "v_omega":
                raise UnsupportedRepresentationError(
                    f"Twist field {self.name!r} must explicitly use representation='v_omega'"
                )

        uses_quaternion = self.representation in ("quaternion", "xyz_quaternion")
        if uses_quaternion and self.quaternion_order not in ("wxyz", "xyzw"):
            raise MissingQuaternionConventionError(
                f"Field {self.name!r} must declare quaternion_order='wxyz' or 'xyzw'"
            )
        if not uses_quaternion and self.quaternion_order is not None:
            raise MissingQuaternionConventionError(
                f"Field {self.name!r} declares quaternion_order without a quaternion representation"
            )

        if self.kind == "relative_pose_spatial" and self.composition != "left":
            raise MissingDeltaCompositionError(
                f"Spatial delta {self.name!r} must declare composition='left'"
            )
        if self.kind == "relative_pose_body" and self.composition != "right":
            raise MissingDeltaCompositionError(
                f"Body delta {self.name!r} must declare composition='right'"
            )
        if self.kind not in ("relative_pose_spatial", "relative_pose_body") and self.composition is not None:
            raise MissingDeltaCompositionError(
                f"Non-relative field {self.name!r} cannot declare composition"
            )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    key_path: Tuple[str, ...] | str
    expected_last_dim: int
    fields: Tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TensorSpec.name must be non-empty")
        object.__setattr__(self, "key_path", normalize_key_path(self.key_path))
        object.__setattr__(self, "fields", tuple(self.fields))
        if self.expected_last_dim <= 0:
            raise SchemaDimensionError("expected_last_dim must be positive")
        if not self.fields:
            raise SchemaCoverageError(f"TensorSpec {self.name!r} has no fields")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError(f"TensorSpec {self.name!r} has duplicate field names")

        ordered = sorted(self.fields, key=lambda field: (field.start, field.end))
        cursor = 0
        for field in ordered:
            if field.end > self.expected_last_dim:
                raise SchemaDimensionError(
                    f"Field {field.name!r} ends at {field.end}, beyond tensor dimension "
                    f"{self.expected_last_dim}"
                )
            if field.start < cursor:
                raise SchemaOverlapError(
                    f"Field {field.name!r} overlaps a previous field in {self.name!r}"
                )
            if field.start > cursor:
                raise SchemaCoverageError(
                    f"TensorSpec {self.name!r} leaves dimensions [{cursor}:{field.start}] uncovered"
                )
            cursor = field.end
        if cursor != self.expected_last_dim:
            raise SchemaCoverageError(
                f"TensorSpec {self.name!r} leaves dimensions [{cursor}:{self.expected_last_dim}] uncovered"
            )


@dataclass(frozen=True)
class SampleSchema:
    name: str
    tensors: Tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SampleSchema.name must be non-empty")
        object.__setattr__(self, "tensors", tuple(self.tensors))
        if not self.tensors:
            raise SchemaCoverageError(f"SampleSchema {self.name!r} has no tensors")
        names = [tensor.name for tensor in self.tensors]
        paths = [tensor.key_path for tensor in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError(f"SampleSchema {self.name!r} has duplicate tensor names")
        if len(set(paths)) != len(paths):
            raise ValueError(f"SampleSchema {self.name!r} has duplicate key paths")
        if () in paths and len(paths) != 1:
            raise ValueError("A root tensor key_path=() cannot coexist with nested tensor paths")


@dataclass(frozen=True)
class CanonicalFrameProfile:
    name: str
    canonical_frame: str
    observation_schema: Optional[SampleSchema] = None
    action_schema: Optional[SampleSchema] = None
    training_schema: Optional[SampleSchema] = None
    config_version: int = 1
    require_metadata: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.canonical_frame:
            raise ValueError("Profile name and canonical_frame must be non-empty")
        if self.config_version != 1:
            raise ValueError(f"Unsupported frame profile config_version={self.config_version}")
        if not any((self.observation_schema, self.action_schema, self.training_schema)):
            raise ValueError("A frame profile must define at least one schema")
