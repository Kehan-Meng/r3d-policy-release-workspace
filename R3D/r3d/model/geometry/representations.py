"""Rotation and pose representation conversion plus frame transformations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .errors import InvalidRotationError, SchemaDimensionError, UnsupportedRepresentationError
from .quaternion import (
    canonicalize_quaternion_wxyz,
    matrix_to_quaternion,
    quaternion_to_matrix,
    reorder_quaternion,
)
from .se3 import (
    make_transform,
    transform_absolute_pose_matrices,
    transform_body_relative_pose_matrices,
    transform_body_twists,
    transform_rotation_matrices,
    transform_spatial_relative_pose_matrices,
    transform_spatial_twists,
)
from .types import ArrayLike, QuaternionOrder
from .validation import is_torch_array, require_floating_geometry, validate_rotation_matrix


OrientationRepresentation = Literal[
    "quaternion",
    "rotation_matrix_9d",
    "axis_angle",
    "rotation_6d_columns",
]

PoseRepresentation = Literal[
    "xyz_quaternion",
    "xyz_axis_angle",
    "xyz_rotation_6d_columns",
    "matrix_16d",
]


def _stack(values, axis: int) -> ArrayLike:
    return torch.stack(values, dim=axis) if is_torch_array(values[0]) else np.stack(values, axis=axis)


def _cat(values, axis: int) -> ArrayLike:
    return torch.cat(values, dim=axis) if is_torch_array(values[0]) else np.concatenate(values, axis=axis)


def _norm(value: ArrayLike, *, keepdims: bool = False) -> ArrayLike:
    if is_torch_array(value):
        return torch.linalg.vector_norm(value, dim=-1, keepdim=keepdims)
    return np.linalg.norm(value, axis=-1, keepdims=keepdims)


def _where(condition, left, right):
    return torch.where(condition, left, right) if is_torch_array(left) else np.where(condition, left, right)


def axis_angle_to_quaternion_wxyz(axis_angle: ArrayLike) -> ArrayLike:
    require_floating_geometry(axis_angle, name="axis_angle")
    if axis_angle.shape[-1] != 3:
        raise SchemaDimensionError(f"axis_angle must have shape [...,3], got {axis_angle.shape}")
    angle = _norm(axis_angle, keepdims=True)
    angle_sq = angle * angle
    if is_torch_array(axis_angle):
        safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
        sin_half_over_angle = torch.where(
            angle > 1e-6,
            torch.sin(angle * 0.5) / safe_angle,
            0.5 - angle_sq / 48 + angle_sq * angle_sq / 3840,
        )
        cosine = torch.cos(angle * 0.5)
    else:
        safe_angle = np.where(angle > 1e-6, angle, 1.0)
        sin_half_over_angle = np.where(
            angle > 1e-6,
            np.sin(angle * 0.5) / safe_angle,
            0.5 - angle_sq / 48 + angle_sq * angle_sq / 3840,
        )
        cosine = np.cos(angle * 0.5)
    return canonicalize_quaternion_wxyz(
        _cat((cosine, axis_angle * sin_half_over_angle), axis=-1)
    )


def quaternion_wxyz_to_axis_angle(quaternion_wxyz: ArrayLike) -> ArrayLike:
    q = canonicalize_quaternion_wxyz(quaternion_wxyz)
    vector = q[..., 1:]
    sin_half = _norm(vector, keepdims=True)
    w = q[..., :1]
    if is_torch_array(q):
        angle = 2 * torch.atan2(sin_half, torch.clamp(w, min=0))
        safe_sin = torch.where(sin_half > 1e-6, sin_half, torch.ones_like(sin_half))
        scale = torch.where(sin_half > 1e-6, angle / safe_sin, 2 + angle * angle / 12)
    else:
        angle = 2 * np.arctan2(sin_half, np.clip(w, a_min=0, a_max=None))
        safe_sin = np.where(sin_half > 1e-6, sin_half, 1.0)
        scale = np.where(sin_half > 1e-6, angle / safe_sin, 2 + angle * angle / 12)
    return vector * scale


def axis_angle_to_matrix(axis_angle: ArrayLike) -> ArrayLike:
    return quaternion_to_matrix(axis_angle_to_quaternion_wxyz(axis_angle), order="wxyz")


def matrix_to_axis_angle(matrix: ArrayLike) -> ArrayLike:
    return quaternion_wxyz_to_axis_angle(matrix_to_quaternion(matrix, order="wxyz"))


def rotation_6d_to_matrix(rotation_6d: ArrayLike, *, eps: float = 1e-8) -> ArrayLike:
    require_floating_geometry(rotation_6d, name="rotation_6d")
    if rotation_6d.shape[-1] != 6:
        raise SchemaDimensionError(f"rotation_6d must have shape [...,6], got {rotation_6d.shape}")
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    finite = torch.isfinite(rotation_6d).all() if is_torch_array(rotation_6d) else np.isfinite(rotation_6d).all()
    if not bool(finite.item() if is_torch_array(rotation_6d) else finite):
        raise InvalidRotationError("rotation_6d contains NaN or infinity")
    first_norm = _norm(first, keepdims=True)
    if is_torch_array(first):
        if bool((first_norm <= eps).any().item()):
            raise InvalidRotationError("rotation_6d first column is degenerate")
    elif bool((first_norm <= eps).any()):
        raise InvalidRotationError("rotation_6d first column is degenerate")
    b1 = first / first_norm
    projection = (b1 * second).sum(dim=-1, keepdim=True) if is_torch_array(first) else np.sum(b1 * second, axis=-1, keepdims=True)
    orthogonal = second - projection * b1
    second_norm = _norm(orthogonal, keepdims=True)
    if is_torch_array(first):
        if bool((second_norm <= eps).any().item()):
            raise InvalidRotationError("rotation_6d columns are collinear")
        b3 = torch.linalg.cross(b1, orthogonal / second_norm, dim=-1)
    else:
        if bool((second_norm <= eps).any()):
            raise InvalidRotationError("rotation_6d columns are collinear")
        b3 = np.cross(b1, orthogonal / second_norm, axis=-1)
    b2 = orthogonal / second_norm
    return _stack((b1, b2, b3), axis=-1)


def matrix_to_rotation_6d(matrix: ArrayLike) -> ArrayLike:
    validate_rotation_matrix(matrix)
    return _cat((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def orientation_to_matrix(
    value: ArrayLike,
    *,
    representation: OrientationRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    if representation == "quaternion":
        if quaternion_order is None:
            raise UnsupportedRepresentationError("quaternion_order is required")
        return quaternion_to_matrix(value, order=quaternion_order)
    if representation == "rotation_matrix_9d":
        if value.shape[-1] != 9:
            raise SchemaDimensionError(f"Expected [...,9], got {value.shape}")
        matrix = value.reshape(value.shape[:-1] + (3, 3))
        validate_rotation_matrix(matrix)
        return matrix
    if representation == "axis_angle":
        return axis_angle_to_matrix(value)
    if representation == "rotation_6d_columns":
        return rotation_6d_to_matrix(value)
    raise UnsupportedRepresentationError(f"Unknown orientation representation: {representation!r}")


def matrix_to_orientation(
    matrix: ArrayLike,
    *,
    representation: OrientationRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    validate_rotation_matrix(matrix)
    if representation == "quaternion":
        if quaternion_order is None:
            raise UnsupportedRepresentationError("quaternion_order is required")
        return matrix_to_quaternion(matrix, order=quaternion_order)
    if representation == "rotation_matrix_9d":
        return matrix.reshape(matrix.shape[:-2] + (9,))
    if representation == "axis_angle":
        return matrix_to_axis_angle(matrix)
    if representation == "rotation_6d_columns":
        return matrix_to_rotation_6d(matrix)
    raise UnsupportedRepresentationError(f"Unknown orientation representation: {representation!r}")


def pose_to_matrix(
    value: ArrayLike,
    *,
    representation: PoseRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    require_floating_geometry(value, name="pose")
    if representation == "matrix_16d":
        if value.shape[-1] != 16:
            raise SchemaDimensionError(f"Expected [...,16], got {value.shape}")
        matrix = value.reshape(value.shape[:-1] + (4, 4))
        from .validation import validate_transform_matrix

        validate_transform_matrix(matrix)
        return matrix

    expected_dim = {
        "xyz_quaternion": 7,
        "xyz_axis_angle": 6,
        "xyz_rotation_6d_columns": 9,
    }.get(representation)
    if expected_dim is None:
        raise UnsupportedRepresentationError(f"Unknown pose representation: {representation!r}")
    if value.shape[-1] != expected_dim:
        raise SchemaDimensionError(f"Expected [...,{expected_dim}], got {value.shape}")

    position = value[..., :3]
    orientation_value = value[..., 3:]
    orientation_rep = {
        "xyz_quaternion": "quaternion",
        "xyz_axis_angle": "axis_angle",
        "xyz_rotation_6d_columns": "rotation_6d_columns",
    }[representation]
    rotation = orientation_to_matrix(
        orientation_value,
        representation=orientation_rep,
        quaternion_order=quaternion_order,
    )
    return make_transform(rotation, position)


def matrix_to_pose(
    matrix: ArrayLike,
    *,
    representation: PoseRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    from .validation import validate_transform_matrix

    validate_transform_matrix(matrix)
    if representation == "matrix_16d":
        return matrix.reshape(matrix.shape[:-2] + (16,))
    orientation_rep = {
        "xyz_quaternion": "quaternion",
        "xyz_axis_angle": "axis_angle",
        "xyz_rotation_6d_columns": "rotation_6d_columns",
    }.get(representation)
    if orientation_rep is None:
        raise UnsupportedRepresentationError(f"Unknown pose representation: {representation!r}")
    orientation = matrix_to_orientation(
        matrix[..., :3, :3],
        representation=orientation_rep,
        quaternion_order=quaternion_order,
    )
    return _cat((matrix[..., :3, 3], orientation), axis=-1)


def transform_orientation(
    value: ArrayLike,
    matrix_target_from_source: ArrayLike,
    *,
    representation: OrientationRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    rotation = orientation_to_matrix(
        value,
        representation=representation,
        quaternion_order=quaternion_order,
    )
    transformed = transform_rotation_matrices(rotation, matrix_target_from_source)
    return matrix_to_orientation(
        transformed,
        representation=representation,
        quaternion_order=quaternion_order,
    )


def transform_absolute_pose(
    value: ArrayLike,
    matrix_target_from_source: ArrayLike,
    *,
    representation: PoseRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    pose = pose_to_matrix(value, representation=representation, quaternion_order=quaternion_order)
    transformed = transform_absolute_pose_matrices(pose, matrix_target_from_source)
    return matrix_to_pose(transformed, representation=representation, quaternion_order=quaternion_order)


def transform_relative_pose_spatial(
    value: ArrayLike,
    matrix_target_from_source: ArrayLike,
    *,
    representation: PoseRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    delta = pose_to_matrix(value, representation=representation, quaternion_order=quaternion_order)
    transformed = transform_spatial_relative_pose_matrices(delta, matrix_target_from_source)
    return matrix_to_pose(transformed, representation=representation, quaternion_order=quaternion_order)


def transform_relative_pose_body(
    value: ArrayLike,
    *,
    representation: PoseRepresentation,
    quaternion_order: QuaternionOrder | None = None,
) -> ArrayLike:
    """Copy a body delta while changing only its external parent frame.

    This is not an arbitrary conversion between two moving body frames. A
    right-multiplicative delta expressed in the same moving body frame is
    invariant only when the external parent/reference frame changes.
    """
    pose_to_matrix(value, representation=representation, quaternion_order=quaternion_order)
    return value.clone() if is_torch_array(value) else np.array(value, copy=True)


def transform_twist(
    value: ArrayLike,
    matrix_target_from_source: ArrayLike | None,
    *,
    kind: Literal["spatial", "body"],
) -> ArrayLike:
    if kind == "spatial":
        if matrix_target_from_source is None:
            raise ValueError("Spatial twist requires matrix_target_from_source")
        return transform_spatial_twists(value, matrix_target_from_source)
    if kind == "body":
        # The body frame itself is unchanged; only its external reference
        # frame may differ. Arbitrary body_A -> body_B conversion is unsupported.
        return transform_body_twists(value)
    raise UnsupportedRepresentationError(f"Unknown twist kind: {kind!r}")
