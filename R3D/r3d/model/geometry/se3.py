"""Backend-neutral SE(3) operations using T_target_from_source semantics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .errors import SchemaDimensionError
from .types import ArrayLike, Transform
from .validation import (
    is_torch_array,
    require_floating_geometry,
    validate_same_backend,
    validate_transform_matrix,
)


def _transpose(value: ArrayLike) -> ArrayLike:
    return value.transpose(-1, -2) if is_torch_array(value) else np.swapaxes(value, -1, -2)


def _clone(value: ArrayLike) -> ArrayLike:
    return value.clone() if is_torch_array(value) else np.array(value, copy=True)


def _zeros(shape: Tuple[int, ...], like: ArrayLike) -> ArrayLike:
    if is_torch_array(like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def _ones(shape: Tuple[int, ...], like: ArrayLike) -> ArrayLike:
    if is_torch_array(like):
        return torch.ones(shape, dtype=like.dtype, device=like.device)
    return np.ones(shape, dtype=like.dtype)


def _cat(values, axis: int) -> ArrayLike:
    return torch.cat(values, dim=axis) if is_torch_array(values[0]) else np.concatenate(values, axis=axis)


def _stack(values, axis: int) -> ArrayLike:
    return torch.stack(values, dim=axis) if is_torch_array(values[0]) else np.stack(values, axis=axis)


def clone_array(value: ArrayLike) -> ArrayLike:
    return _clone(value)


def make_transform(rotation: ArrayLike, translation: ArrayLike) -> ArrayLike:
    require_floating_geometry(rotation, name="rotation")
    require_floating_geometry(translation, name="translation")
    validate_same_backend(rotation, translation)
    if tuple(rotation.shape[-2:]) != (3, 3):
        raise SchemaDimensionError(f"Expected rotation [...,3,3], got {rotation.shape}")
    if translation.shape[-1] != 3:
        raise SchemaDimensionError(f"Expected translation [...,3], got {translation.shape}")

    try:
        batch_shape = np.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    except ValueError as exc:
        raise SchemaDimensionError(
            f"Rotation batch shape {rotation.shape[:-2]} and translation batch shape "
            f"{translation.shape[:-1]} are not broadcastable"
        ) from exc
    if is_torch_array(rotation):
        rotation = rotation.expand(batch_shape + (3, 3))
        translation = translation.expand(batch_shape + (3,))
    else:
        rotation = np.broadcast_to(rotation, batch_shape + (3, 3))
        translation = np.broadcast_to(translation, batch_shape + (3,))

    upper = _cat((rotation, translation[..., None]), axis=-1)
    bottom_left = _zeros(tuple(upper.shape[:-2]) + (1, 3), upper)
    bottom_right = _ones(tuple(upper.shape[:-2]) + (1, 1), upper)
    matrix = _cat((upper, _cat((bottom_left, bottom_right), axis=-1)), axis=-2)
    validate_transform_matrix(matrix)
    return matrix


def identity_matrix(*, like: ArrayLike, batch_shape: Tuple[int, ...] = ()) -> ArrayLike:
    require_floating_geometry(like)
    if is_torch_array(like):
        eye = torch.eye(4, dtype=like.dtype, device=like.device)
        return eye.expand(batch_shape + (4, 4)).clone()
    return np.broadcast_to(np.eye(4, dtype=like.dtype), batch_shape + (4, 4)).copy()


def compose_matrices(
    matrix_target_from_middle: ArrayLike,
    matrix_middle_from_source: ArrayLike,
) -> ArrayLike:
    validate_same_backend(matrix_target_from_middle, matrix_middle_from_source)
    validate_transform_matrix(matrix_target_from_middle)
    validate_transform_matrix(matrix_middle_from_source)
    result = matrix_target_from_middle @ matrix_middle_from_source
    validate_transform_matrix(result)
    return result


def compose_transforms(
    transform_target_from_middle: Transform,
    transform_middle_from_source: Transform,
) -> Transform:
    if transform_target_from_middle.source_frame != transform_middle_from_source.target_frame:
        raise ValueError(
            "Cannot compose transforms: middle frames do not match "
            f"({transform_target_from_middle.source_frame!r} != "
            f"{transform_middle_from_source.target_frame!r})"
        )
    if transform_target_from_middle.length_unit != transform_middle_from_source.length_unit:
        raise ValueError("Cannot compose transforms with different length units")
    return Transform(
        source_frame=transform_middle_from_source.source_frame,
        target_frame=transform_target_from_middle.target_frame,
        matrix=compose_matrices(
            transform_target_from_middle.matrix,
            transform_middle_from_source.matrix,
        ),
        length_unit=transform_target_from_middle.length_unit,
    )


def invert_matrix(matrix_target_from_source: ArrayLike) -> ArrayLike:
    validate_transform_matrix(matrix_target_from_source)
    rotation = matrix_target_from_source[..., :3, :3]
    translation = matrix_target_from_source[..., :3, 3]
    inverse_rotation = _transpose(rotation)
    inverse_translation = -(
        translation[..., None, :] @ _transpose(inverse_rotation)
    ).squeeze(-2)
    return make_transform(inverse_rotation, inverse_translation)


def invert_transform(transform_target_from_source: Transform) -> Transform:
    return Transform(
        source_frame=transform_target_from_source.target_frame,
        target_frame=transform_target_from_source.source_frame,
        matrix=invert_matrix(transform_target_from_source.matrix),
        length_unit=transform_target_from_source.length_unit,
    )


def _expand_transform_for_value(
    matrix: ArrayLike,
    value: ArrayLike,
    *,
    value_event_ndim: int,
) -> ArrayLike:
    """Align transform batch dimensions with the leftmost value dimensions."""

    matrix_batch = tuple(matrix.shape[:-2])
    value_leading = tuple(value.shape[:-value_event_ndim])
    if len(matrix_batch) > len(value_leading):
        raise SchemaDimensionError(
            f"Transform batch shape {matrix_batch} has more dimensions than value "
            f"leading shape {value_leading}"
        )
    for transform_dim, value_dim in zip(matrix_batch, value_leading):
        if transform_dim not in (1, value_dim):
            raise SchemaDimensionError(
                f"Transform batch shape {matrix_batch} is not a broadcastable prefix "
                f"of value leading shape {value_leading}"
            )
    extra = len(value_leading) - len(matrix_batch)
    return matrix.reshape(matrix_batch + (1,) * extra + (4, 4))


def _rotate(value: ArrayLike, rotation: ArrayLike) -> ArrayLike:
    return (value[..., None, :] @ _transpose(rotation)).squeeze(-2)


def transform_points(points_source: ArrayLike, matrix_target_from_source: ArrayLike) -> ArrayLike:
    require_floating_geometry(points_source, name="points")
    validate_same_backend(points_source, matrix_target_from_source)
    validate_transform_matrix(matrix_target_from_source)
    if points_source.shape[-1] != 3:
        raise SchemaDimensionError(f"Points must have shape [...,3], got {points_source.shape}")
    matrix = _expand_transform_for_value(matrix_target_from_source, points_source, value_event_ndim=1)
    return _rotate(points_source, matrix[..., :3, :3]) + matrix[..., :3, 3]


def transform_vectors(vectors_source: ArrayLike, matrix_target_from_source: ArrayLike) -> ArrayLike:
    require_floating_geometry(vectors_source, name="vectors")
    validate_same_backend(vectors_source, matrix_target_from_source)
    validate_transform_matrix(matrix_target_from_source)
    if vectors_source.shape[-1] != 3:
        raise SchemaDimensionError(f"Vectors must have shape [...,3], got {vectors_source.shape}")
    matrix = _expand_transform_for_value(matrix_target_from_source, vectors_source, value_event_ndim=1)
    return _rotate(vectors_source, matrix[..., :3, :3])


def transform_rotation_matrices(
    rotations_source_from_entity: ArrayLike,
    matrix_target_from_source: ArrayLike,
) -> ArrayLike:
    require_floating_geometry(rotations_source_from_entity, name="orientations")
    validate_same_backend(rotations_source_from_entity, matrix_target_from_source)
    validate_transform_matrix(matrix_target_from_source)
    if tuple(rotations_source_from_entity.shape[-2:]) != (3, 3):
        raise SchemaDimensionError(
            f"Orientations must have shape [...,3,3], got {rotations_source_from_entity.shape}"
        )
    matrix = _expand_transform_for_value(
        matrix_target_from_source,
        rotations_source_from_entity,
        value_event_ndim=2,
    )
    return matrix[..., :3, :3] @ rotations_source_from_entity


def transform_absolute_pose_matrices(
    pose_source_from_entity: ArrayLike,
    matrix_target_from_source: ArrayLike,
) -> ArrayLike:
    validate_same_backend(pose_source_from_entity, matrix_target_from_source)
    validate_transform_matrix(pose_source_from_entity)
    validate_transform_matrix(matrix_target_from_source)
    matrix = _expand_transform_for_value(
        matrix_target_from_source,
        pose_source_from_entity,
        value_event_ndim=2,
    )
    return matrix @ pose_source_from_entity


def transform_spatial_relative_pose_matrices(
    delta_pose_source: ArrayLike,
    matrix_target_from_source: ArrayLike,
) -> ArrayLike:
    validate_same_backend(delta_pose_source, matrix_target_from_source)
    validate_transform_matrix(delta_pose_source)
    validate_transform_matrix(matrix_target_from_source)
    matrix = _expand_transform_for_value(
        matrix_target_from_source,
        delta_pose_source,
        value_event_ndim=2,
    )
    return matrix @ delta_pose_source @ invert_matrix(matrix)


def transform_body_relative_pose_matrices(delta_pose_body: ArrayLike) -> ArrayLike:
    """Body/right-multiplicative deltas are invariant to parent-frame changes."""

    validate_transform_matrix(delta_pose_body)
    return _clone(delta_pose_body)


def skew(vector: ArrayLike) -> ArrayLike:
    require_floating_geometry(vector, name="vector")
    if vector.shape[-1] != 3:
        raise SchemaDimensionError(f"Expected [...,3], got {vector.shape}")
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = x * 0
    return _stack(
        (
            _stack((zero, -z, y), axis=-1),
            _stack((z, zero, -x), axis=-1),
            _stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )


def adjoint_matrix(matrix_target_from_source: ArrayLike) -> ArrayLike:
    """Return the SE(3) adjoint for twist order ``[v, omega]``."""

    validate_transform_matrix(matrix_target_from_source)
    rotation = matrix_target_from_source[..., :3, :3]
    translation = matrix_target_from_source[..., :3, 3]
    zero = _zeros(tuple(rotation.shape), rotation)
    upper = _cat((rotation, skew(translation) @ rotation), axis=-1)
    lower = _cat((zero, rotation), axis=-1)
    return _cat((upper, lower), axis=-2)


def transform_spatial_twists(
    twists_source: ArrayLike,
    matrix_target_from_source: ArrayLike,
) -> ArrayLike:
    require_floating_geometry(twists_source, name="twists")
    validate_same_backend(twists_source, matrix_target_from_source)
    validate_transform_matrix(matrix_target_from_source)
    if twists_source.shape[-1] != 6:
        raise SchemaDimensionError(f"Twists must have shape [...,6], got {twists_source.shape}")
    matrix = _expand_transform_for_value(matrix_target_from_source, twists_source, value_event_ndim=1)
    linear_source = twists_source[..., :3]
    angular_source = twists_source[..., 3:]
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    angular_target = _rotate(angular_source, rotation)
    linear_target = _rotate(linear_source, rotation)
    if is_torch_array(twists_source):
        cross = torch.linalg.cross(translation, angular_target, dim=-1)
    else:
        cross = np.cross(translation, angular_target, axis=-1)
    return _cat((linear_target + cross, angular_target), axis=-1)


def transform_body_twists(twists_body: ArrayLike) -> ArrayLike:
    """Right-trivialized body twists are invariant to parent-frame changes."""

    require_floating_geometry(twists_body, name="twists")
    if twists_body.shape[-1] != 6:
        raise SchemaDimensionError(f"Twists must have shape [...,6], got {twists_body.shape}")
    return _clone(twists_body)
