"""Numerical validation helpers shared by all geometry backends."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .errors import (
    InvalidRotationError,
    InvalidTransformError,
    UnsupportedArrayTypeError,
    UnsupportedDTypeError,
)
from .types import ArrayLike


def is_torch_array(value: object) -> bool:
    return isinstance(value, torch.Tensor)


def is_numpy_array(value: object) -> bool:
    return isinstance(value, np.ndarray)


def require_array(value: object, *, name: str = "value") -> ArrayLike:
    if not (is_torch_array(value) or is_numpy_array(value)):
        raise UnsupportedArrayTypeError(
            f"{name} must be a NumPy array or Torch tensor, got {type(value)!r}"
        )
    return value


def require_floating_geometry(value: ArrayLike, *, name: str = "value") -> None:
    require_array(value, name=name)
    if is_torch_array(value):
        if value.dtype not in (torch.float32, torch.float64):
            raise UnsupportedDTypeError(
                f"{name} geometry must use torch.float32/float64, got {value.dtype}"
            )
    elif value.dtype not in (np.float32, np.float64):
        raise UnsupportedDTypeError(
            f"{name} geometry must use np.float32/float64, got {value.dtype}"
        )


def tolerances(value: ArrayLike) -> Tuple[float, float]:
    require_array(value)
    is_float64 = value.dtype in (np.float64, torch.float64)
    return (1e-9, 1e-9) if is_float64 else (2e-5, 2e-5)


def _all_finite(value: ArrayLike) -> bool:
    if is_torch_array(value):
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _eye3_like(value: ArrayLike) -> ArrayLike:
    if is_torch_array(value):
        return torch.eye(3, dtype=value.dtype, device=value.device)
    return np.eye(3, dtype=value.dtype)


def _allclose(a: ArrayLike, b: ArrayLike, *, atol: float, rtol: float) -> bool:
    if is_torch_array(a):
        return bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def validate_rotation_matrix(rotation: ArrayLike) -> None:
    require_floating_geometry(rotation, name="rotation")
    if rotation.ndim < 2 or tuple(rotation.shape[-2:]) != (3, 3):
        raise InvalidRotationError(
            f"Rotation must have shape [..., 3, 3], got {tuple(rotation.shape)}"
        )
    if not _all_finite(rotation):
        raise InvalidRotationError("Rotation contains NaN or infinity")

    atol, rtol = tolerances(rotation)
    transpose = rotation.transpose(-1, -2) if is_torch_array(rotation) else np.swapaxes(rotation, -1, -2)
    gram = transpose @ rotation
    if not _allclose(gram, _eye3_like(rotation), atol=atol, rtol=rtol):
        raise InvalidRotationError("Rotation is not orthonormal")

    determinant = torch.linalg.det(rotation) if is_torch_array(rotation) else np.linalg.det(rotation)
    ones = torch.ones_like(determinant) if is_torch_array(rotation) else np.ones_like(determinant)
    if not _allclose(determinant, ones, atol=atol, rtol=rtol):
        raise InvalidRotationError("Rotation determinant is not +1")


def validate_transform_matrix(matrix: ArrayLike) -> None:
    require_floating_geometry(matrix, name="transform matrix")
    if matrix.ndim < 2 or tuple(matrix.shape[-2:]) != (4, 4):
        raise InvalidTransformError(
            f"Transform must have shape [..., 4, 4], got {tuple(matrix.shape)}"
        )
    if not _all_finite(matrix):
        raise InvalidTransformError("Transform contains NaN or infinity")

    atol, rtol = tolerances(matrix)
    if is_torch_array(matrix):
        expected = torch.zeros_like(matrix[..., 3, :])
    else:
        expected = np.zeros_like(matrix[..., 3, :])
    expected[..., 3] = 1
    if not _allclose(matrix[..., 3, :], expected, atol=atol, rtol=rtol):
        raise InvalidTransformError("Transform bottom row must be [0, 0, 0, 1]")

    try:
        validate_rotation_matrix(matrix[..., :3, :3])
    except InvalidRotationError as exc:
        raise InvalidTransformError(str(exc)) from exc


def is_identity_transform(matrix: ArrayLike) -> bool:
    validate_transform_matrix(matrix)
    atol, rtol = tolerances(matrix)
    if is_torch_array(matrix):
        identity = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    else:
        identity = np.eye(4, dtype=matrix.dtype)
    return _allclose(matrix, identity, atol=atol, rtol=rtol)


def validate_same_backend(*values: ArrayLike) -> None:
    if not values:
        return
    first_is_torch = is_torch_array(values[0])
    for value in values:
        require_array(value)
        if is_torch_array(value) != first_is_torch:
            raise UnsupportedArrayTypeError("NumPy arrays and Torch tensors cannot be mixed")
        if first_is_torch:
            if value.device != values[0].device:
                raise UnsupportedArrayTypeError("Torch geometry values must share a device")
            if value.dtype != values[0].dtype:
                raise UnsupportedDTypeError("Torch geometry values must share a dtype")
        elif value.dtype != values[0].dtype:
            raise UnsupportedDTypeError("NumPy geometry values must share a dtype")
