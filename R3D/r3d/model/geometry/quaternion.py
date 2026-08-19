"""Quaternion operations with explicit ordering and NumPy/Torch parity."""

from __future__ import annotations

import numpy as np
import torch

from .errors import InvalidRotationError, MissingQuaternionConventionError, SchemaDimensionError
from .types import ArrayLike, QuaternionOrder
from .validation import is_torch_array, require_floating_geometry, validate_rotation_matrix


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


def _validate_order(order: str | None) -> QuaternionOrder:
    if order not in ("wxyz", "xyzw"):
        raise MissingQuaternionConventionError(
            f"Quaternion order must be 'wxyz' or 'xyzw', got {order!r}"
        )
    return order


def reorder_quaternion(
    quaternion: ArrayLike,
    *,
    source_order: QuaternionOrder,
    target_order: QuaternionOrder,
) -> ArrayLike:
    require_floating_geometry(quaternion, name="quaternion")
    _validate_order(source_order)
    _validate_order(target_order)
    if quaternion.shape[-1] != 4:
        raise SchemaDimensionError(f"Quaternion must have shape [...,4], got {quaternion.shape}")
    if source_order == target_order:
        return quaternion.clone() if is_torch_array(quaternion) else np.array(quaternion, copy=True)
    if source_order == "xyzw":
        return _cat((quaternion[..., 3:4], quaternion[..., :3]), axis=-1)
    return _cat((quaternion[..., 1:], quaternion[..., 0:1]), axis=-1)


def normalize_quaternion(quaternion: ArrayLike, *, eps: float = 1e-12) -> ArrayLike:
    require_floating_geometry(quaternion, name="quaternion")
    if quaternion.shape[-1] != 4:
        raise SchemaDimensionError(f"Quaternion must have shape [...,4], got {quaternion.shape}")
    finite = torch.isfinite(quaternion).all() if is_torch_array(quaternion) else np.isfinite(quaternion).all()
    if not bool(finite.item() if is_torch_array(quaternion) else finite):
        raise InvalidRotationError("Quaternion contains NaN or infinity")
    norm = _norm(quaternion, keepdims=True)
    if is_torch_array(quaternion):
        if bool((norm <= eps).any().item()):
            raise InvalidRotationError("Cannot normalize a zero or near-zero quaternion")
    elif bool((norm <= eps).any()):
        raise InvalidRotationError("Cannot normalize a zero or near-zero quaternion")
    return quaternion / norm


def canonicalize_quaternion_wxyz(quaternion_wxyz: ArrayLike) -> ArrayLike:
    quaternion_wxyz = normalize_quaternion(quaternion_wxyz)
    sign = _where(
        quaternion_wxyz[..., :1] < 0,
        -quaternion_wxyz[..., :1] * 0 + -1,
        quaternion_wxyz[..., :1] * 0 + 1,
    )
    return quaternion_wxyz * sign


def quaternion_to_matrix(
    quaternion: ArrayLike,
    *,
    order: QuaternionOrder,
) -> ArrayLike:
    _validate_order(order)
    q = reorder_quaternion(
        quaternion,
        source_order=order,
        target_order="wxyz",
    )
    q = normalize_quaternion(q)
    w, x, y, z = (q[..., i] for i in range(4))
    two = q[..., 0] * 0 + 2

    return _stack(
        (
            _stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), axis=-1),
            _stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), axis=-1),
            _stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def _sqrt_positive_part(value: ArrayLike) -> ArrayLike:
    if is_torch_array(value):
        # Avoid the undefined derivative of sqrt at exactly zero. The inactive
        # branch is evaluated too, so clamp it to a finite positive value.
        safe = torch.sqrt(torch.clamp(value, min=torch.finfo(value.dtype).eps))
        return torch.where(value > 0, safe, torch.zeros_like(value))
    return np.sqrt(np.clip(value, a_min=0, a_max=None))


def matrix_to_quaternion(
    matrix: ArrayLike,
    *,
    order: QuaternionOrder,
) -> ArrayLike:
    """Convert rotation matrices using the best-conditioned quaternion branch."""

    _validate_order(order)
    validate_rotation_matrix(matrix)
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        _stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            axis=-1,
        )
    )

    candidates = _stack(
        (
            _stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), axis=-1),
            _stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), axis=-1),
            _stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), axis=-1),
            _stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), axis=-1),
        ),
        axis=-2,
    )
    floor = 1e-8 if matrix.dtype in (np.float64, torch.float64) else 1e-4
    if is_torch_array(matrix):
        denominator = 2 * torch.clamp(q_abs, min=floor)[..., :, None]
        candidates = candidates / denominator
        best = q_abs.argmax(dim=-1)
        gather_index = best[..., None, None].expand(best.shape + (1, 4))
        quaternion_wxyz = torch.gather(candidates, -2, gather_index).squeeze(-2)
    else:
        denominator = 2 * np.maximum(q_abs, floor)[..., :, None]
        candidates = candidates / denominator
        best = np.argmax(q_abs, axis=-1)
        quaternion_wxyz = np.take_along_axis(
            candidates,
            best[..., None, None],
            axis=-2,
        ).squeeze(-2)

    quaternion_wxyz = canonicalize_quaternion_wxyz(quaternion_wxyz)
    return reorder_quaternion(
        quaternion_wxyz,
        source_order="wxyz",
        target_order=order,
    )


def quaternion_multiply(
    left: ArrayLike,
    right: ArrayLike,
    *,
    order: QuaternionOrder,
) -> ArrayLike:
    _validate_order(order)
    left_wxyz = reorder_quaternion(left, source_order=order, target_order="wxyz")
    right_wxyz = reorder_quaternion(right, source_order=order, target_order="wxyz")
    lw, lx, ly, lz = (left_wxyz[..., i] for i in range(4))
    rw, rx, ry, rz = (right_wxyz[..., i] for i in range(4))
    product = _stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    product = normalize_quaternion(product)
    return reorder_quaternion(product, source_order="wxyz", target_order=order)


def quaternion_geodesic_distance(
    left: ArrayLike,
    right: ArrayLike,
    *,
    order: QuaternionOrder,
) -> ArrayLike:
    left_wxyz = normalize_quaternion(
        reorder_quaternion(left, source_order=order, target_order="wxyz")
    )
    right_wxyz = normalize_quaternion(
        reorder_quaternion(right, source_order=order, target_order="wxyz")
    )
    dot = (left_wxyz * right_wxyz).sum(dim=-1) if is_torch_array(left) else np.sum(left_wxyz * right_wxyz, axis=-1)
    if is_torch_array(left):
        dot = torch.clamp(torch.abs(dot), max=1)
        return 2 * torch.acos(dot)
    return 2 * np.arccos(np.clip(np.abs(dot), a_min=0, a_max=1))
