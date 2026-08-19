"""Explicit projection of learned RoboTwin2 EE16 outputs to controller domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EE16ProjectionReport:
    left_quaternion_norm_min: float
    left_quaternion_norm_max: float
    right_quaternion_norm_min: float
    right_quaternion_norm_max: float
    gripper_clip_count: int
    max_abs_correction: float


def project_ee16_to_executable_domain(action, *, quaternion_eps=1e-8):
    """Normalize rotations and clip only grippers; positions stay untouched."""
    value = np.asarray(action)
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("EE16 projection requires floating-point input")
    if value.shape[-1] != 16:
        raise ValueError(f"EE16 action must end in 16, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("EE16 action contains NaN or Inf")

    output = np.array(value, copy=True)
    left_norm = np.linalg.norm(output[..., 3:7], axis=-1, keepdims=True)
    right_norm = np.linalg.norm(output[..., 11:15], axis=-1, keepdims=True)
    if np.any(left_norm < quaternion_eps) or np.any(right_norm < quaternion_eps):
        raise ValueError("EE16 action contains a zero or near-zero quaternion")
    output[..., 3:7] /= left_norm
    output[..., 11:15] /= right_norm

    raw_gripper = output[..., [7, 15]].copy()
    output[..., 7] = np.clip(output[..., 7], 0.0, 1.0)
    output[..., 15] = np.clip(output[..., 15], 0.0, 1.0)
    report = EE16ProjectionReport(
        left_quaternion_norm_min=float(left_norm.min()),
        left_quaternion_norm_max=float(left_norm.max()),
        right_quaternion_norm_min=float(right_norm.min()),
        right_quaternion_norm_max=float(right_norm.max()),
        gripper_clip_count=int(np.count_nonzero(raw_gripper != output[..., [7, 15]])),
        max_abs_correction=float(np.max(np.abs(output - value))),
    )
    return output, report
