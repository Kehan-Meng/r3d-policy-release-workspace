"""Parameter-free spatial multi-scope routing utilities for ACT meta queries."""

from __future__ import annotations

import math

import torch


ROLE_NAMES = ("local", "medium", "broad", "global")


def expand_scope_sigmas(num_queries, sigmas, *, device, dtype):
    if num_queries % len(ROLE_NAMES) != 0:
        raise ValueError(
            f"Spatial multi-scope requires num_queries divisible by 4, got {num_queries}"
        )
    if len(sigmas) != len(ROLE_NAMES):
        raise ValueError(f"Spatial multi-scope requires four sigmas, got {len(sigmas)}")
    values = []
    for index, sigma in enumerate(sigmas):
        if sigma is None or (isinstance(sigma, (int, float)) and math.isinf(sigma)):
            if index != len(sigmas) - 1:
                raise ValueError("Only the final/global spatial scope may use null/inf sigma")
            values.append(float("inf"))
        else:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError(f"Finite spatial scope sigma must be positive, got {sigma}")
            values.append(sigma)
    per_role = num_queries // len(ROLE_NAMES)
    return torch.tensor(values, device=device, dtype=dtype).repeat_interleave(per_role)


def build_spatial_scope_bias(
    cross1_attention,
    point_centers,
    sigmas,
    *,
    detach_centroid=True,
    eps=1e-6,
):
    """Build centered [B,Q,N] Gaussian scope bias from true Cross1 attention."""
    if cross1_attention.ndim != 4:
        raise ValueError(
            "cross1_attention must have shape [B,H,Q,N], got "
            f"{tuple(cross1_attention.shape)}"
        )
    if point_centers.ndim != 3 or point_centers.shape[-1] != 3:
        raise ValueError(
            f"point_centers must have shape [B,N,3], got {tuple(point_centers.shape)}"
        )
    batch, _heads, queries, points = cross1_attention.shape
    if point_centers.shape[:2] != (batch, points):
        raise ValueError(
            "point center axes must match Cross1 attention, got "
            f"{tuple(point_centers.shape[:2])} and {(batch, points)}"
        )

    attention = cross1_attention.mean(dim=1)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(eps)
    centroid = torch.matmul(attention, point_centers)
    if detach_centroid:
        centroid = centroid.detach()

    cloud_center = point_centers.mean(dim=1, keepdim=True)
    cloud_radius = (
        (point_centers - cloud_center).square().sum(dim=-1).mean(dim=1).sqrt()
        .clamp_min(eps)
    )
    distance = (point_centers[:, None] - centroid[:, :, None]).norm(dim=-1)
    normalized_distance = distance / cloud_radius[:, None, None]

    sigma = expand_scope_sigmas(
        queries, sigmas, device=normalized_distance.device, dtype=normalized_distance.dtype
    )
    finite = torch.isfinite(sigma)
    bias = torch.zeros_like(normalized_distance)
    bias[:, finite] = -normalized_distance[:, finite].square() / (
        2.0 * sigma[finite].square()[None, :, None]
    )
    bias = bias - bias.mean(dim=-1, keepdim=True)
    # Preserve exact zero for the global role after centering.
    bias[:, ~finite] = 0.0
    role_ids = torch.arange(len(ROLE_NAMES), device=bias.device).repeat_interleave(
        queries // len(ROLE_NAMES)
    )
    return bias, {
        "role_ids": role_ids,
        "sigmas": sigma,
        "centroid": centroid,
        "cloud_rms_radius": cloud_radius,
        "normalized_distance": normalized_distance,
    }
