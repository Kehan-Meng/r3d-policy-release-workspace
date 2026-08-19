"""Low-coupling diversity regularizers for per-frame Meta Query tokens."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MQDiversityStats:
    loss: torch.Tensor
    mean_abs_cos: torch.Tensor
    mean_cos2: torch.Tensor


def _validate_features(features: torch.Tensor) -> None:
    if not torch.is_tensor(features):
        raise TypeError("MQ features must be a torch.Tensor")
    if features.ndim < 3:
        raise ValueError(
            "MQ features must have shape [..., K, D] with at least one group "
            f"dimension, got {tuple(features.shape)}"
        )
    if features.shape[-1] < 1:
        raise ValueError("MQ feature dimension D must be positive")


def cosine_orthogonality_stats(
    features: torch.Tensor,
    eps: float = 1e-6,
) -> MQDiversityStats:
    """Return squared off-diagonal cosine loss and logging metrics.

    Every leading index identifies an independent group. For example,
    ``[B, T, Q, D]`` computes one Q-by-Q Gram matrix per frame, never across
    frames. Computation is promoted to FP32 for mixed-precision stability.
    """
    _validate_features(features)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    num_queries = features.shape[-2]
    work = features.float()
    if num_queries < 2:
        zero = work.sum() * 0.0
        return MQDiversityStats(zero, zero, zero)

    normalized = F.normalize(work, p=2, dim=-1, eps=eps)
    gram = normalized @ normalized.transpose(-2, -1)
    off_diagonal = ~torch.eye(
        num_queries,
        dtype=torch.bool,
        device=features.device,
    )
    pairwise_cos = gram[..., off_diagonal]
    mean_cos2 = pairwise_cos.square().mean()
    return MQDiversityStats(
        loss=mean_cos2,
        mean_abs_cos=pairwise_cos.abs().mean(),
        mean_cos2=mean_cos2,
    )


def cosine_orthogonality_loss(
    features: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return cosine_orthogonality_stats(features, eps=eps).loss


def project_flow_readable_mq(
    mq_features: torch.Tensor,
    projection: nn.Linear,
    detach_projection_params: bool = True,
    mq_feature_offset: Optional[int] = None,
) -> torch.Tensor:
    """Project the MQ-dependent part of Flow's affine condition projection.

    Current R3D ``cat_on_token=false`` concatenates a diffusion-time prefix
    before the global condition. ``mq_feature_offset`` identifies the MQ
    columns inside that condition, including when later feature blocks such as
    text are appended. The common time/text components are intentionally
    omitted, while the affine bias is retained. This isolates what Flow can
    read *from MQ content*.

    Detaching only weight/bias keeps gradients to MQ and upstream ACT while
    preventing the auxiliary objective from training the projection itself.
    """
    _validate_features(mq_features)
    if not isinstance(projection, nn.Linear):
        raise TypeError(
            "Flow MQ diversity currently requires global_cond_proj to be "
            f"nn.Linear, got {type(projection).__name__}"
        )

    feature_dim = mq_features.shape[-1]
    if mq_feature_offset is None:
        mq_feature_offset = projection.in_features - feature_dim
    mq_feature_end = mq_feature_offset + feature_dim
    if mq_feature_offset < 0 or mq_feature_end > projection.in_features:
        raise ValueError(
            "MQ feature slice is outside global_cond_proj input: "
            f"offset={mq_feature_offset}, feature_dim={feature_dim}, "
            f"in_features={projection.in_features}"
        )

    weight = projection.weight[:, mq_feature_offset:mq_feature_end]
    bias = projection.bias
    if detach_projection_params:
        weight = weight.detach()
        bias = None if bias is None else bias.detach()

    # Projection parameters are normally FP32 even under autocast. Promoting
    # MQ here avoids dtype mismatches and keeps this auxiliary path stable.
    projected = F.linear(mq_features.to(dtype=weight.dtype), weight, bias)
    return projected


def _stats_to_log_dict(
    prefix: str,
    stats: MQDiversityStats,
    weighted: torch.Tensor,
) -> Dict[str, float]:
    return {
        f"loss_mq_div_{prefix}_raw": float(stats.loss.detach().item()),
        f"loss_mq_div_{prefix}_weighted": float(weighted.detach().item()),
        f"mq_{prefix}_mean_abs_cos": float(stats.mean_abs_cos.detach().item()),
        f"mq_{prefix}_mean_cos2": float(stats.mean_cos2.detach().item()),
    }


def add_mq_diversity_loss(
    original_loss: torch.Tensor,
    mq_features_per_frame: Optional[torch.Tensor],
    *,
    raw_enabled: bool = False,
    raw_weight: float = 0.0,
    flow_enabled: bool = False,
    flow_weight: float = 0.0,
    flow_projection: Optional[nn.Linear] = None,
    flow_detach_projection_params: bool = True,
    flow_mq_feature_offset: Optional[int] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Add independently switchable raw and Flow-readable MQ losses.

    The disabled path returns the exact same loss object and no new logs.
    ``mq_features_per_frame`` must retain a separate group dimension for each
    frame, typically ``[B*T, Q, D]`` or ``[B, T, Q, D]``.
    """
    if not raw_enabled and not flow_enabled:
        return original_loss, {}
    if mq_features_per_frame is None:
        raise RuntimeError(
            "MQ diversity is enabled, but per-frame ACT MQ features were not captured"
        )
    if raw_weight < 0 or flow_weight < 0:
        raise ValueError("MQ diversity weights must be non-negative")

    total = original_loss
    logs: Dict[str, float] = {}

    if raw_enabled:
        raw_stats = cosine_orthogonality_stats(mq_features_per_frame, eps=eps)
        weighted_raw = float(raw_weight) * raw_stats.loss
        total = total + weighted_raw
        logs.update(_stats_to_log_dict("raw", raw_stats, weighted_raw))

    if flow_enabled:
        if flow_projection is None:
            raise RuntimeError(
                "Flow-readable MQ diversity requires model.global_cond_proj"
            )
        projected = project_flow_readable_mq(
            mq_features_per_frame,
            flow_projection,
            detach_projection_params=flow_detach_projection_params,
            mq_feature_offset=flow_mq_feature_offset,
        )
        flow_stats = cosine_orthogonality_stats(projected, eps=eps)
        weighted_flow = float(flow_weight) * flow_stats.loss
        total = total + weighted_flow
        logs.update(_stats_to_log_dict("flow", flow_stats, weighted_flow))

    logs["total_loss"] = float(total.detach().item())
    return total, logs
