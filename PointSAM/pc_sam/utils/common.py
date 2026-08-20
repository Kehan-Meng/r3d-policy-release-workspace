"""Small geometry helpers used by the heatmap decoder."""

import dataclasses
import torch


@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor
    features: torch.Tensor
    centers: torch.Tensor
    query_coords: torch.Tensor = None
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None
    pc_cls: torch.Tensor = None
    text_eot: torch.Tensor = None
    text_valid_mask: torch.Tensor = None


def knn_points(
    query: torch.Tensor,
    key: torch.Tensor,
    k: int,
    sorted: bool = False,
    transpose: bool = False,
):
    """Return distances and indices of the k nearest key points per query."""
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

    distance = torch.cdist(query, key)
    if k == 1:
        return torch.min(distance, dim=2, keepdim=True)
    return torch.topk(distance, k, dim=2, largest=False, sorted=sorted)


def compute_interp_weights(query: torch.Tensor, key: torch.Tensor, k=3, eps=1e-8):
    """Compute inverse-distance interpolation weights from key points to query points.

    Args:
        query: [B, Nq, 3] output/query coordinates.
        key: [B, Nk, 3] patch center coordinates.
        k: number of nearest centers to use.

    Returns:
        idx: [B, Nq, K] nearest-center indices.
        weight: [B, Nq, K] normalized interpolation weights.
    """
    dist, idx = knn_points(query, key, k)
    inv_dist = 1.0 / torch.clamp(dist.square(), min=eps)
    weight = inv_dist / torch.sum(inv_dist, dim=2, keepdim=True)
    return idx, weight


def interpolate_features(x: torch.Tensor, index: torch.Tensor, weight: torch.Tensor):
    """Interpolate key features to query positions.

    Args:
        x: [B, Nk, C] key features.
        index: [B, Nq, K] key indices for each query.
        weight: [B, Nq, K] interpolation weights.

    Returns:
        [B, Nq, C] interpolated features.
    """
    batch_size, num_queries, num_neighbors = index.shape
    batch_offset = (
        torch.arange(batch_size, device=x.device).reshape(-1, 1, 1) * x.shape[1]
    )
    index_flat = (index + batch_offset).flatten()
    gathered = x.flatten(0, 1)[index_flat].reshape(
        batch_size, num_queries, num_neighbors, x.shape[-1]
    )
    return (gathered * weight.unsqueeze(-1)).sum(-2)
