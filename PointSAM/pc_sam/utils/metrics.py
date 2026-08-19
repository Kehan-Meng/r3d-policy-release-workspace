"""
LASO-consistent evaluation metrics for affordance heatmap prediction.

Reference: https://github.com/yl3800/LASO

Four metrics:
  - aIoU : Average IoU over 20 evenly-spaced thresholds [0, 1]
  - AUC  : ROC AUC (targets binarized at 0.5, predictions continuous)
  - SIM  : Similarity — intersection of L1-normalized maps
  - MAE  : Mean Absolute Error

All functions accept torch tensors [B, N] or [B, Q, N] and return (mean_value, valid_count).
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ============================================================
#  aIoU — Average Intersection over Union (20 thresholds)
# ============================================================

def compute_aiou(pred: "torch.Tensor", target: "torch.Tensor",
                 num_thresholds: int = 20):
    """Compute LASO-style aIoU for batched predictions.

    Args:
        pred:   [B, N] or [B, Q, N] — sigmoid probabilities in [0, 1]
        target: same shape as pred — ground-truth heatmaps
        num_thresholds: number of evenly-spaced thresholds in [0, 1]

    Returns:
        (mean_aiou, valid_count). mean_aiou is NaN if no valid samples.
        For [B, Q, N] inputs, valid_count counts B*Q pairs, not B.
    """
    if pred.dim() == 3:
        pred = pred.flatten(0, 1)      # [B*Q, N]
        target = target.flatten(0, 1)  # [B*Q, N]

    # LASO-style: binarize target at 0.5
    t_bin = target >= 0.5              # bool, [M, N]
    positive_count = t_bin.sum(dim=-1)
    valid_target = positive_count > 0  # only exclude all-zero GT

    if not valid_target.any():
        return float("nan"), 0

    pred = pred[valid_target]          # [M_valid, N]
    t_bin = t_bin[valid_target]

    thresholds = torch.linspace(0, 1, num_thresholds, device=pred.device)
    per_threshold_ious = []
    for thre in thresholds:
        p_bin = pred >= thre
        intersection = torch.logical_and(p_bin, t_bin).sum(dim=-1)
        union = torch.logical_or(p_bin, t_bin).sum(dim=-1)
        # union > 0 is guaranteed (every valid sample has ≥1 positive in t_bin)
        iou = intersection.float() / union.float()
        per_threshold_ious.append(iou)

    # [T, M_valid] -> mean over thresholds -> [M_valid]
    per_sample_aiou = torch.stack(per_threshold_ious, dim=0).mean(dim=0)

    # 当前batch的平均aIoU, 当前batch参与平均的样本数量
    return per_sample_aiou.mean().item(), per_sample_aiou.numel()


# ============================================================
#  AUC — ROC AUC
# ============================================================

def compute_auc(pred: "torch.Tensor", target: "torch.Tensor"):
    """Compute ROC AUC for batched predictions.

    Target is binarized at 0.5 (consistent with LASO).
    Uses sklearn on CPU.

    Args:
        pred:   [B, N] or [B, Q, N] — sigmoid probabilities
        target: same shape — ground-truth soft heatmaps

    Returns:
        (mean_auc, valid_count). mean_auc is NaN if no valid samples.
    """
    if pred.dim() == 3:
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    aucs = []
    for i in range(pred_np.shape[0]):
        p = np.asarray(pred_np[i], dtype=np.float64)
        t = np.asarray(target_np[i], dtype=np.float64)
        t_bin = (t >= 0.5).astype(int)
        if t_bin.sum() == 0 or t_bin.sum() == len(t_bin):
            aucs.append(float("nan"))
        else:
            try:
                aucs.append(float(roc_auc_score(t_bin, p)))
            except ValueError:
                aucs.append(float("nan"))

    aucs = np.array(aucs, dtype=np.float64)
    valid = np.isfinite(aucs)
    if not valid.any():
        return float("nan"), 0
    return float(aucs[valid].mean()), int(valid.sum())


# ============================================================
#  SIM — Similarity (intersection of L1-normalized maps)
# ============================================================

def compute_sim(pred: "torch.Tensor", target: "torch.Tensor",
                eps: float = 1e-12):
    """Compute SIM for batched predictions.

    Args:
        pred:   [B, N] or [B, Q, N] — sigmoid probabilities
        target: same shape
        eps:    small constant to avoid division by zero

    Returns:
        (mean_sim, count). SIM in [0, 1], higher = more similar.
    """
    if pred.dim() == 3:
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)

    if pred.shape[0] == 0:
        return float("nan"), 0

    pred_n = pred / (pred.sum(dim=-1, keepdim=True) + eps)
    target_n = target / (target.sum(dim=-1, keepdim=True) + eps)

    sim = torch.minimum(pred_n, target_n).sum(dim=-1)  # [M]
    return sim.mean().item(), int(sim.shape[0])


# ============================================================
#  MAE — Mean Absolute Error
# ============================================================

def compute_mae(pred: "torch.Tensor", target: "torch.Tensor"):
    """Compute MAE for batched predictions.

    Args:
        pred:   [B, N] or [B, Q, N] — sigmoid probabilities
        target: same shape

    Returns:
        (mean_mae, count).
    """
    if pred.dim() == 3:
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)

    if pred.shape[0] == 0:
        return float("nan"), 0

    mae = (pred - target).abs().mean(dim=-1)
    return mae.mean().item(), int(mae.shape[0])


# ============================================================
#  All-in-one
# ============================================================

def compute_all_metrics(pred: "torch.Tensor", target: "torch.Tensor") -> dict:
    """Compute all four metrics for batched predictions.

    Args:
        pred:   [B, N] or [B, Q, N] — sigmoid probabilities
        target: same shape — ground-truth soft heatmaps

    Returns:
        dict with keys 'aiou', 'auc', 'sim', 'mae'.
        Each value is (mean, valid_count).
    """
    return {
        "aiou": compute_aiou(pred, target),
        "auc":  compute_auc(pred, target),
        "sim":  compute_sim(pred, target),
        "mae":  compute_mae(pred, target),
    }
