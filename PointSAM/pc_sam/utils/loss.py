"""Losses used by the PointSAM heatmap V2 training path."""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _all_gather_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Gather equal-sized local batches without autograd, as in Uni3D."""
    world_size = _get_world_size()
    if world_size == 1:
        return tensor

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)


class BCEDiceHeatmapLoss(nn.Module):
    """BCEWithLogits + Dice loss for fixed-Q heatmap prediction."""

    def __init__(
        self,
        bce_loss_weight: float = 1.0,
        dice_loss_weight: float = 1.0,
        dice_scale: float = 1000.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.dice_scale = dice_scale
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        """Compute loss for raw heatmap logits shaped [B, 1, N]."""
        if pred_logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_logits.shape} vs target {target.shape}"
            )
        if pred_logits.dim() != 3 or pred_logits.shape[1] != 1:
            raise ValueError(
                f"BCEDiceHeatmapLoss expects fixed Q=1, got {pred_logits.shape}"
            )

        target = target.to(dtype=pred_logits.dtype)

        loss_bce = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction="none"
        )#先对 pred_logits 做 sigmoid，变成概率 , 再和 target 计算 BCE
        loss_bce = loss_bce.flatten(1, 2).mean(1).mean() 
        # 展平[B,1,N]->[B,N];
        #.mean(1)对每个样本的所有点求平均:[B,N]->[B];
        #.mean()对 batch 再求平均 : [B] -> scalar


        pred = pred_logits.sigmoid()
        pred_flat = pred.flatten(1, 2)
        target_flat = target.flatten(1, 2)

        pred_scaled = pred_flat / self.dice_scale
        target_scaled = target_flat / self.dice_scale
        numerator = 2 * (pred_scaled * target_flat).sum(-1)
        denominator = pred_scaled.sum(-1) + target_scaled.sum(-1)
        loss_dice = 1 - (numerator + self.eps) / (denominator + self.eps)
        loss_dice = loss_dice.mean()

        loss = self.bce_loss_weight * loss_bce + self.dice_loss_weight * loss_dice
        aux = {
            "loss_bce_raw": loss_bce.detach(),
            "loss_bce": (self.bce_loss_weight * loss_bce).detach(),
            "loss_dice_raw": loss_dice.detach(),
            "loss_dice": (self.dice_loss_weight * loss_dice).detach(),
        }
        return loss, aux


class Uni3DContrastiveLoss(nn.Module):
    """对投影后的点云 CLS 与文本 EOT 计算对称 InfoNCE loss。"""

    def __init__(self):
        super().__init__()
        # Match Uni3D: store log(1 / temperature) and exponentiate it in forward.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = None
        self.last_local_batch_size = None

    def forward(
        self,
        pc_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """使用本地 query 和跨卡收集的 key 计算点云-文本对比损失。"""
        if pc_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError(
                "Uni3DContrastiveLoss expects [B, D] features, got "
                f"pc_features={tuple(pc_features.shape)}, "
                f"text_features={tuple(text_features.shape)}"
            )
        if pc_features.shape != text_features.shape:
            raise ValueError(
                "Point and text feature shapes must match, got "
                f"{tuple(pc_features.shape)} and {tuple(text_features.shape)}"
            )

        local_batch_size = pc_features.size(0)
        if (
            local_batch_size != self.last_local_batch_size
            or self.labels is None
            or self.labels.device != pc_features.device
        ):
            self.labels = local_batch_size * _get_rank() + torch.arange(
                local_batch_size,
                device=pc_features.device,
            )
            self.last_local_batch_size = local_batch_size

        pc_features = F.normalize(pc_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Uni3D form: local queries contrast against keys gathered from all GPUs.
        pc_features_all = _all_gather_batch(pc_features)
        text_features_all = _all_gather_batch(text_features)

        logit_scale = self.logit_scale.exp()
        logits_per_pc_text = logit_scale * pc_features @ text_features_all.t()
        logits_per_text_pc = logit_scale * text_features @ pc_features_all.t()

        loss_contrastive = (
            F.cross_entropy(logits_per_pc_text, self.labels)
            + F.cross_entropy(logits_per_text_pc, self.labels)
        ) / 2

        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            pc_text_acc = 100 * pred.eq(self.labels).sum() / local_batch_size

        aux = {
            "loss_contrastive_raw": loss_contrastive.detach(),
            "contrastive_logit_scale": logit_scale.detach(),
            "pc_text_acc": pc_text_acc,
        }
        return loss_contrastive, aux
