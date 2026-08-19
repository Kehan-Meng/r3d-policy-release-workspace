"""ACTTextAlignHead: masked CLIP-token cosine reconstruction loss.

Training-only auxiliary supervision分支：
用 ACT compact tokens 作为 Key/Value，通过 cross-attention 从被 mask 的
CLIP text tokens (Query) 中重建原始 token features。

只在训练时 compute_loss() 中参与，推理 predict_action() 不启用。

Diagnostics (zero-act / shuffle-act):
  通过复用同一个 mask_bool，比较 normal / zero-act / shuffle-act 三种
  reconstruction loss，判断 ACT tokens 是否真的被 cross-attention 使用。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ACTTextAlignHead(nn.Module):
    """用 ACT compact tokens 重建被 mask 的 CLIP text token features.

    Input:
        act_tokens:           [B, M, 512]  ACT 压缩后的 compact visual tokens
        text_tokens:          [B, 77, text_input_dim] CLIP token features
        attention_mask:       [B, 77] or None  1=valid token, 0=padding
        special_tokens_mask:  [B, 77] or None  1=special token (BOS/EOS/PAD/CLS)
        mask:                 [B, 77] bool or None  外部指定的 mask（用于 diagnostic）
        return_pred:          bool  是否返回 pred tensor（离线诊断用）

    Output:
        loss:  scalar, cosine distance on masked positions only
        debug: dict with mask_ratio, attention stats, mask stats
        pred:  [B, L, D] (only when return_pred=True)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        text_input_dim: Optional[int] = None,
        mask_ratio: float = 0.3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_input_dim = int(text_input_dim or embed_dim)
        self.mask_ratio = mask_ratio

        # Text tokens may come from a larger CLIP/OpenCLIP tower (e.g. EVA
        # projected tokens are 1024-dim).  The reconstruction head operates in
        # the ACT token space, so project text tokens into embed_dim first.
        if self.text_input_dim == embed_dim:
            self.text_input_proj = nn.Identity()
        else:
            self.text_input_proj = nn.Linear(self.text_input_dim, embed_dim)

        # Learnable mask token — init std=1.0 to match normed text distribution
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=1.0)

        # Pre-norms
        self.text_norm = nn.LayerNorm(embed_dim)
        self.act_norm = nn.LayerNorm(embed_dim)

        # Cross-attention: Q = masked_text, K/V = act_tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Standard residual FFN block
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def _random_mask(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        special_tokens_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Randomly mask ~mask_ratio of valid text tokens.

        Rules:
          - Special tokens (BOS/EOS/PAD/CLS) are never masked, determined by
            special_tokens_mask when available.
          - Each sample masks at least 1 token.
          - Masked positions are filled with self.mask_token.

        Returns:
            masked_text: [B, L, D] with masked positions replaced
            mask_bool:   [B, L] bool, True = masked
            mask_stats:  dict with valid/masked token counts and ratios
        """
        B, L, D = text_tokens.shape
        device = text_tokens.device

        # Build allowable mask: which positions can be masked
        if special_tokens_mask is not None:
            # special_tokens_mask: 1 = special (BOS/EOS/PAD/CLS), 0 = normal
            allowable = (special_tokens_mask == 0)  # [B, L]
        elif attention_mask is not None:
            # Fallback: exclude padding + token 0 (BOS) + token -1 (EOS)
            allowable = attention_mask.bool()
            allowable[:, 0] = False
            allowable[:, -1] = False
        else:
            allowable = torch.ones(B, L, dtype=torch.bool, device=device)
            allowable[:, 0] = False   # never mask BOS
            allowable[:, -1] = False  # never mask EOS

        # Guarantee at least 1 allowable token per sample
        n_allowable = allowable.sum(dim=1)  # [B]
        for b in range(B):
            if n_allowable[b] == 0:
                # Fallback: allow token 1 if everything is excluded
                col = min(1, L - 1)
                allowable[b, col] = True
                n_allowable[b] = 1

        # Number of tokens to mask per sample
        k = torch.clamp((n_allowable * self.mask_ratio).long(), min=1)  # [B]
        k_max = k.max().item()

        # Sample mask positions via random sort on allowable entries
        rand = torch.rand(B, L, device=device)
        rand[~allowable] = 2.0  # push non-allowable beyond any sampled rank
        _, indices = rand.topk(k_max, dim=1, largest=False)  # [B, k_max]

        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, indices[b, : k[b].item()]] = True

        # Replace masked positions with learnable mask token
        masked_text = text_tokens.clone()
        masked_text[mask] = self.mask_token.to(dtype=text_tokens.dtype)

        # --- Mask statistics ---
        n_valid = n_allowable.float()                     # [B]
        n_masked = mask.sum(dim=1).float()                # [B]
        L_f = float(L)

        mask_stats: Dict[str, float] = {
            "valid_tokens_mean": float(n_valid.mean().item()),
            "valid_tokens_min":   float(n_valid.min().item()),
            "valid_tokens_max":   float(n_valid.max().item()),
            "masked_tokens_mean": float(n_masked.mean().item()),
            "masked_tokens_min":  float(n_masked.min().item()),
            "masked_tokens_max":  float(n_masked.max().item()),
            "text_mask_ratio_all":   float((n_masked / L_f).mean().item()),
            "text_mask_ratio_valid": float((n_masked / n_valid.clamp(min=1)).mean().item()),
        }

        return masked_text, mask, mask_stats

    # ------------------------------------------------------------------
    # Attention statistics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_attn_stats(
        attn_weights: torch.Tensor,
        mask_bool: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute attention diagnostics on masked positions.

        Args:
            attn_weights: [B, L, M]  softmax-normalised cross-attention weights
            mask_bool:    [B, L]      True = masked (reconstruction target)

        Returns:
            dict with scalar stats averaged over masked positions
        """
        M = attn_weights.shape[-1]
        eps = 1e-8

        # Select masked positions: [N_masked_total, M]
        masked_w = attn_weights[mask_bool]  # [N, M]

        if masked_w.numel() == 0:
            return {
                "act_text_attn_max":       0.0,
                "act_text_attn_std":       0.0,
                "act_text_attn_entropy":   0.0,
                "act_text_attn_top1_mass": 0.0,
                "act_text_attn_top3_mass": 0.0,
            }

        # Entropy: H = -sum(p * log(p))  (natural log, nats)
        entropy_per_pos = -(masked_w * torch.log(masked_w + eps)).sum(dim=-1)  # [N]

        # Top‑k mass
        top1_vals, _ = masked_w.topk(1, dim=-1)  # [N, 1]
        top3_vals, _ = masked_w.topk(3, dim=-1)  # [N, 3]

        return {
            "act_text_attn_max":       float(masked_w.max().item()),
            "act_text_attn_std":       float(masked_w.std(unbiased=False).item()),
            "act_text_attn_entropy":   float(entropy_per_pos.mean().item()),
            "act_text_attn_top1_mass": float(top1_vals.sum(dim=-1).mean().item()),
            "act_text_attn_top3_mass": float(top3_vals.sum(dim=-1).mean().item()),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        act_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_pred: bool = False,
    ):
        """Forward pass returning (loss, debug_dict) or (loss, debug_dict, pred).

        Args:
            act_tokens:          [B, M, embed_dim]  e.g. [B, 32, 512]
            text_tokens:         [B, L, text_input_dim]
            attention_mask:      [B, L] or None
            special_tokens_mask: [B, L] or None
            mask:                [B, L] bool or None
                                 If provided, use this mask directly (skip _random_mask).
                                 Used for zero-act / shuffle-act diagnostics.
            return_pred:         If True, also return the predicted [B, L, D] tensor.

        Returns:
            loss:  scalar cosine distance on masked positions
            debug: dict with mask/attention stats (includes 'mask' tensor for external reuse)
            pred:  [B, L, D] (only when return_pred=True)
        """
        # Assert dim — first version expects 512
        if act_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"act_tokens last dim must be {self.embed_dim}, "
                f"got {act_tokens.shape[-1]}"
            )
        if text_tokens.shape[-1] != self.text_input_dim:
            raise ValueError(
                f"text_tokens last dim must be {self.text_input_dim}, "
                f"got {text_tokens.shape[-1]}"
            )

        # Detach text_tokens BEFORE constructing masked_text so that the
        # auxiliary loss never updates the CLIP text encoder.
        text_tokens = text_tokens.detach()
        text_tokens = self.text_input_proj(text_tokens)

        # 1. Norm
        text_normed = self.text_norm(text_tokens)       # [B, L, D]
        act_normed = self.act_norm(act_tokens)           # [B, M, D]

        # 2. Build mask (random or external)
        if mask is not None:
            # Use externally provided mask (for diagnostic reuse)
            mask_bool = mask.bool()
            masked_text = text_normed.clone()
            masked_text[mask_bool] = self.mask_token.to(dtype=text_normed.dtype)
            # Compute mask_stats from the external mask for consistent logging
            n_valid = torch.ones(
                text_normed.shape[0], device=text_normed.device
            ).float() * text_normed.shape[1]
            n_masked = mask_bool.sum(dim=1).float()
            L_f = float(text_normed.shape[1])
            mask_stats: Dict[str, float] = {
                "valid_tokens_mean": float(n_valid.mean().item()),
                "valid_tokens_min":   float(n_valid.min().item()),
                "valid_tokens_max":   float(n_valid.max().item()),
                "masked_tokens_mean": float(n_masked.mean().item()),
                "masked_tokens_min":  float(n_masked.min().item()),
                "masked_tokens_max":  float(n_masked.max().item()),
                "text_mask_ratio_all":   float((n_masked / L_f).mean().item()),
                "text_mask_ratio_valid": float((n_masked / n_valid.clamp(min=1)).mean().item()),
            }
        else:
            masked_text, mask_bool, mask_stats = self._random_mask(
                text_normed, attention_mask, special_tokens_mask,
            )

        # 3. Cross-attention: Q = masked_text, K/V = act_tokens
        attn_out, attn_weights = self.cross_attn(
            query=masked_text,
            key=act_normed,
            value=act_normed,
        )  # attn_out: [B, L, D], attn_weights: [B, L, M]

        # 4. Standard residual block
        x = masked_text + attn_out            # residual 1: skip connection
        x = x + self.ffn(self.ffn_norm(x))    # residual 2: FFN with pre-norm
        pred = self.out_norm(x)               # [B, L, D]

        # 5. Cosine feature-reconstruction loss on masked positions only:
        #
        #   L_rec = 1 - mean_{(b,k): k in M_b}
        #                     cos(pred_{b,k}, sg(target_{b,k})).
        #
        # The target is the unmasked CLIP token feature after the same input
        # projection and normalization used by the decoder.  Stop-gradient is
        # deliberately applied to the target, not the prediction; detaching
        # the prediction would prevent the reconstruction head and ACT tokens
        # from receiving this auxiliary supervision.
        target = text_normed.detach()
        pred_masked = pred[mask_bool].float()
        target_masked = target[mask_bool].float()
        cosine = torch.nn.functional.cosine_similarity(
            pred_masked,
            target_masked,
            dim=-1,
            eps=1e-8,
        )
        loss = 1.0 - cosine.mean()

        # 6. Attention statistics (on masked positions)
        attn_stats = self._compute_attn_stats(attn_weights, mask_bool)

        # 7. Debug dict — includes mask for external diagnostic reuse.
        #    mask is NOT logged to wandb (tensor), only consumed in DP3.compute_loss.
        debug: Dict = {
            **mask_stats,
            **attn_stats,
            # Keep mean for backward-compat but not for uniformity judgement
            # (softmax over M tokens → mean ≈ 1/M by construction)
            "act_text_attn_mean": float(attn_weights[mask_bool].mean().item())
                if mask_bool.any() else 0.0,
            "act_text_reconstruction_cosine": float(cosine.detach().mean().item()),
            # Internal mask for diagnostic reuse — excluded from wandb logging
            # by DP3's loss_dict filtering (only floats are logged)
            "mask": mask_bool,
        }

        if return_pred:
            return loss, debug, pred
        return loss, debug
