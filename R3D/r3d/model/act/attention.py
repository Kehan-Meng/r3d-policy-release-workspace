import math

import torch
import torch.nn as nn


class HeatmapGuidedCrossAttention(nn.Module):
    """Batch-first cross-attention with optional heatmap modulation."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        num_heads=8,
        dropout=0.0,
        heatmap_mode="none",
        heatmap_gamma=1.0,
        log_bias_lambda=1.0,
        eps=1e-6,
        bias=True,
        competitive=False,
        competitive_temperature=1.0,
    ):
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim

        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
            )
        if heatmap_mode not in ("none", "multiply", "log_bias"):
            raise ValueError(
                "heatmap_mode must be one of 'none', 'multiply', or 'log_bias', "
                f"got {heatmap_mode}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if heatmap_gamma <= 0:
            raise ValueError(
                f"heatmap_gamma must be positive, got {heatmap_gamma}"
            )
        if competitive_temperature <= 0:
            raise ValueError(
                "competitive_temperature must be positive, got "
                f"{competitive_temperature}"
            )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.heatmap_mode = heatmap_mode
        self.heatmap_gamma = float(heatmap_gamma)
        self.log_bias_lambda = log_bias_lambda
        self.eps = eps
        self.competitive = competitive
        self.competitive_temperature = float(competitive_temperature)

        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _reshape_heads(self, tensor):
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _format_heatmap(self, heatmap, batch_size, num_context_tokens, dtype, device):
        if heatmap.ndim == 2:
            heatmap = heatmap.unsqueeze(-1)
        if heatmap.ndim != 3:
            raise ValueError(
                f"heatmap must have shape [B, N, 1] or [B, N], got {tuple(heatmap.shape)}"
            )
        if heatmap.shape[0] != batch_size:
            raise ValueError(
                f"heatmap batch size must be {batch_size}, got {heatmap.shape[0]}"
            )
        if heatmap.shape[1] != num_context_tokens:
            raise ValueError(
                "heatmap token dimension must match context tokens, got "
                f"{heatmap.shape[1]} and {num_context_tokens}"
            )
        if heatmap.shape[2] != 1:
            raise ValueError(
                f"heatmap last dimension must be 1, got {heatmap.shape[2]}"
            )

        return heatmap.to(device=device, dtype=dtype).transpose(1, 2).unsqueeze(1)

    def _resolve_heatmap_mode(self, heatmap_mode):
        mode = self.heatmap_mode if heatmap_mode is None else heatmap_mode
        if mode not in ("none", "multiply", "log_bias"):
            raise ValueError(
                "heatmap_mode must be one of 'none', 'multiply', or 'log_bias', "
                f"got {mode}"
            )
        return mode

    def _format_heatmap_for_scores(self, scores, heatmap):
        batch_size, _, _, num_context_tokens = scores.shape
        return self._format_heatmap(
            heatmap,
            batch_size=batch_size,
            num_context_tokens=num_context_tokens,
            dtype=scores.dtype,
            device=scores.device,
        )

    def _multiply_attention_by_heatmap(self, attn, heatmap):
        # Treat the heatmap as a non-negative spatial prior over context tokens.
        output_dtype = attn.dtype
        attn = attn.float()
        gate = (heatmap.float().clamp_min(0.0) + self.eps).pow(
            self.heatmap_gamma
        )
        weighted_attn = attn * gate
        normalized_attn = weighted_attn / weighted_attn.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(weighted_attn.dtype).tiny)
        return normalized_attn.to(dtype=output_dtype)

    def forward(
        self, queries, context, heatmap=None, heatmap_mode=None,
        log_bias_lambda=None,
    ):
        """
        Args:
            queries: Tensor with shape [B, M, query_dim].
            context: Tensor with shape [B, N, context_dim].
            heatmap: Optional tensor with shape [B, N, 1].

        Returns:
            output: Tensor with shape [B, M, query_dim].
            attn: Per-head attention weights with shape [B, heads, M, N].
        """
        if queries.ndim != 3:
            raise ValueError(
                f"queries must have shape [B, M, D], got {tuple(queries.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(
                f"context must have shape [B, N, D], got {tuple(context.shape)}"
            )
        if queries.shape[0] != context.shape[0]:
            raise ValueError(
                "queries and context must share the same batch size, got "
                f"{queries.shape[0]} and {context.shape[0]}"
            )
        if queries.shape[-1] != self.query_dim:
            raise ValueError(
                f"queries last dimension must be {self.query_dim}, got {queries.shape[-1]}"
            )
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context last dimension must be {self.context_dim}, got {context.shape[-1]}"
            )

        q = self._reshape_heads(self.q_proj(queries))
        k = self._reshape_heads(self.k_proj(context))
        v = self._reshape_heads(self.v_proj(context))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mode = self._resolve_heatmap_mode(heatmap_mode)

        formatted_heatmap = None
        if heatmap is not None and mode != "none":
            formatted_heatmap = self._format_heatmap_for_scores(scores, heatmap)

        if mode == "log_bias" and formatted_heatmap is not None:
            lambda_ = (
                self.log_bias_lambda
                if log_bias_lambda is None
                else log_bias_lambda
            )
            scores = scores + lambda_ * torch.log(
                formatted_heatmap.clamp_min(self.eps)
            )

        if self.competitive:
            # ── Competitive MQ Assignment ──
            # softmax across MQ/query dimension (dim=2):
            #   for every point n, Q meta-queries compete for ownership
            C = torch.softmax(
                scores / self.competitive_temperature, dim=2
            )  # [B, H, Q, N], Σ_q C[b,h,q,n] = 1

            if mode == "multiply" and formatted_heatmap is not None:
                # multiply point-wise heatmap importance, then normalize over points
                attn = self._multiply_attention_by_heatmap(C, formatted_heatmap)
            else:
                # normalize over point dimension for each MQ
                attn = C / (C.sum(dim=-1, keepdim=True) + self.eps)
        else:
            # ── Baseline: each MQ independently selects points ──
            attn = torch.softmax(scores, dim=-1)
            if mode == "multiply" and formatted_heatmap is not None:
                attn = self._multiply_attention_by_heatmap(attn, formatted_heatmap)

        output = torch.matmul(self.attn_drop(attn), v)
        output = output.transpose(1, 2).contiguous().view(
            queries.shape[0], queries.shape[1], self.query_dim
        )
        output = self.out_proj(output)
        output = self.out_drop(output)
        return output, attn
