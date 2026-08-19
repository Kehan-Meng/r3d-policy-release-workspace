import torch
import torch.nn as nn

from .attention import HeatmapGuidedCrossAttention


class _FramewiseCommonGradientDeflation(torch.autograd.Function):
    """Identity forward with query-common gradient attenuation in backward."""

    @staticmethod
    def forward(ctx, tensor, alpha):
        ctx.alpha = float(alpha)
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        common = gradient.mean(dim=1, keepdim=True)
        return gradient - ctx.alpha * common, None


def framewise_common_gradient_deflation(tensor, alpha):
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"common gradient deflation alpha must be in [0,1], got {alpha}")
    if float(alpha) == 0.0:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(
            "Cross-attention message must have shape [B*T,Q,D], got "
            f"{tuple(tensor.shape)}"
        )
    return _FramewiseCommonGradientDeflation.apply(tensor, float(alpha))


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_ratio=4.0, dropout=0.0):
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError(f"ffn_ratio must be positive, got {ffn_ratio}")

        hidden_dim = max(1, int(dim * ffn_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MetaSelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block for compact meta queries."""

    def __init__(self, dim, num_heads=8, ffn_ratio=4.0, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, ffn_ratio=ffn_ratio, dropout=dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )
        x = residual + self.attn_drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class HeatmapCrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block from meta queries to point patch tokens."""

    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        ffn_ratio=4.0,
        dropout=0.0,
        heatmap_mode="none",
        heatmap_gamma=1.0,
        log_bias_lambda=1.0,
        eps=1e-6,
        competitive=False,
        competitive_temperature=1.0,
        common_gradient_deflation_alpha=0.0,
    ):
        super().__init__()
        context_dim = dim if context_dim is None else context_dim
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.heatmap_mode = heatmap_mode
        self.heatmap_gamma = heatmap_gamma
        self.log_bias_lambda = log_bias_lambda
        self.competitive = competitive
        self.competitive_temperature = float(competitive_temperature)
        self.common_gradient_deflation_alpha = float(common_gradient_deflation_alpha)
        if not 0.0 <= self.common_gradient_deflation_alpha <= 1.0:
            raise ValueError(
                "common_gradient_deflation_alpha must be in [0,1], got "
                f"{self.common_gradient_deflation_alpha}"
            )
        self.norm_query = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.cross_attn = HeatmapGuidedCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            dropout=dropout,
            heatmap_mode=heatmap_mode,
            heatmap_gamma=heatmap_gamma,
            log_bias_lambda=log_bias_lambda,
            eps=eps,
            competitive=competitive,
            competitive_temperature=competitive_temperature,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, ffn_ratio=ffn_ratio, dropout=dropout)

    def forward(
        self, queries, context, heatmap=None, heatmap_mode=None,
        attention_bias=None, attention_bias_mode="absolute",
        attention_bias_strength=1.0, attention_bias_relative_rho=0.0,
        return_score_debug=False,
    ):
        mode = self.heatmap_mode if heatmap_mode is None else heatmap_mode
        normalized_queries = self.norm_query(queries)
        normalized_context = self.norm_context(context)
        if attention_bias is None and not return_score_debug:
            # Preserve the legacy call signature for default-off compatibility
            # and existing read-only audit wrappers.
            result = self.cross_attn(
                normalized_queries,
                normalized_context,
                heatmap=heatmap,
                heatmap_mode=mode,
                log_bias_lambda=self.log_bias_lambda,
            )
        else:
            result = self.cross_attn(
                normalized_queries,
                normalized_context,
                heatmap=heatmap,
                heatmap_mode=mode,
                log_bias_lambda=self.log_bias_lambda,
                attention_bias=attention_bias,
                attention_bias_mode=attention_bias_mode,
                attention_bias_strength=attention_bias_strength,
                attention_bias_relative_rho=attention_bias_relative_rho,
                return_score_debug=return_score_debug,
            )
        attn_out, attn = result[:2]
        attn_out = framewise_common_gradient_deflation(
            attn_out,
            self.common_gradient_deflation_alpha,
        )
        queries = queries + self.attn_drop(attn_out)
        queries = queries + self.ffn(self.norm2(queries))
        if return_score_debug:
            return queries, attn, result[2]
        return queries, attn
