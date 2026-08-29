import torch.nn as nn

from .attention import HeatmapGuidedCrossAttention


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
    ):
        super().__init__()
        context_dim = dim if context_dim is None else context_dim
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.heatmap_mode = heatmap_mode
        self.heatmap_gamma = heatmap_gamma
        self.log_bias_lambda = log_bias_lambda
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
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, ffn_ratio=ffn_ratio, dropout=dropout)

    def forward(self, queries, context, heatmap=None, heatmap_mode=None):
        mode = self.heatmap_mode if heatmap_mode is None else heatmap_mode
        normalized_queries = self.norm_query(queries)
        normalized_context = self.norm_context(context)
        attn_out, attn = self.cross_attn(
            normalized_queries,
            normalized_context,
            heatmap=heatmap,
            heatmap_mode=mode,
            log_bias_lambda=self.log_bias_lambda,
        )
        queries = queries + self.attn_drop(attn_out)
        queries = queries + self.ffn(self.norm2(queries))
        return queries, attn
