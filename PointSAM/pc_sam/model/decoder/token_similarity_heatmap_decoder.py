"""双向交叉注意力驱动的点云/文本相似度热图解码器。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_sam.model.transformer import TwoWayTransformer
from pc_sam.utils.common import AuxInputs, compute_interp_weights, interpolate_features


class TokenSimilarityHeatmapDecoder(nn.Module):
    """双向交叉注意力、上采样、余弦相似度与池化预测头。"""

    def __init__(
        self,
        pc_input_dim: int = 256,
        text_input_dim: int = 256,
        transformer_depth: int = 2,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        normalize: bool = True,
        temperature: float = 0.07,
        interp_k: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        if pc_input_dim != text_input_dim:
            raise ValueError(
                "双向 Transformer 要求点云和文本维度一致，"
                f"实际为 {pc_input_dim} 和 {text_input_dim}"
            )

        self.pc_input_dim = int(pc_input_dim)
        self.text_input_dim = int(text_input_dim)
        self.normalize = bool(normalize)
        self.interp_k = int(interp_k)
        self.eps = float(eps)

        self.two_way_transformer = TwoWayTransformer(
            depth=int(transformer_depth),
            embedding_dim=self.pc_input_dim,
            num_heads=int(transformer_num_heads),
            mlp_dim=int(transformer_mlp_dim),
            attention_downsample_rate=int(attention_downsample_rate),
        )
        # 上采样后按要求再经过一个 256 -> 256 的全连接层。
        self.point_fc = nn.Linear(self.pc_input_dim, self.pc_input_dim)
        self.pool_proj = nn.Linear(2, 1)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / float(temperature))
        )

    def _check_inputs(self, pc_embeddings, text_tokens):
        if pc_embeddings.dim() != 3 or pc_embeddings.shape[-1] != self.pc_input_dim:
            raise ValueError(
                "TokenSimilarityHeatmapDecoder 要求点云特征形状为 "
                f"[B, G, {self.pc_input_dim}]，实际为 {tuple(pc_embeddings.shape)}"
            )
        if text_tokens.dim() != 3 or text_tokens.shape[-1] != self.text_input_dim:
            raise ValueError(
                "TokenSimilarityHeatmapDecoder 要求文本特征形状为 "
                f"[B, L, {self.text_input_dim}]，实际为 {tuple(text_tokens.shape)}"
            )
        if pc_embeddings.shape[0] != text_tokens.shape[0]:
            raise ValueError(
                "点云与文本 batch size 必须一致，实际为 "
                f"{pc_embeddings.shape[0]} 和 {text_tokens.shape[0]}"
            )

    @staticmethod
    def _check_heatmap(heatmaps):
        if heatmaps.dim() != 3 or heatmaps.shape[1] != 1:
            raise ValueError(
                "TokenSimilarityHeatmapDecoder 必须返回 [B, 1, N]，"
                f"实际为 {tuple(heatmaps.shape)}"
            )
        if not torch.isfinite(heatmaps).all():
            raise FloatingPointError("预测热图中包含 NaN 或 Inf。")

    @staticmethod
    def _get_sinusoidal_pe(num_positions: int, dim: int, dtype, device):
        """固定正弦/余弦位置编码（不可学习）。

        与原始 Transformer "Attention Is All You Need" 一致。

        Returns:
            [1, num_positions, dim] 位置编码张量。
        """
        pe = torch.zeros(num_positions, dim, device=device, dtype=dtype)
        position = torch.arange(
            num_positions, device=device, dtype=dtype
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, num_positions, dim]

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
        aux_inputs: AuxInputs,
    ) -> torch.Tensor:
        self._check_inputs(pc_embeddings, text_tokens)
        text_valid = aux_inputs.text_valid_mask
        if text_valid is None:
            text_valid = text_tokens.abs().sum(dim=-1).gt(0)
        if text_valid.shape != text_tokens.shape[:2]:
            raise ValueError(
                "文本有效位掩码应为 [B, L]，"
                f"实际为 {tuple(text_valid.shape)}"
            )

        # Point-SAM 的一次 TwoWayTransformer 调用在每层依次完成：
        # 点云 Q -> 文本 KV，以及文本 Q -> 点云 KV。
        text_pe = self._get_sinusoidal_pe(
            text_tokens.shape[1],
            self.text_input_dim,
            dtype=text_tokens.dtype,
            device=text_tokens.device,
        )
        transformed_pc, transformed_text = self.two_way_transformer(
            pc_embedding=text_tokens,
            pc_pe=text_pe.expand(text_tokens.shape[0], -1, -1),
            point_embedding=pc_embeddings,
            key_padding_mask=text_valid,
        )

        query_coords = (
            aux_inputs.query_coords
            if aux_inputs.query_coords is not None
            else aux_inputs.coords
        )
        if aux_inputs.interp_index is not None and aux_inputs.interp_weight is not None:
            interp_index = aux_inputs.interp_index
            interp_weight = aux_inputs.interp_weight
        else:
            with torch.no_grad():
                interp_index, interp_weight = compute_interp_weights(
                    query_coords,
                    aux_inputs.centers,
                    k=self.interp_k,
                )

        point_tokens = interpolate_features(
            transformed_pc,
            interp_index,
            interp_weight,
        )
        point_tokens = self.point_fc(point_tokens)

        if self.normalize:
            point_tokens = F.normalize(point_tokens, dim=-1, eps=self.eps)
            transformed_text = F.normalize(
                transformed_text,
                dim=-1,
                eps=self.eps,
            )

        similarity = point_tokens @ transformed_text.transpose(-1, -2)
        similarity = similarity * self.logit_scale.exp().clamp(max=100.0)

        valid = text_valid.unsqueeze(1)
        similarity_for_max = similarity.masked_fill(~valid, -1e4)
        max_pool = similarity_for_max.max(dim=-1, keepdim=True).values

        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean_pool = (
            similarity.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
            / valid_count
        )

        pooled = torch.cat((max_pool, mean_pool), dim=-1)
        heatmaps = self.pool_proj(pooled).transpose(1, 2).contiguous()
        self._check_heatmap(heatmaps)
        return heatmaps
