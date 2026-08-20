"""Text-guided point-cloud heatmap model assembly.

This module is the top-level model entrypoint. It wires together:
  - point-cloud encoder
  - text encoder
  - heatmap decoder

Data flow:
  Dataset: coords [N, 3], text ["str"], gt_masks [1, N]
  Collate: coords [B, N, 3], text [["str"], ...], gt_masks [B, 1, N]
  Model:   point heatmaps [B, 1, N] logits by default.
           Optionally also returns patch heatmaps [B, N1].
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from pc_sam.utils.common import AuxInputs

if TYPE_CHECKING:
    from pc_sam.model.decoder.token_similarity_heatmap_decoder import (
        TokenSimilarityHeatmapDecoder,
    )
    from pc_sam.model.encoder.point_encoder import Uni3DPointEncoderForSAM
    from pc_sam.model.encoder.text_encoder import OpenCLIPTokenTextEncoder


class PointCloudSAM(nn.Module):
    def __init__(
        self,
        pc_encoder: "Uni3DPointEncoderForSAM",
        mask_decoder: "TokenSimilarityHeatmapDecoder",
        text_prompt_encoder: "OpenCLIPTokenTextEncoder",
        return_patch_heatmap: bool = False,
        return_contrastive_features: bool = False,
        contrastive_dim: int = 256,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.mask_decoder = mask_decoder
        self.text_prompt_encoder = text_prompt_encoder
        self.return_patch_heatmap = return_patch_heatmap
        self.return_contrastive_features = return_contrastive_features
        self.contrastive_dim = int(contrastive_dim)

        pc_feature_dim = int(self.pc_encoder.embed_dim)
        text_feature_dim = int(self.text_prompt_encoder.text_dim)
        if int(self.mask_decoder.pc_input_dim) != self.contrastive_dim:
            raise ValueError(
                "解码器点云维度与 contrastive_dim 不一致："
                f"pc_input_dim={self.mask_decoder.pc_input_dim}, "
                f"contrastive_dim={self.contrastive_dim}"
            )
        if int(self.mask_decoder.text_input_dim) != self.contrastive_dim:
            raise ValueError(
                "解码器文本维度与 contrastive_dim 不一致："
                f"text_input_dim={self.mask_decoder.text_input_dim}, "
                f"contrastive_dim={self.contrastive_dim}"
            )

        # 同一模态的全局 token 和 patch token 共享投影，保证两者位于同一空间。
        self.pc_projection = nn.Linear(pc_feature_dim, self.contrastive_dim)
        self.text_projection = nn.Linear(text_feature_dim, self.contrastive_dim)
    
    # ===================== Text-Guided Heatmap Forward =====================
    #下采样
    def downsample_heatmap_to_patches(
        self,
        heatmaps: torch.Tensor,
        patches: dict,
    ) -> torch.Tensor:
        """Use FPS+KNN groups to downsample point heatmaps to patch heatmaps."""
        if heatmaps.shape[1] == 1:
            heatmaps = heatmaps.transpose(1, 2).contiguous()  # [B, N, 1]

        if "knn_idx" not in patches:
            raise ValueError(
                "return_patch_heatmap=True requires patches['knn_idx']; "
                "the current Uni3D token encoder returns patch centers only."
            )
        knn_idx = patches["knn_idx"].to(device=heatmaps.device, dtype=torch.long)
        gather_index = knn_idx.unsqueeze(-1).expand(-1, -1, -1, heatmaps.shape[-1])
        heatmaps = heatmaps.unsqueeze(1).expand(-1, knn_idx.shape[1], -1, -1)
        patch_heatmaps = torch.gather(
            heatmaps,
            dim=2,
            index=gather_index,
        ).max(dim=2).values  # [B, N1, 1]
        return patch_heatmaps.squeeze(-1).contiguous()

    def forward_heatmap(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        text,
        query_coords: torch.Tensor = None,
        return_patch_heatmap: bool = None,
        return_contrastive_features: bool = None,
    ):
        """Run the text-guided heatmap forward pass.

        Args:
            coords: [B, Ns, 3] point coordinates for the encoder.
            features: [B, Ns, C] point features for the encoder.
            text: List[List[str]] with one text query per sample.
            query_coords: [B, Ng, 3] output coordinates. Defaults to coords.
            return_contrastive_features: 返回投影后的 CLS/EOT 对比特征，
                默认使用模型初始化设置。

        Returns:
            By default, heatmaps [B, 1, Ng]. Contrastive features are appended
            only when return_contrastive_features=True.
        """
        if return_patch_heatmap is None:
            return_patch_heatmap = self.return_patch_heatmap
        if return_contrastive_features is None:
            return_contrastive_features = self.return_contrastive_features

        # ---- 1. Point cloud encoder ----
        # pc_embeddings: [B, L, D]
        # patches = {"centers": center}
        
        pc_embeddings, patches = self.pc_encoder(coords, features)

        centers = patches["centers"]  # [B, L, 3]

        # ---- 2. Text encoder: [B, 77, 1024]，并始终提取 EOT ----
        text_tokens, text_eot = self.text_prompt_encoder(
            text,
            device=coords.device,
            dtype=pc_embeddings.dtype,
            return_eot=True,
        )

        # 在双向 Transformer 前统一映射到 256 维。
        text_valid_mask = text_tokens.abs().sum(dim=-1).gt(0)
        pc_embeddings = self.pc_projection(pc_embeddings)
        pc_cls = self.pc_projection(patches["cls_embedding"])
        text_tokens = self.text_projection(text_tokens)
        text_tokens = text_tokens.masked_fill(~text_valid_mask.unsqueeze(-1), 0.0)
        text_eot = self.text_projection(text_eot)

        aux_inputs = AuxInputs(
            coords=coords,
            features=features,
            centers=centers,
            query_coords=query_coords,
            pc_cls=pc_cls,
            text_eot=text_eot,
            text_valid_mask=text_valid_mask,
        )

        # ---- 3. Heatmap decoder ----
        heatmaps = self.mask_decoder(
            pc_embeddings=pc_embeddings,
            text_tokens=text_tokens,
            aux_inputs=aux_inputs,
        )
        # heatmaps: [B, 1, Ng]

        contrastive_features = None
        if return_contrastive_features:
            contrastive_features = {
                "pc_cls": pc_cls,
                "text_eot": text_eot,
            }

        if return_patch_heatmap:
            patch_heatmaps = self.downsample_heatmap_to_patches(heatmaps, patches)
            if return_contrastive_features:
                return heatmaps, patch_heatmaps, contrastive_features
            return heatmaps, patch_heatmaps

        if return_contrastive_features:
            return heatmaps, contrastive_features
        return heatmaps

    @torch.no_grad()
    def predict_text_heatmaps(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        text_prompts,
        query_coords: torch.Tensor = None,
        return_patch_heatmap: bool = None,
        return_contrastive_features: bool = None,
    ):
        """Inference helper for text-to-heatmap prediction.

        Args:
            coords: [B, Ns, 3] point coordinates for the encoder.
            features: [B, Ns, C] point features for the encoder.
            text_prompts: str, List[str], or List[List[str]].
            query_coords: [B, Ng, 3] output coordinates. Defaults to coords.
        Returns:
            Point heatmaps plus the always-present CLS/EOT feature dictionary.
        """
        # Normalize text input to List[List[str]] with fixed Q=1.
        if isinstance(text_prompts, str):
            text_prompts = [[text_prompts]]
        elif isinstance(text_prompts, list) and len(text_prompts) > 0:
            if isinstance(text_prompts[0], str):
                text_prompts = [[t] for t in text_prompts]

        for i, prompts in enumerate(text_prompts):
            if not isinstance(prompts, list) or len(prompts) != 1:
                raise ValueError(
                    f"PointCloudSAM expects exactly one text query per point cloud "
                    f"(fixed Q=1), got sample {i}: {prompts}"
                )

        return self.forward_heatmap(
            coords,
            features,
            text_prompts,
            query_coords=query_coords,
            return_patch_heatmap=return_patch_heatmap,
            return_contrastive_features=return_contrastive_features,
        )

    # ===================== Common Forward =====================

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        text=None,
        query_coords: torch.Tensor = None,
        return_patch_heatmap: bool = None,
        return_contrastive_features: bool = None,
        **kwargs,
    ):
        """Forward used by training and evaluation.

        Args:
            coords: [B, Ns, 3] point coordinates for the encoder.
            features: [B, Ns, C] point features for the encoder.
            text: List[List[str]] with one text query per sample.
            query_coords: [B, Ng, 3] output coordinates. Defaults to coords.
        Returns:
            Point heatmaps plus the always-present CLS/EOT feature dictionary.
        """

        return self.forward_heatmap(
            coords=coords,
            features=features,
            text=text,
            query_coords=query_coords,
            return_patch_heatmap=return_patch_heatmap,
            return_contrastive_features=return_contrastive_features,
        )
    
