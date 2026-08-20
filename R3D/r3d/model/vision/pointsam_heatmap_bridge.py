"""
PointSAMHeatmapBridge — R3D × Point-SAM text-guided heatmap bridge.

Connects the R3D-Policy pipeline to the pointsam_last text-guided heatmap model.
Drop-in replacement for DP3Encoder (same forward() interface).

Returns (no heatmap):
    pc_embeddings : [B, 512, 256]    ViT patch tokens (projected from 1024)
    pc_pe         : [B, 512, 256]    random Fourier positional encoding

Returns (with heatmap):
    pc_embeddings : [B, 512, 256]
    pc_pe         : [B, 512, 256]
    heatmap       : [B, 512, 1]      patch-level heatmap (kNN max-pooled, sigmoid)

Data flow:
  pcd [B, N, 3|6]
    → Uni3DPointEncoderForSAM (FPS + kNN + PointNet + ViT + E-SAM adapters)
      → raw [B, 512, 1024], patches {"centers": [B, 512, 3], "knn_idx": [B, 512, 64]}
    → pc_proj: Linear(1024 → 256)
      → pc_embeddings [B, 512, 256]  (aligned with pc_pe)
    → PositionEmbeddingRandom(normalized centers)
      → pc_pe [B, 512, 256]
    → [optional] text encoder + decoder → point heatmap [B, N, 1]
      → _downsample_heatmap_to_patches (kNN max-pool) → patch heatmap [B, 512, 1]
"""

import math
import os
import pathlib
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint


# =============================================================================
# PositionEmbeddingRandom — duplicate so this file stays self-contained
# (same implementation as pointnet_extractor.py and SAM's prompt_encoder.py)
# =============================================================================

class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies.

    Adapted from Segment Anything's prompt_encoder.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self._pe_encoding(coords)


# =============================================================================
# PointSAMHeatmapBridge
# =============================================================================

class PointSAMHeatmapBridge(nn.Module):
    """Standalone PointSAM heatmap adapter that bridges R3D and pointsam.

    Parameters
    ----------
    pointsam_root : str
        Path to the pointsam checkout.
    embed_dim : int
        Dimension of pc_pe positional encoding (default 256 → 128 pos feats).
    out_dim : int or None
        Alias for embed_dim. When set, overrides embed_dim.
    heatmap_config_name : str
        Hydra config name inside pointsam_root/configs/.
    heatmap_config_dir : str or None
        Override config directory; defaults to <pointsam_root>/configs.
    heatmap_as_prob : bool
        Whether to sigmoid the heatmap logits before returning.
    load_checkpoint : bool
        Whether to load a trained PointSAM checkpoint.
    checkpoint_path : str or None
        Path to checkpoint (.safetensors / .pt / directory).
    freeze : bool
        Freeze all PointSAM components.
    freeze_uni3d : bool
        Freeze only the Uni3D backbone (keep E-SAM adapters trainable).
    freeze_out_proj : bool
        Freeze the encoder output projection layer.
    strict_load : bool
        Raise on any checkpoint-or-model key mismatch.
    ignore_mismatched_sizes : bool
        Skip checkpoint keys with incompatible tensor shapes.
    """

    def __init__(
        self,
        pointsam_root: str = None,
        embed_dim: int = 256,
        out_dim: int = None,
        heatmap_config_name: str = 'large_heatmap_uni3d_token_similarity',
        heatmap_config_dir: str = None,
        heatmap_as_prob: bool = True,
        load_checkpoint: bool = False,
        checkpoint_path: str = None,
        freeze: bool = False,
        freeze_uni3d: bool = False,
        freeze_out_proj: bool = False,
        strict_load: bool = False,
        ignore_mismatched_sizes: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = out_dim or embed_dim          # pc_pe output dim (256)
        self.heatmap_as_prob = heatmap_as_prob
        self.freeze_components = freeze
        self.freeze_uni3d_backbone = bool(freeze_uni3d)
        self.supports_heatmap = True

        # ---- resolve paths & import pointsam ----
        pointsam_root = self._prepare_pointsam_import(pointsam_root)

        # ---- build model from Hydra config ----
        self._build_model(
            pointsam_root=pointsam_root,
            config_dir=heatmap_config_dir,
            config_name=heatmap_config_name,
            checkpoint_path=checkpoint_path if load_checkpoint else None,
            strict=strict_load,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        # ---- pc_pe: random Fourier positional encoding for patch centers ----
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)  # 128 → 256

        # ---- pc_proj: project ViT output 1024 → 256 to align with pc_pe ----
        self.pc_proj = nn.Linear(
            int(getattr(self.encoder, 'embed_dim', 1024)),  # encoder out dim
            self.embed_dim,                                  # target = 256
        )

        # ---- pc_embed_dim: projected dim matches pc_pe (256) ----
        self.pc_embed_dim = self.embed_dim

        # ---- freeze policy ----
        if freeze:
            self._set_freeze_all()
        elif freeze_uni3d:
            self._set_freeze_uni3d()

        if freeze_out_proj:
            self._set_freeze_out_proj()

        cprint(
            f"[PointSAMHeatmapBridge] ready "
            f"config={heatmap_config_name}, pe_dim={self.embed_dim}, "
            f"raw_dim={getattr(self.encoder, 'embed_dim', '?')}, "
            f"pc_embed_dim={self.pc_embed_dim}",
            "cyan",
        )

    # =========================================================================
    # Import path management
    # =========================================================================

    @staticmethod
    def _prepare_pointsam_import(pointsam_root: str) -> str:
        """Add pointsam to sys.path and evict stale pc_sam modules."""
        if not pointsam_root:
            try:
                import pc_sam
            except ImportError as exc:
                raise ImportError(
                    "PointSAM is not installed. Run `pip install -e ./PointSAM`."
                ) from exc
            return str(pathlib.Path(pc_sam.__file__).resolve().parent.parent)
        pointsam_root = os.path.abspath(os.path.expanduser(pointsam_root))
        extra_paths = [
            pointsam_root,
            os.path.join(pointsam_root, 'third_party', 'Pointnet2_PyTorch',
                         'pointnet2_ops_lib'),
            os.path.join(pointsam_root, 'third_party', 'torkit3d'),
        ]
        for extra_path in extra_paths:
            if not os.path.exists(extra_path):
                continue
            if extra_path in sys.path:
                sys.path.remove(extra_path)
            sys.path.insert(0, extra_path)

        # Purge pc_sam modules if they were imported from a different root
        loaded_pc_sam = sys.modules.get('pc_sam')
        loaded_path = getattr(loaded_pc_sam, '__file__', None)
        if loaded_path is not None:
            loaded_path = os.path.abspath(loaded_path)
            if not loaded_path.startswith(pointsam_root):
                for module_name in list(sys.modules.keys()):
                    if module_name == 'pc_sam' or module_name.startswith('pc_sam.'):
                        del sys.modules[module_name]

        return pointsam_root

    # =========================================================================
    # Model construction (Hydra)
    # =========================================================================

    def _build_model(
        self,
        pointsam_root: str,
        config_dir: str,
        config_name: str,
        checkpoint_path: str,
        strict: bool,
        ignore_mismatched_sizes: bool,
    ):
        """Instantiate PointCloudSAM from a pointsam Hydra config."""
        config_dir = config_dir or os.path.join(pointsam_root, 'configs')
        config_dir = os.path.abspath(os.path.expanduser(config_dir))

        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        import hydra

        # Clear any pre-existing Hydra context (R3D train.py also uses Hydra)
        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            global_hydra.clear()

        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=[])
            OmegaConf.resolve(cfg)

        # ---- Instantiate the full PointCloudSAM model ----
        self.model: nn.Module = hydra.utils.instantiate(cfg.model)

        # Extract sub-modules for fine-grained control
        self.encoder = self.model.pc_encoder           # Uni3DPointEncoderForSAM
        self.text_encoder = self.model.text_prompt_encoder  # OpenCLIPTokenTextEncoder
        self.decoder = self.model.mask_decoder          # TokenSimilarityHeatmapDecoder

        # ---- Load checkpoint if provided ----
        if checkpoint_path is not None:
            state_dict = self._read_checkpoint(checkpoint_path)
            if state_dict is not None:
                for prefix, module in (
                    ('pc_encoder.', self.encoder),
                    ('text_prompt_encoder.', self.text_encoder),
                    ('mask_decoder.', self.decoder),
                ):
                    self._load_prefixed_module(
                        module, state_dict, prefix,
                        strict=strict,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                    )
            else:
                cprint(
                    "[PointSAMHeatmapBridge] "
                    "checkpoint not found, using initialized weights",
                    "yellow",
                )

        cprint(
            f"[PointSAMHeatmapBridge] "
            f"encoder={self.encoder.__class__.__name__}, "
            f"decoder={self.decoder.__class__.__name__}, "
            f"text_encoder={self.text_encoder.__class__.__name__}",
            "cyan",
        )

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================

    @staticmethod
    def _resolve_checkpoint_path(path: str) -> str:
        path = os.path.abspath(os.path.expanduser(str(path)))
        if os.path.isdir(path):
            for name in ('model.safetensors', 'pytorch_model.bin',
                         'model.pt', 'checkpoint.pt'):
                candidate = os.path.join(path, name)
                if os.path.exists(candidate):
                    return candidate
        return path

    @staticmethod
    def _read_checkpoint(path: str):
        path = PointSAMHeatmapBridge._resolve_checkpoint_path(path)
        if not os.path.exists(path):
            cprint(f"[PointSAMHeatmapBridge] "
                   f"checkpoint not found: {path}", "red")
            return None

        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(path)

        ckpt = torch.load(path, map_location='cpu')
        for key in ('state_dict', 'model', 'module'):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
        return ckpt

    @staticmethod
    def _load_prefixed_module(
        module: nn.Module,
        state_dict: dict,
        prefix: str,
        strict: bool,
        ignore_mismatched_sizes: bool = False,
    ):
        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not module_state:
            if strict:
                raise RuntimeError(
                    f"No checkpoint keys matched prefix '{prefix}'"
                )
            cprint(f"[PointSAMHeatmapBridge] "
                   f"skipped {prefix}: no matched keys", "yellow")
            return

        if ignore_mismatched_sizes and not strict:
            current_state = module.state_dict()
            filtered = {}
            skipped = []
            for key, value in module_state.items():
                cur = current_state.get(key)
                if cur is not None and cur.shape != value.shape:
                    skipped.append(
                        (key, tuple(value.shape), tuple(cur.shape))
                    )
                    continue
                filtered[key] = value
            if skipped:
                cprint(f"  Skipped incompatible {prefix} keys: "
                       f"{skipped[:10]}", "yellow")
            module_state = filtered

        result = module.load_state_dict(module_state, strict=strict)
        cprint(
            f"[PointSAMHeatmapBridge] loaded {prefix}"
            f"missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}",
            "yellow" if result.missing_keys or result.unexpected_keys
            else "cyan",
        )

    # =========================================================================
    # Freeze helpers
    # =========================================================================

    def _set_freeze_all(self):
        for module in (self.encoder, self.text_encoder, self.decoder):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(False)
            module.eval()
        if hasattr(self, "pc_proj"):
            for p in self.pc_proj.parameters():
                p.requires_grad_(False)
            self.pc_proj.eval()
        cprint("[PointSAMHeatmapBridge] "
               "froze encoder/text/decoder/pc_proj", "yellow")

    def _set_freeze_uni3d(self):
        """Freeze Uni3D backbone; keep E-SAM adapters trainable."""
        for name, p in self.encoder.named_parameters():
            keep = (
                '.MLP_Adapter.' in name
                or '.Space_Adapter.' in name
            )
            p.requires_grad_(keep)
        # Keep encoder in eval mode so BatchNorm stats don't drift
        self.encoder.eval()
        cprint("[PointSAMHeatmapBridge] "
               "froze Uni3D backbone, kept adapters trainable", "yellow")

    def _set_freeze_out_proj(self):
        for name, p in self.encoder.named_parameters():
            if name.startswith('trans2embed.'):
                p.requires_grad_(False)
        cprint("[PointSAMHeatmapBridge] "
               "froze trans2embed (out_proj)", "yellow")

    # =========================================================================
    # Coordinate normalization (for pc_pe)
    # =========================================================================

    @staticmethod
    def _normalize_centers(centers: torch.Tensor) -> torch.Tensor:
        """Normalize patch centers to [-1, 1] per point cloud.

        Args:
            centers: [B, G, 3] FPS patch center coordinates.

        Returns:
            [B, G, 3] normalized centers.
        """
        # Per-sample max extent, then scale to [-1, 1]
        max_extent = centers.abs().amax(dim=1, keepdim=True).amax(dim=-1, keepdim=True)
        max_extent = max_extent.clamp(min=1e-8)
        return centers / max_extent

    # =========================================================================
    # Encode points (no text) → pc_embeddings + pc_pe
    # =========================================================================

    def _encode_points(
        self,
        pts: torch.Tensor,
        colors: torch.Tensor,
    ):
        """Run the pc_encoder, project, and compute positional encoding.

        Args:
            pts:    [B, N, 3] point coordinates.
            colors: [B, N, 3] point colors/features.

        Returns:
            pc_embeddings: [B, 512, 256]  projected ViT patch tokens.
            patches:       dict with "centers" [B, 512, 3].
            pc_pe:         [B, 512, 256]  random Fourier PE.
            pc_raw:        [B, 512, 1024] raw ViT output (for decoder).
        """
        pc_raw, patches = self.encoder(pts, colors)
        # pc_raw: [B, 512, 1024]
        # patches: {"centers": [B, 512, 3]}

        centers = patches["centers"]                     # [B, 512, 3]
        centers_norm = self._normalize_centers(centers)  # [B, 512, 3] in [-1,1]
        pc_pe = self.pe_layer(centers_norm)              # [B, 512, 256]

        pc_embeddings = self.pc_proj(pc_raw)             # [B, 512, 256]

        return pc_embeddings, patches, pc_pe, pc_raw

    # =========================================================================
    # Text normalization
    # =========================================================================

    @staticmethod
    def _normalize_text_prompts(text, batch_size: int) -> List[List[str]]:
        """Normalize free-form text input into List[List[str]] with Q=1.

        Args:
            text: str | List[str] | List[List[str]]
            batch_size: expected batch size.

        Returns:
            List of length B, each element is a list of exactly 1 string.
        """
        if text is None:
            raise ValueError(
                "PointSAMHeatmapBridge requires text prompts "
                "for heatmap prediction"
            )
        if isinstance(text, str):
            text = [text] * batch_size
        elif isinstance(text, tuple):
            text = list(text)
        elif not isinstance(text, list):
            text = [str(text)] * batch_size

        if len(text) == 1 and batch_size != 1:
            text = text * batch_size
        if len(text) != batch_size:
            raise ValueError(
                f"text batch size mismatch: got {len(text)} prompts "
                f"for batch_size={batch_size}"
            )

        normalized = []
        for item in text:
            if isinstance(item, str):
                normalized.append([item])
            elif isinstance(item, (list, tuple)) and len(item) == 1:
                normalized.append([str(item[0])])
            else:
                raise ValueError(
                    "PointSAM expects exactly one text query per point cloud, "
                    f"got {item!r}"
                )
        return normalized

    # =========================================================================
    # Heatmap prediction
    # =========================================================================

    @staticmethod
    def _downsample_heatmap_to_patches(
        heatmap: torch.Tensor,
        patches: dict,
    ) -> torch.Tensor:
        """Downsample point-level heatmap [B, N, 1] → patch-level [B, G, 1].

        Uses the FPS+kNN grouping indices from the encoder.  For each of the G
        patches, gathers the heatmap values of all points in its kNN group and
        takes the maximum.  Identical logic to PointCloudSAM's
        ``downsample_heatmap_to_patches`` in the original pointsam codebase.

        Args:
            heatmap: [B, N, 1] point-level heatmap (logits or probabilities).
            patches: dict with ``"knn_idx"`` [B, G, K] — each entry is a point
                     index into [0, N).

        Returns:
            [B, G, 1] patch-level heatmap.
        """
        knn_idx = patches["knn_idx"]                     # [B, G, K]
        knn_idx = knn_idx.to(device=heatmap.device, dtype=torch.long)

        B, G, K = knn_idx.shape
        # gather: [B, G, N, 1] → [B, G, K, 1]
        gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, heatmap.shape[-1])
        gathered = torch.gather(
            heatmap.unsqueeze(1).expand(-1, G, -1, -1),   # [B, G, N, 1]
            dim=2,
            index=gather_idx,
        )                                                  # [B, G, K, 1]
        patch_heatmap = gathered.max(dim=2).values         # [B, G, 1]
        return patch_heatmap

    def _predict_heatmap(
        self,
        pts: torch.Tensor,
        colors: torch.Tensor,
        pc_raw: torch.Tensor,
        patches: dict,
        text,
    ) -> torch.Tensor:
        """Run text encoder + heatmap decoder → patch-level heatmap.

        Decoder produces a point-level heatmap [B, 1, N]; we downsample to
        patch-level [B, G, 1] via FPS+kNN max-pooling, matching the token
        count that ACT expects.

        Args:
            pts:     [B, N, 3] original point coords.
            colors:  [B, N, 3] original point features.
            pc_raw:  [B, 512, 1024] raw ViT patch tokens (before projection).
            patches: dict with ``"centers"`` [B, 512, 3] and
                     ``"knn_idx"`` [B, 512, K].
            text:    str | List[str] | List[List[str]].

        Returns:
            heatmap: [B, G, 1] patch-level heatmap (sigmoid if heatmap_as_prob).
        """
        from pc_sam.utils.common import AuxInputs

        text_prompts = self._normalize_text_prompts(text, pts.shape[0])

        # ---- Text encode ----
        try:
            text_tokens = self.text_encoder(
                text_prompts,
                device=pts.device,
                dtype=pc_raw.dtype,
            )
        except TypeError:
            text_tokens = self.text_encoder(text_prompts)
        # text_tokens: [B, T, 1024]

        # ---- Build aux inputs ----
        centers = patches["centers"]                     # [B, G, 3]
        aux_inputs = AuxInputs(
            coords=pts,
            features=colors,
            centers=centers,
            query_coords=pts,  # decode at original point resolution
        )

        # ---- Decode (uses raw 1024-dim tokens) ----
        heatmap_logits = self.decoder(
            pc_embeddings=pc_raw,
            text_tokens=text_tokens,
            aux_inputs=aux_inputs,
        )
        # heatmap_logits: [B, 1, N]

        # Transpose to R3D convention: [B, 1, N] → [B, N, 1]
        heatmap = heatmap_logits.transpose(1, 2).contiguous()  # [B, N, 1]

        # ---- Downsample: N points → G patches (max-pool over kNN groups) ----
        heatmap = self._downsample_heatmap_to_patches(heatmap, patches)  # [B, G, 1]

        if self.heatmap_as_prob:
            heatmap = heatmap.sigmoid()

        return heatmap

    # =========================================================================
    # Forward — R3D interface
    # =========================================================================

    def forward(
        self,
        pcd: torch.Tensor,
        eval: bool = False,
        text=None,
        return_heatmap: bool = False,
    ):
        """R3D-compatible forward pass.

        Args:
            pcd:  [B, N, C] point cloud (C=3 for XYZ, C≥6 for XYZ+RGB).
            eval: if True, run all components in eval mode.
            text: text prompts for heatmap prediction.
            return_heatmap: if True, also return patch-level heatmap.

        Returns:
            Without heatmap: (pc_embeddings, pc_pe)
                pc_embeddings: [B, 512, 256]
                pc_pe:         [B, 512, 256]

            With heatmap:    (pc_embeddings, pc_pe, heatmap)
                heatmap: [B, 512, 1] patch-level sigmoid/logit heatmap.
        """
        # ---- Eval mode ----
        # DP3 calls model.train() every epoch, which recursively switches this
        # encoder back to train mode. Re-assert eval mode for a frozen Uni3D so
        # its BatchNorm statistics and stochastic backbone stay fixed; adapter
        # parameters still receive gradients in eval mode.
        if self.freeze_components or self.freeze_uni3d_backbone or eval:
            self.encoder.eval()
        if self.text_encoder is not None and (
            self.freeze_components
            or bool(getattr(self.text_encoder, 'freeze_clip', False))
            or eval
        ):
            self.text_encoder.eval()
        if self.decoder is not None and (self.freeze_components or eval):
            self.decoder.eval()

        # ---- Extract XYZ + colors ----
        pts = pcd[..., :3].contiguous()
        if pcd.shape[-1] >= 6:
            colors = pcd[..., 3:6].contiguous()
        else:
            colors = torch.zeros_like(pts)

        # ---- Encode point cloud ----
        pc_embeddings, patches, pc_pe, pc_raw = self._encode_points(pts, colors)
        # pc_embeddings: [B, 512, 256]  (projected)
        # pc_pe:         [B, 512, 256]
        # pc_raw:        [B, 512, 1024] (for decoder)

        # ---- Optional heatmap ----
        if return_heatmap:
            heatmap = self._predict_heatmap(
                pts, colors, pc_raw, patches, text,
            )
            return pc_embeddings, pc_pe, heatmap

        return pc_embeddings, pc_pe


class PointSAMHeatmapBridgeV4(PointSAMHeatmapBridge):
    """V4 alias for the pointsam_last heatmap bridge used by R3D configs."""

    pass
