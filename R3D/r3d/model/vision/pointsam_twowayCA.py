"""
PointSAMTwoWayCA — R3D × Point-SAM two-way cross-attention heatmap bridge.

Connects the R3D-Policy pipeline to the pointsam_adapter_twowayCA_contras
text-guided heatmap model. Drop-in replacement for DP3Encoder (same forward()
interface).

Key difference from PointSAMHeatmapBridge:
  - Decoder expects 256-dim inputs (vs 1024-dim), uses TwoWayTransformer.
  - PointCloudSAM has pc_projection (1024→256) and text_projection (1024→256)
    that must be applied BEFORE the decoder.
  - Text encoder supports return_eot=True (EOT token for contrastive features).
  - Encoder returns cls_embedding in patches dict.

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
      → raw [B, 512, 1024], patches {"centers": [B, 512, 3],
                                       "knn_idx": [B, 512, 64],
                                       "cls_embedding": [B, 1024]}
    → pc_proj: Linear(1024 → 256)
      → pc_embeddings [B, 512, 256]  (aligned with pc_pe)
    → PositionEmbeddingRandom(normalized centers)
      → pc_pe [B, 512, 256]
    → [optional] text encoder (return_eot=True) → tokens [B,77,1024] + eot [B,1024]
      → model.text_projection: Linear(1024→256)
      → model.pc_projection: Linear(1024→256) on pc_raw
      → TwoWayTransformer decoder → point heatmap [B, 1, N]
      → _downsample_heatmap_to_patches (kNN max-pool) → patch heatmap [B, 512, 1]
"""

import math
import os
import pathlib
import warnings
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
# PointSAMTwoWayCA
# =============================================================================

class PointSAMTwoWayCA(nn.Module):
    """Standalone PointSAM two-way-CA heatmap adapter bridging R3D and pointsam.

    Parameters
    ----------
    pointsam_root : str or None
        Deprecated compatibility field. PointSAM is imported as an installed
        package and this value is ignored.
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
    fps_randomization : bool
        Randomize input point order during training so deterministic FPS uses
        a different initial point. Disabled by default for old checkpoints.
    random_point_dropout : bool
        Replace a random fraction of input points with an anchor point during
        training, matching the R3D point-dropout augmentation.
    random_point_dropout_max_ratio : float
        Upper bound of the per-sample dropout ratio.
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
        open_clip_python_path: str = None,
        bpe_path: str = None,
        point_feature_projection: str = "pretrained",
        heatmap_override_mode: str = "none",
        green_oracle_min_normalized: float = -0.375,
        green_oracle_margin_normalized: float = 0.195,
        green_oracle_floor: float = 1e-3,
        heatmap_as_prob: bool = True,
        load_checkpoint: bool = False,
        checkpoint_path: str = None,
        freeze: bool = False,
        freeze_uni3d: bool = False,
        freeze_out_proj: bool = False,
        fps_randomization: bool = False,
        random_point_dropout: bool = False,
        random_point_dropout_max_ratio: float = 0.8,
        strict_load: bool = False,
        ignore_mismatched_sizes: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = out_dim or embed_dim          # pc_pe output dim (256)
        self.heatmap_as_prob = heatmap_as_prob
        self.freeze_components = freeze
        self.freeze_uni3d_backbone = bool(freeze_uni3d)
        self.fps_randomization = bool(fps_randomization)
        self.random_point_dropout = bool(random_point_dropout)
        self.random_point_dropout_max_ratio = float(
            random_point_dropout_max_ratio
        )
        if not 0.0 <= self.random_point_dropout_max_ratio < 1.0:
            raise ValueError(
                "random_point_dropout_max_ratio must be in [0, 1), got "
                f"{random_point_dropout_max_ratio}"
            )
        self.supports_heatmap = True
        self.point_feature_projection = str(point_feature_projection).lower()
        self.heatmap_override_mode = str(heatmap_override_mode).lower()
        if self.heatmap_override_mode not in ("none", "green_rgb"):
            raise ValueError(
                "heatmap_override_mode must be 'none' or 'green_rgb', "
                f"got {heatmap_override_mode!r}"
            )
        self.green_oracle_min_normalized = float(green_oracle_min_normalized)
        self.green_oracle_margin_normalized = float(green_oracle_margin_normalized)
        self.green_oracle_floor = float(green_oracle_floor)
        if self.point_feature_projection not in ("pretrained", "policy"):
            raise ValueError(
                "point_feature_projection must be 'pretrained' or 'policy', "
                f"got {point_feature_projection!r}"
            )

        # PointSAM is a normal package dependency in the public release. Keep
        # the legacy argument only so old checkpoint configs remain loadable.
        if pointsam_root:
            warnings.warn(
                "pointsam_root is ignored; install PointSAM in the active "
                "environment instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        pointsam_root = self._installed_pointsam_root()

        # ---- build model from Hydra config ----
        self._build_model(
            pointsam_root=pointsam_root,
            config_dir=heatmap_config_dir,
            config_name=heatmap_config_name,
            open_clip_python_path=open_clip_python_path,
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path if load_checkpoint else None,
            strict=strict_load,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        # ---- pc_pe: random Fourier positional encoding for patch centers ----
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)  # 128 → 256

        # ---- pc_proj: project ViT output 1024 → 256 to align with pc_pe ----
        # This is the BRIDGE's own projection for R3D token output.
        # PointCloudSAM also has self.model.pc_projection (used for decoder).
        # They are separate layers serving different purposes.
        if self.point_feature_projection == "policy":
            self.pc_proj = nn.Linear(
                int(getattr(self.encoder, 'embed_dim', 1024)),
                self.embed_dim,
            )
        else:
            self.pc_proj = None

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
            f"[PointSAMTwoWayCA] ready "
            f"config={heatmap_config_name}, pe_dim={self.embed_dim}, "
            f"raw_dim={getattr(self.encoder, 'embed_dim', '?')}, "
            f"pc_embed_dim={self.pc_embed_dim}, "
            f"point_feature_projection={self.point_feature_projection}, "
            f"fps_randomization={self.fps_randomization}, "
            f"random_point_dropout={self.random_point_dropout}, "
            f"dropout_max={self.random_point_dropout_max_ratio}",
            "cyan",
        )

    # =========================================================================
    # Installed package discovery
    # =========================================================================

    @staticmethod
    def _installed_pointsam_root() -> str:
        try:
            import pc_sam
        except ImportError as exc:
            raise ImportError(
                "PointSAM is not installed. Run `pip install -e ./PointSAM`."
            ) from exc
        return str(pathlib.Path(pc_sam.__file__).resolve().parent.parent)

    # =========================================================================
    # Model construction (Hydra)
    # =========================================================================

    def _build_model(
        self,
        pointsam_root: str,
        config_dir: str,
        config_name: str,
        open_clip_python_path: str,
        bpe_path: str,
        checkpoint_path: str,
        strict: bool,
        ignore_mismatched_sizes: bool,
    ):
        """Instantiate PointCloudSAM from a pointsam Hydra config."""
        default_config_dir = pathlib.Path(pointsam_root) / "configs"
        requested_config_dir = (
            pathlib.Path(config_dir).expanduser() if config_dir else default_config_dir
        )
        if not requested_config_dir.is_dir():
            warnings.warn(
                f"PointSAM config directory {requested_config_dir} is unavailable; "
                f"using installed configs at {default_config_dir}.",
                RuntimeWarning,
                stacklevel=2,
            )
            requested_config_dir = default_config_dir
        config_dir = str(requested_config_dir.resolve())

        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        import hydra

        # Clear any pre-existing Hydra context (R3D train.py also uses Hydra)
        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            global_hydra.clear()

        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=[])
            cfg.model.text_prompt_encoder.open_clip_python_path = ""
            packaged_bpe = pathlib.Path(pointsam_root) / "resources" / "bpe_simple_vocab_16e6.txt.gz"
            requested_bpe = pathlib.Path(bpe_path).expanduser() if bpe_path else packaged_bpe
            if not requested_bpe.is_file():
                requested_bpe = packaged_bpe
            cfg.model.text_prompt_encoder.bpe_path = str(requested_bpe.resolve())
            # The policy bridge only instantiates the model subtree. Resolving
            # the full training config would unnecessarily require dataset-only
            # environment variables such as AFFOGATO_DATA_ROOT during inference.
            OmegaConf.resolve(cfg.model)

        # ---- Instantiate the full PointCloudSAM model ----
        self.model: nn.Module = hydra.utils.instantiate(cfg.model)

        # Extract sub-modules for fine-grained control
        self.encoder = self.model.pc_encoder           # Uni3DPointEncoderForSAM
        self.text_encoder = self.model.text_prompt_encoder  # OpenCLIPTokenTextEncoder
        self.decoder = self.model.mask_decoder          # TokenSimilarityHeatmapDecoder (TwoWayTransformer)

        # ---- Load checkpoint if provided ----
        if checkpoint_path is not None:
            state_dict = self._read_checkpoint(checkpoint_path)
            if state_dict is not None:
                # Core sub-modules (same as old bridge)
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
                # New projection layers in the adapter (may not exist in old checkpoints)
                for prefix, module in (
                    ('pc_projection.', self.model.pc_projection),
                    ('text_projection.', self.model.text_projection),
                ):
                    if any(key.startswith(prefix) for key in state_dict):
                        self._load_prefixed_module(
                            module, state_dict, prefix,
                            strict=False,  # always lenient for optional projections
                            ignore_mismatched_sizes=True,
                        )
            else:
                cprint(
                    "[PointSAMTwoWayCA] "
                    "checkpoint not found, using initialized weights",
                    "yellow",
                )

        cprint(
            f"[PointSAMTwoWayCA] "
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
        path = PointSAMTwoWayCA._resolve_checkpoint_path(path)
        if not os.path.exists(path):
            cprint(f"[PointSAMTwoWayCA] "
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
            cprint(f"[PointSAMTwoWayCA] "
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
            f"[PointSAMTwoWayCA] loaded {prefix}"
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
        # Also freeze PointCloudSAM-level projection layers
        for proj_name in ('pc_projection', 'text_projection'):
            proj = getattr(self.model, proj_name, None)
            if proj is not None:
                for p in proj.parameters():
                    p.requires_grad_(False)
                proj.eval()
        if self.pc_proj is not None:
            for p in self.pc_proj.parameters():
                p.requires_grad_(False)
            self.pc_proj.eval()
        cprint("[PointSAMTwoWayCA] "
               "froze encoder/text/decoder/pc_projection/text_projection/pc_proj", "yellow")

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
        cprint("[PointSAMTwoWayCA] "
               "froze Uni3D backbone, kept adapters trainable", "yellow")

    def _set_freeze_out_proj(self):
        for name, p in self.encoder.named_parameters():
            if name.startswith('trans2embed.'):
                p.requires_grad_(False)
        cprint("[PointSAMTwoWayCA] "
               "froze trans2embed (out_proj)", "yellow")

    # =========================================================================
    # Coordinate normalization (for pc_pe)
    # =========================================================================

    def _augment_point_cloud(self, pcd: torch.Tensor) -> torch.Tensor:
        """Apply R3D point-level augmentations while preserving tensor shape."""
        if not self.training:
            return pcd

        augmented = pcd
        batch_size, num_points, num_channels = augmented.shape

        if self.fps_randomization:
            # The external PointSAM FPS starts deterministically from input
            # index zero. Permuting first gives it a uniformly random original
            # start point. Keep all channels together so XYZ and RGB remain
            # aligned; the heatmap is computed from this augmented cloud later.
            permutation = torch.rand(
                batch_size, num_points, device=augmented.device
            ).argsort(dim=1)
            augmented = torch.gather(
                augmented,
                dim=1,
                index=permutation.unsqueeze(-1).expand(
                    -1, -1, num_channels
                ),
            )

        if self.random_point_dropout and self.random_point_dropout_max_ratio > 0:
            dropout_ratio = torch.rand(
                batch_size, 1, 1, device=augmented.device
            ) * self.random_point_dropout_max_ratio
            dropout_mask = torch.rand(
                batch_size, num_points, 1, device=augmented.device
            ) < dropout_ratio
            anchor = augmented[:, :1, :].expand(-1, num_points, -1)
            augmented = torch.where(dropout_mask, anchor, augmented)

        return augmented

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
            patches:       dict with "centers" [B, 512, 3], "knn_idx", "cls_embedding".
            pc_pe:         [B, 512, 256]  random Fourier PE.
            pc_raw:        [B, 512, 1024] raw ViT output (for decoder, needs pc_projection).
        """
        pc_raw, patches = self.encoder(pts, colors)
        # pc_raw: [B, 512, 1024]
        # patches: {"centers": [B, 512, 3], "knn_idx": [B, 512, 64],
        #           "cls_embedding": [B, 1024]}  ← new in adapter

        centers = patches["centers"]                     # [B, 512, 3]
        centers_norm = self._normalize_centers(centers)  # [B, 512, 3] in [-1,1]
        pc_pe = self.pe_layer(centers_norm)              # [B, 512, 256]

        if self.point_feature_projection == "pretrained":
            pc_embeddings = self.model.pc_projection(pc_raw)
        else:
            pc_embeddings = self.pc_proj(pc_raw)

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
                "PointSAMTwoWayCA requires text prompts "
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
        takes the maximum.

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

        Adapted for the two-way-CA architecture:
          - Text encoder called with return_eot=True (EOT used for
            contrastive features, not required by decoder).
          - PointCloudSAM's pc_projection / text_projection reduce
            1024-dim to 256-dim before the TwoWayTransformer decoder.
          - text_valid_mask is packed into AuxInputs for the decoder.

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

        # ---- Text encode with return_eot (new adapter interface) ----
        text_tokens, text_eot = self.text_encoder(
            text_prompts,
            device=pts.device,
            dtype=pc_raw.dtype,
            return_eot=True,
        )
        # text_tokens: [B, 77, 1024], text_eot: [B, 1024]

        # ---- Project to 256-dim via PointCloudSAM projections ----
        text_valid_mask = text_tokens.abs().sum(dim=-1).gt(0)           # [B, 77]
        pc_projected = self.model.pc_projection(pc_raw)                 # [B, 512, 1024] → [B, 512, 256]
        text_projected = self.model.text_projection(text_tokens)        # [B, 77, 1024] → [B, 77, 256]
        text_projected = text_projected.masked_fill(
            ~text_valid_mask.unsqueeze(-1), 0.0,
        )

        # ---- Build aux inputs (include new adapter fields) ----
        centers = patches["centers"]                     # [B, G, 3]
        aux_inputs = AuxInputs(
            coords=pts,
            features=colors,
            centers=centers,
            query_coords=pts,  # decode at original point resolution
            text_valid_mask=text_valid_mask,
        )

        # ---- Decode (decoder expects 256-dim inputs) ----
        heatmap_logits = self.decoder(
            pc_embeddings=pc_projected,
            text_tokens=text_projected,
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

    def _build_green_rgb_heatmap(
        self,
        colors: torch.Tensor,
        patches: dict,
    ) -> torch.Tensor:
        """Build an experiment-only oracle from green-dominant input points.

        The policy normalizer maps raw RGB approximately from [0, 255] to
        [-1, 1]. The default thresholds correspond to raw G>=80 and a
        G-vs-R/B margin of about 25. Max pooling uses the same PointSAM patch
        groups as the learned heatmap, so ACT receives the usual [B,G,1]
        shape. This path is disabled unless explicitly requested by config.
        """
        red, green, blue = colors.unbind(dim=-1)
        mask = (
            (green >= self.green_oracle_min_normalized)
            & (green >= red + self.green_oracle_margin_normalized)
            & (green >= blue + self.green_oracle_margin_normalized)
        )
        point_heatmap = mask.to(dtype=colors.dtype).unsqueeze(-1)
        patch_heatmap = self._downsample_heatmap_to_patches(point_heatmap, patches)
        return patch_heatmap.clamp_min(self.green_oracle_floor)

    # =========================================================================
    # Forward — R3D interface
    # =========================================================================

    def forward(
        self,
        pcd: torch.Tensor,
        eval: bool = False,
        text=None,
        return_heatmap: bool = False,
        return_centers: bool = False,
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

        # ---- R3D point-level training augmentations ----
        if not eval:
            pcd = self._augment_point_cloud(pcd)

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
        # pc_raw:        [B, 512, 1024] (for decoder, needs pc_projection)

        # ---- Optional heatmap ----
        if return_heatmap:
            if self.heatmap_override_mode == "green_rgb":
                heatmap = self._build_green_rgb_heatmap(colors, patches)
            else:
                heatmap = self._predict_heatmap(
                    pts, colors, pc_raw, patches, text,
                )
            if return_centers:
                return pc_embeddings, pc_pe, heatmap, patches["centers"]
            return pc_embeddings, pc_pe, heatmap

        if return_centers:
            return pc_embeddings, pc_pe, patches["centers"]
        return pc_embeddings, pc_pe


class PointSAMTwoWayCAV4(PointSAMTwoWayCA):
    """V4 alias for the two-way-CA heatmap bridge used by R3D configs."""

    pass
