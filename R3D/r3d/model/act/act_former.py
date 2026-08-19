import torch
import torch.nn as nn

from r3d.model.common.heatmap_utils import build_pseudo_heatmap_from_text

from .blocks import HeatmapCrossAttentionBlock, MetaSelfAttentionBlock
from .spatial_scope import build_spatial_scope_bias


class AffordanceGuidedCompactorTransformer(nn.Module):
    """Compress dense point patch tokens into compact ACT tokens."""

    def __init__(
        self,
        token_dim,
        pe_dim,
        num_queries=16,
        num_heads=8,
        ffn_ratio=4.0,
        dropout=0.0,
        heatmap_mode="none",
        heatmap_intervention="none",
        heatmap_intervention_seed=0,
        heatmap_intervention_roll=1,
        heatmap_gamma=1.0,
        log_bias_lambda=1.0,
        eps=1e-6,
        use_pc_pe_in_input=True,
        pe_scale_init=1.0,
        use_pseudo_heatmap=False,
        pseudo_text_pool="cls",
        pseudo_heatmap_normalize="minmax",
        return_debug=False,
        drop_cls_token=True,
        competitive_cross1=False,
        competitive_cross2=False,
        competitive_temperature=1.0,
        mq_spatial_scope=None,
        mq_common_gradient_deflation=None,
    ):
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if pe_dim <= 0:
            raise ValueError(f"pe_dim must be positive, got {pe_dim}")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.token_dim = token_dim
        self.pe_dim = pe_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.heatmap_mode = heatmap_mode
        valid_interventions = {"none", "uniform", "shuffle", "spatial_roll", "inverse"}
        if heatmap_intervention not in valid_interventions:
            raise ValueError(
                "heatmap_intervention must be one of "
                f"{sorted(valid_interventions)}, got {heatmap_intervention!r}"
            )
        self.heatmap_intervention = heatmap_intervention
        self.heatmap_intervention_seed = int(heatmap_intervention_seed)
        self.heatmap_intervention_roll = int(heatmap_intervention_roll)
        self.use_pc_pe_in_input = use_pc_pe_in_input
        self.use_pseudo_heatmap = use_pseudo_heatmap
        self.pseudo_text_pool = pseudo_text_pool
        self.pseudo_heatmap_normalize = pseudo_heatmap_normalize
        self.return_debug = return_debug
        self.drop_cls_token = drop_cls_token
        scope = dict(mq_spatial_scope or {})
        deflation = dict(mq_common_gradient_deflation or {})
        self.common_gradient_deflation_enabled = bool(deflation.get("enabled", False))
        self.common_gradient_deflation_location = str(
            deflation.get("location", "cross2_message")
        )
        self.common_gradient_deflation_alpha = float(deflation.get("alpha", 0.0))
        if self.common_gradient_deflation_location != "cross2_message":
            raise ValueError(
                "Only location='cross2_message' is supported for MQ common gradient "
                f"deflation, got {self.common_gradient_deflation_location!r}"
            )
        if not 0.0 <= self.common_gradient_deflation_alpha <= 1.0:
            raise ValueError("MQ common gradient deflation alpha must be in [0,1]")
        cross2_deflation_alpha = (
            self.common_gradient_deflation_alpha
            if self.common_gradient_deflation_enabled
            else 0.0
        )
        self.spatial_scope_enabled = bool(scope.get("enabled", False))
        self.spatial_scope_mode = str(scope.get("mode", "absolute"))
        self.spatial_scope_strength = float(scope.get("strength", 0.0))
        self.spatial_scope_relative_rho = float(scope.get("relative_rho", 0.0))
        self.spatial_scope_sigmas = tuple(scope.get("sigmas", [0.20, 0.45, 0.90, None]))
        self.spatial_scope_detach_centroid = bool(scope.get("detach_cross1_centroid", True))
        if self.spatial_scope_enabled and not competitive_cross2:
            raise ValueError("Spatial multi-scope requires competitive_cross2=True")
        if self.spatial_scope_enabled and num_queries % 4 != 0:
            raise ValueError("Spatial multi-scope requires num_queries divisible by 4")
        if self.spatial_scope_strength < 0:
            raise ValueError("Spatial multi-scope strength must be non-negative")
        if self.spatial_scope_relative_rho < 0:
            raise ValueError("Spatial multi-scope relative_rho must be non-negative")
        if self.spatial_scope_mode not in ("absolute", "relative_rms"):
            raise ValueError(
                "Spatial multi-scope mode must be 'absolute' or 'relative_rms', got "
                f"{self.spatial_scope_mode!r}"
            )

        self.meta_queries = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.trunc_normal_(self.meta_queries, std=0.02)

        if use_pc_pe_in_input:
            self.pc_pe_proj = (
                nn.Identity() if pe_dim == token_dim else nn.Linear(pe_dim, token_dim)
            )
            self.pe_scale = nn.Parameter(
                torch.tensor(float(pe_scale_init), dtype=torch.float32)
            )
        else:
            self.pc_pe_proj = None
            self.register_parameter("pe_scale", None)

        self.self_block1 = MetaSelfAttentionBlock(
            dim=token_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )
        self.cross_block1 = HeatmapCrossAttentionBlock(
            dim=token_dim,
            context_dim=token_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            heatmap_mode=heatmap_mode,
            heatmap_gamma=heatmap_gamma,
            log_bias_lambda=log_bias_lambda,
            eps=eps,
            competitive=competitive_cross1,
            competitive_temperature=competitive_temperature,
        )
        self.self_block2 = MetaSelfAttentionBlock(
            dim=token_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )
        self.cross_block2 = HeatmapCrossAttentionBlock(
            dim=token_dim,
            context_dim=token_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            heatmap_mode=heatmap_mode,
            heatmap_gamma=heatmap_gamma,
            log_bias_lambda=log_bias_lambda,
            eps=eps,
            competitive=competitive_cross2,
            competitive_temperature=competitive_temperature,
            common_gradient_deflation_alpha=cross2_deflation_alpha,
        )

    def _resolve_heatmap_mode(self, heatmap_mode):
        mode = self.heatmap_mode if heatmap_mode is None else heatmap_mode
        if mode not in ("none", "multiply", "log_bias"):
            raise ValueError(
                "heatmap_mode must be one of 'none', 'multiply', or 'log_bias', "
                f"got {mode}"
            )
        return mode

    def _apply_heatmap_intervention(self, heatmap):
        if heatmap is None or self.heatmap_intervention == "none":
            return heatmap
        if self.heatmap_intervention == "uniform":
            return torch.ones_like(heatmap)
        if self.heatmap_intervention == "inverse":
            return 1.0 - heatmap
        if self.heatmap_intervention == "spatial_roll":
            return torch.roll(
                heatmap,
                shifts=self.heatmap_intervention_roll,
                dims=1,
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.heatmap_intervention_seed)
        permutation = torch.randperm(
            heatmap.shape[1],
            generator=generator,
            device="cpu",
        ).to(heatmap.device)
        return heatmap.index_select(1, permutation)

    def _drop_cls_token_if_needed(self, patch_tokens, pc_pe, heatmap):
        if not self.drop_cls_token:
            return patch_tokens, pc_pe, heatmap, {
                "dropped_cls_token": False,
                "original_num_tokens": patch_tokens.shape[1],
                "num_patch_tokens": patch_tokens.shape[1],
            }

        if patch_tokens.shape[1] < 2:
            raise ValueError(
                "drop_cls_token=True requires patch_tokens to contain at least "
                f"2 tokens, got {patch_tokens.shape[1]}"
            )

        original_num_tokens = patch_tokens.shape[1]
        patch_tokens = patch_tokens[:, 1:, :]
        num_patch_tokens = patch_tokens.shape[1]

        if pc_pe.shape[1] == original_num_tokens:
            pc_pe = pc_pe[:, 1:, :]
            pc_pe_alignment = "dropped_cls"
        elif pc_pe.shape[1] == num_patch_tokens:
            pc_pe_alignment = "already_patch_level"
        else:
            raise ValueError(
                "pc_pe token count must match patch_tokens before or after dropping CLS, "
                f"got pc_pe={pc_pe.shape[1]}, original_tokens={original_num_tokens}, "
                f"patch_tokens_after_drop={num_patch_tokens}"
            )

        heatmap_alignment = "none"
        if heatmap is not None:
            if heatmap.shape[1] == original_num_tokens:
                heatmap = heatmap[:, 1:, :]
                heatmap_alignment = "dropped_cls"
            elif heatmap.shape[1] == num_patch_tokens:
                heatmap_alignment = "already_patch_level"
            else:
                raise ValueError(
                    "heatmap token count must match patch_tokens before or after dropping CLS, "
                    f"got heatmap={heatmap.shape[1]}, original_tokens={original_num_tokens}, "
                    f"patch_tokens_after_drop={num_patch_tokens}"
                )

        return patch_tokens, pc_pe, heatmap, {
            "dropped_cls_token": True,
            "original_num_tokens": original_num_tokens,
            "num_patch_tokens": num_patch_tokens,
            "pc_pe_alignment": pc_pe_alignment,
            "heatmap_alignment": heatmap_alignment,
        }

    def _validate_inputs(self, patch_tokens, pc_pe, heatmap):
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must have shape [B, N, D], got {tuple(patch_tokens.shape)}"
            )
        if pc_pe.ndim != 3:
            raise ValueError(f"pc_pe must have shape [B, N, Dpe], got {tuple(pc_pe.shape)}")
        if patch_tokens.shape[0] != pc_pe.shape[0]:
            raise ValueError(
                "patch_tokens and pc_pe must share the same batch size, got "
                f"{patch_tokens.shape[0]} and {pc_pe.shape[0]}"
            )
        if patch_tokens.shape[1] != pc_pe.shape[1]:
            raise ValueError(
                "patch_tokens and pc_pe must share the same token count, got "
                f"{patch_tokens.shape[1]} and {pc_pe.shape[1]}"
            )
        if patch_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"patch_tokens last dimension must be {self.token_dim}, "
                f"got {patch_tokens.shape[-1]}"
            )
        if pc_pe.shape[-1] != self.pe_dim:
            raise ValueError(
                f"pc_pe last dimension must be {self.pe_dim}, got {pc_pe.shape[-1]}"
            )
        if heatmap is not None:
            if heatmap.ndim != 3:
                raise ValueError(
                    f"heatmap must have shape [B, N, 1], got {tuple(heatmap.shape)}"
                )
            if heatmap.shape[0] != patch_tokens.shape[0]:
                raise ValueError(
                    "heatmap and patch_tokens must share the same batch size, got "
                    f"{heatmap.shape[0]} and {patch_tokens.shape[0]}"
                )
            if heatmap.shape[1] != patch_tokens.shape[1]:
                raise ValueError(
                    "heatmap and patch_tokens must share the same token count, got "
                    f"{heatmap.shape[1]} and {patch_tokens.shape[1]}"
                )
            if heatmap.shape[2] != 1:
                raise ValueError(f"heatmap last dimension must be 1, got {heatmap.shape[2]}")

    def _build_heatmap(self, patch_tokens, text_tokens, heatmap):
        if heatmap is not None:
            return heatmap, "provided"
        if not self.use_pseudo_heatmap:
            return None, "none"
        if text_tokens is None:
            return None, "missing_text"
        return build_pseudo_heatmap_from_text(
            patch_tokens,
            text_tokens,
            text_pool=self.pseudo_text_pool,
            normalize=self.pseudo_heatmap_normalize,
        ), "pseudo"

    def _add_pc_pe_to_input(self, patch_tokens, pc_pe):
        if not self.use_pc_pe_in_input:
            return patch_tokens
        pc_pe_for_input = self.pc_pe_proj(pc_pe)
        return patch_tokens + self.pe_scale.to(dtype=patch_tokens.dtype) * pc_pe_for_input

    def forward(
        self, patch_tokens, pc_pe, heatmap=None, text_tokens=None,
        heatmap_mode=None, point_centers=None,
    ):
        """
        Args:
            patch_tokens: Tensor with shape [B, N, token_dim].
            pc_pe: Tensor with shape [B, N, pe_dim].
            heatmap: Optional patch-level heatmap with shape [B, N, 1].
            text_tokens: Optional text tokens for pseudo heatmap, shape [B, M_text, token_dim].

        Returns:
            compact_tokens: Tensor with shape [B, num_queries, token_dim].
            compact_pc_pe: Tensor with shape [B, num_queries, pe_dim].
            debug: Dict with optional detached debug tensors.
        """
        mode = self._resolve_heatmap_mode(heatmap_mode)
        patch_tokens, pc_pe, heatmap, cls_debug = self._drop_cls_token_if_needed(
            patch_tokens,
            pc_pe,
            heatmap,
        )
        if point_centers is not None:
            if point_centers.shape[1] == cls_debug["original_num_tokens"]:
                if cls_debug["dropped_cls_token"]:
                    point_centers = point_centers[:, 1:, :]
            elif point_centers.shape[1] != cls_debug["num_patch_tokens"]:
                raise ValueError(
                    "point_centers token count must match ACT tokens before or after CLS drop"
                )
        heatmap, heatmap_source = self._build_heatmap(patch_tokens, text_tokens, heatmap)
        self._validate_inputs(patch_tokens, pc_pe, heatmap)
        heatmap = self._apply_heatmap_intervention(heatmap)

        point_tokens = self._add_pc_pe_to_input(patch_tokens, pc_pe)
        batch_size = patch_tokens.shape[0]
        queries = self.meta_queries.unsqueeze(0).expand(batch_size, -1, -1)

        queries = self.self_block1(queries)
        queries, attn1 = self.cross_block1(
            queries,
            point_tokens,
            heatmap=heatmap,
            heatmap_mode=mode,
        )
        queries = self.self_block2(queries)
        scope_bias = None
        scope_debug = None
        if self.spatial_scope_enabled:
            if point_centers is None:
                raise ValueError("Spatial multi-scope requires true point_centers [B,N,3]")
            if point_centers.shape[:2] != patch_tokens.shape[:2]:
                raise ValueError("point_centers must align with ACT patch tokens")
            scope_bias, scope_debug = build_spatial_scope_bias(
                attn1,
                point_centers,
                self.spatial_scope_sigmas,
                detach_centroid=self.spatial_scope_detach_centroid,
                eps=1e-6,
            )
        cross2_result = self.cross_block2(
            queries,
            point_tokens,
            heatmap=heatmap,
            heatmap_mode=mode,
            attention_bias=scope_bias,
            attention_bias_mode=self.spatial_scope_mode,
            attention_bias_strength=self.spatial_scope_strength,
            attention_bias_relative_rho=self.spatial_scope_relative_rho,
            return_score_debug=self.return_debug and self.spatial_scope_enabled,
        )
        compact_tokens, attn2 = cross2_result[:2]

        compact_attn = attn2.mean(dim=1)
        compact_pc_pe = torch.matmul(compact_attn, pc_pe)

        debug = {}
        if self.return_debug:
            debug = {
                "heatmap_mode": mode,
                "heatmap_source": heatmap_source,
                "heatmap_intervention": self.heatmap_intervention,
                "used_heatmap": heatmap is not None and mode != "none",
                "attn1": attn1.detach(),
                "attn2": attn2.detach(),
                "compact_attn": compact_attn.detach(),
            }
            debug.update(cls_debug)
            if heatmap is not None:
                debug["heatmap"] = heatmap.detach()
            if scope_debug is not None:
                debug["spatial_scope"] = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in scope_debug.items()
                }
                debug["spatial_scope"]["raw_centered_bias"] = scope_bias.detach()
                debug["spatial_scope"]["bias"] = cross2_result[2][
                    "applied_attention_bias"
                ]
                debug["spatial_scope"].update(cross2_result[2])

        return compact_tokens, compact_pc_pe, debug
