from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from r3d.model.common.normalizer import LinearNormalizer
from r3d.policy.base_policy import BasePolicy
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.diffusion.mask_generator import LowdimMaskGenerator
from r3d.common.pytorch_util import dict_apply
from r3d.common.model_util import print_params
from r3d.model.common.heatmap_utils import build_pseudo_heatmap_from_text
from r3d.model.vision.pointnet_extractor import DP3Encoder
from r3d.model.text import CLIPTextEncoder
from r3d.model.text.act_text_align_head import ACTTextAlignHead
from r3d.model.act import AffordanceGuidedCompactorTransformer
from r3d.model.flow_matching import (
    compute_consistency_flow_matching_loss,
    flow_ode_sample,
)
from r3d.model.losses import add_mq_diversity_loss

class DP3(BasePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            fps_random_config=None,
            transformer_config=None,
            use_act=False,
            act_config=None,
            use_target_ee=False,
            cat_on_token=False,
            # text module parameters
            use_text: bool = False,
            use_text_for_global_cond: bool = True,
            text_json_path: Optional[str] = None,
            task_name: Optional[str] = None,
            clip_model_name: str = "openai/clip-vit-base-patch32",
            text_feat_dim: int = 64,
            freeze_clip: bool = True,
            strict_text_lookup: bool = False,
            # ACTTextAlignHead — training-only auxiliary supervision
            use_act_text_align: bool = False,
            act_text_align_config: Optional[Dict] = None,
            # training-only MQ diversity auxiliary objectives
            mq_diversity: Optional[Dict] = None,
            # action generation objective
            generation_type: str = "diffusion",
            flow_matching: Optional[Dict] = None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        legacy_dense_reader = kwargs.pop("dense_residual_reader", None)
        if legacy_dense_reader and bool(legacy_dense_reader.get("enabled", False)):
            raise ValueError(
                "Enabled dense_residual_reader checkpoints are no longer supported"
            )

        self.use_target_ee = use_target_ee

        self.cat_on_token = cat_on_token

        cprint(f"[Diffusion] cat on token: {self.cat_on_token}", "green")

        self.condition_type = condition_type

        cprint(f"[Diffusion] condition_type: {self.condition_type}", "green")

        self.generation_type = generation_type
        if self.generation_type not in ("diffusion", "flow_matching"):
            raise ValueError(
                "generation_type must be either 'diffusion' or 'flow_matching', "
                f"got {self.generation_type}"
            )
        cprint(f"[DP3] generation_type: {self.generation_type}", "green")

        flow_matching = flow_matching or {}
        default_flow_time_scale = noise_scheduler.config.num_train_timesteps - 1
        self.flow_eps = flow_matching.get("eps", 1e-2)
        self.flow_initial_noise_scale = float(
            flow_matching.get("initial_noise_scale", 1.0)
        )
        if not math.isfinite(self.flow_initial_noise_scale) or self.flow_initial_noise_scale < 0:
            raise ValueError(
                "flow_matching.initial_noise_scale must be finite and non-negative"
            )
        self.flow_rollout_endpoint_weight = float(
            flow_matching.get("rollout_endpoint_weight", 0.0)
        )
        self.flow_rollout_endpoint_final_weight = float(
            flow_matching.get(
                "rollout_endpoint_final_weight",
                self.flow_rollout_endpoint_weight,
            )
        )
        self.flow_rollout_endpoint_ramp_start_epoch = int(
            flow_matching.get("rollout_endpoint_ramp_start_epoch", 0)
        )
        self.flow_rollout_endpoint_ramp_end_epoch = int(
            flow_matching.get("rollout_endpoint_ramp_end_epoch", 0)
        )
        self.flow_rollout_endpoint_batch_size = int(
            flow_matching.get("rollout_endpoint_batch_size", 4)
        )
        self.flow_rollout_endpoint_num_steps = int(
            flow_matching.get("rollout_endpoint_num_steps", 4)
        )
        if self.flow_rollout_endpoint_weight < 0:
            raise ValueError("rollout_endpoint_weight must be non-negative")
        if self.flow_rollout_endpoint_final_weight < 0:
            raise ValueError("rollout_endpoint_final_weight must be non-negative")
        if self.flow_rollout_endpoint_ramp_end_epoch < self.flow_rollout_endpoint_ramp_start_epoch:
            raise ValueError(
                "rollout_endpoint_ramp_end_epoch must be >= ramp_start_epoch"
            )
        if self.flow_rollout_endpoint_batch_size <= 0:
            raise ValueError("rollout_endpoint_batch_size must be positive")
        if self.flow_rollout_endpoint_num_steps <= 0:
            raise ValueError("rollout_endpoint_num_steps must be positive")
        self.flow_delta = flow_matching.get("delta", 1e-2)
        self.flow_num_segments = flow_matching.get("num_segments", 2)
        self.flow_boundary = flow_matching.get("boundary", 1)
        self.flow_alpha = flow_matching.get("alpha", 1e-5)
        self.flow_consistency_weight = flow_matching.get("consistency_weight", 1.0)
        self.flow_consistency_schedule = flow_matching.get(
            "consistency_schedule", "constant"
        )
        self.flow_consistency_final_weight = flow_matching.get(
            "consistency_final_weight", self.flow_consistency_weight
        )
        self.flow_consistency_ramp_start_epoch = flow_matching.get(
            "consistency_ramp_start_epoch", 0
        )
        self.flow_consistency_ramp_end_epoch = flow_matching.get(
            "consistency_ramp_end_epoch", 0
        )
        self.flow_consistency_schedule_power = float(
            flow_matching.get("consistency_schedule_power", 1.0)
        )
        self.flow_adaptive_consistency_mode = flow_matching.get(
            "adaptive_consistency_mode", "off"
        )
        self.flow_adaptive_target_ratio = float(
            flow_matching.get("adaptive_target_ratio", 0.12)
        )
        self.flow_adaptive_ratio_min = float(
            flow_matching.get("adaptive_ratio_min", 0.08)
        )
        self.flow_adaptive_ratio_max = float(
            flow_matching.get("adaptive_ratio_max", 0.20)
        )
        self.flow_adaptive_cosine_increase_min = float(
            flow_matching.get("adaptive_cosine_increase_min", 0.25)
        )
        self.flow_adaptive_cosine_hold_min = float(
            flow_matching.get("adaptive_cosine_hold_min", 0.10)
        )
        self.flow_adaptive_warmup_epochs = int(
            flow_matching.get("adaptive_warmup_epochs", 200)
        )
        self.flow_adaptive_measure_interval = int(
            flow_matching.get("adaptive_measure_interval", 10)
        )
        self.flow_adaptive_update_interval = int(
            flow_matching.get("adaptive_update_interval", 50)
        )
        self.flow_adaptive_ema_decay = float(
            flow_matching.get("adaptive_ema_decay", 0.9)
        )
        self.flow_adaptive_update_rate = float(
            flow_matching.get("adaptive_update_rate", 0.1)
        )
        self.flow_adaptive_beta_min = float(
            flow_matching.get("adaptive_beta_min", 1.0)
        )
        self.flow_adaptive_beta_max = float(
            flow_matching.get("adaptive_beta_max", 100.0)
        )
        self.flow_adaptive_max_relative_change = float(
            flow_matching.get("adaptive_max_relative_change", 0.2)
        )
        self.flow_adaptive_conflict_protection = bool(
            flow_matching.get("adaptive_conflict_protection", True)
        )
        self.flow_couple_alpha_to_consistency = bool(
            flow_matching.get("couple_alpha_to_consistency", False)
        )
        self.flow_alpha_to_consistency_ratio = float(
            flow_matching.get("alpha_to_consistency_ratio", 1e-5)
        )
        # S6-audit: per-joint FK Jacobian weighting + dual-arm differential loss
        self._use_joint_fk_weighting = bool(
            flow_matching.get("use_joint_fk_weighting", False)
        )
        self._dual_arm_diff_weight = float(
            flow_matching.get("dual_arm_diff_weight", 0.0)
        )
        if self._use_joint_fk_weighting:
            # FK Jacobian sensitivity (mm/rad) from S6-4, normalized to mean=1, floor=0.05
            _FK_RAW = [300.8, 289.3, 311.5, 94.1, 30.9, 0.0, 1.0,
                       289.2, 296.9, 310.7, 94.0, 30.9, 0.0, 1.0]
            _FK_MEAN = sum(_FK_RAW) / 14.0
            self._flow_dim_weights = tuple(
                max(w / _FK_MEAN, 0.05) for w in _FK_RAW
            )
        else:
            self._flow_dim_weights = None
        self.flow_stop_gradient_target = flow_matching.get(
            "stop_gradient_target", True
        )
        routing_cfg = flow_matching.get("consistency_routing", {}) or {}
        self.flow_routing_enabled = bool(routing_cfg.get("enabled", False))
        self.flow_routing_t_bins = tuple(
            float(value) for value in routing_cfg.get(
                "t_bins", (0.0, 0.25, 0.5, 0.75, 1.0)
            )
        )
        self.flow_routing_warmup_epochs = int(routing_cfg.get("warmup_epochs", 200))
        self.flow_routing_measure_interval = int(routing_cfg.get("measure_interval", 50))
        self.flow_routing_ema_decay = float(routing_cfg.get("ema_decay", 0.8))
        self.flow_routing_update_rate = float(routing_cfg.get("update_rate", 0.1))
        self.flow_routing_multiplier_min = float(routing_cfg.get("multiplier_min", 0.25))
        self.flow_routing_multiplier_max = float(routing_cfg.get("multiplier_max", 2.0))
        self.flow_routing_normalize_mean = bool(
            routing_cfg.get("normalize_mean_multiplier", True)
        )
        self.flow_routing_nfe = int(routing_cfg.get("nfe", 4))
        self.flow_routing_val_batches = int(routing_cfg.get("val_batches", 8))
        self.flow_routing_val_batch_size = int(routing_cfg.get("val_batch_size", 2))
        self.flow_routing_noise_seeds = tuple(
            int(value) for value in routing_cfg.get(
                "noise_seeds", (1101, 1102, 1103, 1104)
            )
        )
        if len(self.flow_routing_t_bins) < 2 or any(
            left >= right
            for left, right in zip(self.flow_routing_t_bins, self.flow_routing_t_bins[1:])
        ):
            raise ValueError("consistency_routing.t_bins must be strictly increasing")
        if not 0 <= self.flow_routing_ema_decay < 1:
            raise ValueError("consistency_routing.ema_decay must be in [0, 1)")
        if not 0 < self.flow_routing_update_rate <= 1:
            raise ValueError("consistency_routing.update_rate must be in (0, 1]")
        if not 0 < self.flow_routing_multiplier_min <= self.flow_routing_multiplier_max:
            raise ValueError("consistency_routing multiplier bounds are invalid")
        if not isinstance(self.flow_stop_gradient_target, bool):
            raise TypeError("flow_matching.stop_gradient_target must be boolean")
        if self.flow_consistency_weight < 0:
            raise ValueError("flow_matching.consistency_weight must be non-negative")
        if self.flow_consistency_schedule not in ("constant", "linear", "geometric"):
            raise ValueError(
                "flow_matching.consistency_schedule must be constant, linear, "
                f"or geometric, got {self.flow_consistency_schedule}"
            )
        if self.flow_adaptive_consistency_mode not in ("off", "band", "upper_cap"):
            raise ValueError(
                "flow_matching.adaptive_consistency_mode must be off, band, "
                f"or upper_cap, got {self.flow_adaptive_consistency_mode}"
            )
        if not 0 < self.flow_adaptive_ratio_min <= self.flow_adaptive_target_ratio <= self.flow_adaptive_ratio_max:
            raise ValueError(
                "adaptive consistency ratios must satisfy "
                "0 < ratio_min <= target_ratio <= ratio_max"
            )
        if self.flow_adaptive_measure_interval <= 0 or self.flow_adaptive_update_interval <= 0:
            raise ValueError("adaptive measure/update intervals must be positive")
        if not 0 <= self.flow_adaptive_ema_decay < 1:
            raise ValueError("adaptive_ema_decay must be in [0, 1)")
        if self.flow_adaptive_update_rate <= 0:
            raise ValueError("adaptive_update_rate must be positive")
        if not 0 < self.flow_adaptive_beta_min <= self.flow_adaptive_beta_max:
            raise ValueError("adaptive beta bounds must be positive and ordered")
        if not 0 < self.flow_adaptive_max_relative_change < 1:
            raise ValueError("adaptive_max_relative_change must be in (0, 1)")
        if self.flow_consistency_ramp_end_epoch < self.flow_consistency_ramp_start_epoch:
            raise ValueError(
                "consistency_ramp_end_epoch must be >= consistency_ramp_start_epoch"
            )
        if self.flow_consistency_schedule_power <= 0:
            raise ValueError("consistency_schedule_power must be positive")
        if self.flow_consistency_schedule == "geometric" and (
            self.flow_consistency_weight <= 0
            or self.flow_consistency_final_weight <= 0
        ):
            raise ValueError("geometric consistency scheduling requires positive weights")
        self.flow_direct_velocity_weight = flow_matching.get(
            "direct_velocity_weight", 0.0
        )
        self.flow_direct_velocity_schedule = flow_matching.get(
            "direct_velocity_schedule", "constant"
        )
        self.flow_direct_velocity_final_weight = flow_matching.get(
            "direct_velocity_final_weight", self.flow_direct_velocity_weight
        )
        self.flow_direct_velocity_decay_start_epoch = flow_matching.get(
            "direct_velocity_decay_start_epoch", 0
        )
        self.flow_direct_velocity_decay_end_epoch = flow_matching.get(
            "direct_velocity_decay_end_epoch", 0
        )
        self._training_epoch = 0
        self._flow_adaptive_measure_pending = False
        self.register_buffer(
            "_flow_adaptive_beta",
            torch.tensor(float(self.flow_consistency_weight), dtype=torch.float32),
        )
        self.register_buffer("_flow_adaptive_ratio_ema", torch.tensor(float("nan")))
        self.register_buffer("_flow_adaptive_cosine_ema", torch.tensor(float("nan")))
        self.register_buffer("_flow_adaptive_last_measure_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_flow_adaptive_last_update_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_flow_adaptive_update_count", torch.tensor(0, dtype=torch.long))
        num_routing_bins = len(self.flow_routing_t_bins) - 1
        self.register_buffer("_flow_routing_multipliers", torch.ones(num_routing_bins))
        self.register_buffer("_flow_routing_cosine_ema", torch.full((num_routing_bins,), float("nan")))
        self.register_buffer("_flow_routing_ratio_ema", torch.full((num_routing_bins,), float("nan")))
        self.register_buffer("_flow_routing_last_measure_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_flow_routing_update_count", torch.tensor(0, dtype=torch.long))
        if self.flow_direct_velocity_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "flow_matching.direct_velocity_schedule must be constant, linear, "
                f"or cosine, got {self.flow_direct_velocity_schedule}"
            )
        if self.flow_direct_velocity_decay_end_epoch < self.flow_direct_velocity_decay_start_epoch:
            raise ValueError(
                "direct_velocity_decay_end_epoch must be >= decay_start_epoch"
            )
        self.flow_num_inference_steps = flow_matching.get("num_inference_steps", 1)
        self.flow_solver = flow_matching.get("solver", "euler")
        if self.flow_solver not in ("euler", "heun", "rk4"):
            raise ValueError(f"Unsupported flow_matching.solver: {self.flow_solver}")
        self.flow_time_scale = flow_matching.get("time_scale", default_flow_time_scale)

        if self.generation_type == "flow_matching":
            if self.condition_type != "one_way_transformer":
                raise NotImplementedError(
                    "flow_matching is currently only implemented for "
                    "condition_type='one_way_transformer'"
                )
            if not obs_as_global_cond:
                raise NotImplementedError(
                    "flow_matching currently requires obs_as_global_cond=True"
                )
            if self.cat_on_token:
                raise NotImplementedError(
                    "flow_matching currently requires cat_on_token=False"
                )
            if (
                self.flow_adaptive_consistency_mode != "off"
                and self.flow_direct_velocity_weight <= 0
            ):
                raise ValueError(
                    "adaptive consistency weighting requires "
                    "flow_matching.direct_velocity_weight > 0"
                )
            cprint(
                "[FlowMatching] "
                f"eps={self.flow_eps}, delta={self.flow_delta}, "
                f"num_segments={self.flow_num_segments}, boundary={self.flow_boundary}, "
                f"alpha={self.flow_alpha}, "
                f"consistency_weight={self.flow_consistency_weight}, "
                f"consistency_schedule={self.flow_consistency_schedule}, "
                f"consistency_final_weight={self.flow_consistency_final_weight}, "
                f"consistency_ramp_epochs="
                f"{self.flow_consistency_ramp_start_epoch}:"
                f"{self.flow_consistency_ramp_end_epoch}, "
                f"consistency_schedule_power={self.flow_consistency_schedule_power}, "
                f"adaptive_consistency_mode={self.flow_adaptive_consistency_mode}, "
                f"adaptive_ratio="
                f"{self.flow_adaptive_ratio_min}:"
                f"{self.flow_adaptive_target_ratio}:"
                f"{self.flow_adaptive_ratio_max}, "
                f"couple_alpha_to_consistency={self.flow_couple_alpha_to_consistency}, "
                f"stop_gradient_target={self.flow_stop_gradient_target}, "
                f"direct_velocity_weight={self.flow_direct_velocity_weight}, "
                f"direct_velocity_schedule={self.flow_direct_velocity_schedule}, "
                f"direct_velocity_final_weight={self.flow_direct_velocity_final_weight}, "
                f"direct_velocity_decay_epochs="
                f"{self.flow_direct_velocity_decay_start_epoch}:"
                f"{self.flow_direct_velocity_decay_end_epoch}, "
                f"num_inference_steps={self.flow_num_inference_steps}, "
                f"solver={self.flow_solver}, "
                f"time_scale={self.flow_time_scale}",
                "cyan",
            )

        _feature_mode = pointcloud_encoder_cfg.get('feature_mode', None)
        self.pc_encoder_extract_global_feature = _feature_mode != 'pointsam'
        if self.generation_type == "flow_matching" and self.pc_encoder_extract_global_feature:
            raise NotImplementedError(
                "flow_matching currently requires a patch-token point cloud encoder "
                "that returns pc_pe"
            )

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            fps_random_config=fps_random_config,
            cat_on_token=cat_on_token,
        )

        encoder_patch_dim = getattr(obs_encoder.extractor, "embed_dim", None)
        if encoder_patch_dim is None and pointcloud_encoder_cfg is not None:
            encoder_patch_dim = pointcloud_encoder_cfg.get(
                "out_dim", pointcloud_encoder_cfg.get("embed_dim", None)
            )
        if encoder_patch_dim is None:
            encoder_patch_dim = obs_encoder.output_shape()
        self.encoder_patch_dim = int(encoder_patch_dim)

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape() # embed_dim + robot_state_embed_dim = 512
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_text = use_text
        self.use_text_for_global_cond = use_text_for_global_cond
        self.text_feat_dim = text_feat_dim
        self.text_encoder: Optional[CLIPTextEncoder] = None
        if use_text:
            if cat_on_token:
                raise NotImplementedError("Text conditioning is not supported with cat_on_token=True")
            if not obs_as_global_cond:
                raise NotImplementedError("Text conditioning currently requires obs_as_global_cond=True")
            if global_cond_dim is None:
                raise ValueError("Text conditioning requires a global condition dimension")

            cprint("[DP3] Initializing CLIP text encoder", "cyan")
            cprint(f"[DP3]   text_json_path: {text_json_path}", "cyan")
            cprint(f"[DP3]   task_name: {task_name}", "cyan")
            cprint(f"[DP3]   text_feat_dim: {text_feat_dim}", "cyan")
            cprint(f"[DP3]   use_text_for_global_cond: {use_text_for_global_cond}", "cyan")
            self.text_encoder = CLIPTextEncoder(
                clip_model_name=clip_model_name,
                text_json_path=text_json_path,
                task_name=task_name,
                text_feat_dim=text_feat_dim,
                freeze_clip=freeze_clip,
                strict_text_lookup=strict_text_lookup,
            )
            if self.use_text_for_global_cond:
                global_cond_dim += text_feat_dim
                cprint(f"[DP3] Updated global_cond_dim with text: {global_cond_dim}", "cyan")
            else:
                cprint("[DP3] Text encoder enabled for auxiliary uses only; global_cond is unchanged", "cyan")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # Hint: ensure encoder_output_dim matches Uni3D output dimension
        if pointnet_type in ["uni3d", "uni3d_pretrained"]:
            cprint(f"[DP3] Uni3D encoder detected, ensure encoder_output_dim matches Uni3D output dim", "cyan")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
            transformer_config=transformer_config,
            use_target_ee=self.use_target_ee,
            cat_on_token=self.cat_on_token,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.act = None
        self.act_text_proj = None
        self.encoder_heatmap_text_proj = None
        self.use_act = False
        self.last_act_debug = {}
        self.use_pointsam_heatmap = False
        self.use_encoder_clip_heatmap = False
        self.act_pseudo_heatmap_source = "act"
        self.encoder_heatmap_text_pool = "cls"
        self.encoder_heatmap_normalize = "minmax"
        if use_act:
            if self.condition_type != "one_way_transformer":
                cprint(
                    "[ACT] use_act=True ignored because condition_type is not one_way_transformer",
                    "yellow",
                )
            elif self.pc_encoder_extract_global_feature:
                raise ValueError("ACT requires patch-token point cloud features, not global features")
            else:
                act_config = dict(act_config or {})
                act_use_pseudo_heatmap = bool(act_config.get("use_pseudo_heatmap", False))
                self.act_pseudo_heatmap_source = act_config.pop("pseudo_heatmap_source", "act")
                self.encoder_heatmap_text_pool = act_config.get("pseudo_text_pool", "cls")
                self.encoder_heatmap_normalize = act_config.get("pseudo_heatmap_normalize", "minmax")
                if self.act_pseudo_heatmap_source in ("encoder_clip", "clip_similarity", "uni3d_clip"):
                    self.use_encoder_clip_heatmap = act_use_pseudo_heatmap
                    act_config["use_pseudo_heatmap"] = False
                elif self.act_pseudo_heatmap_source != "act":
                    raise ValueError(
                        "act_config.pseudo_heatmap_source must be one of "
                        "'act', 'encoder_clip', 'clip_similarity', or 'uni3d_clip', "
                        f"got {self.act_pseudo_heatmap_source}"
                    )

                act_token_dim = act_config.pop("token_dim", obs_feature_dim)
                if act_token_dim != obs_feature_dim:
                    raise ValueError(
                        "ACT token_dim must match obs_encoder output dim / global_cond_dim, got "
                        f"{act_token_dim} and {obs_feature_dim}"
                    )

                default_pe_dim = None
                if pointcloud_encoder_cfg is not None:
                    default_pe_dim = pointcloud_encoder_cfg.get("embed_dim", None)
                if default_pe_dim is None and transformer_config is not None:
                    default_pe_dim = transformer_config.get("embedding_dim", None)
                if default_pe_dim is None:
                    default_pe_dim = encoder_output_dim

                act_pe_dim = act_config.pop("pe_dim", default_pe_dim)
                self.act = AffordanceGuidedCompactorTransformer(
                    token_dim=act_token_dim,
                    pe_dim=act_pe_dim,
                    **act_config,
                )
                if self.act.use_pseudo_heatmap:
                    if not self.use_text or self.text_encoder is None:
                        cprint(
                            "[ACT] use_pseudo_heatmap=True but text encoder is disabled; "
                            "ACT will run without pseudo heatmap until text is enabled.",
                            "yellow",
                        )
                    elif text_feat_dim == act_token_dim:
                        self.act_text_proj = nn.Identity()
                    else:
                        self.act_text_proj = nn.Linear(text_feat_dim, act_token_dim)
                    if self.act_text_proj is not None:
                        cprint(
                            f"[ACT] pseudo heatmap text projection: {text_feat_dim} -> {act_token_dim}",
                            "cyan",
                        )
                if self.use_encoder_clip_heatmap:
                    if not self.use_text or self.text_encoder is None:
                        raise ValueError(
                            "act_config.pseudo_heatmap_source=encoder_clip requires "
                            "policy.use_text=true"
                        )
                    if text_feat_dim == encoder_patch_dim:
                        self.encoder_heatmap_text_proj = nn.Identity()
                    else:
                        self.encoder_heatmap_text_proj = nn.Linear(text_feat_dim, encoder_patch_dim)
                    cprint(
                        "[ACT] encoder-CLIP pseudo heatmap enabled: "
                        f"text {text_feat_dim} -> patch {encoder_patch_dim}",
                        "cyan",
                    )
                self.use_act = True
                cprint(
                    f"[ACT] enabled: token_dim={act_token_dim}, pe_dim={act_pe_dim}, "
                    f"num_queries={self.act.num_queries}, num_heads={self.act.num_heads}",
                    "green",
                )
        if self.use_act and self.act is not None:
            self.use_pointsam_heatmap = (
                getattr(obs_encoder.extractor, "supports_heatmap", False)
                and getattr(self.act, "heatmap_mode", "none") != "none"
            )
            if self.use_pointsam_heatmap:
                cprint("[ACT] using PointSAM predicted heatmap", "cyan")

        mq_diversity = dict(mq_diversity or {})
        raw_diversity = dict(mq_diversity.get("raw") or {})
        legacy_flow_diversity = dict(mq_diversity.get("flow") or {})
        if bool(legacy_flow_diversity.get("enabled", False)):
            raise ValueError(
                "mq_diversity.flow was removed; only mq_diversity.raw is supported"
            )
        self.mq_diversity_raw_enabled = bool(raw_diversity.get("enabled", False))
        self.mq_diversity_raw_weight = float(raw_diversity.get("weight", 0.0))
        self.mq_diversity_eps = float(mq_diversity.get("eps", 1e-6))
        self.mq_diversity_enabled = self.mq_diversity_raw_enabled
        if self.mq_diversity_raw_weight < 0:
            raise ValueError("MQ raw diversity weight must be non-negative")
        if self.mq_diversity_eps <= 0:
            raise ValueError("mq_diversity.eps must be positive")
        if self.mq_diversity_enabled and not self.use_act:
            raise ValueError("MQ diversity requires use_act=True")
        if self.mq_diversity_enabled:
            cprint(
                "[MQDiversity] enabled: "
                f"raw={self.mq_diversity_raw_enabled} "
                f"(weight={self.mq_diversity_raw_weight})",
                "green",
            )

        # --- ACTTextAlignHead (training-only auxiliary supervision) ---
        self.use_act_text_align = use_act_text_align
        self.act_text_align_head: Optional[ACTTextAlignHead] = None
        self.act_text_align_lambda_max = 0.0
        self.act_text_align_warmup_steps = 1000
        # Temporarily replaced with plain attribute for backward-compat eval.
        # DO NOT COMMIT: revert to register_buffer after eval.
        self.training_step = torch.tensor(0, dtype=torch.long)
        # self.register_buffer(
        #     "training_step", torch.tensor(0, dtype=torch.long), persistent=True,
        # )

        if use_act_text_align:
            if not use_text:
                raise ValueError(
                    "use_act_text_align=True requires use_text=True"
                )
            if not use_act:
                raise ValueError(
                    "use_act_text_align=True requires use_act=True"
                )
            if self.text_encoder is None:
                raise RuntimeError(
                    "text_encoder is None despite use_text=True"
                )

            align_cfg = dict(act_text_align_config or {})
            self.act_text_align_lambda_max = float(
                align_cfg.get("lambda_max", 0.0)
            )
            self.act_text_align_warmup_steps = int(
                align_cfg.get("warmup_steps", 1000)
            )
            align_embed_dim = int(align_cfg.get("embed_dim", act_token_dim))

            self.act_text_align_head = ACTTextAlignHead(
                embed_dim=align_embed_dim,
                text_input_dim=align_cfg.get("text_input_dim", 512),
                mask_ratio=float(align_cfg.get("mask_ratio", 0.3)),
                num_heads=int(align_cfg.get("num_heads", 8)),
                mlp_ratio=float(align_cfg.get("mlp_ratio", 4.0)),
            )

            # --- Diagnostic config ---
            diag_cfg = dict(align_cfg.get("diagnostic", {}))
            self.act_text_align_diag_enabled = bool(
                diag_cfg.get("enabled", True)
            )
            self.act_text_align_diag_every_n = int(
                diag_cfg.get("every_n_steps", 200)
            )
            self.act_text_align_diag_zero_act = bool(
                diag_cfg.get("compute_zero_act", True)
            )
            self.act_text_align_diag_shuffle_act = bool(
                diag_cfg.get("compute_shuffle_act", True)
            )

            cprint(
                "[ACTTextAlignHead] enabled: "
                f"lambda_max={self.act_text_align_lambda_max}, "
                f"warmup_steps={self.act_text_align_warmup_steps}, "
                f"diag_every_n={self.act_text_align_diag_every_n}",
                "green",
            )

        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)

    def _apply_act(
            self, nobs_features, pc_pe, heatmap=None, text_tokens=None):
        compact_features, compact_pc_pe, act_debug = self.act(
            nobs_features,
            pc_pe,
            heatmap=heatmap,
            text_tokens=text_tokens,
        )
        if compact_features.shape[:2] != compact_pc_pe.shape[:2]:
            raise RuntimeError(
                "ACT compact token axis mismatch: compact_features has shape "
                f"{tuple(compact_features.shape)}, compact_pc_pe has shape "
                f"{tuple(compact_pc_pe.shape)}"
            )
        self.last_act_debug = act_debug
        return compact_features, compact_pc_pe

    def _add_mq_diversity_loss(
            self,
            loss: torch.Tensor,
            loss_dict: Dict,
            mq_features_per_frame: Optional[torch.Tensor]):
        if not self.training or not self.mq_diversity_enabled:
            return loss, loss_dict
        loss, diversity_logs = add_mq_diversity_loss(
            loss,
            mq_features_per_frame,
            raw_enabled=self.mq_diversity_raw_enabled,
            raw_weight=self.mq_diversity_raw_weight,
            eps=self.mq_diversity_eps,
        )
        loss_dict.update(diversity_logs)
        return loss, loss_dict

    def set_text_command(self, command):
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is disabled. Set policy.use_text=true first.")
        self.text_encoder.set_command(command)

    def lookup_text(self, command) -> str:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is disabled. Set policy.use_text=true first.")
        return self.text_encoder.lookup_text(command)

    def _compute_text_feature(self, batch_size: int, texts=None, commands=None) -> Optional[torch.Tensor]:
        if self.use_text and self.text_encoder is not None:
            return self.text_encoder.forward(batch_size, texts=texts, commands=commands)
        return None

    def _compute_text_features_full(self, batch_size: int, texts=None, commands=None):
        """一次 CLIP forward 返回 pooled + token-level features + masks.

        Returns:
            pooled_proj:        [B, text_feat_dim] or None
            last_hidden_state:  [B, 77, 512] or None
            attention_mask:     [B, 77] or None
            special_tokens_mask: [B, 77] or None
        """
        if self.use_text and self.text_encoder is not None:
            return self.text_encoder.forward_with_tokens(
                batch_size, texts=texts, commands=commands
            )
        return None, None, None, None

    def _compute_act_text_align_diagnostics(
        self,
        normal_loss: torch.Tensor,
        align_debug: Dict,
        global_cond: torch.Tensor,
        text_tokens_full: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        special_tokens_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute zero-act and shuffle-act diagnostic losses.

        Reuses the same mask_bool from the normal forward pass so that the
        only difference is which ACT tokens serve as K/V.

        Returns:
            dict of diagnostic metrics (empty if diag is disabled or skipped)
        """
        diag_dict: Dict[str, float] = {}

        if not self.act_text_align_diag_enabled:
            return diag_dict
        if self.act_text_align_head is None:
            return diag_dict

        step = self.training_step.item()
        if step % self.act_text_align_diag_every_n != 0:
            return diag_dict

        mask_bool = align_debug.get("mask")
        if mask_bool is None:
            return diag_dict

        B = global_cond.shape[0]

        # --- Zero-act ---
        if self.act_text_align_diag_zero_act:
            zero_act = torch.zeros_like(global_cond)
            zero_loss, _ = self.act_text_align_head(
                act_tokens=zero_act,
                text_tokens=text_tokens_full,
                attention_mask=text_attention_mask,
                special_tokens_mask=special_tokens_mask,
                mask=mask_bool,
            )
            diag_dict["act_text_align_loss_zero_act"] = float(zero_loss.item())
            diag_dict["act_text_align_gap_zero_minus_normal"] = float(
                (zero_loss - normal_loss).item()
            )
            diag_dict["act_text_align_ratio_zero_over_normal"] = float(
                (zero_loss / normal_loss.clamp(min=1e-8)).item()
            )

        # --- Shuffle-act ---
        if self.act_text_align_diag_shuffle_act and B > 1:
            perm = torch.randperm(B, device=global_cond.device)
            shuffle_act = global_cond[perm]
            shuffle_loss, _ = self.act_text_align_head(
                act_tokens=shuffle_act,
                text_tokens=text_tokens_full,
                attention_mask=text_attention_mask,
                special_tokens_mask=special_tokens_mask,
                mask=mask_bool,
            )
            diag_dict["act_text_align_loss_shuffle_act"] = float(shuffle_loss.item())
            diag_dict["act_text_align_gap_shuffle_minus_normal"] = float(
                (shuffle_loss - normal_loss).item()
            )
            diag_dict["act_text_align_ratio_shuffle_over_normal"] = float(
                (shuffle_loss / normal_loss.clamp(min=1e-8)).item()
            )

        return diag_dict

    def _resolve_text_prompts(self, batch_size: int, texts=None, commands=None):
        if self.text_encoder is not None:
            return self.text_encoder._resolve_texts(
                batch_size,
                texts=texts,
                commands=commands,
            )

        if texts is not None:
            if isinstance(texts, str):
                resolved = [texts]
            elif isinstance(texts, tuple):
                resolved = list(texts)
            elif isinstance(texts, list):
                resolved = texts
            else:
                resolved = [str(texts)]
        elif commands is not None:
            if isinstance(commands, str):
                resolved = [commands]
            elif isinstance(commands, tuple):
                resolved = list(commands)
            elif isinstance(commands, list):
                resolved = commands
            else:
                resolved = [str(commands)]
        else:
            resolved = [""]

        resolved = [str(item) for item in resolved]
        if len(resolved) == batch_size:
            return resolved
        if len(resolved) == 1:
            return resolved * batch_size
        raise ValueError(
            f"text batch size mismatch: got {len(resolved)} prompts for batch_size={batch_size}"
        )

    @staticmethod
    def _expand_prompts_for_obs_steps(prompts, n_obs_steps: int):
        return [
            prompt
            for prompt in prompts
            for _ in range(n_obs_steps)
        ]

    def _build_encoder_clip_heatmap(
            self,
            text_feat: Optional[torch.Tensor],
            pc_embedding: torch.Tensor,
            n_obs_steps: int,
            target_dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if not self.use_encoder_clip_heatmap:
            return None
        if text_feat is None or self.encoder_heatmap_text_proj is None:
            return None

        text_tokens = self.encoder_heatmap_text_proj(text_feat)
        if target_dtype is not None:
            text_tokens = text_tokens.to(dtype=target_dtype)
        text_tokens = text_tokens.unsqueeze(1).expand(-1, n_obs_steps, -1)
        text_tokens = text_tokens.reshape(text_tokens.shape[0] * n_obs_steps, 1, -1)
        return build_pseudo_heatmap_from_text(
            pc_embedding,
            text_tokens,
            text_pool=self.encoder_heatmap_text_pool,
            normalize=self.encoder_heatmap_normalize,
        )

    def _build_act_text_tokens(
            self,
            text_feat: Optional[torch.Tensor],
            n_obs_steps: int,
            target_dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if not self.use_act or self.act is None:
            return None
        if not getattr(self.act, "use_pseudo_heatmap", False):
            return None
        if text_feat is None or self.act_text_proj is None:
            return None

        text_tokens = self.act_text_proj(text_feat)
        if target_dtype is not None:
            text_tokens = text_tokens.to(dtype=target_dtype)
        text_tokens = text_tokens.unsqueeze(1).expand(-1, n_obs_steps, -1)
        text_tokens = text_tokens.reshape(text_tokens.shape[0] * n_obs_steps, 1, -1)
        return text_tokens

    def _concat_text_to_global_cond(
            self,
            global_cond: torch.Tensor,
            text_feat: torch.Tensor) -> torch.Tensor:
        text_feat = text_feat.to(device=global_cond.device, dtype=global_cond.dtype)
        if global_cond.dim() == 3:
            text_feat = text_feat.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
        return torch.cat([global_cond, text_feat], dim=-1)

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            pc_pe=None,
            n_obs_steps=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

            model_output = model(sample=trajectory,
                                timestep=t,
                                local_cond=local_cond, global_cond=global_cond, pc_pe=pc_pe,
                                n_obs_steps=n_obs_steps)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

        return trajectory

    def flow_conditional_sample(self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            pc_pe=None,
            n_obs_steps=None,
            generator=None,
            **kwargs
            ):
        sample_n_obs_steps = self.n_obs_steps if n_obs_steps is None else n_obs_steps
        return flow_ode_sample(
            self.model,
            condition_data,
            condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            n_obs_steps=sample_n_obs_steps,
            eps=self.flow_eps,
            num_inference_steps=self.flow_num_inference_steps,
            solver=self.flow_solver,
            time_scale=self.flow_time_scale,
            initial_noise_scale=self.flow_initial_noise_scale,
            generator=generator,
        )


    def predict_action(self, obs_dict, command=None, text=None, task_name=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Clip point cloud to ensure it's within [-1-1e-6, 1+1e-6]
        if 'point_cloud' in nobs:
            nobs['point_cloud'] = torch.clamp(nobs['point_cloud'], min=-1-1e-6, max=1+1e-6)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        pc_pe = None
        if self.obs_as_global_cond:
            text_feat = self._compute_text_feature(
                B,
                texts=text,
                commands=command if command is not None else task_name,
            )
            pointsam_text = None
            if self.use_pointsam_heatmap:
                pointsam_text = self._expand_prompts_for_obs_steps(
                    self._resolve_text_prompts(
                        B,
                        texts=text,
                        commands=command if command is not None else task_name,
                    ),
                    self.n_obs_steps,
                )

            # condition through global feature
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

            if not self.pc_encoder_extract_global_feature:
                if self.use_pointsam_heatmap:
                    nobs_features, pc_pe, heatmap = self.obs_encoder(
                        this_nobs,
                        eval=True,
                        text=pointsam_text,
                        return_heatmap=True,
                    )
                else:
                    if self.use_encoder_clip_heatmap:
                        nobs_features, pc_pe, pc_embedding = self.obs_encoder(
                            this_nobs,
                            eval=True,
                            return_pc_embedding=True,
                        )
                        heatmap = self._build_encoder_clip_heatmap(
                            text_feat,
                            pc_embedding,
                            self.n_obs_steps,
                            target_dtype=pc_embedding.dtype,
                        )
                    else:
                        nobs_features, pc_pe = self.obs_encoder(this_nobs, eval=True)
                        heatmap = None
                if self.use_act:
                    act_text_tokens = self._build_act_text_tokens(
                        text_feat,
                        self.n_obs_steps,
                        target_dtype=nobs_features.dtype,
                    )
                    nobs_features, pc_pe = self._apply_act(
                        nobs_features,
                        pc_pe,
                        heatmap=heatmap,
                        text_tokens=act_text_tokens,
                    )
                num_patches = pc_pe.shape[1]
                num_tokens = nobs_features.shape[1]
            else:
                nobs_features = self.obs_encoder(this_nobs, eval=True)

            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                # treat as a sequence
                if not self.pc_encoder_extract_global_feature:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps * num_tokens, -1)
                    pc_pe = pc_pe.reshape(B, self.n_obs_steps * num_patches, -1)
                else:
                    global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)

            if text_feat is not None and self.use_text_for_global_cond:
                global_cond = self._concat_text_to_global_cond(global_cond, text_feat)

            # Initialize empty action data
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))

            nobs_features = self.obs_encoder(this_nobs, eval=True)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        # run sampling
        if self.generation_type == "flow_matching":
            nsample = self.flow_conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                pc_pe=pc_pe,
                n_obs_steps=self.n_obs_steps,
                **self.kwargs)
        else:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                pc_pe=pc_pe,
                n_obs_steps=self.n_obs_steps,
                **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # If ee auxiliary task is enabled, only return the joint portion for execution
        if self.use_target_ee:
            ee_dim = Da // 2
            joint_dim = Da - ee_dim

            # First half of action dims = joint, second half = ee
            joint_action = action[:, :, :joint_dim]  # (B, Ta, 14)
            ee_action = action[:, :, joint_dim:]     # (B, Ta, 14)

            result = {
                'action': joint_action,           # joint only for execution
                'action_pred': action_pred,       # full prediction (28-dim)
                'ee_pred': ee_action,             # ee prediction (for logging)
            }
            
        else:
            result = {
                'action': action,                 # (B, Ta, 14)
                'action_pred': action_pred,       # (B, T, 14)
            }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_training_progress(self, epoch: int, num_epochs: int):
        self._training_epoch = int(epoch)
        adaptive_enabled = self.flow_adaptive_consistency_mode != "off"
        due = (
            adaptive_enabled
            and self._training_epoch >= self.flow_adaptive_warmup_epochs
            and self._training_epoch % self.flow_adaptive_measure_interval == 0
            and self._training_epoch != int(self._flow_adaptive_last_measure_epoch.item())
        )
        self._flow_adaptive_measure_pending = due

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        # Keep checkpoints created before adaptive beta buffers were introduced loadable.
        for name in (
            "_flow_adaptive_beta",
            "_flow_adaptive_ratio_ema",
            "_flow_adaptive_cosine_ema",
            "_flow_adaptive_last_measure_epoch",
            "_flow_adaptive_last_update_epoch",
            "_flow_adaptive_update_count",
            "_flow_routing_multipliers",
            "_flow_routing_cosine_ema",
            "_flow_routing_ratio_ema",
            "_flow_routing_last_measure_epoch",
            "_flow_routing_update_count",
        ):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def flow_routing_due(self):
        return (
            self.flow_routing_enabled
            and self._training_epoch >= self.flow_routing_warmup_epochs
            and self._training_epoch % self.flow_routing_measure_interval == 0
            and self._training_epoch != int(self._flow_routing_last_measure_epoch.item())
        )

    @torch.no_grad()
    def update_flow_routing(self, cosine_of_mean_gradients, weighted_ratios):
        cosines = torch.as_tensor(
            cosine_of_mean_gradients,
            device=self._flow_routing_multipliers.device,
            dtype=torch.float32,
        )
        ratios = torch.as_tensor(
            weighted_ratios,
            device=self._flow_routing_multipliers.device,
            dtype=torch.float32,
        )
        if cosines.shape != self._flow_routing_multipliers.shape:
            raise ValueError("Router metric shape does not match configured t-bins")
        decay = self.flow_routing_ema_decay
        for target, value in (
            (self._flow_routing_cosine_ema, cosines),
            (self._flow_routing_ratio_ema, ratios),
        ):
            missing = torch.isnan(target)
            target.copy_(torch.where(missing, value, decay * target + (1.0 - decay) * value))

        quality = self._flow_routing_cosine_ema.clamp_min(0.0)
        if float(quality.sum()) > 1e-12:
            target = quality / quality.mean().clamp_min(1e-12)
            rate = self.flow_routing_update_rate
            proposed = (1.0 - rate) * self._flow_routing_multipliers + rate * target
            proposed.clamp_(self.flow_routing_multiplier_min, self.flow_routing_multiplier_max)
            if self.flow_routing_normalize_mean:
                proposed.div_(proposed.mean().clamp_min(1e-12))
                proposed.clamp_(self.flow_routing_multiplier_min, self.flow_routing_multiplier_max)
            self._flow_routing_multipliers.copy_(proposed)
        self._flow_routing_last_measure_epoch.fill_(self._training_epoch)
        self._flow_routing_update_count.add_(1)

    def flow_routing_log(self):
        result = {
            "flow_routing_update_count": int(self._flow_routing_update_count.item()),
        }
        for index in range(len(self.flow_routing_t_bins) - 1):
            result[f"flow_routing_bin{index}_multiplier"] = float(
                self._flow_routing_multipliers[index].item()
            )
            result[f"flow_routing_bin{index}_cosine_ema"] = float(
                self._flow_routing_cosine_ema[index].item()
            )
            result[f"flow_routing_bin{index}_ratio_ema"] = float(
                self._flow_routing_ratio_ema[index].item()
            )
        return result

    @staticmethod
    def _ema_update(current, value, decay):
        if not math.isfinite(current):
            return value
        return decay * current + (1.0 - decay) * value

    def _update_adaptive_consistency(self, loss_dict):
        if "flow_grad_ratio_unweighted" not in loss_dict:
            return

        beta = float(self._flow_adaptive_beta.item())
        direct_weight = float(self._get_direct_velocity_weight())
        ratio = (
            beta * float(loss_dict["flow_grad_ratio_unweighted"])
            / max(direct_weight, 1e-12)
        )
        cosine = float(loss_dict["flow_grad_cosine"])
        ratio_ema = self._ema_update(
            float(self._flow_adaptive_ratio_ema.item()),
            ratio,
            self.flow_adaptive_ema_decay,
        )
        cosine_ema = self._ema_update(
            float(self._flow_adaptive_cosine_ema.item()),
            cosine,
            self.flow_adaptive_ema_decay,
        )
        self._flow_adaptive_ratio_ema.fill_(ratio_ema)
        self._flow_adaptive_cosine_ema.fill_(cosine_ema)
        self._flow_adaptive_last_measure_epoch.fill_(self._training_epoch)
        self._flow_adaptive_measure_pending = False

        action = "measure"
        due_update = (
            self._training_epoch % self.flow_adaptive_update_interval == 0
            and self._training_epoch != int(self._flow_adaptive_last_update_epoch.item())
        )
        if due_update:
            desired_ratio = None
            if self.flow_adaptive_conflict_protection and cosine_ema < 0.0:
                new_beta = max(
                    self.flow_adaptive_beta_min,
                    beta * (1.0 - self.flow_adaptive_max_relative_change),
                )
                self._flow_adaptive_beta.fill_(new_beta)
                action = "decrease_conflict"
            elif ratio_ema > self.flow_adaptive_ratio_max:
                desired_ratio = (
                    self.flow_adaptive_target_ratio
                    if self.flow_adaptive_consistency_mode == "band"
                    else self.flow_adaptive_ratio_max
                )
                action = "decrease_above_cap"
            elif (
                self.flow_adaptive_consistency_mode == "band"
                and ratio_ema < self.flow_adaptive_ratio_min
                and cosine_ema >= self.flow_adaptive_cosine_increase_min
            ):
                desired_ratio = self.flow_adaptive_target_ratio
                action = "increase_below_band"
            elif cosine_ema < self.flow_adaptive_cosine_hold_min:
                action = "hold_low_cosine"
            elif self.flow_adaptive_consistency_mode == "upper_cap":
                nominal_beta = self._get_scheduled_consistency_weight()
                if beta < nominal_beta:
                    new_beta = min(
                        nominal_beta,
                        beta * (1.0 + self.flow_adaptive_max_relative_change),
                    )
                    self._flow_adaptive_beta.fill_(new_beta)
                    action = "follow_schedule_below_cap"
            else:
                action = "hold_in_band"

            if desired_ratio is not None and ratio_ema > 0:
                multiplier = math.exp(
                    self.flow_adaptive_update_rate
                    * math.log(desired_ratio / ratio_ema)
                )
                max_change = self.flow_adaptive_max_relative_change
                multiplier = min(1.0 + max_change, max(1.0 - max_change, multiplier))
                new_beta = min(
                    self.flow_adaptive_beta_max,
                    max(self.flow_adaptive_beta_min, beta * multiplier),
                )
                self._flow_adaptive_beta.fill_(new_beta)
            self._flow_adaptive_last_update_epoch.fill_(self._training_epoch)
            self._flow_adaptive_update_count.add_(1)

        loss_dict.update({
            "flow_adaptive_beta": float(self._flow_adaptive_beta.item()),
            "flow_adaptive_grad_ratio": ratio,
            "flow_adaptive_grad_ratio_ema": ratio_ema,
            "flow_adaptive_grad_cosine_ema": cosine_ema,
            "flow_adaptive_update_count": int(self._flow_adaptive_update_count.item()),
            "flow_adaptive_action": action,
        })

    def _get_direct_velocity_weight(self):
        initial = float(self.flow_direct_velocity_weight)
        final = float(self.flow_direct_velocity_final_weight)
        schedule = self.flow_direct_velocity_schedule
        start = int(self.flow_direct_velocity_decay_start_epoch)
        end = int(self.flow_direct_velocity_decay_end_epoch)
        epoch = int(self._training_epoch)

        if schedule == "constant" or end <= start or epoch <= start:
            return initial
        if epoch >= end:
            return final

        progress = (epoch - start) / float(end - start)
        if schedule == "linear":
            factor = 1.0 - progress
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final + (initial - final) * factor

    def _get_scheduled_consistency_weight(self):
        initial = float(self.flow_consistency_weight)
        final = float(self.flow_consistency_final_weight)
        schedule = self.flow_consistency_schedule
        start = int(self.flow_consistency_ramp_start_epoch)
        end = int(self.flow_consistency_ramp_end_epoch)
        epoch = int(self._training_epoch)

        if schedule == "constant" or end <= start or epoch <= start:
            consistency_weight = initial
        elif epoch >= end:
            consistency_weight = final
        else:
            progress = (epoch - start) / float(end - start)
            progress = progress ** self.flow_consistency_schedule_power
            if schedule == "linear":
                consistency_weight = initial + (final - initial) * progress
            else:
                consistency_weight = math.exp(
                    (1.0 - progress) * math.log(initial)
                    + progress * math.log(final)
                )
        return consistency_weight

    def _get_consistency_weights(self):
        if self.flow_adaptive_consistency_mode != "off":
            consistency_weight = float(self._flow_adaptive_beta.item())
        else:
            consistency_weight = self._get_scheduled_consistency_weight()

        alpha = float(self.flow_alpha)
        if self.flow_couple_alpha_to_consistency:
            alpha = consistency_weight * self.flow_alpha_to_consistency_ratio
        return consistency_weight, alpha

    def _compute_flow_matching_loss(
            self,
            trajectory,
            cond_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            pc_pe=None,
            dim_weights=None,
            ):
        consistency_weight, alpha = self._get_consistency_weights()
        loss, loss_dict = compute_consistency_flow_matching_loss(
            self.model,
            trajectory,
            condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            n_obs_steps=self.n_obs_steps,
            eps=self.flow_eps,
            delta=self.flow_delta,
            num_segments=self.flow_num_segments,
            boundary=self.flow_boundary,
            alpha=alpha,
            consistency_weight=consistency_weight,
            stop_gradient_target=self.flow_stop_gradient_target,
            direct_velocity_weight=self._get_direct_velocity_weight(),
            time_scale=self.flow_time_scale,
            compute_gradient_diagnostics=(
                self.training and self._flow_adaptive_measure_pending
            ),
            consistency_routing_multipliers=(
                self._flow_routing_multipliers if self.flow_routing_enabled else None
            ),
            consistency_routing_t_bins=(
                self.flow_routing_t_bins if self.flow_routing_enabled else None
            ),
            consistency_routing_stratified_t=self.flow_routing_enabled,
            dim_weights=dim_weights,
        )
        if self.flow_adaptive_consistency_mode != "off":
            self._update_adaptive_consistency(loss_dict)
            loss_dict.setdefault(
                "flow_adaptive_beta", float(self._flow_adaptive_beta.item())
            )
            loss_dict.setdefault(
                "flow_adaptive_grad_ratio_ema",
                float(self._flow_adaptive_ratio_ema.item()),
            )
            loss_dict.setdefault(
                "flow_adaptive_grad_cosine_ema",
                float(self._flow_adaptive_cosine_ema.item()),
            )
        if self.flow_routing_enabled:
            loss_dict.update(self.flow_routing_log())
        return loss, loss_dict

    def _get_flow_rollout_endpoint_weight(self):
        start = self.flow_rollout_endpoint_ramp_start_epoch
        end = self.flow_rollout_endpoint_ramp_end_epoch
        if end <= start or self._training_epoch <= start:
            return self.flow_rollout_endpoint_weight
        if self._training_epoch >= end:
            return self.flow_rollout_endpoint_final_weight
        progress = (self._training_epoch - start) / float(end - start)
        return (
            self.flow_rollout_endpoint_weight
            + progress * (
                self.flow_rollout_endpoint_final_weight
                - self.flow_rollout_endpoint_weight
            )
        )

    def _compute_flow_rollout_endpoint_loss(
            self,
            target,
            local_cond=None,
            global_cond=None,
            pc_pe=None,
            ):
        """Differentiate through the deployed Euler rollout toward clean action."""
        batch_size = min(self.flow_rollout_endpoint_batch_size, target.shape[0])
        target = target[:batch_size]

        def take_batch(value):
            return None if value is None else value[:batch_size]

        local_cond = take_batch(local_cond)
        global_cond = take_batch(global_cond)
        pc_pe = take_batch(pc_pe)
        sample = torch.randn_like(target)
        dt = (1.0 - self.flow_eps) / self.flow_rollout_endpoint_num_steps
        for index in range(self.flow_rollout_endpoint_num_steps):
            num_t = self.flow_eps + index * dt
            timestep = torch.full(
                (batch_size,),
                fill_value=num_t * self.flow_time_scale,
                device=sample.device,
                dtype=sample.dtype,
            )
            velocity = self.model(
                sample=sample,
                timestep=timestep,
                local_cond=local_cond,
                global_cond=global_cond,
                pc_pe=pc_pe,
                n_obs_steps=self.n_obs_steps,
            )
            sample = sample + dt * velocity
        return F.mse_loss(sample, target)

    def compute_loss(self, batch):
        # normalize input
        obs_dict = batch['obs']
        nobs = self.normalizer.normalize(obs_dict)

        # Clip point cloud to ensure it's within [-1-1e-6, 1+1e-6]
        if 'point_cloud' in nobs:
            nobs['point_cloud'] = torch.clamp(nobs['point_cloud'], min=-1-1e-6, max=1+1e-6)

        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        align_loss = None
        align_debug = None
        trajectory = nactions
        cond_data = trajectory
        pc_pe = None
        mq_features_per_frame = None

        if self.obs_as_global_cond:
            # Main policy text always uses the canonical dataset instruction.
            # ACTTextAlignHead may use a separate paraphrased align_text below.
            text_tokens_full = None
            text_attention_mask = None
            special_tokens_mask = None
            text_feat = self._compute_text_feature(
                batch_size,
                texts=batch.get('text'),
                commands=batch.get('command', batch.get('task_name')),
            )
            if self.use_act_text_align and self.act_text_align_head is not None:
                align_text = batch.get('align_text')
                if align_text is not None:
                    _, text_tokens_full, text_attention_mask, special_tokens_mask = \
                        self._compute_text_features_full(
                            batch_size,
                            texts=align_text,
                            commands=None,
                        )
            pointsam_text = None
            if self.use_pointsam_heatmap:
                pointsam_text = self._expand_prompts_for_obs_steps(
                    self._resolve_text_prompts(
                        batch_size,
                        texts=batch.get('text'),
                        commands=batch.get('command', batch.get('task_name')),
                    ),
                    self.n_obs_steps,
                )

            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))

            if not self.pc_encoder_extract_global_feature:
                if self.use_pointsam_heatmap:
                    nobs_features, pc_pe, heatmap = self.obs_encoder(
                        this_nobs,
                        text=pointsam_text,
                        return_heatmap=True,
                    )
                else:
                    if self.use_encoder_clip_heatmap:
                        nobs_features, pc_pe, pc_embedding = self.obs_encoder(
                            this_nobs,
                            return_pc_embedding=True,
                        )
                        heatmap = self._build_encoder_clip_heatmap(
                            text_feat,
                            pc_embedding,
                            self.n_obs_steps,
                            target_dtype=pc_embedding.dtype,
                        )
                    else:
                        nobs_features, pc_pe = self.obs_encoder(this_nobs)
                        heatmap = None
                if self.use_act:
                    act_text_tokens = self._build_act_text_tokens(
                        text_feat,
                        self.n_obs_steps,
                        target_dtype=nobs_features.dtype,
                    )
                    nobs_features, pc_pe = self._apply_act(
                        nobs_features,
                        pc_pe,
                        heatmap=heatmap,
                        text_tokens=act_text_tokens,
                    )
                    if self.training and self.mq_diversity_enabled:
                        # Shape [B*T, Q, D]: each frame remains an independent
                        # orthogonality group before temporal concatenation.
                        mq_features_per_frame = nobs_features
                num_patches = pc_pe.shape[1]
                num_tokens = nobs_features.shape[1]
            else:
                nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type or self.condition_type == "one_way_transformer":
                # treat as a sequence
                if not self.pc_encoder_extract_global_feature:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps * num_tokens, -1)
                    pc_pe = pc_pe.reshape(batch_size, self.n_obs_steps * num_patches, -1)
                else:
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

            if text_feat is not None and self.use_text_for_global_cond:
                global_cond = self._concat_text_to_global_cond(global_cond, text_feat)

            # --- ACTTextAlignHead: compute auxiliary alignment loss ---
            align_loss = None
            align_debug = None
            align_diag_dict: Dict[str, float] = {}
            if text_tokens_full is not None and self.act_text_align_head is not None:
                # Skip if text tokens are all zeros (empty text fallback)
                if text_tokens_full.sum() != 0:
                    align_loss, align_debug = self.act_text_align_head(
                        act_tokens=global_cond,
                        text_tokens=text_tokens_full,
                        attention_mask=text_attention_mask,
                        special_tokens_mask=special_tokens_mask,
                    )
                    # --- ACTTextAlignHead diagnostics (zero-act / shuffle-act) ---
                    align_diag_dict = self._compute_act_text_align_diagnostics(
                        normal_loss=align_loss,
                        align_debug=align_debug,
                        global_cond=global_cond,
                        text_tokens_full=text_tokens_full,
                        text_attention_mask=text_attention_mask,
                        special_tokens_mask=special_tokens_mask,
                    )
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))

            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        if self.generation_type == "flow_matching":
            loss, loss_dict = self._compute_flow_matching_loss(
                trajectory=trajectory,
                cond_data=cond_data,
                condition_mask=condition_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                pc_pe=pc_pe,
                dim_weights=self._flow_dim_weights,
            )
            endpoint_weight = self._get_flow_rollout_endpoint_weight()
            if endpoint_weight > 0:
                endpoint_loss = self._compute_flow_rollout_endpoint_loss(
                    target=trajectory,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    pc_pe=pc_pe,
                )
                weighted_endpoint_loss = (
                    endpoint_weight * endpoint_loss
                )
                loss = loss + weighted_endpoint_loss
                loss_dict.update({
                    "flow_rollout_endpoint_loss": float(endpoint_loss.detach().item()),
                    "flow_weighted_rollout_endpoint_loss": float(
                        weighted_endpoint_loss.detach().item()
                    ),
                    "flow_rollout_endpoint_weight": float(
                        endpoint_weight
                    ),
                    "flow_rollout_endpoint_batch_size": int(
                        min(self.flow_rollout_endpoint_batch_size, trajectory.shape[0])
                    ),
                    "flow_rollout_endpoint_num_steps": int(
                        self.flow_rollout_endpoint_num_steps
                    ),
                })
            loss, loss_dict = self._add_mq_diversity_loss(
                loss,
                loss_dict,
                mq_features_per_frame,
            )
            # S6-audit: dual-arm differential diagnostic logging
            if self._dual_arm_diff_weight > 0:
                with torch.no_grad():
                    _act_phys = self.normalizer['action'].unnormalize(
                        trajectory[:, 0, :].detach()
                    )
                    _ap_phys = self.normalizer['agent_pos'].unnormalize(
                        nobs['agent_pos'][:, -1, :].detach()
                    )
                    _dq = _act_phys - _ap_phys
                    _dq_L = _dq[:, :6]; _dq_R = _dq[:, 7:13]
                    _diff = (_dq_L - _dq_R) / 2.0
                    _common = (_dq_L + _dq_R) / 2.0
                    _diff_norm = torch.norm(_diff, dim=-1).mean()
                    _common_norm = torch.norm(_common, dim=-1).mean()
                    loss_dict["dual_arm_expert_diff_norm"] = float(_diff_norm.item())
                    loss_dict["dual_arm_expert_common_norm"] = float(_common_norm.item())
                    loss_dict["dual_arm_diff_fraction"] = float(
                        (_diff_norm / (_common_norm + _diff_norm + 1e-8)).item()
                    )
            if global_cond is not None:
                loss_dict["condition_num_tokens"] = int(global_cond.shape[1]) if global_cond.dim() == 3 else 1
                loss_dict["condition_token_dim"] = int(global_cond.shape[-1])
            if self.use_act and self.last_act_debug:
                loss_dict["act_used_heatmap"] = float(bool(self.last_act_debug.get("used_heatmap", False)))
                loss_dict["act_num_queries"] = int(self.act.num_queries) if self.act is not None else 0
                heatmap = self.last_act_debug.get("heatmap")
                if torch.is_tensor(heatmap):
                    heatmap_float = heatmap.detach().float()
                    loss_dict["act_heatmap_min"] = heatmap_float.min().item()
                    loss_dict["act_heatmap_max"] = heatmap_float.max().item()
                    loss_dict["act_heatmap_mean"] = heatmap_float.mean().item()
                    loss_dict["act_heatmap_std"] = heatmap_float.std(unbiased=False).item()
            # --- ACTTextAlignHead: weight and log ---
            if align_loss is not None and align_debug is not None:
                step = self.training_step.item()
                lambda_text = self.act_text_align_lambda_max * min(
                    1.0, step / max(1, self.act_text_align_warmup_steps)
                )
                if self.training:
                    loss = loss + lambda_text * align_loss
                # Per-step keys from align_debug (scalars only, exclude "mask" tensor)
                _debug_log = {
                    k: float(v) for k, v in align_debug.items()
                    if k != "mask" and not torch.is_tensor(v)
                }
                loss_dict.update({
                    "act_text_align_loss_raw": float(align_loss.item()),
                    "act_text_align_lambda": float(lambda_text),
                    "act_text_align_loss_weighted": float(lambda_text * align_loss.item()),
                    "total_loss": float(loss.item()),
                    "text_mask_ratio": float(align_debug.get("text_mask_ratio_all", 0.0)),
                    "text_mask_ratio_all": float(align_debug.get("text_mask_ratio_all", 0.0)),
                    "text_mask_ratio_valid": float(align_debug.get("text_mask_ratio_valid", 0.0)),
                    "text_valid_tokens_mean": float(align_debug.get("valid_tokens_mean", 0.0)),
                    "text_valid_tokens_min": float(align_debug.get("valid_tokens_min", 0.0)),
                    "text_valid_tokens_max": float(align_debug.get("valid_tokens_max", 0.0)),
                    "text_masked_tokens_mean": float(align_debug.get("masked_tokens_mean", 0.0)),
                    "text_masked_tokens_min": float(align_debug.get("masked_tokens_min", 0.0)),
                    "text_masked_tokens_max": float(align_debug.get("masked_tokens_max", 0.0)),
                    "act_text_attn_mean": float(align_debug.get("act_text_attn_mean", 0.0)),
                    "act_text_attn_max": float(align_debug.get("act_text_attn_max", 0.0)),
                    "act_text_attn_std": float(align_debug.get("act_text_attn_std", 0.0)),
                    "act_text_attn_entropy": float(align_debug.get("act_text_attn_entropy", 0.0)),
                    "act_text_attn_top1_mass": float(align_debug.get("act_text_attn_top1_mass", 0.0)),
                    "act_text_attn_top3_mass": float(align_debug.get("act_text_attn_top3_mass", 0.0)),
                })
                # Diagnostic keys (only present every N steps)
                loss_dict.update(align_diag_dict)
            return loss, loss_dict

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory,
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond,
                        pc_pe=pc_pe,
                        n_obs_steps=self.n_obs_steps)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)

        # If ee auxiliary task is enabled, compute joint and ee losses separately
        if self.use_target_ee:
            ee_dim = self.action_dim // 2
            joint_dim = self.action_dim - ee_dim
            joint_loss = loss[:, :, :joint_dim]  # first half: joint dims
            ee_loss = loss[:, :, joint_dim:]     # second half: ee dims

            joint_loss_mean = reduce(joint_loss, 'b ... -> b (...)', 'mean').mean()
            ee_loss_mean = reduce(ee_loss, 'b ... -> b (...)', 'mean').mean()

            ee_loss_weight = 1
            total_loss = joint_loss_mean + ee_loss_weight * ee_loss_mean
            
            loss_dict = {
                'bc_loss': total_loss.item(),
                'joint_loss': joint_loss_mean.item(),
                'ee_loss': ee_loss_mean.item(),
            }
            
            loss = total_loss

        else:
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
            

            loss_dict = {
                'bc_loss': loss.item(),
            }

        loss, loss_dict = self._add_mq_diversity_loss(
            loss,
            loss_dict,
            mq_features_per_frame,
        )
        if global_cond is not None:
            loss_dict["condition_num_tokens"] = int(global_cond.shape[1]) if global_cond.dim() == 3 else 1
            loss_dict["condition_token_dim"] = int(global_cond.shape[-1])
        if self.use_act and self.last_act_debug:
            loss_dict["act_used_heatmap"] = float(bool(self.last_act_debug.get("used_heatmap", False)))
            loss_dict["act_num_queries"] = int(self.act.num_queries) if self.act is not None else 0
            heatmap = self.last_act_debug.get("heatmap")
            if torch.is_tensor(heatmap):
                heatmap_float = heatmap.detach().float()
                loss_dict["act_heatmap_min"] = heatmap_float.min().item()
                loss_dict["act_heatmap_max"] = heatmap_float.max().item()
                loss_dict["act_heatmap_mean"] = heatmap_float.mean().item()
                loss_dict["act_heatmap_std"] = heatmap_float.std(unbiased=False).item()

        # --- ACTTextAlignHead: weight and log ---
        if align_loss is not None and align_debug is not None:
            step = self.training_step.item()
            lambda_text = self.act_text_align_lambda_max * min(
                1.0, step / max(1, self.act_text_align_warmup_steps)
            )
            if self.training:
                loss = loss + lambda_text * align_loss
            # Per-step keys from align_debug (scalars only, exclude "mask" tensor)
            loss_dict.update({
                "act_text_align_loss_raw": float(align_loss.item()),
                "act_text_align_lambda": float(lambda_text),
                "act_text_align_loss_weighted": float(lambda_text * align_loss.item()),
                "total_loss": float(loss.item()),
                "text_mask_ratio": float(align_debug.get("text_mask_ratio_all", 0.0)),
                "text_mask_ratio_all": float(align_debug.get("text_mask_ratio_all", 0.0)),
                "text_mask_ratio_valid": float(align_debug.get("text_mask_ratio_valid", 0.0)),
                "text_valid_tokens_mean": float(align_debug.get("valid_tokens_mean", 0.0)),
                "text_valid_tokens_min": float(align_debug.get("valid_tokens_min", 0.0)),
                "text_valid_tokens_max": float(align_debug.get("valid_tokens_max", 0.0)),
                "text_masked_tokens_mean": float(align_debug.get("masked_tokens_mean", 0.0)),
                "text_masked_tokens_min": float(align_debug.get("masked_tokens_min", 0.0)),
                "text_masked_tokens_max": float(align_debug.get("masked_tokens_max", 0.0)),
                "act_text_attn_mean": float(align_debug.get("act_text_attn_mean", 0.0)),
                "act_text_attn_max": float(align_debug.get("act_text_attn_max", 0.0)),
                "act_text_attn_std": float(align_debug.get("act_text_attn_std", 0.0)),
                "act_text_attn_entropy": float(align_debug.get("act_text_attn_entropy", 0.0)),
                "act_text_attn_top1_mass": float(align_debug.get("act_text_attn_top1_mass", 0.0)),
                "act_text_attn_top3_mass": float(align_debug.get("act_text_attn_top3_mass", 0.0)),
            })
            # Diagnostic keys (only present every N steps)
            loss_dict.update(align_diag_dict)
        return loss, loss_dict
