from typing import Dict, Optional
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy

from r3d.model.common.normalizer import LinearNormalizer
from r3d.policy.base_policy import BasePolicy
from r3d.model.diffusion.mask_generator import LowdimMaskGenerator
from r3d.model.diffusion.objective import compute_diffusion_policy_loss
from r3d.common.pytorch_util import dict_apply
from r3d.common.model_util import print_params
from r3d.model.flow_matching.config import FlowMatchingConfig
from r3d.policy.conditioning import encode_global_condition
from r3d.policy.dp3_builders import (
    build_action_model,
    build_act_compactor,
    build_observation_encoder,
    build_text_encoder,
)
from r3d.policy.objectives import (
    add_condition_shape_metrics,
    compute_flow_policy_loss,
)
from r3d.policy.sampling import diffusion_sample, flow_sample
from r3d.policy.text_conditioning import (
    compute_text_feature,
    concat_text_to_global_condition,
    expand_prompts_for_obs_steps,
    resolve_text_prompts,
)

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
            # action generation objective
            generation_type: str = "diffusion",
            flow_matching: Optional[Dict] = None,
            # parameters passed to step
            **kwargs):
        super().__init__()

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

        default_flow_time_scale = noise_scheduler.config.num_train_timesteps - 1
        flow_config = FlowMatchingConfig.from_mapping(
            flow_matching,
            default_time_scale=default_flow_time_scale,
        )
        self._flow_config = flow_config
        self.flow_eps = flow_config.eps
        self.flow_initial_noise_scale = flow_config.initial_noise_scale
        self.flow_delta = flow_config.delta
        self.flow_num_segments = flow_config.num_segments
        self.flow_boundary = flow_config.boundary
        self.flow_alpha = flow_config.alpha
        self.flow_consistency_weight = flow_config.consistency_weight
        self.flow_consistency_schedule = flow_config.consistency_schedule
        self.flow_consistency_final_weight = flow_config.consistency_final_weight
        self.flow_consistency_ramp_start_epoch = flow_config.consistency_ramp_start_epoch
        self.flow_consistency_ramp_end_epoch = flow_config.consistency_ramp_end_epoch
        self.flow_consistency_schedule_power = flow_config.consistency_schedule_power
        self.flow_stop_gradient_target = flow_config.stop_gradient_target
        self.flow_direct_velocity_weight = flow_config.direct_velocity_weight
        self._training_epoch = 0
        self.flow_num_inference_steps = flow_config.num_inference_steps
        self.flow_solver = flow_config.solver
        self.flow_time_scale = flow_config.time_scale

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
                f"stop_gradient_target={self.flow_stop_gradient_target}, "
                f"direct_velocity_weight={self.flow_direct_velocity_weight}, "
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

        obs_encoder, self.encoder_patch_dim = build_observation_encoder(
            observation_space=obs_dict,
            crop_shape=crop_shape,
            encoder_output_dim=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            fps_random_config=fps_random_config,
            cat_on_token=cat_on_token,
        )

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
        if use_text:
            cprint("[DP3] Initializing CLIP text encoder", "cyan")
            cprint(f"[DP3]   text_json_path: {text_json_path}", "cyan")
            cprint(f"[DP3]   task_name: {task_name}", "cyan")
            cprint(f"[DP3]   text_feat_dim: {text_feat_dim}", "cyan")
            cprint(f"[DP3]   use_text_for_global_cond: {use_text_for_global_cond}", "cyan")
        self.text_encoder, global_cond_dim = build_text_encoder(
            enabled=use_text,
            cat_on_token=cat_on_token,
            obs_as_global_cond=obs_as_global_cond,
            global_cond_dim=global_cond_dim,
            clip_model_name=clip_model_name,
            text_json_path=text_json_path,
            task_name=task_name,
            text_feat_dim=text_feat_dim,
            freeze_clip=freeze_clip,
            strict_text_lookup=strict_text_lookup,
            use_for_global_cond=self.use_text_for_global_cond,
        )
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        # Hint: ensure encoder_output_dim matches Uni3D output dimension
        if pointnet_type in ["uni3d", "uni3d_pretrained"]:
            cprint(f"[DP3] Uni3D encoder detected, ensure encoder_output_dim matches Uni3D output dim", "cyan")

        model = build_action_model(
            input_dim=input_dim,
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
        self.act = build_act_compactor(
            enabled=use_act,
            condition_type=self.condition_type,
            extracts_global_feature=self.pc_encoder_extract_global_feature,
            obs_feature_dim=obs_feature_dim,
            encoder_output_dim=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            transformer_config=transformer_config,
            act_config=act_config,
        )
        self.use_act = self.act is not None
        self.use_pointsam_heatmap = False
        if self.use_act and self.act is not None:
            self.use_pointsam_heatmap = (
                getattr(obs_encoder.extractor, "supports_heatmap", False)
                and getattr(self.act, "heatmap_mode", "none") != "none"
            )
            if self.use_pointsam_heatmap:
                cprint("[ACT] using PointSAM predicted heatmap", "cyan")

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

    def set_text_command(self, command):
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is disabled. Set policy.use_text=true first.")
        self.text_encoder.set_command(command)

    def lookup_text(self, command) -> str:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is disabled. Set policy.use_text=true first.")
        return self.text_encoder.lookup_text(command)

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
        return diffusion_sample(
            self.model,
            self.noise_scheduler,
            condition_data,
            condition_mask,
            num_inference_steps=self.num_inference_steps,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            n_obs_steps=n_obs_steps,
        )

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
        return flow_sample(
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
            text_feat = compute_text_feature(
                self.text_encoder if self.use_text else None,
                B,
                texts=text,
                commands=command if command is not None else task_name,
            )
            pointsam_text = None
            if self.use_pointsam_heatmap:
                pointsam_text = expand_prompts_for_obs_steps(
                    resolve_text_prompts(
                        self.text_encoder,
                        B,
                        texts=text,
                        commands=command if command is not None else task_name,
                    ),
                    self.n_obs_steps,
                )

            encoded = encode_global_condition(
                nobs=nobs,
                batch_size=B,
                n_obs_steps=self.n_obs_steps,
                obs_encoder=self.obs_encoder,
                act=self.act,
                condition_type=self.condition_type,
                extracts_global_feature=self.pc_encoder_extract_global_feature,
                use_pointsam_heatmap=self.use_pointsam_heatmap,
                pointsam_text=pointsam_text,
                eval_mode=True,
            )
            global_cond = encoded.global_cond
            pc_pe = encoded.pc_pe
            if self.use_text_for_global_cond:
                global_cond = concat_text_to_global_condition(global_cond, text_feat)

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
        trajectory = nactions
        cond_data = trajectory
        pc_pe = None

        if self.obs_as_global_cond:
            text_feat = compute_text_feature(
                self.text_encoder if self.use_text else None,
                batch_size,
                texts=batch.get('text'),
                commands=batch.get('command', batch.get('task_name')),
            )
            pointsam_text = None
            if self.use_pointsam_heatmap:
                pointsam_text = expand_prompts_for_obs_steps(
                    resolve_text_prompts(
                        self.text_encoder,
                        batch_size,
                        texts=batch.get('text'),
                        commands=batch.get('command', batch.get('task_name')),
                    ),
                    self.n_obs_steps,
                )

            encoded = encode_global_condition(
                nobs=nobs,
                batch_size=batch_size,
                n_obs_steps=self.n_obs_steps,
                obs_encoder=self.obs_encoder,
                act=self.act,
                condition_type=self.condition_type,
                extracts_global_feature=self.pc_encoder_extract_global_feature,
                use_pointsam_heatmap=self.use_pointsam_heatmap,
                pointsam_text=pointsam_text,
            )
            global_cond = encoded.global_cond
            pc_pe = encoded.pc_pe
            if self.use_text_for_global_cond:
                global_cond = concat_text_to_global_condition(global_cond, text_feat)
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
            loss, loss_dict = compute_flow_policy_loss(
                self.model,
                trajectory,
                condition_mask,
                n_obs_steps=self.n_obs_steps,
                config=self._flow_config,
                training_epoch=self._training_epoch,
                local_cond=local_cond,
                global_cond=global_cond,
                pc_pe=pc_pe,
            )
            add_condition_shape_metrics(loss_dict, global_cond)
            return loss, loss_dict

        loss, loss_dict = compute_diffusion_policy_loss(
            self.model,
            self.noise_scheduler,
            trajectory,
            condition_mask,
            cond_data,
            action_dim=self.action_dim,
            use_target_ee=self.use_target_ee,
            device=self.device,
            n_obs_steps=self.n_obs_steps,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
        )

        add_condition_shape_metrics(loss_dict, global_cond)
        return loss, loss_dict
