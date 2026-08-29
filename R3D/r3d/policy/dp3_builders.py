from __future__ import annotations

from typing import Optional

from termcolor import cprint

from r3d.model.act import AffordanceGuidedCompactorTransformer
from r3d.model.diffusion.diffusion_backbone import ConditionalUnet1D
from r3d.model.text import CLIPTextEncoder
from r3d.model.vision.pointnet_extractor import DP3Encoder


def build_observation_encoder(
    *,
    observation_space,
    crop_shape,
    encoder_output_dim,
    pointcloud_encoder_cfg,
    use_pc_color,
    pointnet_type,
    fps_random_config,
    cat_on_token,
):
    encoder = DP3Encoder(
        observation_space=observation_space,
        img_crop_shape=crop_shape,
        out_channel=encoder_output_dim,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        use_pc_color=use_pc_color,
        pointnet_type=pointnet_type,
        fps_random_config=fps_random_config,
        cat_on_token=cat_on_token,
    )
    patch_dim = getattr(encoder.extractor, "embed_dim", None)
    if patch_dim is None and pointcloud_encoder_cfg is not None:
        patch_dim = pointcloud_encoder_cfg.get(
            "out_dim",
            pointcloud_encoder_cfg.get("embed_dim"),
        )
    return encoder, int(patch_dim or encoder.output_shape())


def build_text_encoder(
    *,
    enabled: bool,
    cat_on_token: bool,
    obs_as_global_cond: bool,
    global_cond_dim: Optional[int],
    clip_model_name: str,
    text_json_path: Optional[str],
    task_name: Optional[str],
    text_feat_dim: int,
    freeze_clip: bool,
    strict_text_lookup: bool,
    use_for_global_cond: bool,
):
    if not enabled:
        return None, global_cond_dim
    if cat_on_token:
        raise NotImplementedError("Text conditioning is not supported with cat_on_token=True")
    if not obs_as_global_cond:
        raise NotImplementedError("Text conditioning currently requires obs_as_global_cond=True")
    if global_cond_dim is None:
        raise ValueError("Text conditioning requires a global condition dimension")

    encoder = CLIPTextEncoder(
        clip_model_name=clip_model_name,
        text_json_path=text_json_path,
        task_name=task_name,
        text_feat_dim=text_feat_dim,
        freeze_clip=freeze_clip,
        strict_text_lookup=strict_text_lookup,
    )
    if use_for_global_cond:
        global_cond_dim += text_feat_dim
    return encoder, global_cond_dim


def build_action_model(
    *,
    input_dim,
    global_cond_dim,
    diffusion_step_embed_dim,
    down_dims,
    kernel_size,
    n_groups,
    condition_type,
    use_down_condition,
    use_mid_condition,
    use_up_condition,
    transformer_config,
    use_target_ee,
    cat_on_token,
):
    return ConditionalUnet1D(
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
        use_target_ee=use_target_ee,
        cat_on_token=cat_on_token,
    )


def build_act_compactor(
    *,
    enabled: bool,
    condition_type: str,
    extracts_global_feature: bool,
    obs_feature_dim: int,
    encoder_output_dim: int,
    pointcloud_encoder_cfg,
    transformer_config,
    act_config,
):
    if not enabled:
        return None
    if condition_type != "one_way_transformer":
        cprint(
            "[ACT] use_act=True ignored because condition_type is not one_way_transformer",
            "yellow",
        )
        return None
    if extracts_global_feature:
        raise ValueError("ACT requires patch-token point cloud features, not global features")

    config = dict(act_config or {})
    token_dim = config.pop("token_dim", obs_feature_dim)
    if token_dim != obs_feature_dim:
        raise ValueError(
            "ACT token_dim must match obs_encoder output dim / global_cond_dim, "
            f"got {token_dim} and {obs_feature_dim}"
        )
    pe_dim = None
    if pointcloud_encoder_cfg is not None:
        pe_dim = pointcloud_encoder_cfg.get("embed_dim")
    if pe_dim is None and transformer_config is not None:
        pe_dim = transformer_config.get("embedding_dim")
    pe_dim = config.pop("pe_dim", pe_dim or encoder_output_dim)
    return AffordanceGuidedCompactorTransformer(
        token_dim=token_dim,
        pe_dim=pe_dim,
        **config,
    )

