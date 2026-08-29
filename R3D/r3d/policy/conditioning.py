from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from r3d.common.pytorch_util import dict_apply


@dataclass
class EncodedCondition:
    global_cond: torch.Tensor
    pc_pe: Optional[torch.Tensor]


def apply_act(act, features, pc_pe, *, heatmap=None):
    compact_features, compact_pc_pe = act(features, pc_pe, heatmap=heatmap)
    if compact_features.shape[:2] != compact_pc_pe.shape[:2]:
        raise RuntimeError(
            "ACT compact token axis mismatch: compact_features has shape "
            f"{tuple(compact_features.shape)}, compact_pc_pe has shape "
            f"{tuple(compact_pc_pe.shape)}"
        )
    return compact_features, compact_pc_pe


def encode_global_condition(
    *,
    nobs,
    batch_size: int,
    n_obs_steps: int,
    obs_encoder,
    act,
    condition_type: str,
    extracts_global_feature: bool,
    use_pointsam_heatmap: bool,
    pointsam_text=None,
    eval_mode: bool = False,
) -> EncodedCondition:
    flat_obs = dict_apply(
        nobs,
        lambda value: value[:, :n_obs_steps, ...].reshape(
            -1,
            *value.shape[2:],
        ),
    )
    encoder_kwargs = {"eval": True} if eval_mode else {}

    pc_pe = None
    if extracts_global_feature:
        features = obs_encoder(flat_obs, **encoder_kwargs)
    else:
        if use_pointsam_heatmap:
            features, pc_pe, heatmap = obs_encoder(
                flat_obs,
                text=pointsam_text,
                return_heatmap=True,
                **encoder_kwargs,
            )
        else:
            features, pc_pe = obs_encoder(flat_obs, **encoder_kwargs)
            heatmap = None

        if act is not None:
            features, pc_pe = apply_act(
                act,
                features,
                pc_pe,
                heatmap=heatmap,
            )

    sequence_condition = (
        "cross_attention" in condition_type
        or condition_type == "one_way_transformer"
    )
    if sequence_condition:
        if extracts_global_feature:
            global_cond = features.reshape(batch_size, n_obs_steps, -1)
        else:
            global_cond = features.reshape(
                batch_size,
                n_obs_steps * features.shape[1],
                -1,
            )
            pc_pe = pc_pe.reshape(
                batch_size,
                n_obs_steps * pc_pe.shape[1],
                -1,
            )
    else:
        global_cond = features.reshape(batch_size, -1)

    return EncodedCondition(
        global_cond=global_cond,
        pc_pe=pc_pe,
    )
