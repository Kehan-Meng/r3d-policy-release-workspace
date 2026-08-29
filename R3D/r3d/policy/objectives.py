from __future__ import annotations

from r3d.model.flow_matching import compute_consistency_flow_matching_loss


def compute_flow_policy_loss(
    model,
    trajectory,
    condition_mask,
    *,
    n_obs_steps,
    config,
    training_epoch: int,
    local_cond=None,
    global_cond=None,
    pc_pe=None,
):
    return compute_consistency_flow_matching_loss(
        model,
        trajectory,
        condition_mask,
        local_cond=local_cond,
        global_cond=global_cond,
        pc_pe=pc_pe,
        n_obs_steps=n_obs_steps,
        eps=config.eps,
        delta=config.delta,
        num_segments=config.num_segments,
        boundary=config.boundary,
        alpha=config.alpha,
        consistency_weight=config.consistency_weight_at(training_epoch),
        stop_gradient_target=config.stop_gradient_target,
        direct_velocity_weight=config.direct_velocity_weight,
        time_scale=config.time_scale,
    )


def add_condition_shape_metrics(loss_dict, global_cond):
    if global_cond is not None:
        loss_dict["condition_num_tokens"] = (
            int(global_cond.shape[1]) if global_cond.dim() == 3 else 1
        )
        loss_dict["condition_token_dim"] = int(global_cond.shape[-1])
