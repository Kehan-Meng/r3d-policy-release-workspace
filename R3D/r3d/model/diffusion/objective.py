from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import reduce


def compute_diffusion_policy_loss(
    model,
    scheduler,
    trajectory,
    condition_mask,
    condition_data,
    *,
    action_dim: int,
    use_target_ee: bool,
    device,
    n_obs_steps: int,
    local_cond=None,
    global_cond=None,
    pc_pe=None,
):
    noise = torch.randn_like(trajectory)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (trajectory.shape[0],),
        device=trajectory.device,
    ).long()
    noisy_trajectory = scheduler.add_noise(trajectory, noise, timesteps)
    loss_mask = ~condition_mask
    noisy_trajectory[condition_mask] = condition_data[condition_mask]

    prediction = model(
        sample=noisy_trajectory,
        timestep=timesteps,
        local_cond=local_cond,
        global_cond=global_cond,
        pc_pe=pc_pe,
        n_obs_steps=n_obs_steps,
    )

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = trajectory
    elif prediction_type == "v_prediction":
        scheduler.alpha_t = scheduler.alpha_t.to(device)
        scheduler.sigma_t = scheduler.sigma_t.to(device)
        alpha_t = scheduler.alpha_t[timesteps].unsqueeze(-1).unsqueeze(-1)
        sigma_t = scheduler.sigma_t[timesteps].unsqueeze(-1).unsqueeze(-1)
        target = alpha_t * noise - sigma_t * trajectory
    else:
        raise ValueError(f"Unsupported prediction type {prediction_type}")

    element_loss = F.mse_loss(prediction, target, reduction="none")
    element_loss = element_loss * loss_mask.to(element_loss.dtype)
    if not use_target_ee:
        loss = reduce(element_loss, "b ... -> b (...)", "mean").mean()
        return loss, {"bc_loss": loss.item()}

    ee_dim = action_dim // 2
    joint_dim = action_dim - ee_dim
    joint_loss = reduce(
        element_loss[:, :, :joint_dim],
        "b ... -> b (...)",
        "mean",
    ).mean()
    ee_loss = reduce(
        element_loss[:, :, joint_dim:],
        "b ... -> b (...)",
        "mean",
    ).mean()
    loss = joint_loss + ee_loss
    return loss, {
        "bc_loss": loss.item(),
        "joint_loss": joint_loss.item(),
        "ee_loss": ee_loss.item(),
    }

