from __future__ import annotations

import torch

from r3d.model.flow_matching import flow_ode_sample


def diffusion_sample(
    model,
    scheduler,
    condition_data,
    condition_mask,
    *,
    num_inference_steps,
    local_cond=None,
    global_cond=None,
    pc_pe=None,
    n_obs_steps=None,
):
    trajectory = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
    )
    scheduler.set_timesteps(num_inference_steps)
    for timestep in scheduler.timesteps:
        trajectory[condition_mask] = condition_data[condition_mask].to(
            trajectory.dtype
        )
        model_output = model(
            sample=trajectory,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            n_obs_steps=n_obs_steps,
        )
        trajectory = scheduler.step(
            model_output,
            timestep,
            trajectory,
        ).prev_sample

    trajectory[condition_mask] = condition_data[condition_mask].to(
        trajectory.dtype
    )
    return trajectory


def flow_sample(
    model,
    condition_data,
    condition_mask,
    *,
    n_obs_steps,
    eps,
    num_inference_steps,
    solver,
    time_scale,
    initial_noise_scale,
    local_cond=None,
    global_cond=None,
    pc_pe=None,
    generator=None,
):
    return flow_ode_sample(
        model,
        condition_data,
        condition_mask,
        local_cond=local_cond,
        global_cond=global_cond,
        pc_pe=pc_pe,
        n_obs_steps=n_obs_steps,
        eps=eps,
        num_inference_steps=num_inference_steps,
        solver=solver,
        time_scale=time_scale,
        initial_noise_scale=initial_noise_scale,
        generator=generator,
    )

