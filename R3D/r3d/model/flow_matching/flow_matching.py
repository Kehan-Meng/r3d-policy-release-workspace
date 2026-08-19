import math

import torch


def _validate_flow_condition(condition_mask, global_cond, pc_pe):
    if condition_mask is None:
        raise RuntimeError(
            "flow_matching requires an explicit condition_mask; the first "
            "implementation expects an all-False mask from obs_as_global_cond=True"
        )
    if global_cond is None:
        raise RuntimeError(
            "flow_matching with one_way_transformer requires global_cond from "
            "the encoder/ACT condition pipeline"
        )
    if pc_pe is None:
        raise RuntimeError(
            "flow_matching with one_way_transformer requires pc_pe from the "
            "encoder/ACT condition pipeline"
        )
    if condition_mask is not None and condition_mask.any().item():
        raise NotImplementedError(
            "flow_matching currently only supports obs_as_global_cond=True "
            "without inpainting/action conditioning masks"
        )


def compute_consistency_flow_matching_loss(
        model,
        trajectory,
        condition_mask,
        *,
        local_cond=None,
        global_cond=None,
        pc_pe=None,
        dense_cond=None,
        dense_pe=None,
        n_obs_steps=None,
        eps=1e-2,
        delta=1e-2,
        num_segments=2,
        boundary=1,
        alpha=1e-5,
        consistency_weight=1.0,
        stop_gradient_target=True,
        direct_velocity_weight=0.0,
        time_scale=99.0,
        compute_gradient_diagnostics=False,
        consistency_routing_multipliers=None,
        consistency_routing_t_bins=None,
        consistency_routing_stratified_t=False,
        dim_weights=None,
        ):
    _validate_flow_condition(condition_mask, global_cond, pc_pe)

    batch_size = trajectory.shape[0]
    a1 = trajectory
    a0 = torch.randn_like(a1)

    routing_bin_indices = None
    if consistency_routing_stratified_t:
        if consistency_routing_t_bins is None:
            raise ValueError("Stratified t sampling requires routing t_bins")
        t_bins = torch.as_tensor(
            consistency_routing_t_bins, device=a1.device, dtype=a1.dtype
        )
        if t_bins.ndim != 1 or t_bins.numel() < 2:
            raise ValueError("routing t_bins must be a one-dimensional boundary list")
        if not torch.all(t_bins[1:] > t_bins[:-1]):
            raise ValueError("routing t_bins must be strictly increasing")
        num_bins = t_bins.numel() - 1
        # Balanced bin assignment, shuffled so a dataset sample is not tied to one bin.
        routing_bin_indices = torch.arange(
            batch_size, device=a1.device, dtype=torch.long
        ).remainder(num_bins)
        routing_bin_indices = routing_bin_indices[torch.randperm(batch_size, device=a1.device)]
        low = t_bins[routing_bin_indices].clamp_min(float(eps))
        high = t_bins[routing_bin_indices + 1].clamp_max(1.0)
        if torch.any(high <= low):
            raise ValueError("routing t_bins contain an empty interval after eps clipping")
        t = low + (high - low) * torch.rand(
            batch_size, device=a1.device, dtype=a1.dtype
        )
    else:
        t = torch.rand(
            batch_size,
            device=a1.device,
            dtype=a1.dtype,
        ) * (1.0 - eps) + eps
    r = torch.clamp(t + delta, max=1.0)

    t_expand = t.view(batch_size, 1, 1).expand_as(a1)
    r_expand = r.view(batch_size, 1, 1).expand_as(a1)

    xt = t_expand * a1 + (1.0 - t_expand) * a0
    xr = r_expand * a1 + (1.0 - r_expand) * a0

    vt = model(
        sample=xt,
        timestep=t * time_scale,
        local_cond=local_cond,
        global_cond=global_cond,
        pc_pe=pc_pe,
        dense_cond=dense_cond,
        dense_pe=dense_pe,
        n_obs_steps=n_obs_steps,
    )
    vr_kwargs = dict(
        sample=xr,
        timestep=r * time_scale,
        local_cond=local_cond,
        global_cond=global_cond,
        pc_pe=pc_pe,
        dense_cond=dense_cond,
        dense_pe=dense_pe,
        n_obs_steps=n_obs_steps,
    )
    if stop_gradient_target:
        # Fixed target branch used by consistency-style training.
        with torch.no_grad():
            vr = model(**vr_kwargs)
    else:
        # Matches the robot-policy compute_loss in the official FlowPolicy code.
        vr = model(**vr_kwargs)

    segments = torch.linspace(
        0,
        1,
        num_segments + 1,
        device=a1.device,
        dtype=a1.dtype,
    )
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    segment_ends = segments[seg_indices]
    segment_ends_expand = segment_ends.view(batch_size, 1, 1).expand_as(a1)
    x_at_segment_ends = (
        segment_ends_expand * a1
        + (1.0 - segment_ends_expand) * a0
    )

    ft = xt + (segment_ends_expand - t_expand) * vt
    if boundary == 0:
        fr = x_at_segment_ends
    else:
        fr_euler = xr + (segment_ends_expand - r_expand) * vr
        less_than_threshold = r_expand < boundary
        fr = torch.where(less_than_threshold, fr_euler, x_at_segment_ends)

    losses_f = torch.square(ft - fr).reshape(batch_size, -1).mean(dim=-1)

    if dim_weights is not None:
        w = torch.as_tensor(dim_weights, device=losses_f.device, dtype=losses_f.dtype)
        w = w.view(1, 1, -1).expand(batch_size, trajectory.shape[1], -1)
        losses_f_w = torch.square(ft - fr) * w
        losses_f = losses_f_w.reshape(batch_size, -1).mean(dim=-1)

    if boundary == 0:
        losses_v = torch.zeros_like(losses_f)
    else:
        less_than_threshold = t_expand < boundary
        far_from_segment_ends = (
            (segment_ends - t) > 1.01 * delta
        ).view(batch_size, 1, 1).expand_as(a1)
        losses_v_tensor = torch.square(vt - vr)
        losses_v_tensor = (
            losses_v_tensor
            * less_than_threshold.to(losses_v_tensor.dtype)
            * far_from_segment_ends.to(losses_v_tensor.dtype)
        )
        losses_v = losses_v_tensor.reshape(batch_size, -1).mean(dim=-1)

    direct_velocity_target = a1 - a0
    losses_direct = torch.square(vt - direct_velocity_target)
    losses_direct = losses_direct.reshape(batch_size, -1).mean(dim=-1)

    gradient_diagnostics = {}
    if compute_gradient_diagnostics:
        grad_f = torch.autograd.grad(
            outputs=losses_f.mean(),
            inputs=vt,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad_direct = torch.autograd.grad(
            outputs=losses_direct.mean(),
            inputs=vt,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        grad_f = grad_f.detach().float().reshape(-1)
        grad_direct = grad_direct.detach().float().reshape(-1)
        grad_f_norm = torch.linalg.vector_norm(grad_f)
        grad_direct_norm = torch.linalg.vector_norm(grad_direct)
        denominator = (grad_f_norm * grad_direct_norm).clamp_min(1e-12)
        gradient_diagnostics = {
            "flow_grad_f_norm": grad_f_norm.item(),
            "flow_grad_direct_norm": grad_direct_norm.item(),
            "flow_grad_ratio_unweighted": (
                grad_f_norm / grad_direct_norm.clamp_min(1e-12)
            ).item(),
            "flow_grad_cosine": (
                torch.dot(grad_f, grad_direct) / denominator
            ).clamp(-1.0, 1.0).item(),
        }

    routing_sample_multipliers = torch.ones_like(losses_f)
    if consistency_routing_multipliers is not None:
        if routing_bin_indices is None:
            raise ValueError("routing multipliers require stratified t sampling")
        route_values = torch.as_tensor(
            consistency_routing_multipliers,
            device=losses_f.device,
            dtype=losses_f.dtype,
        )
        if route_values.numel() != int(t_bins.numel() - 1):
            raise ValueError(
                "routing multiplier count must equal len(t_bins)-1, got "
                f"{route_values.numel()} and {t_bins.numel() - 1}"
            )
        routing_sample_multipliers = route_values[routing_bin_indices]

    weighted_losses_f = (
        consistency_weight * routing_sample_multipliers * losses_f
    )
    weighted_losses_v = alpha * losses_v
    weighted_losses_direct = direct_velocity_weight * losses_direct
    loss = torch.mean(
        weighted_losses_f + weighted_losses_v + weighted_losses_direct
    )
    loss_dict = {
        "bc_loss": loss.item(),
        "flow_loss": loss.item(),
        "flow_f_loss": losses_f.mean().item(),
        "flow_weighted_f_loss": weighted_losses_f.mean().item(),
        "flow_v_loss": losses_v.mean().item(),
        "flow_weighted_v_loss": weighted_losses_v.mean().item(),
        "flow_direct_velocity_loss": losses_direct.mean().item(),
        "flow_weighted_direct_velocity_loss": weighted_losses_direct.mean().item(),
        "flow_consistency_weight": float(consistency_weight),
        "flow_alpha": float(alpha),
        "flow_stop_gradient_target": float(stop_gradient_target),
        "flow_direct_velocity_weight": float(direct_velocity_weight),
    }
    if routing_bin_indices is not None:
        loss_dict["flow_routing_multiplier_mean"] = (
            routing_sample_multipliers.detach().float().mean().item()
        )
        for bin_index in range(int(t_bins.numel() - 1)):
            selected = routing_bin_indices == bin_index
            if selected.any():
                loss_dict[f"flow_routing_bin{bin_index}_f_loss"] = (
                    losses_f[selected].detach().float().mean().item()
                )
                loss_dict[f"flow_routing_bin{bin_index}_sample_fraction"] = (
                    selected.detach().float().mean().item()
                )
            loss_dict[f"flow_routing_bin{bin_index}_multiplier"] = float(
                consistency_routing_multipliers[bin_index]
            )
    loss_dict.update(gradient_diagnostics)
    return loss, loss_dict


def flow_solver_nfe(solver, num_inference_steps):
    stages = {"euler": 1, "heun": 2, "rk4": 4}
    if solver not in stages:
        raise ValueError(f"Unknown flow ODE solver: {solver}")
    return stages[solver] * num_inference_steps


@torch.no_grad()
def flow_ode_sample(
        model,
        condition_data,
        condition_mask,
        *,
        local_cond=None,
        global_cond=None,
        pc_pe=None,
        dense_cond=None,
        dense_pe=None,
        n_obs_steps=None,
        eps=1e-2,
        num_inference_steps=1,
        solver="euler",
        time_scale=99.0,
        initial_noise_scale=1.0,
        generator=None,
        ):
    _validate_flow_condition(condition_mask, global_cond, pc_pe)
    if num_inference_steps <= 0:
        raise ValueError(
            "flow_matching.num_inference_steps must be positive, "
            f"got {num_inference_steps}"
        )
    flow_solver_nfe(solver, num_inference_steps)
    if not math.isfinite(initial_noise_scale) or initial_noise_scale < 0:
        raise ValueError(
            "initial_noise_scale must be finite and non-negative, got "
            f"{initial_noise_scale}"
        )

    trajectory = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator,
    ) * initial_noise_scale
    trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

    # Integrate exactly from t=eps to t=1.  Using 1 / N here would
    # overshoot the intended terminal time by eps.
    dt = (1.0 - eps) / num_inference_steps
    def velocity(sample, num_t):
        t = torch.full(
            (trajectory.shape[0],),
            fill_value=num_t,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        return model(
            sample=sample,
            timestep=t * time_scale,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_pe=pc_pe,
            dense_cond=dense_cond,
            dense_pe=dense_pe,
            n_obs_steps=n_obs_steps,
        )

    def enforce_condition(sample):
        sample[condition_mask] = condition_data[condition_mask].to(sample.dtype)
        return sample

    for i in range(num_inference_steps):
        num_t = eps + i * dt
        if solver == "euler":
            k1 = velocity(trajectory, num_t)
            trajectory = trajectory + dt * k1
        elif solver == "heun":
            k1 = velocity(trajectory, num_t)
            predictor = enforce_condition(trajectory + dt * k1)
            k2 = velocity(predictor, num_t + dt)
            trajectory = trajectory + 0.5 * dt * (k1 + k2)
        else:  # rk4
            k1 = velocity(trajectory, num_t)
            k2 = velocity(
                enforce_condition(trajectory + 0.5 * dt * k1),
                num_t + 0.5 * dt,
            )
            k3 = velocity(
                enforce_condition(trajectory + 0.5 * dt * k2),
                num_t + 0.5 * dt,
            )
            k4 = velocity(
                enforce_condition(trajectory + dt * k3),
                num_t + dt,
            )
            trajectory = trajectory + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)

    return trajectory


def flow_euler_sample(*args, **kwargs):
    """Backward-compatible Euler sampler wrapper."""
    kwargs["solver"] = "euler"
    return flow_ode_sample(*args, **kwargs)
