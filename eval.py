import argparse
import csv
import importlib.util
import json
import os
import pathlib
import random
import re
import sys
import time

import dill
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from release_utils import (
    REPO_ROOT,
    configure_native_libraries,
    load_experiment_config,
    prepend_env_path,
    resolve_repo_path,
)

def load_training_workspace_class(r3d_dir):
    """Load the training workspace entry module without mutating ``sys.path``."""
    train_path = pathlib.Path(r3d_dir) / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(f"R3D training entry not found: {train_path}")
    spec = importlib.util.spec_from_file_location("r3d_training_entry", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load R3D training entry: {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainDP3Workspace


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def discover_tasks(raw_dir):
    raw_dir = resolve_repo_path(raw_dir)
    return sorted(path.name[:-5] for path in raw_dir.glob("*.zarr"))


def select_tasks(cfg, tasks_arg, max_tasks):
    discover_task_dir = nested_cfg_get(cfg, ("eval", "discover_task_dir"), None)
    configured_tasks = nested_cfg_get(cfg, ("eval", "task_names"), None)
    if tasks_arg:
        if tasks_arg.strip().lower() == "discover":
            if not discover_task_dir:
                raise ValueError("--tasks discover requires cfg.eval.discover_task_dir")
            tasks = discover_tasks(discover_task_dir)
        else:
            tasks = [item.strip() for item in tasks_arg.split(",") if item.strip()]
    elif configured_tasks:
        tasks = list(configured_tasks)
    elif discover_task_dir:
        tasks = discover_tasks(discover_task_dir)
    else:
        tasks = [cfg.task.name]
    # A multi-domain dataset may contain several entries for the same task
    # (for example clean and randomized hammer). RoboTwin2Runner evaluates all
    # matching entries itself, so invoking the same task name twice here would
    # duplicate the complete rollout and overwrite its result under one key.
    tasks = list(dict.fromkeys(tasks))
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def latest_checkpoint(output_dir):
    ckpt_dir = resolve_repo_path(output_dir) / "checkpoints"
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    train_loss_candidates = []
    for path in candidates:
        match = re.search(r"train_loss[_=]([0-9.eE+-]+)", path.stem)
        if match:
            train_loss_candidates.append((float(match.group(1)), path))
    if train_loss_candidates:
        return min(train_loss_candidates, key=lambda item: item[0])[1]

    def score(path):
        stem = path.stem
        return int(stem) if stem.isdigit() else -1

    numbered = [path for path in candidates if path.stem.isdigit()]
    return max(numbered, key=score) if numbered else candidates[-1]


def load_policy(cfg, checkpoint_path, device, policy_source="auto"):
    runtime = cfg.get("runtime", {})
    project_dir = resolve_repo_path(runtime.get("project_dir"), default=REPO_ROOT)
    r3d_dir = resolve_repo_path(runtime.get("r3d_dir"), default="R3D")
    os.chdir(project_dir)

    TrainDP3Workspace = load_training_workspace_class(r3d_dir)

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    train_cfg = payload["cfg"]
    with open_dict(train_cfg):
        train_cfg.training.device = device
        train_cfg.training.use_ddp = False
        # Evaluation-only ACT interventions live in the external experiment
        # config, not in the checkpoint's saved training config. Propagate
        # them before constructing the workspace so causal heatmap ablations
        # actually reach AffordanceGuidedCompactorTransformer.
        eval_act_cfg = nested_cfg_get(cfg, ("policy", "act"), None)
        train_act_cfg = nested_cfg_get(train_cfg, ("policy", "act_config"), None)
        if eval_act_cfg is not None and train_act_cfg is not None:
            for key in (
                "heatmap_intervention",
                "heatmap_intervention_seed",
                "heatmap_intervention_roll",
            ):
                value = cfg_get(eval_act_cfg, key, None)
                if value is not None:
                    train_act_cfg[key] = value

    intervention = nested_cfg_get(
        train_cfg,
        ("policy", "act_config", "heatmap_intervention"),
        "none",
    )
    print(f"[EVAL] ACT heatmap_intervention={intervention}", flush=True)

    workspace = TrainDP3Workspace(train_cfg, output_dir=str(pathlib.Path(cfg.experiment.output_dir)))
    workspace.load_payload(payload)
    if policy_source == "auto":
        policy_source = "ema" if train_cfg.training.use_ema else "model"
    if policy_source == "ema":
        if workspace.ema_model is None:
            raise ValueError("Requested policy_source=ema, but checkpoint/config has no ema_model.")
        policy = workspace.ema_model
    elif policy_source == "model":
        policy = workspace.model
    else:
        raise ValueError(f"Unknown policy_source: {policy_source}")
    policy.to(torch.device(device))
    policy.eval()
    return policy, train_cfg, policy_source, payload


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return default


def nested_cfg_get(cfg, keys, default=None):
    cur = cfg
    for key in keys:
        cur = cfg_get(cur, key, default=None)
        if cur is None:
            return default
    return cur


def is_robotwin_runner(train_cfg):
    target = str(nested_cfg_get(train_cfg, ("task", "env_runner", "_target_"), ""))
    return "robotwin" in target.lower()


def is_adroit_runner(train_cfg):
    target = str(nested_cfg_get(train_cfg, ("task", "env_runner", "_target_"), ""))
    if "adroit" in target.lower():
        return True
    task_config = str(nested_cfg_get(train_cfg, ("task", "config"), ""))
    return "adroit" in task_config.lower()


def is_maniskill_runner(train_cfg):
    target = str(nested_cfg_get(train_cfg, ("task", "env_runner", "_target_"), ""))
    if "maniskill" in target.lower():
        return True
    dataset_target = str(
        nested_cfg_get(train_cfg, ("task", "dataset", "_target_"), "")
    )
    if "maniskill" in dataset_target.lower():
        return True
    task_config = str(nested_cfg_get(train_cfg, ("task", "config"), ""))
    if "maniskill" in task_config.lower():
        return True
    task_name = str(nested_cfg_get(train_cfg, ("task_name",), ""))
    return task_name.endswith("-v0") and any(
        marker in task_name.lower()
        for marker in ("cube", "peginsertion", "peg-insertion")
    )


def is_peg_assembly_runner(train_cfg):
    target = str(nested_cfg_get(train_cfg, ("task", "env_runner", "_target_"), ""))
    if "peg_assembly" in target.lower() and "franka_grasp" not in target.lower():
        return True
    task_config = str(nested_cfg_get(train_cfg, ("task", "config"), ""))
    return task_config.lower() == "peg_assembly"


def is_franka_grasp_runner(train_cfg):
    target = str(nested_cfg_get(train_cfg, ("task", "env_runner", "_target_"), ""))
    if "franka_grasp_peg_assembly" in target.lower():
        return True
    task_config = str(nested_cfg_get(train_cfg, ("task", "config"), ""))
    return task_config.lower() == "franka_grasp_peg_assembly"


def checkpoint_epoch_tag(checkpoint_path):
    stem = pathlib.Path(checkpoint_path).stem
    for pattern in (r"epoch[_=]?(\d+)", r"^(\d+)$"):
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))
    return 0


def resolve_robotwin_task_config(cfg, train_cfg, task_name):
    task_settings = nested_cfg_get(cfg, ("eval", "task_settings"), None)
    if task_settings is not None and task_name in task_settings:
        return task_settings[task_name]
    return (
        nested_cfg_get(cfg, ("eval", "task_setting"), None)
        or nested_cfg_get(cfg, ("task", "setting"), None)
        or nested_cfg_get(cfg, ("data", "task_setting"), None)
        or nested_cfg_get(train_cfg, ("task", "env_runner", "task_config"), None)
        or "demo_clean"
    )


def build_runner(
        cfg,
        train_cfg,
        task_name,
        output_dir,
        device,
        eval_episodes,
        eval_seed=None,
        episode_start=0,
        robotwin_lean_observation=False,
        robotwin_profile=False,
        robotwin_defer_intermediate_render=False,
        robotwin_rt_spp=None,
        robotwin_camera_shader=None,
        fast=False,
        ):
    if is_robotwin_runner(train_cfg):
        from r3d.env_runner.robotwin2_runner import RoboTwin2Runner

        task_entries = nested_cfg_get(train_cfg, ("task", "env_runner", "task_entries"), None)

        return RoboTwin2Runner(
            output_dir=str(output_dir),
            task_name=task_name,
            task_entries=task_entries,
            eval_task_name=task_name if task_entries is not None else None,
            seed=int(nested_cfg_get(train_cfg, ("task", "env_runner", "seed"), 0)),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 2000)),
            n_obs_steps=train_cfg.n_obs_steps,
            n_action_steps=train_cfg.n_action_steps,
            task_config=resolve_robotwin_task_config(cfg, train_cfg, task_name),
            instruction_type=(
                nested_cfg_get(cfg, ("eval", "instruction_type"), None)
                or nested_cfg_get(cfg, ("task", "instruction_type"), None)
                or nested_cfg_get(train_cfg, ("task", "env_runner", "instruction_type"), "unseen")
            ),
            action_space_type=nested_cfg_get(train_cfg, ("task", "env_runner", "action_space_type"), "joint"),
            head_camera_type=nested_cfg_get(train_cfg, ("task", "env_runner", "head_camera_type"), "D435"),
            save_video=False,
            tqdm_interval_sec=nested_cfg_get(train_cfg, ("task", "env_runner", "tqdm_interval_sec"), 5.0),
            episode_start=episode_start,
            deterministic_eval_seed=eval_seed,
            lean_observation=robotwin_lean_observation,
            profile_eval=robotwin_profile,
            defer_intermediate_render=robotwin_defer_intermediate_render,
            rt_samples_per_pixel=robotwin_rt_spp,
            camera_shader=robotwin_camera_shader,
        )

    if is_adroit_runner(train_cfg):
        from r3d.env_runner.adroit_runner import AdroitRunner

        # Adroit uses short task name (e.g. "door") from task_name field
        adroit_task_name = nested_cfg_get(train_cfg, ("task", "task_name"), task_name)
        render_device_id = int(
            os.environ.get("MKH_RENDER_GPU_ID", str(device).split(":")[-1])
        )
        return AdroitRunner(
            output_dir=str(output_dir),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 600)),
            n_obs_steps=train_cfg.n_obs_steps,
            n_action_steps=train_cfg.n_action_steps,
            fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
            task_name=adroit_task_name,
            use_point_crop=nested_cfg_get(train_cfg, ("policy", "use_point_crop"), True),
            deterministic_eval_seed=eval_seed,
            render_device_id=render_device_id,
        )

    if is_maniskill_runner(train_cfg):
        from r3d.env_runner.maniskill_runner import ManiskillRunner

        runner_cfg = nested_cfg_get(train_cfg, ("task", "env_runner"), None)
        env_task_name = str(cfg_get(runner_cfg, "task_name", task_name))
        camera_ee_contract = bool(
            cfg_get(runner_cfg, "camera_ee_contract", False)
            or nested_cfg_get(
                train_cfg, ("task", "dataset", "camera_ee_contract"), False
            )
        )
        camera_profile_path = (
            cfg_get(runner_cfg, "camera_profile_path", None)
            or nested_cfg_get(
                train_cfg, ("task", "dataset", "camera_profile_path"), None
            )
        )
        if camera_profile_path:
            camera_profile_path = str(resolve_repo_path(camera_profile_path))
        if camera_ee_contract and not camera_profile_path:
            raise ValueError("camera-EE evaluation requires camera_profile_path")
        if camera_ee_contract:
            import hashlib

            expected_hash = nested_cfg_get(
                train_cfg, ("task", "dataset", "camera_profile_sha256"), None
            )
            if not expected_hash:
                raise ValueError("camera-EE checkpoint lacks camera_profile_sha256")
            with open(camera_profile_path, "rb") as handle:
                actual_hash = hashlib.sha256(handle.read()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    "camera-EE profile hash mismatch: "
                    f"checkpoint={expected_hash}, file={actual_hash}"
                )
        return ManiskillRunner(
            output_dir=str(output_dir),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 200)),
            n_obs_steps=int(train_cfg.n_obs_steps),
            n_action_steps=int(train_cfg.n_action_steps),
            fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
            task_name=env_task_name,
            device=device,
            use_point_crop=bool(nested_cfg_get(train_cfg, ("policy", "use_point_crop"), True)),
            num_points=int(nested_cfg_get(cfg, ("task", "num_points"), 1024)),
            save_video=not fast,
            deterministic_eval_seed=eval_seed,
            episode_start=episode_start,
            camera_ee_contract=camera_ee_contract,
            camera_profile_path=camera_profile_path,
        )

    if is_franka_grasp_runner(train_cfg):
        from r3d.env_runner.franka_grasp_peg_assembly_runner import FrankaGraspPegAssemblyRunner

        runner_cfg = nested_cfg_get(train_cfg, ("task", "env_runner"), None)
        default_seed = nested_cfg_get(
            train_cfg,
            ("task", "env_runner", "deterministic_eval_seed"),
            100000,
        )
        resolved_seed = eval_seed if eval_seed is not None else default_seed
        if resolved_seed is not None:
            resolved_seed = int(resolved_seed) + int(episode_start)
        return FrankaGraspPegAssemblyRunner(
            output_dir=str(output_dir),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 400)),
            n_obs_steps=train_cfg.n_obs_steps,
            n_action_steps=train_cfg.n_action_steps,
            fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
            shape_name=task_name,
            clearance=float(cfg_get(runner_cfg, "clearance", 5.0e-4)),
            distribution=str(
                nested_cfg_get(
                    cfg,
                    ("eval", "distribution"),
                    cfg_get(runner_cfg, "distribution", "id"),
                )
            ),
            num_points=int(
                nested_cfg_get(
                    cfg,
                    ("task", "num_points"),
                    cfg_get(runner_cfg, "num_points", 512),
                )
            ),
            image_size=int(cfg_get(runner_cfg, "image_size", 128)),
            policy_command=str(cfg_get(runner_cfg, "policy_command", "peg_assembly")),
            save_video=False,
            deterministic_eval_seed=resolved_seed,
            observation_mode=str(cfg_get(runner_cfg, "observation_mode", "rgbd")),
        )

    if is_peg_assembly_runner(train_cfg):
        from r3d.env_runner.peg_assembly_runner import PegAssemblyRunner

        runner_cfg = nested_cfg_get(train_cfg, ("task", "env_runner"), None)
        default_seed = nested_cfg_get(
            train_cfg,
            ("task", "env_runner", "deterministic_eval_seed"),
            100000,
        )
        resolved_seed = eval_seed if eval_seed is not None else default_seed
        if resolved_seed is not None:
            resolved_seed = int(resolved_seed) + int(episode_start)
        return PegAssemblyRunner(
            output_dir=str(output_dir),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 400)),
            n_obs_steps=train_cfg.n_obs_steps,
            n_action_steps=train_cfg.n_action_steps,
            fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
            shape_name=task_name,
            clearance=float(cfg_get(runner_cfg, "clearance", 5.0e-4)),
            distribution=str(
                nested_cfg_get(
                    cfg,
                    ("eval", "distribution"),
                    cfg_get(runner_cfg, "distribution", "id"),
                )
            ),
            num_points=int(
                nested_cfg_get(
                    cfg,
                    ("task", "num_points"),
                    cfg_get(runner_cfg, "num_points", 512),
                )
            ),
            image_size=int(cfg_get(runner_cfg, "image_size", 128)),
            policy_command=str(cfg_get(runner_cfg, "policy_command", "peg_assembly")),
            save_video=False,
            deterministic_eval_seed=resolved_seed,
        )

    if fast:
        from r3d.env_runner.metaworld_runner import MetaworldRunnerFast
        print("[EVAL] Using MetaworldRunnerFast (no RGB render, no video)", flush=True)
        return MetaworldRunnerFast(
            output_dir=str(output_dir),
            eval_episodes=eval_episodes,
            max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 600)),
            n_obs_steps=train_cfg.n_obs_steps,
            n_action_steps=train_cfg.n_action_steps,
            fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
            task_name=task_name,
            device=device,
            use_point_crop=train_cfg.policy.use_point_crop,
            num_points=cfg.task.num_points,
        )

    from r3d.env_runner.metaworld_runner import MetaworldRunner

    return MetaworldRunner(
        output_dir=str(output_dir),
        eval_episodes=eval_episodes,
        max_steps=int(nested_cfg_get(cfg, ("eval", "max_steps"), 600)),
        n_obs_steps=train_cfg.n_obs_steps,
        n_action_steps=train_cfg.n_action_steps,
        fps=int(nested_cfg_get(cfg, ("eval", "fps"), 10)),
        task_name=task_name,
        device=device,
        use_point_crop=train_cfg.policy.use_point_crop,
        num_points=cfg.task.num_points,
    )


def run_runner(runner, policy, cfg, train_cfg, checkpoint_path, task_name):
    if is_robotwin_runner(train_cfg):
        task_config = resolve_robotwin_task_config(cfg, train_cfg, task_name)
        return runner.run(
            policy,
            epoch=checkpoint_epoch_tag(checkpoint_path),
            task_config=task_config,
        )
    if is_adroit_runner(train_cfg):
        return runner.run(policy)
    if is_maniskill_runner(train_cfg):
        return runner.run(policy)
    if is_franka_grasp_runner(train_cfg):
        return runner.run(policy)
    if is_peg_assembly_runner(train_cfg):
        return runner.run(policy)
    return runner.run(policy, save_video=False)


def extract_success_metric(log):
    if "success_rate" in log:
        return float(log["success_rate"])
    if "mean_success_rates" in log:
        return float(log["mean_success_rates"])
    if "test_mean_score" in log:
        return float(log["test_mean_score"])
    for key, value in log.items():
        if key.endswith(": success_rate"):
            return float(value)
    return None


def extract_reward_metric(log):
    if "mean_traj_rewards" in log:
        return float(log["mean_traj_rewards"])
    for key, value in log.items():
        if key.endswith(": mean_reward"):
            return float(value)
    return None


def resolve_domain_roles(cfg, tasks):
    """Return the explicit ID/shape-OOD label for each requested task."""

    raw = nested_cfg_get(cfg, ("eval", "domain_roles"), None)
    if raw is None:
        return {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    roles = {str(name): str(role) for name, role in dict(raw).items()}
    return {task: roles[task] for task in tasks if task in roles}


def validate_peg_domain_protocol(tasks, domain_roles, reset_distribution):
    formal_roles = {
        "square": "id",
        "triangle": "shape_ood",
        "rectangle": "shape_ood",
    }
    unexpected_tasks = [task for task in tasks if task not in formal_roles]
    if unexpected_tasks:
        raise ValueError(
            "Formal PegAssembly tasks are square, triangle, and rectangle; "
            f"unexpected={unexpected_tasks}"
        )
    missing_roles = [task for task in tasks if task not in domain_roles]
    if missing_roles:
        raise ValueError(
            "PegAssembly evaluation requires eval.domain_roles for every shape; "
            f"missing={missing_roles}"
        )
    wrong_roles = {
        task: domain_roles[task]
        for task in tasks
        if domain_roles[task] != formal_roles[task]
    }
    if wrong_roles:
        raise ValueError(
            "PegAssembly domain roles violate the formal protocol: "
            f"expected={formal_roles}, actual={wrong_roles}"
        )
    if str(reset_distribution) != "id":
        raise ValueError(
            "Formal PegAssembly ID/shape-OOD evaluation requires "
            "eval.distribution=id for every shape; use a separate experiment "
            "for pose or compound OOD"
        )


def summarize_domain_groups(task_logs, domain_roles):
    grouped = {}
    for task_name, task_log in task_logs.items():
        role = domain_roles.get(task_name)
        if role is None:
            continue
        group = grouped.setdefault(role, {"tasks": [], "success": [], "reward": []})
        group["tasks"].append(task_name)
        success = extract_success_metric(task_log)
        reward = extract_reward_metric(task_log)
        if success is not None:
            group["success"].append(success)
        if reward is not None:
            group["reward"].append(reward)
    return {
        role: {
            "tasks": values["tasks"],
            "mean_success_rate": (
                float(np.mean(values["success"])) if values["success"] else None
            ),
            "mean_traj_reward": (
                float(np.mean(values["reward"])) if values["reward"] else None
            ),
        }
        for role, values in grouped.items()
    }


def run_eval(
        cfg,
        checkpoint_path,
        tasks,
        device,
        eval_episodes,
        policy_source="auto",
        flow_inference_steps=None,
        flow_solver=None,
        n_action_steps=None,
        eval_seed=None,
        episode_start=0,
        robotwin_lean_observation=False,
        robotwin_profile=False,
        robotwin_defer_intermediate_render=False,
        robotwin_rt_spp=None,
        fast=False,
        robotwin_camera_shader=None,
        ):
    policy, train_cfg, resolved_policy_source, payload = load_policy(
        cfg, checkpoint_path, device, policy_source=policy_source,
    )
    if n_action_steps is not None:
        n_action_steps = int(n_action_steps)
        action_start = int(policy.n_obs_steps) - 1
        max_action_steps = int(policy.horizon) - action_start
        if not 1 <= n_action_steps <= max_action_steps:
            raise ValueError(
                f"--n-action-steps must be in [1, {max_action_steps}] for "
                f"horizon={policy.horizon}, n_obs_steps={policy.n_obs_steps}"
            )
        policy.n_action_steps = n_action_steps
        with open_dict(train_cfg):
            train_cfg.n_action_steps = n_action_steps
        print(
            f"[EVAL] override n_action_steps={n_action_steps}; "
            f"effective_executed_action_steps={n_action_steps}",
            flush=True,
        )
    if flow_inference_steps is not None:
        if getattr(policy, "generation_type", None) != "flow_matching":
            raise ValueError("--flow-inference-steps requires a flow_matching checkpoint")
        policy.flow_num_inference_steps = flow_inference_steps
    if flow_solver is not None:
        if getattr(policy, "generation_type", None) != "flow_matching":
            raise ValueError("--flow-solver requires a flow_matching checkpoint")
        policy.flow_solver = flow_solver
    base_policy = policy
    from r3d.env_runner.frame_adapter_wrapper import (
        frame_evaluation_metadata,
        maybe_wrap_policy_for_environment,
    )
    eval_frame_config = cfg_get(cfg, "frame_adapter", None)
    frame_config = eval_frame_config
    frame_config_source = "evaluation_config"
    if frame_config is None:
        frame_config = cfg_get(train_cfg, "frame_adapter", None)
        frame_config_source = "checkpoint_training_config"
    policy = maybe_wrap_policy_for_environment(
        base_policy,
        frame_config,
        checkpoint_metadata=payload.get("frame_adapter"),
        require_checkpoint_metadata=bool(
            cfg_get(frame_config, "enabled", False)
        ),
    )
    frame_eval_metadata = frame_evaluation_metadata(
        policy,
        frame_config,
        payload.get("frame_adapter"),
        config_source=frame_config_source,
    )
    domain_roles = resolve_domain_roles(cfg, tasks)
    if is_franka_grasp_runner(train_cfg) or is_peg_assembly_runner(train_cfg):
        reset_distribution = str(nested_cfg_get(cfg, ("eval", "distribution"), "id"))
        validate_peg_domain_protocol(tasks, domain_roles, reset_distribution)
    task_logs = {}
    for task_name in tasks:
        output_dir = resolve_repo_path(cfg.experiment.results_dir) / "eval_videos" / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        runner = build_runner(
            cfg,
            train_cfg,
            task_name,
            output_dir,
            device,
            eval_episodes,
            eval_seed=eval_seed,
            episode_start=episode_start,
            robotwin_lean_observation=robotwin_lean_observation,
            robotwin_profile=robotwin_profile,
            robotwin_defer_intermediate_render=robotwin_defer_intermediate_render,
            robotwin_rt_spp=robotwin_rt_spp,
            robotwin_camera_shader=robotwin_camera_shader,
            fast=fast,
        )
        print(f"[EVAL] running {task_name} ({eval_episodes} episodes)", flush=True)
        infer_stats = {"calls": 0, "total_sec": 0.0}
        original_predict_action = policy.predict_action

        def timed_predict_action(*args, **kwargs):
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_predict_action(*args, **kwargs)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            infer_stats["calls"] += 1
            infer_stats["total_sec"] += time.perf_counter() - start
            return output

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        policy.predict_action = timed_predict_action
        eval_start = time.perf_counter()
        try:
            # RoboTwin initializes each episode with an expert planner/IK routine
            # that uses autograd. The runner itself already wraps policy inference
            # in no_grad, so do not disable gradients around RoboTwin environment
            # initialization.
            if is_robotwin_runner(train_cfg):
                task_log = json_safe(run_runner(runner, policy, cfg, train_cfg, checkpoint_path, task_name))
            else:
                with torch.no_grad():
                    task_log = json_safe(run_runner(runner, policy, cfg, train_cfg, checkpoint_path, task_name))
        finally:
            eval_wall_time = time.perf_counter() - eval_start
            policy.predict_action = original_predict_action
            # Each peg shape owns a native MuJoCo renderer/context.  Close it
            # before constructing the next shape so three-domain evaluation
            # does not accumulate GPU contexts and frame buffers.
            if (
                is_franka_grasp_runner(train_cfg)
                or is_peg_assembly_runner(train_cfg)
                or is_maniskill_runner(train_cfg)
            ) and hasattr(runner, "close"):
                runner.close()

        task_log["inference_calls"] = infer_stats["calls"]
        task_log["inference_total_sec"] = infer_stats["total_sec"]
        task_log["inference_time_per_call_sec"] = (
            infer_stats["total_sec"] / infer_stats["calls"] if infer_stats["calls"] else None
        )
        task_log["eval_wall_time_sec"] = eval_wall_time
        task_log["eval_time_per_predict_call_sec"] = (
            eval_wall_time / infer_stats["calls"] if infer_stats["calls"] else None
        )
        if task_name in domain_roles:
            task_log["domain_role"] = domain_roles[task_name]
            task_log["reset_distribution"] = str(
                nested_cfg_get(cfg, ("eval", "distribution"), "id")
            )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            task_log["gpu_mem_peak_allocated_mb"] = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 2)
            task_log["gpu_mem_peak_reserved_mb"] = torch.cuda.max_memory_reserved(torch.device(device)) / (1024 ** 2)
        task_logs[task_name] = task_log

    success_values = [
        value for value in (extract_success_metric(log) for log in task_logs.values())
        if value is not None
    ]
    reward_values = [
        value for value in (extract_reward_metric(log) for log in task_logs.values())
        if value is not None
    ]
    overall = {
        "num_tasks": len(task_logs),
        "mean_success_rate_across_tasks": float(np.mean(success_values)) if success_values else None,
        "mean_traj_reward_across_tasks": float(np.mean(reward_values)) if reward_values else None,
    }
    domain_groups = summarize_domain_groups(task_logs, domain_roles)
    for role, summary in domain_groups.items():
        overall[f"{role}_mean_success_rate"] = summary["mean_success_rate"]
    result = {
        "overall": overall,
        "domain_groups": domain_groups,
        "domain_roles": domain_roles,
        "tasks": task_logs,
        "policy_source": resolved_policy_source,
        "frame_adapter": payload.get(
            "frame_adapter", {"frame_adapter_enabled": False}
        ),
        "frame_evaluation": frame_eval_metadata,
    }
    if getattr(base_policy, "generation_type", None) == "flow_matching":
        solver = getattr(base_policy, "flow_solver", "euler")
        steps = int(base_policy.flow_num_inference_steps)
        result["flow_solver"] = solver
        result["flow_inference_steps"] = steps
        result["flow_actual_nfe"] = steps * {"euler": 1, "heun": 2, "rk4": 4}[solver]
    return result


def write_csv(path, task_logs):
    metric_names = sorted({
        key
        for log in task_logs.values()
        for key, value in log.items()
        if isinstance(value, (int, float))
    })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_name", *metric_names])
        writer.writeheader()
        for task_name, log in task_logs.items():
            row = {"task_name": task_name}
            row.update({key: log.get(key) for key in metric_names})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task list. Overrides config eval tasks.")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--policy-source", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument("--flow-inference-steps", type=int, default=None)
    parser.add_argument("--flow-solver", choices=["euler", "heun", "rk4"], default=None)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "Evaluation-only executed action chunk length. The checkpoint horizon "
            "and predicted action tensor are unchanged."
        ),
    )
    parser.add_argument(
        "--heatmap-intervention",
        choices=["none", "uniform", "shuffle", "spatial_roll", "inverse"],
        default=None,
        help="Evaluation-only intervention applied to ACT's heatmap before attention.",
    )
    parser.add_argument("--heatmap-intervention-seed", type=int, default=260727)
    parser.add_argument("--heatmap-intervention-roll", type=int, default=17)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Optional deterministic evaluation seed. For Adroit this fixes "
            "each episode reset and each policy-call Flow noise across NFE sweeps."
        ),
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="RoboTwin episode offset used for deterministic, non-overlapping eval shards.",
    )
    parser.add_argument(
        "--robotwin-lean-observation",
        action="store_true",
        help="Disable unused RoboTwin RGB extraction and wrist-camera capture; keep head-camera xyz+rgb point cloud.",
    )
    parser.add_argument(
        "--robotwin-profile",
        action="store_true",
        help="Record RoboTwin action, rendering, camera, RGB, and point-cloud timing.",
    )
    parser.add_argument(
        "--robotwin-defer-intermediate-render",
        action="store_true",
        help="Skip renderer updates inside RoboTwin's 250Hz physics loop; render only when observations are requested.",
    )
    parser.add_argument(
        "--robotwin-rt-spp",
        type=int,
        default=None,
        help="Experimental ray-tracing samples per pixel; default RoboTwin protocol is 32.",
    )
    parser.add_argument(
        "--robotwin-camera-shader",
        choices=["rt", "default"],
        default=None,
        help="Experimental SAPIEN camera shader; formal RoboTwin protocol uses rt.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override experiment.results_dir without modifying the experiment YAML.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use the domain's no-video fast path. MetaWorld skips unused RGB; "
            "ManiSkill disables per-episode FFmpeg recording."
        ),
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.heatmap_intervention is not None:
        with open_dict(cfg):
            cfg.policy.act.heatmap_intervention = args.heatmap_intervention
            cfg.policy.act.heatmap_intervention_seed = args.heatmap_intervention_seed
            cfg.policy.act.heatmap_intervention_roll = args.heatmap_intervention_roll
    if args.results_dir:
        cfg.experiment.results_dir = str(resolve_repo_path(args.results_dir))
    results_dir = resolve_repo_path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(cfg, args.tasks, args.max_tasks)
    if str(nested_cfg_get(cfg, ("task", "config"), "")).lower() == "peg_assembly":
        validate_peg_domain_protocol(
            tasks,
            resolve_domain_roles(cfg, tasks),
            nested_cfg_get(cfg, ("eval", "distribution"), "id"),
        )
    eval_episodes = args.eval_episodes or cfg.eval.eval_episodes
    device = args.device or cfg.training.device
    if args.checkpoint:
        checkpoint_path = resolve_repo_path(args.checkpoint)
    else:
        try:
            checkpoint_path = latest_checkpoint(cfg.experiment.output_dir)
        except FileNotFoundError:
            if not args.dry_run:
                raise
            checkpoint_path = resolve_repo_path(cfg.experiment.output_dir) / "checkpoints" / "<latest>.ckpt"

    print(f"[EVAL] config={args.config}", flush=True)
    print(f"[EVAL] checkpoint={checkpoint_path}", flush=True)
    print(f"[EVAL] device={device}", flush=True)
    print(f"[EVAL] policy_source={args.policy_source}", flush=True)
    print(f"[EVAL] eval_episodes={eval_episodes}", flush=True)
    print(f"[EVAL] eval_seed={args.eval_seed}", flush=True)
    print(f"[EVAL] n_action_steps={args.n_action_steps or 'checkpoint-default'}", flush=True)
    print(f"[EVAL] heatmap_intervention={args.heatmap_intervention or 'checkpoint-default'}", flush=True)
    print(f"[EVAL] episode_start={args.episode_start}", flush=True)
    print(f"[EVAL] robotwin_lean_observation={args.robotwin_lean_observation}", flush=True)
    print(f"[EVAL] robotwin_profile={args.robotwin_profile}", flush=True)
    print(f"[EVAL] robotwin_defer_intermediate_render={args.robotwin_defer_intermediate_render}", flush=True)
    print(f"[EVAL] robotwin_rt_spp={args.robotwin_rt_spp or 32}", flush=True)
    print(f"[EVAL] robotwin_camera_shader={args.robotwin_camera_shader or 'rt'}", flush=True)
    print(f"[EVAL] tasks={tasks}", flush=True)
    if args.dry_run:
        return

    # Respect user-provided HF offline settings; do not force them on first use.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    prepend_env_path(os.environ, "PATH", pathlib.Path(sys.executable).parent)
    configure_native_libraries(os.environ)
    if args.eval_seed is not None:
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_seed)

    result = run_eval(
        cfg,
        str(checkpoint_path),
        tasks,
        device,
        eval_episodes,
        policy_source=args.policy_source,
        flow_inference_steps=args.flow_inference_steps,
        flow_solver=args.flow_solver,
        n_action_steps=args.n_action_steps,
        eval_seed=args.eval_seed,
        episode_start=args.episode_start,
        robotwin_lean_observation=args.robotwin_lean_observation,
        robotwin_profile=args.robotwin_profile,
        robotwin_defer_intermediate_render=args.robotwin_defer_intermediate_render,
        robotwin_rt_spp=args.robotwin_rt_spp,
        robotwin_camera_shader=args.robotwin_camera_shader,
        fast=args.fast,
    )
    result["experiment"] = OmegaConf.to_container(cfg.experiment, resolve=True)
    result["checkpoint"] = str(checkpoint_path)
    result["eval_episodes"] = eval_episodes
    result["eval_seed"] = args.eval_seed
    result["n_action_steps_override"] = args.n_action_steps
    result["heatmap_intervention"] = args.heatmap_intervention or "checkpoint-default"
    result["heatmap_intervention_seed"] = args.heatmap_intervention_seed
    result["heatmap_intervention_roll"] = args.heatmap_intervention_roll
    result["episode_start"] = args.episode_start
    result["robotwin_lean_observation"] = args.robotwin_lean_observation
    result["robotwin_profile"] = args.robotwin_profile
    result["robotwin_defer_intermediate_render"] = args.robotwin_defer_intermediate_render
    result["robotwin_rt_spp"] = args.robotwin_rt_spp or 32
    result["robotwin_camera_shader"] = args.robotwin_camera_shader or "rt"
    solver = result.get("flow_solver", args.flow_solver or "euler")

    suffix = f"{checkpoint_path.stem}-{result['policy_source']}"
    # A Flow checkpoint can be evaluated with several Euler step counts. Keep
    # each result instead of silently overwriting the previous step setting.
    if result.get("flow_inference_steps") is not None:
        suffix += f"-flow_{solver}_steps_{result['flow_inference_steps']:02d}"
    json_path = results_dir / f"eval_{suffix}.json"
    csv_path = results_dir / f"eval_{suffix}_per_task.csv"
    json_path.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, result["tasks"])
    print(f"[EVAL] json saved to {json_path}", flush=True)
    print(f"[EVAL] csv saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
