import argparse
import os
import pathlib
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from release_utils import (
    REPO_ROOT,
    configure_native_libraries,
    load_experiment_config,
    prepend_env_path,
    resolve_repo_path,
)

PATH_LIKE_ENCODER_KEYS = {
    "bpe_path",
    "checkpoint_path",
    "heatmap_config_dir",
    "pretrained_weights_path",
}


def bool_str(value):
    return "true" if bool(value) else "false"


def hydra_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return bool_str(value)
    if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return str(value)


def _build_nested_overrides(prefix, d):
    """Recursively build hydra CLI overrides for nested dicts.

    Example:
        _build_nested_overrides("policy.act_text_align_config", {
            "lambda_max": 1e-5,
            "diagnostic": {"enabled": True, "every_n_steps": 200},
        })
        # => [
        #   "policy.act_text_align_config.lambda_max=1e-05",
        #   "policy.act_text_align_config.diagnostic.enabled=true",
        #   "policy.act_text_align_config.diagnostic.every_n_steps=200",
        # ]
    """
    overrides = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}"
        if isinstance(value, dict) or OmegaConf.is_dict(value):
            overrides.extend(_build_nested_overrides(full_key, value))
        else:
            overrides.append(f"{full_key}={hydra_value(value)}")
    return overrides


def build_train_command(cfg):
    exp = cfg.experiment
    train = cfg.training
    task = cfg.task
    data = cfg.data
    policy = cfg.policy
    removed_options = []
    for key in ("mq_spatial_scope", "mq_common_gradient_deflation"):
        if policy.act.get(key) is not None:
            removed_options.append(f"policy.act.{key}")
    if policy.get("dense_residual_reader") is not None:
        removed_options.append("policy.dense_residual_reader")
    mq_diversity = policy.get("mq_diversity")
    if mq_diversity is not None and mq_diversity.get("flow") is not None:
        removed_options.append("policy.mq_diversity.flow")
    if removed_options:
        raise ValueError(
            "Removed MQ experiment options are not supported: "
            + ", ".join(removed_options)
            + ". Keep competitive MQ and policy.mq_diversity.raw only."
        )
    performance = cfg.get("performance", {})
    is_multitask = (
        "multitask" in str(cfg.hydra.get("config_name", ""))
        or "multitask" in str(task.get("config", ""))
    )
    task_setting = task.get("setting", data.get("task_setting", None))
    instruction_type = task.get("instruction_type", data.get("instruction_type", None))

    overrides = [
        f"--config-name={cfg.hydra.config_name}",
        f"task={task.config}",
        f"task_name={task.name}",
        f"hydra.run.dir={resolve_repo_path(exp.output_dir)}",
        f"training.device={train.device}",
        f"training.seed={train.seed}",
        f"training.debug={bool_str(train.debug)}",
        f"training.resume={bool_str(train.resume)}",
        f"training.use_ddp={bool_str(train.use_ddp)}",
        f"training.use_bfloat16={bool_str(train.use_bfloat16)}",
        f"training.num_epochs={train.num_epochs}",
        f"training.gradient_accumulate_every={train.get('gradient_accumulate_every', 1)}",
        f"training.max_train_steps={hydra_value(train.max_train_steps)}",
        f"training.max_val_steps={hydra_value(train.max_val_steps)}",
        f"training.rollout_every={train.rollout_every}",
        f"training.checkpoint_every={train.checkpoint_every}",
        f"training.val_every={train.val_every}",
        f"training.sample_every={train.sample_every}",
        f"dataloader.batch_size={train.batch_size}",
        f"dataloader.num_workers={train.num_workers}",
        f"val_dataloader.batch_size={train.val_batch_size}",
        f"val_dataloader.num_workers={train.num_workers}",
        f"logging.mode={train.logging_mode}",
        f"checkpoint.save_ckpt={bool_str(train.save_ckpt)}",
        *([f"++task.env_runner.num_points={task.num_points}"] if (
            not str(task.get("config", "")).startswith("adroit")
        ) else []),
        f"task.shape_meta.obs.point_cloud.shape=[{task.num_points},6]",
        f"policy.use_text={bool_str(policy.use_text)}",
        f"++policy.use_text_for_global_cond={bool_str(policy.get('use_text_for_global_cond', True))}",
        f"policy.task_name={task.name}",
        f"policy.use_act={bool_str(policy.use_act)}",
        f"policy.act_config.num_queries={policy.act.num_queries}",
        f"policy.act_config.num_heads={policy.act.num_heads}",
        f"policy.act_config.heatmap_mode={policy.act.heatmap_mode}",
        f"policy.act_config.use_pseudo_heatmap={bool_str(policy.act.use_pseudo_heatmap)}",
        f"policy.act_config.drop_cls_token={bool_str(policy.act.get('drop_cls_token', True))}",
        f"exp_name={exp.name}",
    ]
    if not is_multitask:
        overrides.append(
            f"task.dataset.zarr_path={resolve_repo_path(data.train_zarr_path)}"
        )
        for key in ("max_train_episodes", "val_ratio"):
            if data.get(key) is not None:
                overrides.append(f"task.dataset.{key}={hydra_value(data.get(key))}")

    data_augmentation = cfg.get("data_augmentation")
    if data_augmentation is not None:
        overrides.extend(_build_nested_overrides(
            "data_augmentation", data_augmentation
        ))

    sequence = cfg.get("sequence")
    if sequence is not None:
        if sequence.get("horizon") is not None:
            overrides.append(f"horizon={sequence.horizon}")
        if sequence.get("n_obs_steps") is not None:
            overrides.append(f"n_obs_steps={sequence.n_obs_steps}")
        if sequence.get("n_action_steps") is not None:
            overrides.append(f"n_action_steps={sequence.n_action_steps}")

    for key in (
        "rollout_eval_episodes",
        "rollout_milestone_epochs",
        "rollout_milestone_eval_episodes",
        "checkpoint_milestone_epochs",
        "checkpoint_start_epoch",
        "encoder_lr_scale",
        "encoder_unfreeze_epoch",
    ):
        if train.get(key) is not None:
            overrides.append(f"++training.{key}={hydra_value(train.get(key))}")

    if train.get("rollout_save_video") is not None:
        overrides.append(
            f"++task.env_runner.save_video={bool_str(train.get('rollout_save_video'))}"
        )

    # Pure offline training does not need to construct a simulator-backed
    # runner. This is especially useful for ManiSkill, whose SAPIEN runtime is
    # otherwise initialized even when periodic rollouts are disabled.
    if train.get("disable_env_runner", False):
        overrides.append("task.env_runner=null")

    if policy.get("generation_type") is not None:
        overrides.append(f"++policy.generation_type={policy.generation_type}")

    frame_adapter = cfg.get("frame_adapter")
    if frame_adapter is not None:
        overrides.extend(_build_nested_overrides(
            "++frame_adapter", frame_adapter
        ))

    if policy.get("num_inference_steps") is not None:
        overrides.append(f"policy.num_inference_steps={policy.num_inference_steps}")

    if policy.get("flow_matching") is not None:
        for item in _build_nested_overrides(
            "policy.flow_matching", policy.flow_matching
        ):
            overrides.append("++" + item)

    if task_setting is not None:
        overrides.extend([
            f"setting={task_setting}",
            f"task.env_runner.task_config={task_setting}",
        ])

    if instruction_type is not None:
        overrides.append(f"task.env_runner.instruction_type={instruction_type}")

    if policy.get("pointnet_type") is not None:
        overrides.append(f"policy.pointnet_type={policy.pointnet_type}")

    if policy.get("clip_model_name") is not None:
        overrides.append(f"policy.clip_model_name={policy.clip_model_name}")
    if policy.get("freeze_clip") is not None:
        overrides.append(f"policy.freeze_clip={bool_str(policy.freeze_clip)}")
    if policy.get("text_feat_dim") is not None:
        overrides.append(f"policy.text_feat_dim={policy.text_feat_dim}")

    if policy.act.get("pseudo_heatmap_source") is not None:
        overrides.append(f"++policy.act_config.pseudo_heatmap_source={policy.act.pseudo_heatmap_source}")
    if policy.act.get("heatmap_gamma") is not None:
        overrides.append(
            f"++policy.act_config.heatmap_gamma={hydra_value(policy.act.heatmap_gamma)}"
        )
    for key in (
        "competitive_cross1",
        "competitive_cross2",
        "competitive_temperature",
    ):
        if policy.act.get(key) is not None:
            overrides.append(
                f"++policy.act_config.{key}={hydra_value(policy.act.get(key))}"
            )
    for key in (
        "heatmap_intervention",
        "heatmap_intervention_seed",
        "heatmap_intervention_roll",
    ):
        if policy.act.get(key) is not None:
            overrides.append(
                f"++policy.act_config.{key}={hydra_value(policy.act.get(key))}"
            )

    if policy.get("use_act_text_align") is not None:
        overrides.append(f"policy.use_act_text_align={bool_str(policy.use_act_text_align)}")
    if policy.get("act_text_align_config") is not None:
        overrides.extend(_build_nested_overrides(
            "++policy.act_text_align_config", policy.act_text_align_config
        ))

    if policy.get("mq_diversity") is not None:
        overrides.extend(_build_nested_overrides(
            "++policy.mq_diversity", policy.mq_diversity
        ))

    if policy.get("pointsam_checkpoint") is not None:
        overrides.append(
            "policy.pointcloud_encoder_cfg.pretrained_weights_path="
            f"{resolve_repo_path(policy.pointsam_checkpoint)}"
        )

    if policy.get("pointcloud_encoder") is not None:
        for key, value in policy.pointcloud_encoder.items():
            if key == "pointsam_root" or key == "open_clip_python_path":
                raise ValueError(
                    f"policy.pointcloud_encoder.{key} is a legacy path injection and "
                    "is not supported in the public package. Install PointSAM and "
                    "OpenCLIP in the release environment instead."
                )
            if key in PATH_LIKE_ENCODER_KEYS and value is not None:
                value = resolve_repo_path(value)
            overrides.append(f"++policy.pointcloud_encoder_cfg.{key}={hydra_value(value)}")

    if performance.get("persistent_workers") is not None:
        persistent = bool_str(performance.get("persistent_workers"))
        overrides.extend([
            f"dataloader.persistent_workers={persistent}",
            f"val_dataloader.persistent_workers={persistent}",
        ])

    if policy.use_text:
        overrides.extend([
            f"++task.dataset.return_text=true",
            f"++task.dataset.text_json_path={resolve_repo_path(data.text_json_path)}",
            f"policy.text_json_path={resolve_repo_path(data.text_json_path)}",
        ])
        if not is_multitask:
            overrides.extend([
                f"++task.dataset.task_name={task.name}",
                f"++task.dataset.text_command={task.name}",
            ])

    # --- instruction prompt augmentation ---
    inst_aug = cfg.get("instruction_aug")
    if inst_aug and inst_aug.get("enabled"):
        overrides.extend([
            f"++task.dataset.instruction_aug.enabled={bool_str(inst_aug['enabled'])}",
            "++task.dataset.instruction_aug.bank_path="
            f"{resolve_repo_path(inst_aug['bank_path'])}",
            f"++task.dataset.instruction_aug.apply_in_train={bool_str(inst_aug.get('apply_in_train', True))}",
            f"++task.dataset.instruction_aug.apply_in_val={bool_str(inst_aug.get('apply_in_val', False))}",
        ])

    if "checkpoint" in cfg:
        ckpt = cfg.checkpoint
        overrides.extend([
            f"checkpoint.topk.monitor_key={ckpt.monitor_key}",
            f"checkpoint.topk.mode={ckpt.mode}",
            f"checkpoint.topk.k={ckpt.k}",
        ])

    runtime = cfg.get("runtime", {})
    python = str(runtime.get("python") or sys.executable)
    r3d_dir = resolve_repo_path(runtime.get("r3d_dir"), default="R3D")
    return [python, str(r3d_dir / "train.py"), *overrides]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    skip_marker = pathlib.Path(f"{args.config}.skip")
    if skip_marker.exists():
        print(f"[TRAIN] skipped by marker: {skip_marker}", flush=True)
        return

    cfg = load_experiment_config(args.config)
    output_dir = resolve_repo_path(cfg.experiment.output_dir)
    results_dir = resolve_repo_path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    gpu_id = os.environ.get("MKH_FORCE_GPU_ID", str(cfg.runtime.gpu_id))

    command = build_train_command(cfg)
    printable = f"CUDA_VISIBLE_DEVICES={gpu_id} " + shlex.join(command)
    command_path = results_dir / "train_command.txt"
    command_path.write_text(printable + "\n", encoding="utf-8")

    print(printable, flush=True)
    print(f"[TRAIN] command saved to {command_path}", flush=True)
    if not args.execute:
        print("[TRAIN] dry-run only. Add --execute to start training.", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    performance = cfg.get("performance", {})
    if performance.get("timm_fused_attention") is not None:
        env["TIMM_FUSED_ATTN"] = "1" if performance.get("timm_fused_attention") else "0"
    # Public configs point at a repository-local CLIP snapshot. Custom configs
    # may use a Hugging Face model id, so do not force offline mode here.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MUJOCO_GL", "egl")
    if cfg.training.logging_mode in ("online", "offline", "disabled"):
        env.setdefault("WANDB_MODE", str(cfg.training.logging_mode))
    prepend_env_path(env, "PATH", pathlib.Path(command[0]).expanduser().resolve().parent)
    runtime = cfg.get("runtime", {})
    r3d_dir = resolve_repo_path(runtime.get("r3d_dir"), default="R3D")
    if str(cfg.task.get("config", "")).lower().startswith("maniskill"):
        maniskill_root = r3d_dir / "r3d" / "env" / "maniskill2"
        prepend_env_path(env, "PYTHONPATH", maniskill_root)
        prepend_env_path(env, "PYTHONPATH", maniskill_root / "warp_maniskill")
    configure_native_libraries(env)

    stdout_path = results_dir / "train_stdout.log"
    stderr_path = results_dir / "train_stderr.log"
    print(f"[TRAIN] stdout → {stdout_path}", flush=True)
    print(f"[TRAIN] stderr → {stderr_path}", flush=True)
    log_mode = "a" if cfg.training.resume else "w"
    with open(stdout_path, log_mode, encoding="utf-8") as f_out, \
         open(stderr_path, log_mode, encoding="utf-8") as f_err:
        if log_mode == "a":
            f_out.write("\n# Cached/resumed launch\n")
            f_err.write("\n# Cached/resumed launch\n")
        f_out.write(f"# Command: {printable}\n\n")
        f_out.flush()
        subprocess.run(
            command,
            cwd=str(resolve_repo_path(runtime.get("project_dir"), default=REPO_ROOT)),
            env=env,
            stdout=f_out,
            stderr=f_err,
            check=True,
        )


if __name__ == "__main__":
    main()
