#!/usr/bin/env python3
"""Prepare RoboTwin2 data for the current R3D codebase.

The pipeline is intentionally conservative:
- existing zarr files are validated first;
- RoboTwin2 simulation collection is opt-in via --collect;
- HDF5-to-zarr conversion is opt-in via --convert-missing or explicit raw dirs;
- generated Hydra configs use absolute zarr/text paths to avoid cwd issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
R3D_ROOT = REPO_ROOT / "R3D"
ROBOTWIN_ROOT = R3D_ROOT / "r3d" / "env" / "robotwin2"
DEFAULT_TASKS_JSON = SCRIPT_DIR / "example_tasks.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "third_party" / "robotwin2"

OBJECT_NAMES = {
    "move_playingcard_away": "the playing card",
    "beat_block_hammer": "the hammer",
    "lift_pot": "the pot",
}

ARM_NAMES = {
    "move_playingcard_away": "the arm",
    "beat_block_hammer": "the arm",
    "lift_pot": "both arms",
}

FALLBACK_TEXT = {
    "move_playingcard_away": "move the playing card away from its initial position",
    "beat_block_hammer": "use the hammer to beat the block",
    "lift_pot": "grasp the pot and lift it up",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_path(path: Optional[str], base: Path = REPO_ROOT) -> Optional[Path]:
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_task_entries(tasks_json: Path) -> List[Dict]:
    payload = load_json(tasks_json)
    tasks = payload.get("tasks", payload)
    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError(f"{tasks_json} must contain a non-empty tasks list")
    normalized = []
    for item in tasks:
        task = dict(item)
        if "task_name" not in task:
            raise ValueError(f"Task entry is missing task_name: {task}")
        task.setdefault("setting", "demo_clean")
        task.setdefault("expert_data_num", 50)
        task.setdefault(
            "zarr_path",
            str(R3D_ROOT / "data" / f"{task['task_name']}-{task['setting']}-{task['expert_data_num']}.zarr"),
        )
        normalized.append(task)
    return normalized


def fill_instruction_template(task_name: str, text: str) -> str:
    text = text.replace("{A}", OBJECT_NAMES.get(task_name, "the object"))
    text = text.replace("{a}", ARM_NAMES.get(task_name, "the arm"))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_instruction(
    task_name: str,
    robotwin_root: Path,
    instruction_type: str,
    instruction_index: int,
    base_text: Optional[Dict[str, str]] = None,
) -> str:
    instruction_file = robotwin_root / "description" / "task_instruction" / f"{task_name}.json"
    if instruction_file.exists():
        data = load_json(instruction_file)
        if instruction_type in ("seen", "unseen"):
            candidates = data.get(instruction_type) or []
            if candidates:
                idx = min(max(instruction_index, 0), len(candidates) - 1)
                return fill_instruction_template(task_name, str(candidates[idx]))
        if instruction_type == "full_description" and data.get("full_description"):
            return fill_instruction_template(task_name, str(data["full_description"]))
        if data.get("full_description"):
            return fill_instruction_template(task_name, str(data["full_description"]))

    if base_text and task_name in base_text:
        return str(base_text[task_name])
    return FALLBACK_TEXT.get(task_name, task_name.replace("_", " "))


def build_text_json(
    tasks: List[Dict],
    robotwin_root: Path,
    instruction_type: str,
    instruction_index: int,
    base_text_json: Optional[Path],
    include_aliases: bool = True,
) -> Dict[str, str]:
    base_text = load_json(base_text_json) if base_text_json and base_text_json.exists() else {}
    output = {}
    for task in tasks:
        task_name = task["task_name"]
        setting = task["setting"]
        expert_num = int(task.get("expert_data_num", 50))
        text = choose_instruction(
            task_name=task_name,
            robotwin_root=robotwin_root,
            instruction_type=instruction_type,
            instruction_index=instruction_index,
            base_text=base_text,
        )
        output[task_name] = text
        if include_aliases:
            output[f"{task_name}-{setting}"] = text
            output[f"{task_name}-{setting}-{expert_num}"] = text
            output[f"{task_name}_{setting}"] = text
            output[f"{task_name}_{setting}_{expert_num}"] = text
    return output


def sorted_episode_files(raw_dir: Path) -> List[Path]:
    data_dir = raw_dir / "data"
    if not data_dir.exists():
        return []
    files = []
    for path in data_dir.glob("episode*.hdf5"):
        match = re.search(r"episode(\d+)\.hdf5$", path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def hdf5_episode_to_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    import h5py

    with h5py.File(path, "r") as f:
        if "pointcloud" not in f:
            raise KeyError(f"{path} is missing dataset pointcloud")
        if "joint_action" not in f or "vector" not in f["joint_action"]:
            raise KeyError(f"{path} is missing dataset joint_action/vector")

        point_cloud = np.asarray(f["pointcloud"][()], dtype=np.float32)
        state = np.asarray(f["joint_action"]["vector"][()], dtype=np.float32)

        target_ee = None
        if "endpose" in f:
            endpose = f["endpose"]
            required = ["left_endpose", "left_gripper", "right_endpose", "right_gripper"]
            if all(key in endpose for key in required):
                left_endpose = np.asarray(endpose["left_endpose"][()], dtype=np.float32)
                right_endpose = np.asarray(endpose["right_endpose"][()], dtype=np.float32)
                left_gripper = np.asarray(endpose["left_gripper"][()], dtype=np.float32).reshape(-1, 1)
                right_gripper = np.asarray(endpose["right_gripper"][()], dtype=np.float32).reshape(-1, 1)
                target_ee = np.concatenate(
                    [left_endpose, left_gripper, right_endpose, right_gripper],
                    axis=-1,
                )

    if point_cloud.ndim != 3:
        raise ValueError(f"{path} pointcloud must be [T,N,C], got {point_cloud.shape}")
    if state.ndim != 2:
        raise ValueError(f"{path} joint_action/vector must be [T,D], got {state.shape}")

    length = min(point_cloud.shape[0], state.shape[0])
    if target_ee is not None:
        length = min(length, target_ee.shape[0])
    if length < 2:
        raise ValueError(f"{path} has too few frames: {length}")

    point_cloud = point_cloud[:length]
    state = state[:length]
    action = np.empty_like(state, dtype=np.float32)
    action[:-1] = state[1:]
    action[-1] = state[-1]

    if target_ee is not None:
        target_ee = target_ee[:length]
        shifted_target_ee = np.empty_like(target_ee, dtype=np.float32)
        shifted_target_ee[:-1] = target_ee[1:]
        shifted_target_ee[-1] = target_ee[-1]
        target_ee = shifted_target_ee

    return state, action, point_cloud, target_ee


def convert_hdf5_to_zarr(
    raw_dir: Path,
    output_zarr: Path,
    overwrite: bool = False,
    include_target_ee: bool = True,
) -> Dict:
    import zarr

    episode_files = sorted_episode_files(raw_dir)
    if not episode_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {raw_dir / 'data'}")

    states = []
    actions = []
    point_clouds = []
    target_ees = []
    episode_ends = []
    total = 0
    for path in episode_files:
        state, action, point_cloud, target_ee = hdf5_episode_to_arrays(path)
        states.append(state)
        actions.append(action)
        point_clouds.append(point_cloud)
        if target_ee is not None:
            target_ees.append(target_ee)
        total += state.shape[0]
        episode_ends.append(total)

    state_arr = np.concatenate(states, axis=0).astype(np.float32)
    action_arr = np.concatenate(actions, axis=0).astype(np.float32)
    point_cloud_arr = np.concatenate(point_clouds, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)
    target_ee_arr = None
    if include_target_ee and len(target_ees) == len(episode_files):
        target_ee_arr = np.concatenate(target_ees, axis=0).astype(np.float32)

    if output_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"{output_zarr} already exists; pass --overwrite-zarr to replace it")
        if output_zarr.suffix != ".zarr":
            raise ValueError(f"Refusing to overwrite non-zarr path: {output_zarr}")
        shutil.rmtree(output_zarr)

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(output_zarr))
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    data.create_dataset(
        "state",
        data=state_arr,
        chunks=(min(100, state_arr.shape[0]), state_arr.shape[1]),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    data.create_dataset(
        "action",
        data=action_arr,
        chunks=(min(100, action_arr.shape[0]), action_arr.shape[1]),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    data.create_dataset(
        "point_cloud",
        data=point_cloud_arr,
        chunks=(min(100, point_cloud_arr.shape[0]), point_cloud_arr.shape[1], point_cloud_arr.shape[2]),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    if target_ee_arr is not None:
        data.create_dataset(
            "target_ee",
            data=target_ee_arr,
            chunks=(min(100, target_ee_arr.shape[0]), target_ee_arr.shape[1]),
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )
    meta.create_dataset(
        "episode_ends",
        data=episode_ends_arr,
        chunks=(min(100, episode_ends_arr.shape[0]),),
        dtype="int64",
        compressor=compressor,
        overwrite=True,
    )

    return {
        "raw_dir": str(raw_dir),
        "zarr_path": str(output_zarr),
        "num_episodes": int(len(episode_files)),
        "num_steps": int(total),
        "state_shape": list(state_arr.shape),
        "action_shape": list(action_arr.shape),
        "point_cloud_shape": list(point_cloud_arr.shape),
        "has_target_ee": target_ee_arr is not None,
    }


def simple_text_lookup(text_map: Dict[str, str], task_name: str, zarr_path: Path) -> Optional[str]:
    candidates = [
        task_name,
        task_name.replace("_", "-"),
        zarr_path.name,
        zarr_path.name[:-5] if zarr_path.name.endswith(".zarr") else zarr_path.name,
    ]
    suffixes = [
        "-demo_clean-50",
        "-demo_randomized-50",
        "_demo_clean_50",
        "_demo_randomized_50",
        "-demo_clean",
        "-demo_randomized",
        "_demo_clean",
        "_demo_randomized",
        "-50",
        "_50",
    ]
    expanded = []
    for item in candidates:
        expanded.extend([item, item.replace("_", "-"), item.replace("-", "_")])
    for item in list(expanded):
        for suffix in suffixes:
            if item.endswith(suffix):
                stripped = item[: -len(suffix)]
                expanded.extend([stripped, stripped.replace("_", "-"), stripped.replace("-", "_")])
    seen = set()
    for key in expanded:
        if key in seen:
            continue
        seen.add(key)
        if key in text_map:
            return text_map[key]
    return None


def validate_zarr(zarr_path: Path, task_name: str, text_map: Dict[str, str]) -> Dict:
    import zarr

    row = {
        "ok": False,
        "task_name": task_name,
        "zarr_path": str(zarr_path),
        "errors": "",
    }
    errors = []
    if not zarr_path.exists():
        errors.append("missing zarr")
        row["errors"] = "; ".join(errors)
        return row

    try:
        root = zarr.open(str(zarr_path), "r")
        data = root["data"]
        meta = root["meta"]
        for key in ["state", "action", "point_cloud"]:
            if key not in data:
                errors.append(f"missing data/{key}")
        if "episode_ends" not in meta:
            errors.append("missing meta/episode_ends")
        if errors:
            row["errors"] = "; ".join(errors)
            return row

        state_shape = tuple(data["state"].shape)
        action_shape = tuple(data["action"].shape)
        point_cloud_shape = tuple(data["point_cloud"].shape)
        episode_ends_shape = tuple(meta["episode_ends"].shape)
        row.update({
            "state_shape": list(state_shape),
            "action_shape": list(action_shape),
            "point_cloud_shape": list(point_cloud_shape),
            "episode_ends_shape": list(episode_ends_shape),
            "num_steps": int(state_shape[0]),
            "num_episodes": int(episode_ends_shape[0]),
            "has_target_ee": "target_ee" in data,
        })

        if state_shape[0] != action_shape[0] or state_shape[0] != point_cloud_shape[0]:
            errors.append("state/action/point_cloud first dimension mismatch")
        if len(state_shape) != 2 or state_shape[1] != 14:
            errors.append(f"expected state [T,14], got {state_shape}")
        if len(action_shape) != 2 or action_shape[1] != 14:
            errors.append(f"expected action [T,14], got {action_shape}")
        if len(point_cloud_shape) != 3 or point_cloud_shape[1:] != (1024, 6):
            errors.append(f"expected point_cloud [T,1024,6], got {point_cloud_shape}")
        if episode_ends_shape[0] == 0:
            errors.append("no episodes")
        else:
            last_end = int(meta["episode_ends"][-1])
            if last_end != state_shape[0]:
                errors.append(f"last episode_end {last_end} != num_steps {state_shape[0]}")

        text = simple_text_lookup(text_map, task_name, zarr_path)
        row["text"] = text or ""
        if not text:
            errors.append("missing text prompt")

    except Exception as exc:
        errors.append(repr(exc))

    row["errors"] = "; ".join(errors)
    row["ok"] = len(errors) == 0
    return row


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ok",
        "task_name",
        "num_episodes",
        "num_steps",
        "state_shape",
        "action_shape",
        "point_cloud_shape",
        "episode_ends_shape",
        "has_target_ee",
        "text",
        "zarr_path",
        "errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def yaml_quote(value: str) -> str:
    return json.dumps(str(value))


def task_config_yaml(tasks: List[Dict], text_json_path: Path) -> str:
    lines = [
        "name: robotwin2_move_card_beat_hammer_lift_pot",
        "",
        "task_entries:",
    ]
    for task in tasks:
        zarr_path = resolve_path(task["zarr_path"])
        lines.extend([
            f"  - task_name: {task['task_name']}",
            f"    setting: {task['setting']}",
            f"    difficulty: {task.get('difficulty', '')}",
            f"    zarr_path: {yaml_quote(str(zarr_path))}",
            f"    text_command: {task['task_name']}",
        ])
    lines.extend([
        "",
        "shape_meta: &shape_meta",
        "  obs:",
        "    point_cloud:",
        "      shape: [1024, 6]",
        "      type: point_cloud",
        "    agent_pos:",
        "      shape: [14]",
        "      type: low_dim",
        "    task_onehot:",
        f"      shape: [{len(tasks)}]",
        "      type: low_dim",
        "  action:",
        "    shape: [14]",
        "",
        "env_runner:",
        "  _target_: r3d.env_runner.robotwin2_runner.RoboTwin2Runner",
        "  output_dir: ${hydra:run.dir}",
        "  task_name: null",
        "  task_entries: ${task.task_entries}",
        "  eval_task_name: ${eval_task_name}",
        "  seed: 0",
        "  eval_episodes: 100",
        "  max_steps: 2000",
        "  n_obs_steps: ${n_obs_steps}",
        "  n_action_steps: ${n_action_steps}",
        "  task_config: demo_clean",
        "  instruction_type: unseen",
        "  action_space_type: joint",
        "  head_camera_type: D435",
        "  save_video: true",
        "  tqdm_interval_sec: 5.0",
        "",
        "dataset:",
        "  _target_: r3d.dataset.multi_robotwin_dataset.MultiRobotwinDataset",
        "  tasks: ${task.task_entries}",
        "  horizon: ${horizon}",
        "  pad_before: ${eval:'${n_obs_steps}-1'}",
        "  pad_after: ${eval:'${n_action_steps}-1'}",
        "  seed: 42",
        "  val_ratio: 0.02",
        "  max_train_episodes: null",
        "  balanced_sampling: true",
        "  use_data_augmentation: ${data_augmentation.use_augmentation}",
        "  pc_xyz_noise_std: ${data_augmentation.pc_xyz_noise_std}",
        "  pc_rgb_noise_std: ${data_augmentation.pc_rgb_noise_std}",
        "  agent_pos_noise_std: ${data_augmentation.agent_pos_noise_std}",
        "  use_color_jitter: ${data_augmentation.use_color_jitter}",
        "  brightness_range: ${data_augmentation.brightness_range}",
        "  contrast_range: ${data_augmentation.contrast_range}",
        "  saturation_range: ${data_augmentation.saturation_range}",
        "  use_target_ee: ${policy.use_target_ee}",
        f"  text_json_path: {yaml_quote(str(text_json_path))}",
        "  return_text: ${policy.use_text}",
        "  strict_text_lookup: false",
        "",
    ])
    return "\n".join(lines)


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def train_commands(tasks: List[Dict], text_json_path: Path, installed_task_name: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"REPO_ROOT={yaml_quote(str(REPO_ROOT))}",
        "cd \"$REPO_ROOT/R3D\"",
        "",
        "# Single-task examples",
    ]
    for task in tasks:
        zarr_path = resolve_path(task["zarr_path"])
        run_name = f"robotwin2_{task['task_name']}_{task['setting']}_seed0"
        lines.extend([
            "",
            f"# {task.get('display_name', task['task_name'])}",
            "python train.py \\",
            "  --config-name=r3d_robotwin2.yaml \\",
            "  task=robotwin2_demo_task \\",
            f"  task_name={task['task_name']} \\",
            f"  setting={task['setting']} \\",
            f"  hydra.run.dir={yaml_quote(str(REPO_ROOT / 'experiments' / 'runs' / 'robotwin2' / run_name))} \\",
            f"  task.dataset.zarr_path={yaml_quote(str(zarr_path))} \\",
            f"  task.dataset.text_json_path={yaml_quote(str(text_json_path))} \\",
            f"  policy.text_json_path={yaml_quote(str(text_json_path))} \\",
            "  training.device=cuda:0 \\",
            "  training.use_ddp=false \\",
            "  logging.mode=offline \\",
            "  checkpoint.save_ckpt=true",
        ])
    lines.extend([
        "",
        "# Multi-task three-task example. Run --install-task-config first, or install",
        "# the generated YAML into R3D/r3d/config/task with the same name.",
        "python train.py \\",
        "  --config-name=r3d_robotwin2_multitask.yaml \\",
        f"  task={installed_task_name} \\",
        "  task_name=robotwin2_move_card_beat_hammer_lift_pot \\",
        f"  hydra.run.dir={yaml_quote(str(REPO_ROOT / 'experiments' / 'runs' / 'robotwin2' / 'robotwin2_three_tasks_seed0'))} \\",
        f"  policy.text_json_path={yaml_quote(str(text_json_path))} \\",
        "  training.device=cuda:0 \\",
        "  training.use_ddp=false \\",
        "  logging.mode=offline \\",
        "  checkpoint.save_ckpt=true",
        "",
    ])
    return "\n".join(lines)


def collect_task(task: Dict, robotwin_root: Path, gpu: Optional[str] = None) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(robotwin_root / "script" / "collect_data.py"),
        task["task_name"],
        task["setting"],
    ]
    subprocess.run(cmd, cwd=str(robotwin_root), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN2_ROOT", DEFAULT_RUNTIME_ROOT)),
        help="Pinned RoboTwin checkout created by environment/install_benchmarks.sh.",
    )
    parser.add_argument("--base-text-json", type=Path, default=R3D_ROOT / "data" / "text.json")
    parser.add_argument("--instruction-type", choices=["seen", "unseen", "full_description"], default="unseen")
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--collect", action="store_true", help="Run RoboTwin2 simulation collection before validation")
    parser.add_argument("--collect-gpu", default=None)
    parser.add_argument("--convert-missing", action="store_true", help="Convert raw HDF5 to zarr if zarr is missing")
    parser.add_argument("--overwrite-zarr", action="store_true")
    parser.add_argument("--install-task-config", action="store_true")
    parser.add_argument(
        "--installed-task-name",
        default="robotwin2_move_card_beat_hammer_lift_pot",
        help="Hydra task config name to install/use for multi-task training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_task_entries(args.tasks_json)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    text_json_path = output_dir / "robotwin2_three_tasks_text.json"
    text_map = build_text_json(
        tasks=tasks,
        robotwin_root=args.robotwin_root.resolve(),
        instruction_type=args.instruction_type,
        instruction_index=args.instruction_index,
        base_text_json=args.base_text_json,
    )
    write_json(text_json_path, text_map)

    conversions = []
    for task in tasks:
        task_name = task["task_name"]
        setting = task["setting"]
        zarr_path = resolve_path(task["zarr_path"])
        if args.collect:
            collect_task(task, args.robotwin_root.resolve(), gpu=args.collect_gpu)
        if not zarr_path.exists() and args.convert_missing:
            raw_dir = resolve_path(task.get("raw_dir")) or (
                args.robotwin_root.resolve() / "data" / task_name / setting
            )
            conversions.append(convert_hdf5_to_zarr(
                raw_dir=raw_dir,
                output_zarr=zarr_path,
                overwrite=args.overwrite_zarr,
            ))

    rows = []
    for task in tasks:
        rows.append(validate_zarr(resolve_path(task["zarr_path"]), task["task_name"], text_map))

    summary = {
        "tasks_json": str(args.tasks_json.resolve()),
        "text_json": str(text_json_path),
        "num_tasks": len(tasks),
        "conversions": conversions,
        "valid_zarrs": int(sum(1 for row in rows if row["ok"])),
        "bad_zarrs": int(sum(1 for row in rows if not row["ok"])),
        "rows": rows,
    }
    write_json(output_dir / "robotwin2_three_tasks_validate.json", summary)
    write_csv(output_dir / "robotwin2_three_tasks_validate.csv", rows)

    generated_task_config = output_dir / f"{args.installed_task_name}.yaml"
    write_text(generated_task_config, task_config_yaml(tasks, text_json_path))

    if args.install_task_config:
        install_path = R3D_ROOT / "r3d" / "config" / "task" / f"{args.installed_task_name}.yaml"
        write_text(install_path, generated_task_config.read_text(encoding="utf-8"))
        summary["installed_task_config"] = str(install_path)
        write_json(output_dir / "robotwin2_three_tasks_validate.json", summary)

    commands_path = output_dir / "run_train_robotwin2_three_tasks.sh"
    write_text(commands_path, train_commands(tasks, text_json_path, args.installed_task_name), executable=True)

    print(json.dumps({
        "text_json": str(text_json_path),
        "validate_json": str(output_dir / "robotwin2_three_tasks_validate.json"),
        "validate_csv": str(output_dir / "robotwin2_three_tasks_validate.csv"),
        "task_config": str(generated_task_config),
        "train_commands": str(commands_path),
        "valid_zarrs": summary["valid_zarrs"],
        "bad_zarrs": summary["bad_zarrs"],
    }, indent=2))
    return 0 if summary["bad_zarrs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
