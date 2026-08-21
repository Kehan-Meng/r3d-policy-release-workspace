#!/usr/bin/env python3
"""Build a self-contained ManiSkill2 camera-EE dataset without mutating source data.

The generated policy contract is:

* point_cloud: XYZ expressed in the static base-camera OpenCV frame, RGB unchanged;
* state: achieved TCP position (3), rotation 6D (6), finger qpos (2), and,
  for PickCube only, goal position in the same camera frame (3);
* action: spatial target-position delta (3), spatial target-rotation delta as
  axis-angle (3), and the native binary gripper command (1).

The deltas reproduce ``pd_ee_target_delta_pose`` semantics.  At the first row
of every episode the previous target is the achieved FK pose.  Afterwards it
is the preceding commanded target.  No state is carried across episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr
from numcodecs import Blosc

TASKS = {
    "PickCube": {
        "env_id": "PickCube-v0",
        "profile": "R3D/r3d/config/frame_transform/maniskill2_pickcube_base_camera_v1.yaml",
        "native_state_dim": 12,
        "policy_state_dim": 14,
    },
    "StackCube": {
        "env_id": "StackCube-v0",
        "profile": "R3D/r3d/config/frame_transform/maniskill2_stackcube_base_camera_v1.yaml",
        "native_state_dim": 9,
        "policy_state_dim": 11,
    },
    "PegInsertionSide": {
        "env_id": "PegInsertionSide-v0",
        "profile": "R3D/r3d/config/frame_transform/maniskill2_peginsertionside_base_camera_v1.yaml",
        "native_state_dim": 9,
        "policy_state_dim": 11,
    },
}

COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_matrix(repo_root: Path, relative_path: str) -> tuple[np.ndarray, Path]:
    path = repo_root / relative_path
    profile = yaml.safe_load(path.read_text())
    transforms = profile.get("transforms", [])
    if len(transforms) != 1:
        raise ValueError(f"Expected one static transform in {path}, got {len(transforms)}")
    provider = transforms[0].get("provider", {})
    if provider.get("type") != "static":
        raise ValueError(f"Only a static camera profile is valid for offline conversion: {path}")
    matrix = np.asarray(provider["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Invalid camera matrix shape {matrix.shape} in {path}")
    return matrix, path


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Stable matrix logarithm with principal angle in [0, pi]."""
    matrices = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    result = np.empty((len(matrices), 3), dtype=np.float64)
    for index, rotation in enumerate(matrices):
        cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        if angle < 1e-8:
            result[index] = 0.5 * np.array(
                [rotation[2, 1] - rotation[1, 2],
                 rotation[0, 2] - rotation[2, 0],
                 rotation[1, 0] - rotation[0, 1]]
            )
        elif np.pi - angle < 1e-5:
            # Eigenvector is stable at pi where the skew formula is singular.
            values, vectors = np.linalg.eig(rotation)
            axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
            axis /= np.linalg.norm(axis)
            result[index] = axis * angle
        else:
            axis = np.array(
                [rotation[2, 1] - rotation[1, 2],
                 rotation[0, 2] - rotation[2, 0],
                 rotation[1, 0] - rotation[0, 1]]
            ) / (2.0 * np.sin(angle))
            result[index] = axis * angle
    return result.reshape(np.asarray(matrix).shape[:-2] + (3,))


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angle, dtype=np.float64).reshape(-1, 3)
    result = np.empty((len(vectors), 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for index, vector in enumerate(vectors):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-10:
            x, y, z = vector
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            result[index] = identity + skew
            continue
        axis = vector / angle
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        result[index] = identity + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    return result.reshape(np.asarray(axis_angle).shape[:-1] + (3, 3))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _transform_cloud_preserving_zeros(cloud: np.ndarray, transform: np.ndarray) -> np.ndarray:
    output = np.asarray(cloud, dtype=np.float32).copy()
    xyz = output[..., :3]
    valid = np.any(xyz != 0.0, axis=-1)
    xyz[valid] = _transform_points(xyz[valid], transform).astype(np.float32)
    return output


def _episode_bounds(episode_ends: np.ndarray, count: int | None) -> tuple[list[tuple[int, int]], int]:
    selected = len(episode_ends) if count is None else min(count, len(episode_ends))
    bounds: list[tuple[int, int]] = []
    start = 0
    for end in episode_ends[:selected]:
        bounds.append((start, int(end)))
        start = int(end)
    return bounds, start


def _make_fk(task: str):
    # Imported lazily so --help and offline profile inspection do not require
    # the benchmark runtime.
    try:
        import gymnasium as gym
        import mani_skill2.envs  # noqa: F401 - registers environments
    except ImportError as exc:
        raise RuntimeError(
            "ManiSkill2 is required for FK conversion. Run "
            "environment/install_benchmarks.sh maniskill2 first."
        ) from exc

    env_id = TASKS[task]["env_id"]
    env = gym.make(
        env_id,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        shader_dir="trivial",
        renderer_kwargs={"offscreen_only": True},
    )
    env.reset(seed=0)
    native = env.unwrapped
    robot = native.agent.robot
    link_index = next(
        index for index, link in enumerate(robot.get_links()) if link.name == "panda_hand_tcp"
    )
    return env, robot.create_pinocchio_model(), robot.pose, link_index


def _fk_world(pin_model: Any, robot_pose: Any, link_index: int, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((len(qpos), 3), dtype=np.float64)
    rotations = np.empty((len(qpos), 3, 3), dtype=np.float64)
    for row, configuration in enumerate(qpos):
        pin_model.compute_forward_kinematics(configuration)
        world_pose = robot_pose * pin_model.get_link_pose(link_index)
        positions[row] = world_pose.p
        rotations[row] = _quat_wxyz_to_matrix(np.asarray(world_pose.q))
    return positions, rotations


def _create_array(group: Any, name: str, shape: tuple[int, ...], dtype: Any, chunks: tuple[int, ...]):
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=COMPRESSOR,
        overwrite=False,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    spec = TASKS[args.task]
    source_path = Path(args.source or repo_root / "R3D/data" / f"{spec['env_id']}.zarr").resolve()
    output_path = Path(
        args.output
        or repo_root / "R3D/data/maniskill2" / f"{spec['env_id']}_camera_ee_v1.zarr"
    ).resolve()
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform, profile_path = _profile_matrix(repo_root, spec["profile"])
    rotation_cw = transform[:3, :3]
    source = zarr.open(str(source_path), mode="r")
    episode_ends_native = np.asarray(source["meta/episode_ends"], dtype=np.int64)
    bounds, total_rows = _episode_bounds(episode_ends_native, args.episodes)
    episode_ends = np.asarray([end for _, end in bounds], dtype=np.int64)

    native_state = source["data/state"]
    native_action = source["data/action"]
    target_ee_world = source["data/target_ee"]
    source_cloud = source["data/point_cloud"]
    if native_state.shape[1] != spec["native_state_dim"]:
        raise ValueError(f"Unexpected native state shape: {native_state.shape}")
    if native_action.shape[1] != 8 or target_ee_world.shape[1] != 7:
        raise ValueError("Expected native action8 and target_ee7")

    output = zarr.open(str(output_path), mode="w")
    data = output.create_group("data")
    meta = output.create_group("meta")
    row_chunk = min(args.row_chunk, max(total_rows, 1))
    state_out = _create_array(data, "state", (total_rows, spec["policy_state_dim"]), np.float32, (row_chunk, spec["policy_state_dim"]))
    action_out = _create_array(data, "action", (total_rows, 7), np.float32, (row_chunk, 7))
    cloud_out = _create_array(data, "point_cloud", (total_rows,) + source_cloud.shape[1:], np.float32, (min(args.cloud_chunk, max(total_rows, 1)),) + source_cloud.shape[1:])
    target_out = _create_array(data, "target_ee", (total_rows, 9), np.float32, (row_chunk, 9))
    native_state_out = _create_array(data, "native_state", (total_rows, native_state.shape[1]), np.float32, (row_chunk, native_state.shape[1]))
    native_action_out = _create_array(data, "native_action", (total_rows, 8), np.float32, (row_chunk, 8))
    target_world_out = _create_array(data, "target_ee_world", (total_rows, 7), np.float32, (row_chunk, 7))
    episode_out = _create_array(meta, "episode_ends", episode_ends.shape, np.int64, (min(100, len(episode_ends)),))
    episode_out[:] = episode_ends

    env, pin_model, robot_pose, ee_link_index = _make_fk(args.task)
    max_delta_reconstruction_position = 0.0
    max_delta_reconstruction_rotation_deg = 0.0
    max_pose_roundtrip_position = 0.0
    action_position_norms: list[np.ndarray] = []
    action_rotation_norms: list[np.ndarray] = []

    for episode_index, (start, end) in enumerate(bounds):
        states = np.asarray(native_state[start:end], dtype=np.float64)
        actions = np.asarray(native_action[start:end], dtype=np.float64)
        targets_world = np.asarray(target_ee_world[start:end], dtype=np.float64)
        achieved_position_w, achieved_rotation_w = _fk_world(
            pin_model, robot_pose, ee_link_index, states[:, :9]
        )
        achieved_position_c = _transform_points(achieved_position_w, transform)
        achieved_rotation_c = rotation_cw @ achieved_rotation_w
        target_position_c = _transform_points(targets_world[:, :3], transform)
        target_rotation_w = _quat_wxyz_to_matrix(targets_world[:, 3:7])
        target_rotation_c = rotation_cw @ target_rotation_w

        policy_state = np.concatenate(
            [
                achieved_position_c,
                _matrix_to_rotation_6d(achieved_rotation_c),
                states[:, 7:9],
            ],
            axis=-1,
        )
        if args.task == "PickCube":
            goal_c = _transform_points(states[:, 9:12], transform)
            policy_state = np.concatenate([policy_state, goal_c], axis=-1)

        previous_position_c = np.empty_like(target_position_c)
        previous_rotation_c = np.empty_like(target_rotation_c)
        previous_position_c[0] = achieved_position_c[0]
        previous_rotation_c[0] = achieved_rotation_c[0]
        if len(states) > 1:
            previous_position_c[1:] = target_position_c[:-1]
            previous_rotation_c[1:] = target_rotation_c[:-1]
        delta_position_c = target_position_c - previous_position_c
        delta_rotation_c = target_rotation_c @ np.swapaxes(previous_rotation_c, -1, -2)
        delta_axis_angle_c = _matrix_to_axis_angle(delta_rotation_c)
        policy_action = np.concatenate(
            [delta_position_c, delta_axis_angle_c, actions[:, 7:8]], axis=-1
        )
        target_camera = np.concatenate(
            [target_position_c, _matrix_to_rotation_6d(target_rotation_c)], axis=-1
        )

        reconstructed_position = previous_position_c + delta_position_c
        reconstructed_rotation = _axis_angle_to_matrix(delta_axis_angle_c) @ previous_rotation_c
        position_error = np.linalg.norm(reconstructed_position - target_position_c, axis=-1)
        rotation_cosine = np.clip(
            (np.trace(np.swapaxes(reconstructed_rotation, -1, -2) @ target_rotation_c, axis1=-2, axis2=-1) - 1) * 0.5,
            -1.0,
            1.0,
        )
        rotation_error_deg = np.degrees(np.arccos(rotation_cosine))
        world_again = (achieved_position_c - transform[:3, 3]) @ rotation_cw
        pose_roundtrip = np.linalg.norm(world_again - achieved_position_w, axis=-1)
        max_delta_reconstruction_position = max(max_delta_reconstruction_position, float(position_error.max()))
        max_delta_reconstruction_rotation_deg = max(max_delta_reconstruction_rotation_deg, float(rotation_error_deg.max()))
        max_pose_roundtrip_position = max(max_pose_roundtrip_position, float(pose_roundtrip.max()))
        action_position_norms.append(np.linalg.norm(delta_position_c, axis=-1))
        action_rotation_norms.append(np.linalg.norm(delta_axis_angle_c, axis=-1))

        state_out[start:end] = policy_state.astype(np.float32)
        action_out[start:end] = policy_action.astype(np.float32)
        target_out[start:end] = target_camera.astype(np.float32)
        native_state_out[start:end] = states.astype(np.float32)
        native_action_out[start:end] = actions.astype(np.float32)
        target_world_out[start:end] = targets_world.astype(np.float32)

        for cloud_start in range(start, end, args.cloud_chunk):
            cloud_end = min(cloud_start + args.cloud_chunk, end)
            cloud = np.asarray(source_cloud[cloud_start:cloud_end], dtype=np.float32)
            cloud_out[cloud_start:cloud_end] = _transform_cloud_preserving_zeros(cloud, transform)

        if (episode_index + 1) % args.progress_every == 0 or episode_index + 1 == len(bounds):
            print(f"[{args.task}] episodes {episode_index + 1}/{len(bounds)}, rows {end}/{total_rows}", flush=True)

    env.close()
    position_norms = np.concatenate(action_position_norms)
    rotation_norms = np.concatenate(action_rotation_norms)
    report = {
        "status": "passed" if (
            max_delta_reconstruction_position < 1e-8
            and max_delta_reconstruction_rotation_deg < 1e-4
            and max_pose_roundtrip_position < 1e-10
        ) else "failed",
        "task": args.task,
        "source": str(source_path),
        "output": str(output_path),
        "episodes": len(bounds),
        "rows": total_rows,
        "contract": {
            "controller": "pd_ee_target_delta_pose",
            "point_cloud": "camera_xyz3_plus_rgb3",
            "state": "camera_achieved_tcp_xyz3_plus_rot6d6_plus_finger_qpos2" + ("_plus_camera_goal_xyz3" if args.task == "PickCube" else ""),
            "action": "camera_spatial_target_delta_xyz3_plus_axis_angle3_plus_gripper1",
            "first_step_reference": "achieved_tcp_fk",
            "later_step_reference": "previous_commanded_target_ee",
        },
        "profile": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "validation": {
            "max_delta_reconstruction_position_m": max_delta_reconstruction_position,
            "max_delta_reconstruction_rotation_deg": max_delta_reconstruction_rotation_deg,
            "max_pose_roundtrip_position_m": max_pose_roundtrip_position,
        },
        "action_distribution": {
            "translation_norm_p50": float(np.quantile(position_norms, 0.50)),
            "translation_norm_p95": float(np.quantile(position_norms, 0.95)),
            "translation_norm_max": float(position_norms.max()),
            "rotation_norm_rad_p50": float(np.quantile(rotation_norms, 0.50)),
            "rotation_norm_rad_p95": float(np.quantile(rotation_norms, 0.95)),
            "rotation_norm_rad_max": float(rotation_norms.max()),
        },
    }
    output.attrs.update(
        {
            "contract_version": "maniskill2_camera_ee_v1",
            "contract_json": json.dumps(report["contract"], sort_keys=True),
            "source_zarr": str(source_path),
            "camera_profile": str(profile_path),
            "camera_profile_sha256": report["profile_sha256"],
            "validation_status": report["status"],
        }
    )
    report_path = output_path.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if report["status"] != "passed":
        raise RuntimeError(f"Converted data failed validation; see {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int, help="Convert only the first N episodes for smoke testing")
    parser.add_argument("--row-chunk", type=int, default=100)
    parser.add_argument("--cloud-chunk", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
