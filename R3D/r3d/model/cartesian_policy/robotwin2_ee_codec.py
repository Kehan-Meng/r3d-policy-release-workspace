"""Camera-frame policy codec for the frozen RoboTwin2 world EE16 contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from r3d.model.geometry.representations import transform_absolute_pose
from r3d.model.geometry.se3 import invert_matrix, transform_points
from r3d.model.geometry.serialization import stable_sha256


@dataclass(frozen=True)
class Robotwin2EE16CameraCodecContract:
    version: str = "robotwin2_ee16_camera_quaternion_v1"
    source_action_contract: str = "robotwin2_executable_ee16_world_v1"
    source_dataset_contract: str = "robotwin2_ee16_learning_dataset_v1"
    source_frame: str = "world"
    policy_frame: str = "head_camera_cv"
    action_dim: int = 16
    policy_state_dim: int = 30
    quaternion_order: str = "wxyz"
    point_cloud_xyz_dim: int = 3
    intrinsic_joint_dim: int = 14
    policy_action_representation: str = "absolute_dual_ee16_quaternion"
    quaternion_sign_rule: str = "episode_temporal_dot_nonnegative"

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata.update({
            "action_layout": (
                "left_xyz3+left_quaternion_wxyz4+left_gripper1+"
                "right_xyz3+right_quaternion_wxyz4+right_gripper1"
            ),
            "policy_state_layout": (
                "current_ee16_camera+native_intrinsic_joint14_command"
            ),
            "extrinsic_semantics": (
                "T_head_camera_cv_from_world; homogeneous extension of stored 3x4"
            ),
            "execution_order": (
                "policy camera EE16 -> inverse codec world EE16 -> native controller"
            ),
        })
        metadata["codec_hash"] = stable_sha256(metadata)
        return metadata


ROBOTWIN2_EE16_CAMERA_CODEC_CONTRACT = Robotwin2EE16CameraCodecContract()


def homogeneous_extrinsic_cv(extrinsic_cv: np.ndarray) -> np.ndarray:
    """Convert T_camera_from_world to SE(3), removing storage round-off only."""
    extrinsic = np.asarray(extrinsic_cv)
    if not np.issubdtype(extrinsic.dtype, np.floating):
        raise TypeError("extrinsic_cv must use a floating dtype")
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic_cv must have shape [...,3,4], got {extrinsic.shape}")
    if not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic_cv contains NaN or Inf")
    rotation = extrinsic[..., :3, :3]
    identity = np.eye(3, dtype=extrinsic.dtype)
    if not np.allclose(
        np.swapaxes(rotation, -1, -2) @ rotation,
        identity,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("extrinsic_cv rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("extrinsic_cv rotation determinant is not +1")
    # SAPIEN stores this calibration at float32 precision even when the Zarr
    # container later promotes it to float64. Project only the resulting
    # ~1e-8 round-off to the nearest SO(3) matrix before strict geometry use.
    u, _, vh = np.linalg.svd(rotation)
    projected = u @ vh
    negative = np.linalg.det(projected) < 0.0
    if np.any(negative):
        u = np.array(u, copy=True)
        u[..., :, -1] = np.where(negative[..., None], -u[..., :, -1], u[..., :, -1])
        projected = u @ vh
    matrix = np.zeros(extrinsic.shape[:-2] + (4, 4), dtype=extrinsic.dtype)
    matrix[..., :3, :3] = projected
    matrix[..., :3, 3] = extrinsic[..., :3, 3]
    matrix[..., 3, 3] = 1.0
    return matrix


def _validate_ee16(action: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(action)
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype")
    if value.shape[-1] != 16:
        raise ValueError(f"{name} must have last dimension 16, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def canonicalize_ee16_temporally(
    action: np.ndarray,
    episode_ends: np.ndarray,
) -> np.ndarray:
    """Choose q/-q signs continuously without crossing episode boundaries."""
    output = np.array(_validate_ee16(action, "action"), copy=True)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or ends[-1] != len(output):
        raise ValueError("episode_ends must be non-empty and end at action length")
    if np.any(np.diff(np.concatenate(([0], ends))) <= 0):
        raise ValueError("episode_ends must be strictly increasing")
    start = 0
    for end in ends:
        for quaternion_slice in (slice(3, 7), slice(11, 15)):
            for index in range(start + 1, int(end)):
                if np.dot(
                    output[index - 1, quaternion_slice],
                    output[index, quaternion_slice],
                ) < 0.0:
                    output[index, quaternion_slice] *= -1.0
        start = int(end)
    return output


class Robotwin2EE16CameraCodec:
    """Exact world/camera codec; it owns no parameters and performs no clipping."""

    contract = ROBOTWIN2_EE16_CAMERA_CODEC_CONTRACT

    @staticmethod
    def _transform_ee16(
        action: np.ndarray,
        matrix_target_from_source: np.ndarray,
    ) -> np.ndarray:
        value = _validate_ee16(action, "ee16")
        matrix = np.asarray(matrix_target_from_source, dtype=value.dtype)
        left = transform_absolute_pose(
            value[..., :7],
            matrix,
            representation="xyz_quaternion",
            quaternion_order="wxyz",
        )
        right = transform_absolute_pose(
            value[..., 8:15],
            matrix,
            representation="xyz_quaternion",
            quaternion_order="wxyz",
        )
        return np.concatenate(
            (left, value[..., 7:8].copy(), right, value[..., 15:16].copy()),
            axis=-1,
        )

    def point_cloud_to_camera(
        self,
        point_cloud_world: np.ndarray,
        extrinsic_cv: np.ndarray,
    ) -> np.ndarray:
        point_cloud = np.asarray(point_cloud_world)
        if point_cloud.ndim < 2 or point_cloud.shape[-1] < 3:
            raise ValueError(
                f"point_cloud_world must have shape [...,P,C>=3], got {point_cloud.shape}"
            )
        matrix = homogeneous_extrinsic_cv(
            np.asarray(extrinsic_cv, dtype=point_cloud.dtype)
        )
        xyz = transform_points(point_cloud[..., :3], matrix)
        return np.concatenate((xyz, point_cloud[..., 3:].copy()), axis=-1)

    def ee16_world_to_camera(
        self,
        action_world: np.ndarray,
        extrinsic_cv: np.ndarray,
    ) -> np.ndarray:
        value = _validate_ee16(action_world, "action_world")
        matrix = homogeneous_extrinsic_cv(
            np.asarray(extrinsic_cv, dtype=value.dtype)
        )
        return self._transform_ee16(value, matrix)

    def ee16_camera_to_world(
        self,
        action_camera: np.ndarray,
        extrinsic_cv: np.ndarray,
    ) -> np.ndarray:
        value = _validate_ee16(action_camera, "action_camera")
        matrix = homogeneous_extrinsic_cv(
            np.asarray(extrinsic_cv, dtype=value.dtype)
        )
        return self._transform_ee16(value, invert_matrix(matrix))

    def policy_state(
        self,
        current_ee16_camera: np.ndarray,
        intrinsic_joint14: np.ndarray,
    ) -> np.ndarray:
        ee = _validate_ee16(current_ee16_camera, "current_ee16_camera")
        joint = np.asarray(intrinsic_joint14)
        if joint.shape[:-1] != ee.shape[:-1] or joint.shape[-1] != 14:
            raise ValueError(
                "intrinsic_joint14 must match EE leading dimensions and end in 14"
            )
        if not np.isfinite(joint).all():
            raise ValueError("intrinsic_joint14 contains NaN or Inf")
        return np.concatenate((ee, joint), axis=-1)
