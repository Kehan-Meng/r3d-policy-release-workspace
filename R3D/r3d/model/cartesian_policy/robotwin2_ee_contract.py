"""Frozen RoboTwin2 executable EE16 and demonstration-data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Robotwin2EE16Contract:
    version: str = "robotwin2_executable_ee16_world_v1"
    dataset_schema_version: str = "commanded_ee16_world_v1"
    action_dim: int = 16
    representation: str = "absolute_dual_ee_pose"
    source_frame: str = "world"
    quaternion_order: str = "wxyz"
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    controller_contract: str = "robotwin2_ee16_sequential_v1"
    point_cloud_frame: str = "world"
    extrinsic_cv_shape: tuple[int, int] = (3, 4)
    cam2world_gl_shape: tuple[int, int] = (4, 4)
    left_pose_slice: tuple[int, int] = (0, 7)
    left_gripper_index: int = 7
    right_pose_slice: tuple[int, int] = (8, 15)
    right_gripper_index: int = 15

    @property
    def action_layout(self) -> str:
        return (
            "left_xyz3+left_quaternion_wxyz4+left_gripper1+"
            "right_xyz3+right_quaternion_wxyz4+right_gripper1"
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["action_layout"] = self.action_layout
        metadata["extrinsic_cv_semantics"] = (
            "maps homogeneous world xyz1 to OpenCV camera xyz"
        )
        metadata["sample_semantics"] = (
            "observation immediately before a variable-duration macro EE command"
        )
        return metadata


ROBOTWIN2_EE16_CONTRACT = Robotwin2EE16Contract()


@dataclass(frozen=True)
class Robotwin2EE16FixedBudgetContract:
    version: str = "robotwin2_ee16_fixed_budget_v1"
    dataset_schema_version: str = "commanded_ee16_fixed_budget_world_v1"
    action_contract_version: str = ROBOTWIN2_EE16_CONTRACT.version
    representation: str = "absolute_target_hold"
    source_frame: str = "world"
    quaternion_order: str = "wxyz"
    physics_rate_hz: int = 250
    action_dim: int = 16
    joint_dim: int = 14

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata.update({
            "action_layout": ROBOTWIN2_EE16_CONTRACT.action_layout,
            "decision_semantics": (
                "observe, send one absolute EE16 target, execute at most H "
                "physics steps, then observe again"
            ),
            "incomplete_target_semantics": (
                "discard the remaining planned trajectory and replan the same "
                "immutable absolute target from the next observation"
            ),
            "short_trajectory_semantics": (
                "hold the final controller target until H physics steps have "
                "elapsed or the environment terminates"
            ),
            "target_reached_semantics": (
                "both planned arm trajectories were completely consumed; "
                "geometric tracking error is recorded separately"
            ),
        })
        return metadata


ROBOTWIN2_EE16_FIXED_BUDGET_CONTRACT = Robotwin2EE16FixedBudgetContract()


@dataclass(frozen=True)
class Robotwin2EE16LearningDatasetContract:
    version: str = "robotwin2_ee16_learning_dataset_v1"
    dataset_schema_version: str = "robotwin2_ee16_fixed_budget_zarr_v1"
    execution_contract_version: str = (
        ROBOTWIN2_EE16_FIXED_BUDGET_CONTRACT.version
    )
    physics_rate_hz: int = 250
    physics_step_budget: int = 250
    arm_joint_dim: int = 12
    command_joint_dim: int = 14
    position_tolerance_m: float = 0.03
    rotation_tolerance_rad: float = float(np.deg2rad(3.0))
    gripper_completion_semantics: str = (
        "command_schedule_consumed_not_physical_position_feedback"
    )

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata.update({
            "policy_interval_seconds": (
                self.physics_step_budget / self.physics_rate_hz
            ),
            "joint14_command_semantics": (
                "12 arm drive targets plus two normalized gripper commands"
            ),
            "arm_joint_position12_semantics": "physical articulation qpos",
            "arm_joint_velocity12_semantics": "physical articulation qvel",
            "macro_target_complete_semantics": (
                "both arm trajectories and both gripper command schedules "
                "consumed, and both EE poses within explicit tolerance"
            ),
            "acceptance_semantics": (
                "collection success and independent fresh replay success"
            ),
        })
        return metadata


ROBOTWIN2_EE16_LEARNING_DATASET_CONTRACT = (
    Robotwin2EE16LearningDatasetContract()
)


@dataclass(frozen=True)
class CommandedEE16Validation:
    status: str
    schema_version: str
    sample_count: int
    failures: tuple[str, ...]
    arrays: Mapping[str, Mapping[str, Any]]

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "failures": list(self.failures),
            "arrays": {key: dict(value) for key, value in self.arrays.items()},
        }


_REQUIRED_ARRAYS = (
    "point_cloud",
    "current_ee16_world",
    "commanded_ee16_world",
    "controller_received_ee16_world",
    "achieved_ee16_world",
    "extrinsic_cv",
    "cam2world_gl",
)

_FIXED_BUDGET_REQUIRED_ARRAYS = (
    "point_cloud",
    "achieved_ee16_world_before",
    "commanded_ee16_world",
    "controller_received_ee16_world",
    "achieved_ee16_world_after",
    "joint14_before",
    "joint14_after",
    "head_camera_extrinsic_cv",
    "cam2world_gl",
    "scheduled_physics_steps",
    "executed_physics_steps",
    "planned_trajectory_steps",
    "trajectory_prefix_steps",
    "hold_steps",
    "target_id",
    "expert_stage_id",
    "target_reached",
    "target_repeated",
    "planner_status_left",
    "planner_status_right",
    "controller_status",
    "terminal",
    "success",
    "transition_valid",
)


def _array_summary(arrays: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for key, value in arrays.items()
    }


def validate_commanded_ee16_episode(
    arrays: Mapping[str, Any],
    *,
    contract: Robotwin2EE16Contract = ROBOTWIN2_EE16_CONTRACT,
    quaternion_atol: float = 1e-6,
) -> CommandedEE16Validation:
    """Validate one macro-command episode and fail closed on ambiguous data."""
    failures: list[str] = []
    missing = [key for key in _REQUIRED_ARRAYS if key not in arrays]
    if missing:
        failures.append(f"missing required arrays: {missing}")
        return CommandedEE16Validation(
            status="failed",
            schema_version=contract.dataset_schema_version,
            sample_count=0,
            failures=tuple(failures),
            arrays=_array_summary(arrays),
        )

    point_cloud = np.asarray(arrays["point_cloud"])
    current = np.asarray(arrays["current_ee16_world"])
    commanded = np.asarray(arrays["commanded_ee16_world"])
    received = np.asarray(arrays["controller_received_ee16_world"])
    achieved = np.asarray(arrays["achieved_ee16_world"])
    extrinsic = np.asarray(arrays["extrinsic_cv"])
    cam2world = np.asarray(arrays["cam2world_gl"])
    sample_count = int(commanded.shape[0]) if commanded.ndim > 0 else 0

    expected_action_shape = (sample_count, contract.action_dim)
    if commanded.ndim != 2 or commanded.shape[1:] != (contract.action_dim,):
        failures.append(
            f"commanded_ee16_world has shape {commanded.shape}, expected [N,{contract.action_dim}]"
        )
    if current.shape != expected_action_shape:
        failures.append(
            f"current_ee16_world has shape {current.shape}, expected {expected_action_shape}"
        )
    if received.shape != expected_action_shape:
        failures.append(
            "controller_received_ee16_world shape differs from commanded action"
        )
    elif commanded.shape == received.shape and not np.array_equal(commanded, received):
        failures.append(
            "recorded command differs from controller received command"
        )
    if achieved.shape != expected_action_shape:
        failures.append(
            f"achieved_ee16_world has shape {achieved.shape}, expected {expected_action_shape}"
        )

    expected_extrinsic_shape = (sample_count, *contract.extrinsic_cv_shape)
    expected_cam2world_shape = (sample_count, *contract.cam2world_gl_shape)
    if extrinsic.shape != expected_extrinsic_shape:
        failures.append(
            f"extrinsic_cv has shape {extrinsic.shape}, expected {expected_extrinsic_shape}"
        )
    if cam2world.shape != expected_cam2world_shape:
        failures.append(
            f"cam2world_gl has shape {cam2world.shape}, expected {expected_cam2world_shape}"
        )
    if point_cloud.ndim != 3 or point_cloud.shape[0] != sample_count or point_cloud.shape[-1] < 3:
        failures.append(
            f"point_cloud has shape {point_cloud.shape}, expected [N,P,C>=3]"
        )

    finite_arrays = {
        "current_ee16_world": current,
        "commanded_ee16_world": commanded,
        "controller_received_ee16_world": received,
        "achieved_ee16_world": achieved,
        "extrinsic_cv": extrinsic,
        "cam2world_gl": cam2world,
        "point_cloud": point_cloud,
    }
    for name, value in finite_arrays.items():
        if not np.isfinite(value).all():
            failures.append(f"{name} contains NaN or Inf")

    if commanded.ndim == 2 and commanded.shape[1] == contract.action_dim:
        for arm, quaternion in (
            ("left", commanded[:, 3:7]),
            ("right", commanded[:, 11:15]),
        ):
            norm = np.linalg.norm(quaternion, axis=-1)
            if not np.allclose(norm, 1.0, atol=quaternion_atol):
                failures.append(f"{arm} command quaternion is not normalized")
        grippers = commanded[:, [
            contract.left_gripper_index,
            contract.right_gripper_index,
        ]]
        if not np.logical_and(
            grippers >= contract.gripper_min,
            grippers <= contract.gripper_max,
        ).all():
            failures.append(
                f"gripper command is outside [{contract.gripper_min},{contract.gripper_max}]"
            )

    return CommandedEE16Validation(
        status="passed" if not failures else "failed",
        schema_version=contract.dataset_schema_version,
        sample_count=sample_count,
        failures=tuple(failures),
        arrays=_array_summary(arrays),
    )


def validate_fixed_budget_ee16_episode(
    arrays: Mapping[str, Any],
    *,
    expected_physics_step_budget: int | None = None,
    contract: Robotwin2EE16FixedBudgetContract = (
        ROBOTWIN2_EE16_FIXED_BUDGET_CONTRACT
    ),
    quaternion_atol: float = 1e-6,
) -> CommandedEE16Validation:
    """Validate one fixed-budget EE16 episode and fail on temporal ambiguity."""
    failures: list[str] = []
    missing = [key for key in _FIXED_BUDGET_REQUIRED_ARRAYS if key not in arrays]
    if missing:
        failures.append(f"missing required arrays: {missing}")
        return CommandedEE16Validation(
            status="failed",
            schema_version=contract.dataset_schema_version,
            sample_count=0,
            failures=tuple(failures),
            arrays=_array_summary(arrays),
        )

    commanded = np.asarray(arrays["commanded_ee16_world"])
    sample_count = int(commanded.shape[0]) if commanded.ndim > 0 else 0
    action_shape = (sample_count, contract.action_dim)
    joint_shape = (sample_count, contract.joint_dim)

    action_arrays = {
        "achieved_ee16_world_before": np.asarray(
            arrays["achieved_ee16_world_before"]
        ),
        "commanded_ee16_world": commanded,
        "controller_received_ee16_world": np.asarray(
            arrays["controller_received_ee16_world"]
        ),
        "achieved_ee16_world_after": np.asarray(
            arrays["achieved_ee16_world_after"]
        ),
    }
    for name, value in action_arrays.items():
        if value.shape != action_shape:
            failures.append(f"{name} has shape {value.shape}, expected {action_shape}")

    received = action_arrays["controller_received_ee16_world"]
    if commanded.shape == received.shape and not np.array_equal(commanded, received):
        failures.append("recorded command differs from controller received command")

    for name in ("joint14_before", "joint14_after"):
        value = np.asarray(arrays[name])
        if value.shape != joint_shape:
            failures.append(f"{name} has shape {value.shape}, expected {joint_shape}")

    point_cloud = np.asarray(arrays["point_cloud"])
    if (
        point_cloud.ndim != 3
        or point_cloud.shape[0] != sample_count
        or point_cloud.shape[-1] < 3
    ):
        failures.append(
            f"point_cloud has shape {point_cloud.shape}, expected [N,P,C>=3]"
        )

    extrinsic = np.asarray(arrays["head_camera_extrinsic_cv"])
    expected_extrinsic = (
        sample_count,
        *ROBOTWIN2_EE16_CONTRACT.extrinsic_cv_shape,
    )
    if extrinsic.shape != expected_extrinsic:
        failures.append(
            f"head_camera_extrinsic_cv has shape {extrinsic.shape}, "
            f"expected {expected_extrinsic}"
        )
    cam2world = np.asarray(arrays["cam2world_gl"])
    expected_cam2world = (
        sample_count,
        *ROBOTWIN2_EE16_CONTRACT.cam2world_gl_shape,
    )
    if cam2world.shape != expected_cam2world:
        failures.append(
            f"cam2world_gl has shape {cam2world.shape}, expected {expected_cam2world}"
        )

    scalar_names = _FIXED_BUDGET_REQUIRED_ARRAYS[9:]
    scalar_arrays: dict[str, np.ndarray] = {}
    for name in scalar_names:
        value = np.asarray(arrays[name])
        scalar_arrays[name] = value
        if value.shape != (sample_count,):
            failures.append(f"{name} has shape {value.shape}, expected {(sample_count,)}")

    finite_names = (
        "point_cloud",
        "achieved_ee16_world_before",
        "commanded_ee16_world",
        "controller_received_ee16_world",
        "achieved_ee16_world_after",
        "joint14_before",
        "joint14_after",
        "head_camera_extrinsic_cv",
        "cam2world_gl",
        "scheduled_physics_steps",
        "executed_physics_steps",
        "planned_trajectory_steps",
        "trajectory_prefix_steps",
        "hold_steps",
        "target_id",
        "expert_stage_id",
    )
    for name in finite_names:
        if not np.isfinite(np.asarray(arrays[name])).all():
            failures.append(f"{name} contains NaN or Inf")

    if commanded.shape == action_shape:
        for arm, quaternion in (
            ("left", commanded[:, 3:7]),
            ("right", commanded[:, 11:15]),
        ):
            if not np.allclose(
                np.linalg.norm(quaternion, axis=-1),
                1.0,
                atol=quaternion_atol,
            ):
                failures.append(f"{arm} command quaternion is not normalized")
        grippers = commanded[:, [
            ROBOTWIN2_EE16_CONTRACT.left_gripper_index,
            ROBOTWIN2_EE16_CONTRACT.right_gripper_index,
        ]]
        if not np.logical_and(grippers >= 0.0, grippers <= 1.0).all():
            failures.append("gripper command is outside [0.0,1.0]")

    if all(
        scalar_arrays[name].shape == (sample_count,)
        for name in (
            "scheduled_physics_steps",
            "executed_physics_steps",
            "planned_trajectory_steps",
            "trajectory_prefix_steps",
            "hold_steps",
            "target_reached",
            "planner_status_left",
            "planner_status_right",
            "terminal",
            "success",
            "transition_valid",
        )
    ):
        scheduled = scalar_arrays["scheduled_physics_steps"].astype(np.int64)
        executed = scalar_arrays["executed_physics_steps"].astype(np.int64)
        planned = scalar_arrays["planned_trajectory_steps"].astype(np.int64)
        prefix = scalar_arrays["trajectory_prefix_steps"].astype(np.int64)
        hold = scalar_arrays["hold_steps"].astype(np.int64)
        terminal = scalar_arrays["terminal"].astype(bool)
        success = scalar_arrays["success"].astype(bool)
        valid = scalar_arrays["transition_valid"].astype(bool)
        reached = scalar_arrays["target_reached"].astype(bool)

        if np.any(scheduled <= 0):
            failures.append("scheduled_physics_steps must be positive")
        if expected_physics_step_budget is not None and not np.all(
            scheduled == int(expected_physics_step_budget)
        ):
            failures.append(
                "scheduled_physics_steps differs from the episode budget"
            )
        if np.any(executed < 0) or np.any(executed > scheduled):
            failures.append("executed_physics_steps must be in [0, scheduled]")
        if np.any(prefix < 0) or np.any(hold < 0) or np.any(planned < 0):
            failures.append("trajectory and hold step counts must be non-negative")
        if np.any(prefix > planned):
            failures.append(
                "trajectory_prefix_steps cannot exceed planned_trajectory_steps"
            )
        if np.any(prefix + hold != executed):
            failures.append(
                "trajectory_prefix_steps + hold_steps must equal executed_physics_steps"
            )
        if np.any(valid & ~terminal & (executed != scheduled)):
            failures.append(
                "every valid non-terminal transition must execute exactly H steps"
            )
        if np.any(~valid):
            failures.append(
                "episode contains transition_valid=false and is not trainable"
            )
        if np.any(success & ~terminal):
            failures.append("success transitions must also be terminal")
        if np.any(valid & reached & (prefix != planned)):
            failures.append(
                "target_reached requires the complete planned trajectory prefix"
            )
        if np.any(valid & (hold > 0) & ~reached):
            failures.append("hold_steps require target_reached=true")

        planner_left = scalar_arrays["planner_status_left"].astype(str)
        planner_right = scalar_arrays["planner_status_right"].astype(str)
        planner_success = np.logical_and(
            planner_left == "Success", planner_right == "Success"
        )
        if np.any(valid & ~planner_success):
            failures.append(
                "valid transitions require Success from both arm planners"
            )

    if all(
        scalar_arrays[name].shape == (sample_count,)
        for name in (
            "target_id",
            "expert_stage_id",
            "target_reached",
            "target_repeated",
        )
    ):
        target_id = scalar_arrays["target_id"].astype(np.int64)
        expert_stage_id = scalar_arrays["expert_stage_id"].astype(np.int64)
        reached = scalar_arrays["target_reached"].astype(bool)
        repeated = scalar_arrays["target_repeated"].astype(bool)
        if np.any(target_id < 0):
            failures.append("target_id must be non-negative")
        if np.any(expert_stage_id < 0):
            failures.append("expert_stage_id must be non-negative")
        repeated_indices = np.flatnonzero(repeated)
        if np.any(repeated_indices == 0) or any(
            target_id[index] != target_id[index - 1]
            for index in repeated_indices
        ):
            failures.append(
                "target_repeated requires the same target_id as the previous transition"
            )
        if sample_count > 0 and bool(repeated[0]):
            failures.append("the first transition cannot repeat a target")
        for index in range(1, sample_count):
            same_target = bool(target_id[index] == target_id[index - 1])
            if bool(repeated[index]) != same_target:
                failures.append(
                    "target_repeated must exactly identify adjacent equal target_id values"
                )
                break
            if not same_target and target_id[index] != target_id[index - 1] + 1:
                failures.append(
                    "target_id may only repeat or advance by exactly one"
                )
                break
            if not same_target and not bool(reached[index - 1]):
                failures.append(
                    "a new target cannot begin before the previous target is reached"
                )
                break
            if expert_stage_id[index] < expert_stage_id[index - 1]:
                failures.append("expert_stage_id must be monotonic")
                break
            if same_target and expert_stage_id[index] != expert_stage_id[index - 1]:
                failures.append(
                    "a repeated target cannot change expert_stage_id"
                )
                break
            if same_target and not np.array_equal(
                commanded[index], commanded[index - 1]
            ):
                failures.append(
                    "a repeated target must preserve the immutable EE16 command"
                )
                break
        if sample_count > 0:
            expected_ids = np.arange(int(target_id.max()) + 1)
            if target_id[0] != 0 or not np.array_equal(
                np.unique(target_id), expected_ids
            ):
                failures.append("target_id must be contiguous and start at zero")

    return CommandedEE16Validation(
        status="passed" if not failures else "failed",
        schema_version=contract.dataset_schema_version,
        sample_count=sample_count,
        failures=tuple(failures),
        arrays=_array_summary(arrays),
    )


_LEARNING_REQUIRED_ARRAYS = (
    "arm_joint_position12_before",
    "arm_joint_position12_after",
    "arm_joint_velocity12_before",
    "arm_joint_velocity12_after",
    "gripper_command2_before",
    "gripper_command2_after",
    "left_position_error_m",
    "left_rotation_error_rad",
    "right_position_error_m",
    "right_rotation_error_rad",
    "arm_trajectory_consumed",
    "gripper_command_consumed",
    "pose_within_tolerance",
    "macro_target_complete",
)


def validate_learning_ee16_episode(
    arrays: Mapping[str, Any],
    *,
    contract: Robotwin2EE16LearningDatasetContract = (
        ROBOTWIN2_EE16_LEARNING_DATASET_CONTRACT
    ),
) -> CommandedEE16Validation:
    """Validate the truthful Phase 4E learning schema on top of Phase 4D."""
    base = validate_fixed_budget_ee16_episode(
        arrays,
        expected_physics_step_budget=contract.physics_step_budget,
    )
    failures = list(base.failures)
    missing = [key for key in _LEARNING_REQUIRED_ARRAYS if key not in arrays]
    if missing:
        failures.append(f"missing Phase 4E arrays: {missing}")
        return CommandedEE16Validation(
            status="failed",
            schema_version=contract.dataset_schema_version,
            sample_count=base.sample_count,
            failures=tuple(failures),
            arrays=_array_summary(arrays),
        )

    n = base.sample_count
    for name in (
        "arm_joint_position12_before",
        "arm_joint_position12_after",
        "arm_joint_velocity12_before",
        "arm_joint_velocity12_after",
    ):
        value = np.asarray(arrays[name])
        if value.shape != (n, contract.arm_joint_dim):
            failures.append(
                f"{name} has shape {value.shape}, expected {(n, contract.arm_joint_dim)}"
            )
        if not np.isfinite(value).all():
            failures.append(f"{name} contains NaN or Inf")

    for name in ("gripper_command2_before", "gripper_command2_after"):
        value = np.asarray(arrays[name])
        if value.shape != (n, 2):
            failures.append(f"{name} has shape {value.shape}, expected {(n, 2)}")
        if not np.isfinite(value).all():
            failures.append(f"{name} contains NaN or Inf")
        if np.any(value < 0.0) or np.any(value > 1.0):
            failures.append(f"{name} is outside [0,1]")

    error_names = (
        "left_position_error_m",
        "left_rotation_error_rad",
        "right_position_error_m",
        "right_rotation_error_rad",
    )
    errors = {}
    for name in error_names:
        value = np.asarray(arrays[name], dtype=np.float64)
        errors[name] = value
        if value.shape != (n,):
            failures.append(f"{name} has shape {value.shape}, expected {(n,)}")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            failures.append(f"{name} must be finite and non-negative")

    boolean_names = (
        "arm_trajectory_consumed",
        "gripper_command_consumed",
        "pose_within_tolerance",
        "macro_target_complete",
    )
    booleans = {}
    for name in boolean_names:
        value = np.asarray(arrays[name], dtype=bool)
        booleans[name] = value
        if value.shape != (n,):
            failures.append(f"{name} has shape {value.shape}, expected {(n,)}")

    if all(value.shape == (n,) for value in errors.values()):
        expected_pose = (
            (errors["left_position_error_m"] <= contract.position_tolerance_m)
            & (errors["right_position_error_m"] <= contract.position_tolerance_m)
            & (errors["left_rotation_error_rad"] <= contract.rotation_tolerance_rad)
            & (errors["right_rotation_error_rad"] <= contract.rotation_tolerance_rad)
        )
        if booleans["pose_within_tolerance"].shape == (n,) and not np.array_equal(
            booleans["pose_within_tolerance"], expected_pose
        ):
            failures.append("pose_within_tolerance disagrees with recorded errors")

    if all(value.shape == (n,) for value in booleans.values()):
        expected_complete = (
            booleans["arm_trajectory_consumed"]
            & booleans["gripper_command_consumed"]
            & booleans["pose_within_tolerance"]
        )
        if not np.array_equal(
            booleans["macro_target_complete"], expected_complete
        ):
            failures.append("macro_target_complete violates its frozen definition")

        target_ids = np.asarray(arrays["target_id"], dtype=np.int64)
        for index in range(1, n):
            if (
                target_ids[index] != target_ids[index - 1]
                and not booleans["macro_target_complete"][index - 1]
            ):
                failures.append(
                    "a new target cannot begin before macro_target_complete"
                )
                break

    return CommandedEE16Validation(
        status="passed" if not failures else "failed",
        schema_version=contract.dataset_schema_version,
        sample_count=n,
        failures=tuple(failures),
        arrays=_array_summary(arrays),
    )
