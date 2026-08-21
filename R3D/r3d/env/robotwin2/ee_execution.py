"""Fail-closed execution helpers for RoboTwin2 Cartesian action chunks."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np


EE_ACTION_DIM = 16
EE_ACTION_LAYOUT = (
    "left_xyz3+left_quaternion_wxyz4+left_gripper1+"
    "right_xyz3+right_quaternion_wxyz4+right_gripper1"
)
EE_EXECUTION_CONTRACT_VERSION = "robotwin2_ee16_sequential_v1"
EE_FIXED_BUDGET_CONTRACT_VERSION = "robotwin2_ee16_fixed_budget_v1"


def validate_ee_action_chunk(actions: Any) -> np.ndarray:
    """Return a finite ``[K, 16]`` EE action chunk or fail closed."""
    array = np.asarray(actions)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != EE_ACTION_DIM:
        raise ValueError(
            f"RoboTwin2 EE actions must have shape [K,{EE_ACTION_DIM}], "
            f"got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("RoboTwin2 EE action chunks cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError("RoboTwin2 EE action chunk contains NaN or Inf")
    # Planners may normalize quaternion views in place. Own this buffer so
    # internal controller operations cannot mutate the policy output or the
    # immutable command-provenance record.
    return np.array(array, copy=True)


def _normalize_target_result(
    raw_result: Any,
    target_index: int,
    requested_target: np.ndarray,
) -> dict[str, Any]:
    if not isinstance(raw_result, Mapping):
        raise RuntimeError(
            "RoboTwin2 task.take_action did not return the structured controller "
            "result required by the EE16 execution contract"
        )
    result = dict(raw_result)
    if "controller_success" not in result:
        raise RuntimeError(
            "RoboTwin2 EE controller result is missing controller_success"
        )
    if "received_action_ee16" not in result:
        raise RuntimeError(
            "RoboTwin2 EE controller result is missing received_action_ee16"
        )
    received = np.asarray(result["received_action_ee16"])
    if received.shape != (EE_ACTION_DIM,) or not np.array_equal(
        received, requested_target
    ):
        raise RuntimeError(
            "RoboTwin2 EE controller boundary changed the requested EE16 target"
        )
    result["received_action_ee16"] = received.tolist()
    result["target_index"] = int(target_index)
    result["command_exact"] = True
    return result


def execute_ee_action_chunk(
    task: Any,
    actions: Any,
    *,
    action_type: str = "ee",
    stop_on_failure: bool = True,
    physics_step_budget: int | None = None,
) -> dict[str, Any]:
    """Execute every Cartesian target exactly once and retain controller status.

    ``Base_Task.take_action`` accepts one Cartesian target per call. Keeping the
    chunk loop here prevents the old behavior where only index zero influenced
    the arm planners while all gripper values were interpolated.
    """
    if action_type not in {"ee", "delta_ee"}:
        raise ValueError(f"Unsupported Cartesian action type: {action_type}")

    if physics_step_budget is not None:
        if (
            isinstance(physics_step_budget, bool)
            or not isinstance(physics_step_budget, Integral)
            or physics_step_budget <= 0
        ):
            raise ValueError(
                "physics_step_budget must be a positive integer or None"
            )
        physics_step_budget = int(physics_step_budget)

    chunk = validate_ee_action_chunk(actions)
    target_results: list[dict[str, Any]] = []
    stopped_on_controller_failure = False
    stopped_for_environment_success = False

    for target_index, target in enumerate(chunk):
        requested_target = target.copy()
        take_action_kwargs = {"action_type": action_type}
        if physics_step_budget is not None:
            take_action_kwargs["physics_step_budget"] = physics_step_budget
        raw_result = task.take_action(
            requested_target[None, :].copy(), **take_action_kwargs
        )
        result = _normalize_target_result(
            raw_result, target_index, requested_target
        )
        target_results.append(result)

        if bool(getattr(task, "eval_success", False)):
            stopped_for_environment_success = True
            break
        if not bool(result["controller_success"]) and stop_on_failure:
            stopped_on_controller_failure = True
            break

    fully_planned = sum(
        int(bool(item["controller_success"])) for item in target_results
    )
    completed_all_targets = len(target_results) == len(chunk)
    controller_success = (
        not stopped_on_controller_failure
        and all(bool(item["controller_success"]) for item in target_results)
    )
    total_executed_physics_steps = sum(
        int(item.get("executed_physics_steps", item.get("physics_steps", 0)))
        for item in target_results
    )
    reached_target_count = sum(
        int(bool(item.get("target_reached", False)))
        for item in target_results
    )

    return {
        "contract_version": (
            EE_FIXED_BUDGET_CONTRACT_VERSION
            if physics_step_budget is not None
            else EE_EXECUTION_CONTRACT_VERSION
        ),
        "action_type": action_type,
        "action_layout": EE_ACTION_LAYOUT,
        "physics_step_budget_per_target": physics_step_budget,
        "requested_target_count": int(len(chunk)),
        "attempted_target_count": int(len(target_results)),
        "fully_planned_target_count": int(fully_planned),
        "reached_target_count": int(reached_target_count),
        "completed_all_targets": bool(completed_all_targets),
        "completed_all_trajectories": bool(
            completed_all_targets and reached_target_count == len(chunk)
        ),
        "controller_success": bool(controller_success),
        "stopped_on_controller_failure": bool(stopped_on_controller_failure),
        "stopped_for_environment_success": bool(stopped_for_environment_success),
        "executed_physics_steps": int(total_executed_physics_steps),
        "target_results": target_results,
    }


__all__ = [
    "EE_ACTION_DIM",
    "EE_ACTION_LAYOUT",
    "EE_EXECUTION_CONTRACT_VERSION",
    "EE_FIXED_BUDGET_CONTRACT_VERSION",
    "execute_ee_action_chunk",
    "validate_ee_action_chunk",
]
