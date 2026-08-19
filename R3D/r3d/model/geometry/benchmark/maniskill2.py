"""ManiSkill2 joint-control contracts with world-to-camera observations."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder, expect_field


_TASK_SPECS = {
    "pickcube": {
        "task": "PickCube",
        "state_dim": 12,
        "state_semantics": "panda_qpos9_plus_goal_position_world3",
        "canonical_frame": "maniskill_pickplace_base_camera_cv",
    },
    "stackcube": {
        "task": "StackCube",
        "state_dim": 9,
        "state_semantics": "panda_qpos9",
        "canonical_frame": "maniskill_pickplace_base_camera_cv",
    },
    "peginsertionside": {
        "task": "PegInsertionSide",
        "state_dim": 9,
        "state_semantics": "panda_qpos9",
        "canonical_frame": "maniskill_peginsertion_base_camera_cv",
    },
}


def normalize_maniskill2_task(task: str) -> str:
    normalized = str(task).lower().replace("_", "").replace("-", "")
    if normalized.endswith("v0"):
        normalized = normalized[:-2]
    if normalized not in _TASK_SPECS:
        from ..errors import ProfileContractError

        raise ProfileContractError(f"Unsupported ManiSkill2 task: {task!r}")
    return normalized


class ManiSkill2SemanticDecoder(BenchmarkSemanticDecoder):
    def __init__(self, task: str):
        self.task_key = normalize_maniskill2_task(task)
        spec = _TASK_SPECS[self.task_key]
        super().__init__(
            BenchmarkNativeContract(
                benchmark="maniskill2",
                task=spec["task"],
                point_cloud_dim=6,
                state_dim=spec["state_dim"],
                action_dim=8,
                point_cloud_frame="world",
                state_semantics=spec["state_semantics"],
                action_semantics="panda_pd_joint_pos7_plus_gripper1",
            )
        )

    def validate_profile_config(self, config: Mapping[str, Any]) -> None:
        super().validate_profile_config(config)
        spec = _TASK_SPECS[self.task_key]
        canonical = spec["canonical_frame"]
        if config.get("canonical_frame") != canonical:
            from ..errors import ProfileContractError

            raise ProfileContractError(
                f"ManiSkill2 {self.contract.task} canonical frame must be {canonical!r}"
            )

        for schema in ("observation", "training"):
            expect_field(
                config,
                schema,
                "point_cloud",
                "xyz",
                kind="point",
                source_frame="world",
                target_frame=canonical,
                preserve_zero_sentinel=True,
            )
            expect_field(config, schema, "point_cloud", "rgb", kind="passthrough")
            expect_field(config, schema, "agent_pos", "qpos", kind="joint")
            if self.task_key == "pickcube":
                expect_field(
                    config,
                    schema,
                    "agent_pos",
                    "goal_position",
                    kind="point",
                    source_frame="world",
                    target_frame=canonical,
                )
        for schema in ("action", "training"):
            expect_field(config, schema, "action", "pd_joint_pos", kind="joint")


__all__ = ["ManiSkill2SemanticDecoder", "normalize_maniskill2_task"]
