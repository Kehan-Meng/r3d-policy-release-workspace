"""Frozen RoboTwin2 joint14 execution contract."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder, expect_field


class Robotwin2Joint14SemanticDecoder(BenchmarkSemanticDecoder):
    def __init__(self):
        super().__init__(
            BenchmarkNativeContract(
                benchmark="robotwin2",
                task="*",
                point_cloud_dim=6,
                state_dim=14,
                action_dim=14,
                point_cloud_frame="world",
                state_semantics="dual_arm_joint_targets6_plus_gripper1_each14",
                action_semantics="executed_joint14",
            )
        )

    def validate_profile_config(self, config: Mapping[str, Any]) -> None:
        super().validate_profile_config(config)
        native = config["native_contract"]
        if native.get("executed_representation") != "joint14":
            from ..errors import ProfileContractError

            raise ProfileContractError("RoboTwin2 Phase 2 only supports executed joint14")
        if native.get("target_ee_is_executed") is not False:
            from ..errors import ProfileContractError

            raise ProfileContractError("target_ee must be declared non-executed")

        for schema in ("observation", "training"):
            expect_field(
                config,
                schema,
                "point_cloud",
                "xyz",
                kind="point",
                source_frame="world",
                target_frame="world",
            )
            expect_field(config, schema, "point_cloud", "rgb", kind="passthrough")
            expect_field(config, schema, "agent_pos", "joint_state", kind="joint")
        for schema in ("action", "training"):
            expect_field(config, schema, "action", "joint_command", kind="joint")


__all__ = ["Robotwin2Joint14SemanticDecoder"]
