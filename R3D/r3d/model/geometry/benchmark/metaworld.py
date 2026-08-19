"""MetaWorld 9D state / 4D delta-action semantic contract."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder, expect_field


class MetaWorldSemanticDecoder(BenchmarkSemanticDecoder):
    def __init__(self):
        super().__init__(
            BenchmarkNativeContract(
                benchmark="metaworld",
                task="*",
                point_cloud_dim=6,
                state_dim=9,
                action_dim=4,
                point_cloud_frame="metaworld_obs_aligned",
                state_semantics="eef_right_finger_left_finger_world_positions9",
                action_semantics="preclip_xyz_command_world_plus_gripper4",
            )
        )

    def validate_profile_config(self, config: Mapping[str, Any]) -> None:
        super().validate_profile_config(config)
        canonical = str(config.get("canonical_frame", ""))
        if canonical not in ("metaworld_obs_aligned", "corner2_cv_camera"):
            from ..errors import ProfileContractError

            raise ProfileContractError(
                f"Unsupported MetaWorld canonical frame: {canonical!r}"
            )

        point_target = canonical
        for schema in ("observation", "training"):
            expect_field(
                config,
                schema,
                "point_cloud",
                "xyz",
                kind="point",
                source_frame="metaworld_obs_aligned",
                target_frame=point_target,
            )
            expect_field(config, schema, "point_cloud", "rgb", kind="passthrough")
            for field in ("eef", "right_finger", "left_finger"):
                expect_field(
                    config,
                    schema,
                    "agent_pos",
                    field,
                    kind="point",
                    source_frame="world",
                    target_frame=canonical,
                )
        for schema in ("action", "training"):
            expect_field(
                config,
                schema,
                "action",
                "translation_command",
                kind="vector",
                source_frame="world",
                target_frame=canonical,
            )
            expect_field(config, schema, "action", "gripper", kind="scalar")


__all__ = ["MetaWorldSemanticDecoder"]
