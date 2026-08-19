"""Adroit task-specific actuator contracts with observation-only geometry."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder, expect_field


_ACTION_DIMS = {"door": 28, "hammer": 26, "pen": 24}


def normalize_adroit_task(task: str) -> str:
    normalized = task.lower().removeprefix("adroit_").replace("-", "_")
    if normalized == "relocate":
        from ..errors import UnsupportedRepresentationError

        raise UnsupportedRepresentationError(
            "This Adroit task is excluded by the frozen paper evaluation scope; "
            "no frame profile is provided"
        )
    if normalized not in _ACTION_DIMS:
        from ..errors import ProfileContractError

        raise ProfileContractError(f"Unsupported Adroit task: {task!r}")
    return normalized


class AdroitSemanticDecoder(BenchmarkSemanticDecoder):
    def __init__(self, task: str):
        task = normalize_adroit_task(task)
        self.task = task
        super().__init__(
            BenchmarkNativeContract(
                benchmark="adroit",
                task=task,
                point_cloud_dim=6,
                state_dim=24,
                action_dim=_ACTION_DIMS[task],
                point_cloud_frame="legacy_synthetic_frame",
                state_semantics=f"adroit_{task}_joint_proprioception24",
                action_semantics=f"adroit_{task}_normalized_mujoco_actuator{_ACTION_DIMS[task]}",
            )
        )

    def validate_profile_config(self, config: Mapping[str, Any]) -> None:
        super().validate_profile_config(config)
        canonical = f"adroit_{self.task}_cv_camera"
        if config.get("canonical_frame") != canonical:
            from ..errors import ProfileContractError

            raise ProfileContractError(
                f"Adroit {self.task} canonical frame must be {canonical!r}"
            )
        for schema in ("observation", "training"):
            expect_field(
                config,
                schema,
                "point_cloud",
                "xyz",
                kind="point",
                source_frame="legacy_synthetic_frame",
                target_frame=canonical,
            )
            expect_field(config, schema, "point_cloud", "rgb", kind="passthrough")
            expect_field(config, schema, "agent_pos", "proprioception", kind="joint")
        for schema in ("action", "training"):
            expect_field(config, schema, "action", "native_actuator", kind="actuator")


__all__ = ["AdroitSemanticDecoder", "normalize_adroit_task"]
