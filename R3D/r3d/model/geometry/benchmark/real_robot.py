"""Semantic decoder for versioned real-robot frame profiles."""

from __future__ import annotations

from typing import Any, Mapping

from ..errors import ProfileContractError
from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder


class RealRobotSemanticDecoder(BenchmarkSemanticDecoder):
    """Decode real-robot arrays using dimensions frozen in the profile.

    Hardware-specific field meanings remain in the profile schema.  This class
    only establishes the same shape-checked native boundary used by simulation
    benchmarks.
    """

    def __init__(self, config: Mapping[str, Any]):
        native = config.get("native_contract")
        if not isinstance(native, Mapping):
            raise ProfileContractError("Real-robot profile is missing native_contract")

        def positive_int(key: str) -> int:
            try:
                value = int(native[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProfileContractError(
                    f"native_contract.{key} must be a positive integer"
                ) from exc
            if value <= 0:
                raise ProfileContractError(
                    f"native_contract.{key} must be a positive integer"
                )
            return value

        def required_text(key: str) -> str:
            value = native.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ProfileContractError(
                    f"native_contract.{key} must be a non-empty string"
                )
            return value

        task = str(config.get("task", "")).strip()
        if not task:
            raise ProfileContractError("Real-robot profile task must be non-empty")
        super().__init__(
            BenchmarkNativeContract(
                benchmark="real_robot",
                task=task,
                point_cloud_dim=positive_int("point_cloud_dim"),
                state_dim=positive_int("state_dim"),
                action_dim=positive_int("action_dim"),
                point_cloud_frame=required_text("point_cloud_frame"),
                state_semantics=required_text("state_semantics"),
                action_semantics=required_text("action_semantics"),
            )
        )


__all__ = ["RealRobotSemanticDecoder"]
