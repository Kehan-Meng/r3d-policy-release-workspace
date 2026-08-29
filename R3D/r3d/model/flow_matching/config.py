from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional


@dataclass(frozen=True)
class FlowMatchingConfig:
    eps: float
    initial_noise_scale: float
    delta: float
    num_segments: int
    boundary: int
    alpha: float
    consistency_weight: float
    consistency_schedule: str
    consistency_final_weight: float
    consistency_ramp_start_epoch: int
    consistency_ramp_end_epoch: int
    consistency_schedule_power: float
    stop_gradient_target: bool
    direct_velocity_weight: float
    num_inference_steps: int
    solver: str
    time_scale: float

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping],
        *,
        default_time_scale: float,
    ) -> "FlowMatchingConfig":
        values = dict(values or {})
        initial_weight = float(values.get("consistency_weight", 1.0))
        config = cls(
            eps=float(values.get("eps", 1e-2)),
            initial_noise_scale=float(values.get("initial_noise_scale", 1.0)),
            delta=float(values.get("delta", 1e-2)),
            num_segments=int(values.get("num_segments", 2)),
            boundary=int(values.get("boundary", 1)),
            alpha=float(values.get("alpha", 1e-5)),
            consistency_weight=initial_weight,
            consistency_schedule=str(values.get("consistency_schedule", "constant")),
            consistency_final_weight=float(
                values.get("consistency_final_weight", initial_weight)
            ),
            consistency_ramp_start_epoch=int(
                values.get("consistency_ramp_start_epoch", 0)
            ),
            consistency_ramp_end_epoch=int(
                values.get("consistency_ramp_end_epoch", 0)
            ),
            consistency_schedule_power=float(
                values.get("consistency_schedule_power", 1.0)
            ),
            stop_gradient_target=values.get("stop_gradient_target", True),
            direct_velocity_weight=float(values.get("direct_velocity_weight", 0.0)),
            num_inference_steps=int(values.get("num_inference_steps", 1)),
            solver=str(values.get("solver", "euler")),
            time_scale=float(values.get("time_scale", default_time_scale)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not math.isfinite(self.initial_noise_scale) or self.initial_noise_scale < 0:
            raise ValueError(
                "flow_matching.initial_noise_scale must be finite and non-negative"
            )
        if not isinstance(self.stop_gradient_target, bool):
            raise TypeError("flow_matching.stop_gradient_target must be boolean")
        if self.consistency_weight < 0:
            raise ValueError("flow_matching.consistency_weight must be non-negative")
        if self.consistency_schedule not in ("constant", "linear", "geometric"):
            raise ValueError(
                "flow_matching.consistency_schedule must be constant, linear, "
                f"or geometric, got {self.consistency_schedule}"
            )
        if self.consistency_ramp_end_epoch < self.consistency_ramp_start_epoch:
            raise ValueError(
                "consistency_ramp_end_epoch must be >= consistency_ramp_start_epoch"
            )
        if self.consistency_schedule_power <= 0:
            raise ValueError("consistency_schedule_power must be positive")
        if self.consistency_schedule == "geometric" and (
            self.consistency_weight <= 0 or self.consistency_final_weight <= 0
        ):
            raise ValueError(
                "geometric consistency scheduling requires positive weights"
            )
        if self.solver not in ("euler", "heun", "rk4"):
            raise ValueError(f"Unsupported flow_matching.solver: {self.solver}")

    def consistency_weight_at(self, epoch: int) -> float:
        initial = self.consistency_weight
        final = self.consistency_final_weight
        start = self.consistency_ramp_start_epoch
        end = self.consistency_ramp_end_epoch

        if self.consistency_schedule == "constant" or end <= start or epoch <= start:
            return initial
        if epoch >= end:
            return final

        progress = (epoch - start) / float(end - start)
        progress = progress ** self.consistency_schedule_power
        if self.consistency_schedule == "linear":
            return initial + (final - initial) * progress
        return math.exp(
            (1.0 - progress) * math.log(initial) + progress * math.log(final)
        )

