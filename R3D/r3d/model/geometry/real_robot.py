"""Real-robot deployment contracts for canonical-frame policies.

This module contains no vendor SDK calls.  It validates a versioned hardware
contract, builds runtime context for dynamic camera transforms, and runs a
small numerical round-trip before a profile is allowed onto a robot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .benchmark import decoder_from_profile_config
from .registry import build_adapter
from .representations import axis_angle_to_matrix, matrix_to_rotation_6d
from .serialization import stable_sha256


_CARTESIAN_FIELD_KINDS = frozenset(
    {
        "point",
        "vector",
        "direction",
        "orientation",
        "absolute_pose",
        "relative_pose_spatial",
        "twist_spatial",
    }
)
_PLACEHOLDER_TOKENS = ("REPLACE", "TODO", "UNKNOWN", "UNSET")


def _nested_get(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _nested_set(mapping: dict[str, Any], path: str, value: Any) -> None:
    current = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _plain_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_copy(item) for item in value)
    return value


def _has_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        upper = value.upper()
        return any(token in upper for token in _PLACEHOLDER_TOKENS)
    return False


@dataclass(frozen=True)
class RealRobotPreflightReport:
    status: str
    profile_name: Optional[str]
    profile_hash: Optional[str]
    calibration_hash: Optional[str]
    checks: Mapping[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    roundtrip: Mapping[str, float]
    information_inventory: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RealRobotRuntimeContextBuilder:
    """Extract validated dynamic frame transforms from an observation."""

    def __init__(
        self,
        *,
        observation_key: str = "frame_context",
        required_keys: Sequence[str] = (),
        matrix_keys: Sequence[str] = (),
    ):
        self.observation_key = str(observation_key)
        self.required_keys = tuple(str(key) for key in required_keys)
        self.matrix_keys = tuple(str(key) for key in matrix_keys)
        if not self.observation_key:
            raise ValueError("observation_key must be non-empty")

    @classmethod
    def from_profile_config(cls, config: Mapping[str, Any]):
        contract = config.get("real_robot_contract", {})
        runtime = contract.get("runtime_context", {}) if isinstance(contract, Mapping) else {}
        return cls(
            observation_key=runtime.get("observation_key", "frame_context"),
            required_keys=runtime.get("required_keys", ()),
            matrix_keys=runtime.get("matrix_keys", ()),
        )

    def normalize_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(context, Mapping):
            raise TypeError("Real-robot frame_context must be a mapping")
        output = _plain_copy(context)
        for key in self.required_keys:
            try:
                _nested_get(output, key)
            except KeyError as exc:
                raise KeyError(f"frame_context is missing required key {key!r}") from exc
        for key in self.matrix_keys:
            try:
                raw = _nested_get(output, key)
            except KeyError as exc:
                raise KeyError(f"frame_context is missing matrix key {key!r}") from exc
            matrix = np.asarray(raw, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"frame_context.{key} must have shape [4, 4]")
            _nested_set(output, key, matrix)
        return output

    def __call__(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        try:
            context = _nested_get(observation, self.observation_key)
        except KeyError as exc:
            raise KeyError(
                f"Observation is missing runtime context {self.observation_key!r}"
            ) from exc
        return self.normalize_context(context)


def _required_mapping(
    parent: Mapping[str, Any], key: str, errors: list[str], path: str
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{path}.{key} must be a mapping")
        return {}
    return value


def _required_text(parent: Mapping[str, Any], key: str, errors: list[str], path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip() or _has_placeholder(value):
        errors.append(f"{path}.{key} must be filled with a non-placeholder string")
        return ""
    return value.strip()


def _required_positive(parent, key, errors, path, *, integer=False) -> float:
    value = parent.get(key)
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError):
        errors.append(f"{path}.{key} must be a positive number")
        return 0
    if number <= 0:
        errors.append(f"{path}.{key} must be a positive number")
    return number


def _validate_matrix3(value: Any, errors: list[str], path: str) -> None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f"{path} must be a numeric 3x3 camera matrix")
        return
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        errors.append(f"{path} must be a finite numeric 3x3 camera matrix")
        return
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2] - 1.0) > 1e-9:
        errors.append(f"{path} has invalid focal lengths or homogeneous row")


def _schema_fields(config: Mapping[str, Any], schema_name: str, tensor_name: str):
    schema = config.get("schemas", {}).get(schema_name, {})
    for tensor in schema.get("tensors", ()):
        if tensor.get("name") == tensor_name:
            return tuple(tensor.get("fields", ()))
    return ()


def _synthetic_tensor(spec, rng: np.random.Generator, count: int):
    leading = (count, 16) if spec.name == "point_cloud" else (count,)
    chunks = []
    for field in sorted(spec.fields, key=lambda item: item.start):
        width = field.end - field.start
        shape = leading + (width,)
        if field.kind in ("scalar", "joint", "actuator", "passthrough"):
            chunk = rng.uniform(-0.5, 0.5, size=shape)
        elif field.kind in ("point", "vector", "direction"):
            chunk = rng.uniform(-0.4, 0.4, size=shape)
        elif field.kind == "orientation":
            axis_angle = rng.normal(0.0, 0.2, size=leading + (3,))
            rotation = axis_angle_to_matrix(axis_angle)
            if field.representation == "rotation_6d_columns":
                chunk = matrix_to_rotation_6d(rotation)
            elif field.representation == "rotation_matrix_9d":
                chunk = rotation.reshape(shape)
            elif field.representation == "axis_angle":
                chunk = axis_angle
            else:
                quaternion = np.zeros(shape, dtype=np.float64)
                quaternion[..., 0 if field.quaternion_order == "wxyz" else 3] = 1.0
                chunk = quaternion
        elif field.kind in ("absolute_pose", "relative_pose_spatial", "relative_pose_body"):
            position = rng.uniform(-0.2, 0.2, size=leading + (3,))
            axis_angle = rng.normal(0.0, 0.1, size=leading + (3,))
            if field.representation == "xyz_axis_angle":
                chunk = np.concatenate((position, axis_angle), axis=-1)
            elif field.representation == "xyz_rotation_6d_columns":
                rotation6d = matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))
                chunk = np.concatenate((position, rotation6d), axis=-1)
            elif field.representation == "xyz_quaternion":
                quaternion = np.zeros(leading + (4,), dtype=np.float64)
                quaternion[..., 0 if field.quaternion_order == "wxyz" else 3] = 1.0
                chunk = np.concatenate((position, quaternion), axis=-1)
            else:
                matrices = np.broadcast_to(np.eye(4), leading + (4, 4)).copy()
                matrices[..., :3, 3] = position
                chunk = matrices.reshape(shape)
        elif field.kind in ("twist_spatial", "twist_body"):
            chunk = rng.normal(0.0, 0.1, size=shape)
        else:
            raise ValueError(f"Unsupported synthetic field kind {field.kind!r}")
        chunks.append(np.asarray(chunk, dtype=np.float64))
    return np.concatenate(chunks, axis=-1)


def _synthetic_sample(schema, rng, count):
    sample: dict[str, Any] = {}
    for spec in schema.tensors:
        if len(spec.key_path) != 1:
            raise ValueError("Real-robot preflight currently expects one-level tensor key paths")
        sample[spec.key_path[0]] = _synthetic_tensor(spec, rng, count)
    return sample


def _max_abs_nested(left: Any, right: Any) -> float:
    if isinstance(left, Mapping):
        return max(_max_abs_nested(left[key], right[key]) for key in left)
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def preflight_real_robot_profile(
    config: Mapping[str, Any],
    *,
    runtime_context: Optional[Mapping[str, Any]] = None,
    roundtrip_samples: int = 32,
) -> RealRobotPreflightReport:
    """Validate hardware metadata, profile semantics and numerical reversibility."""

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    roundtrip: dict[str, float] = {}
    contract = _required_mapping(config, "real_robot_contract", errors, "profile")
    readiness = contract.get("readiness")
    if readiness != "ready":
        errors.append("real_robot_contract.readiness must be 'ready'")
    if config.get("status") != "ready":
        errors.append("profile status must be 'ready'")

    if str(config.get("benchmark", "")).lower() != "real_robot":
        errors.append("benchmark must be 'real_robot'")
    if not bool(config.get("require_metadata", False)):
        errors.append("Real-robot profiles must set require_metadata: true")

    hardware = _required_mapping(contract, "hardware", errors, "real_robot_contract")
    _required_text(hardware, "robot_model", errors, "real_robot_contract.hardware")
    _required_text(hardware, "robot_serial", errors, "real_robot_contract.hardware")
    _required_text(
        hardware, "robot_description_hash", errors, "real_robot_contract.hardware"
    )
    arm_count = int(
        _required_positive(
            hardware, "arm_count", errors, "real_robot_contract.hardware", integer=True
        )
    )
    base_frames = hardware.get("base_frames")
    ee_frames = hardware.get("ee_frames")
    for key, value in (("base_frames", base_frames), ("ee_frames", ee_frames)):
        if not isinstance(value, list) or len(value) != arm_count or any(
            not isinstance(item, str) or not item or _has_placeholder(item) for item in value
        ):
            errors.append(
                f"real_robot_contract.hardware.{key} must contain one frame per arm"
            )

    camera = _required_mapping(contract, "camera", errors, "real_robot_contract")
    mounting = camera.get("mounting")
    if mounting not in ("fixed_external", "eye_in_hand"):
        errors.append("real_robot_contract.camera.mounting must be fixed_external or eye_in_hand")
    _required_text(camera, "serial", errors, "real_robot_contract.camera")
    optical_frame = _required_text(
        camera, "optical_frame", errors, "real_robot_contract.camera"
    )
    if camera.get("optical_axis_convention") != "opencv_x_right_y_down_z_forward":
        errors.append(
            "real_robot_contract.camera.optical_axis_convention must be "
            "opencv_x_right_y_down_z_forward"
        )
    resolution = camera.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2 or any(
        not isinstance(item, int) or item <= 0 for item in resolution
    ):
        errors.append("real_robot_contract.camera.resolution must be [width, height]")
    _validate_matrix3(camera.get("intrinsics"), errors, "real_robot_contract.camera.intrinsics")
    _required_text(camera, "distortion_model", errors, "real_robot_contract.camera")
    if not isinstance(camera.get("distortion_coefficients"), list):
        errors.append("real_robot_contract.camera.distortion_coefficients must be a list")
    _required_positive(
        camera, "depth_scale_m_per_unit", errors, "real_robot_contract.camera"
    )
    if camera.get("depth_registered_to_color") is not True:
        errors.append(
            "real_robot_contract.camera.depth_registered_to_color must be true; "
            "otherwise add an explicit depth-to-color transform before using this profile"
        )

    calibration = _required_mapping(
        contract, "calibration", errors, "real_robot_contract"
    )
    _required_text(calibration, "method", errors, "real_robot_contract.calibration")
    _required_text(
        calibration, "artifact_sha256", errors, "real_robot_contract.calibration"
    )
    if calibration.get("stored_transform_convention") != "T_target_from_source":
        errors.append(
            "real_robot_contract.calibration.stored_transform_convention must be "
            "T_target_from_source"
        )
    measured_at = _required_text(
        calibration, "measured_at_utc", errors, "real_robot_contract.calibration"
    )
    if measured_at:
        try:
            datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("real_robot_contract.calibration.measured_at_utc must be ISO-8601")
    sample_count = _required_positive(
        calibration,
        "sample_count",
        errors,
        "real_robot_contract.calibration",
        integer=True,
    )
    quality = {}
    for key in (
        "translation_rmse_m",
        "rotation_rmse_deg",
        "reprojection_rmse_px",
        "independent_point_rmse_m",
    ):
        try:
            value = float(calibration.get(key))
        except (TypeError, ValueError):
            errors.append(f"real_robot_contract.calibration.{key} must be non-negative")
            continue
        if value < 0 or not np.isfinite(value):
            errors.append(f"real_robot_contract.calibration.{key} must be non-negative")
        quality[key] = value
    acceptance = _required_mapping(
        calibration, "acceptance", errors, "real_robot_contract.calibration"
    )
    thresholds = {
        "translation_rmse_m": "max_translation_rmse_m",
        "rotation_rmse_deg": "max_rotation_rmse_deg",
        "reprojection_rmse_px": "max_reprojection_rmse_px",
        "independent_point_rmse_m": "max_independent_point_rmse_m",
    }
    for metric, threshold_key in thresholds.items():
        threshold = _required_positive(
            acceptance,
            threshold_key,
            errors,
            "real_robot_contract.calibration.acceptance",
        )
        if metric in quality and threshold > 0 and quality[metric] > threshold:
            errors.append(
                f"Calibration {metric}={quality[metric]:.6g} exceeds {threshold_key}={threshold:.6g}"
            )
    min_samples = _required_positive(
        acceptance,
        "min_sample_count",
        errors,
        "real_robot_contract.calibration.acceptance",
        integer=True,
    )
    if sample_count and min_samples and sample_count < min_samples:
        errors.append("Calibration sample_count is below the declared acceptance minimum")
    try:
        axis_cosine = float(calibration.get("axis_alignment_min_cosine"))
    except (TypeError, ValueError):
        errors.append(
            "real_robot_contract.calibration.axis_alignment_min_cosine must be numeric"
        )
        axis_cosine = -1.0
    try:
        min_axis_cosine = float(acceptance.get("min_axis_alignment_cosine"))
    except (TypeError, ValueError):
        errors.append(
            "real_robot_contract.calibration.acceptance.min_axis_alignment_cosine "
            "must be numeric"
        )
        min_axis_cosine = 1.0
    if not -1.0 <= axis_cosine <= 1.0:
        errors.append("calibration.axis_alignment_min_cosine must be in [-1, 1]")
    elif axis_cosine < min_axis_cosine:
        errors.append(
            "Calibration axis_alignment_min_cosine is below the declared acceptance minimum"
        )

    timing = _required_mapping(contract, "timing", errors, "real_robot_contract")
    _required_positive(timing, "observation_frequency_hz", errors, "real_robot_contract.timing")
    _required_positive(timing, "control_frequency_hz", errors, "real_robot_contract.timing")
    max_skew_s = _required_positive(
        timing, "max_camera_robot_skew_s", errors, "real_robot_contract.timing"
    )
    _required_text(timing, "camera_timestamp_source", errors, "real_robot_contract.timing")
    _required_text(timing, "robot_timestamp_source", errors, "real_robot_contract.timing")

    controller = _required_mapping(contract, "controller", errors, "real_robot_contract")
    _required_text(
        controller, "controller_config_hash", errors, "real_robot_contract.controller"
    )
    if controller.get("action_type") != "cartesian_delta_pose_spatial":
        errors.append(
            "real_robot_contract.controller.action_type must be cartesian_delta_pose_spatial"
        )
    if controller.get("translation_unit") != "meter":
        errors.append("real_robot_contract.controller.translation_unit must be meter")
    if controller.get("rotation_representation") != "axis_angle":
        errors.append("real_robot_contract.controller.rotation_representation must be axis_angle")
    if controller.get("delta_composition") != "left":
        errors.append("real_robot_contract.controller.delta_composition must be left")
    command_frames = controller.get("command_frames")
    if isinstance(base_frames, list) and command_frames != base_frames:
        errors.append("controller.command_frames must exactly match hardware.base_frames")
    _required_text(controller, "gripper_semantics", errors, "real_robot_contract.controller")

    safety = _required_mapping(contract, "safety", errors, "real_robot_contract")
    bounds = safety.get("workspace_bounds_m")
    if not isinstance(bounds, list) or len(bounds) != 3 or any(
        not isinstance(axis, list)
        or len(axis) != 2
        or not all(isinstance(value, (int, float)) for value in axis)
        or axis[0] >= axis[1]
        for axis in (bounds or [])
    ):
        errors.append("real_robot_contract.safety.workspace_bounds_m must be 3 [min,max] pairs")
    _required_positive(safety, "max_translation_step_m", errors, "real_robot_contract.safety")
    _required_positive(safety, "max_rotation_step_rad", errors, "real_robot_contract.safety")
    _required_positive(safety, "watchdog_timeout_s", errors, "real_robot_contract.safety")

    runtime_builder = RealRobotRuntimeContextBuilder.from_profile_config(config)
    normalized_context = None
    if mounting == "eye_in_hand":
        if runtime_context is None:
            errors.append("Eye-in-hand preflight requires one synchronized runtime_context sample")
        else:
            try:
                normalized_context = runtime_builder.normalize_context(runtime_context)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        if max_skew_s and not runtime_builder.required_keys:
            errors.append("Eye-in-hand profile must declare runtime_context.required_keys")
        eye_configs = [
            item.get("provider", {})
            for item in config.get("transforms", ())
            if item.get("provider", {}).get("type") == "eye_in_hand"
        ]
        if len(eye_configs) != 1:
            errors.append("Eye-in-hand profile must declare exactly one eye_in_hand provider")
        elif max_skew_s and abs(float(eye_configs[0].get("timestamp_tolerance", -1)) - max_skew_s) > 1e-12:
            errors.append(
                "eye_in_hand timestamp_tolerance must match timing.max_camera_robot_skew_s"
            )
    elif mounting == "fixed_external":
        if any(
            item.get("provider", {}).get("type") == "eye_in_hand"
            for item in config.get("transforms", ())
        ):
            errors.append("fixed_external profile cannot contain an eye_in_hand provider")

    native = config.get("native_contract", {})
    if native.get("point_cloud_frame") != optical_frame:
        errors.append("native_contract.point_cloud_frame must match camera.optical_frame")
    if config.get("canonical_frame") != optical_frame:
        errors.append("canonical_frame must match camera.optical_frame")

    observation_fields = _schema_fields(config, "observation", "agent_pos")
    action_fields = _schema_fields(config, "action", "action")
    point_fields = _schema_fields(config, "observation", "point_cloud")
    if not observation_fields or not action_fields or not point_fields:
        errors.append("observation/action schemas must define point_cloud, agent_pos and action")
    for field in point_fields:
        if field.get("name") == "xyz" and field.get("target_frame") != optical_frame:
            errors.append("Point-cloud xyz must target the camera optical frame")
    geometric_state = [field for field in observation_fields if field.get("kind") in _CARTESIAN_FIELD_KINDS]
    geometric_action = [field for field in action_fields if field.get("kind") in _CARTESIAN_FIELD_KINDS]
    if not geometric_state:
        errors.append("agent_pos must contain at least one Cartesian field")
    if not geometric_action:
        errors.append("action must contain at least one Cartesian field")
    if any(field.get("target_frame") != optical_frame for field in geometric_state + geometric_action):
        errors.append("All Cartesian state/action fields must target the camera optical frame")
    if any(field.get("kind") in ("joint", "actuator") for field in action_fields):
        errors.append("A full camera-space real-robot action contract cannot contain joint/actuator fields")
    if not any(field.get("kind") == "relative_pose_spatial" for field in action_fields):
        errors.append("Action schema must include a relative_pose_spatial field")

    adapter = None
    if not errors:
        try:
            decoder = decoder_from_profile_config(config)
            decoder.validate_profile_config(config)
            adapter = build_adapter(config)
            checks["profile_build"] = True
        except Exception as exc:
            errors.append(f"Profile build failed: {exc}")
            checks["profile_build"] = False

    if adapter is not None:
        try:
            rng = np.random.default_rng(20260820)
            observation = _synthetic_sample(
                adapter.profile.observation_schema, rng, roundtrip_samples
            )
            native_action = _synthetic_tensor(
                adapter.profile.action_schema.tensors[0], rng, roundtrip_samples
            )
            policy_observation = adapter.observation_to_policy_with_metadata(
                observation,
                adapter.native_metadata(),
                runtime_context=normalized_context,
            )
            recovered_observation = adapter.observation_to_native_with_metadata(
                policy_observation.data,
                policy_observation.metadata,
                runtime_context=normalized_context,
            )
            policy_action = adapter.action_to_policy_with_metadata(
                native_action,
                adapter.native_metadata(),
                runtime_context=normalized_context,
            )
            recovered_action = adapter.action_to_environment_with_metadata(
                policy_action.data,
                policy_action.metadata,
                runtime_context=normalized_context,
            )
            roundtrip = {
                "observation_max_abs": _max_abs_nested(
                    observation, recovered_observation.data
                ),
                "action_max_abs": _max_abs_nested(native_action, recovered_action.data),
            }
            checks["synthetic_roundtrip"] = max(roundtrip.values()) < 1e-8
            if not checks["synthetic_roundtrip"]:
                errors.append(f"Synthetic round-trip exceeded tolerance: {roundtrip}")
        except Exception as exc:
            checks["synthetic_roundtrip"] = False
            errors.append(f"Synthetic round-trip failed: {exc}")

    checks["hardware_contract"] = not any("hardware" in item for item in errors)
    checks["camera_contract"] = not any("camera" in item.lower() for item in errors)
    checks["calibration_contract"] = not any("calibration" in item.lower() for item in errors)
    checks["timing_contract"] = not any("timing" in item.lower() or "timestamp" in item.lower() for item in errors)
    checks["controller_contract"] = not any("controller" in item.lower() or "action schema" in item.lower() for item in errors)
    checks["safety_contract"] = not any("safety" in item.lower() for item in errors)
    if quality and quality.get("reprojection_rmse_px", 0.0) > 1.0:
        warnings.append("Calibration reprojection RMSE exceeds 1 px; inspect overlay before collection")

    return RealRobotPreflightReport(
        status="passed" if not errors else "failed",
        profile_name=adapter.profile.name if adapter is not None else config.get("name"),
        profile_hash=adapter.profile_hash if adapter is not None else None,
        calibration_hash=(
            stable_sha256(config.get("transforms", [])) if adapter is not None else None
        ),
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
        roundtrip=roundtrip,
        information_inventory={
            "robot_model": hardware.get("robot_model"),
            "robot_serial": hardware.get("robot_serial"),
            "arm_count": arm_count,
            "camera_serial": camera.get("serial"),
            "camera_mounting": mounting,
            "camera_optical_frame": optical_frame,
            "state_dim": native.get("state_dim"),
            "action_dim": native.get("action_dim"),
            "controller_action_type": controller.get("action_type"),
        },
    )


__all__ = [
    "RealRobotPreflightReport",
    "RealRobotRuntimeContextBuilder",
    "preflight_real_robot_profile",
]
