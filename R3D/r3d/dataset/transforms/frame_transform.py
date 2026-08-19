"""Low-coupling canonical-frame hook shared by production datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from r3d.model.geometry.benchmark.profile import load_profile_bundle
from r3d.model.geometry.serialization import stable_sha256, to_primitive


def _plain_mapping(config: Any) -> Mapping[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return config
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    raise TypeError(f"frame_adapter config must be a mapping, got {type(config)!r}")


def _resolve_profile_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    r3d_root = Path(__file__).resolve().parents[3]
    project_root = r3d_root.parent
    candidates = (Path.cwd() / path, r3d_root / path, project_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Frame profile {raw_path!r} was not found; tried {attempted}")


@dataclass(frozen=True)
class FrameAdapterSettings:
    enabled: bool = False
    profile_path: Optional[str] = None
    augmentation_frame: str = "policy"
    normalizer_train_split_only: bool = True
    normalizer_chunk_size: int = 2048
    allow_frame_config_override: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "FrameAdapterSettings":
        raw = _plain_mapping(config)
        normalizer = _plain_mapping(raw.get("normalizer"))
        settings = cls(
            enabled=bool(raw.get("enabled", False)),
            profile_path=raw.get("profile_path"),
            augmentation_frame=str(raw.get("augmentation_frame", "policy")),
            normalizer_train_split_only=bool(
                normalizer.get("train_split_only", True)
            ),
            normalizer_chunk_size=int(normalizer.get("chunk_size", 2048)),
            allow_frame_config_override=bool(
                raw.get("allow_frame_config_override", False)
            ),
        )
        if settings.enabled and not settings.profile_path:
            raise ValueError("frame_adapter.profile_path is required when enabled=true")
        if settings.augmentation_frame != "policy":
            raise ValueError(
                "Phase 3 only supports augmentation_frame='policy'; native-frame "
                "augmentation would require synchronized action transformation"
            )
        if settings.normalizer_chunk_size <= 0:
            raise ValueError("frame_adapter.normalizer.chunk_size must be positive")
        return settings


class DatasetFrameTransform:
    """Transforms decoded raw samples before augmentation and normalization."""

    def __init__(self, settings: FrameAdapterSettings):
        self.settings = settings
        self.profile_path = _resolve_profile_path(str(settings.profile_path))
        self.bundle = load_profile_bundle(self.profile_path)
        self.adapter = self.bundle.adapter
        self._native_metadata = self.adapter.native_metadata()
        resolved_config = to_primitive(self.bundle.config)
        native_contract = dict(self.bundle.config.get("native_contract", {}))
        observation_schema = self.adapter.profile.observation_schema
        action_schema = self.adapter.profile.action_schema

        def tensor_fields(schema, tensor_name):
            if schema is None:
                return ()
            for tensor in schema.tensors:
                if tensor.name == tensor_name:
                    return tensor.fields
            return ()

        state_sources = {
            field.source_frame
            for field in tensor_fields(observation_schema, "agent_pos")
            if field.source_frame is not None
        }
        action_sources = {
            field.source_frame
            for field in tensor_fields(action_schema, "action")
            if field.source_frame is not None
        }
        action_targets = {
            field.target_frame
            for field in tensor_fields(action_schema, "action")
            if field.target_frame is not None
        }
        action_semantics = native_contract.get("action_semantics")
        self.metadata = {
            "frame_adapter_enabled": True,
            "frame_profile": self.adapter.profile.name,
            "canonical_frame": self.adapter.profile.canonical_frame,
            "profile_path": str(self.profile_path),
            "resolved_frame_config": resolved_config,
            "frame_config_hash": self.adapter.profile_hash,
            "resolved_config_hash": stable_sha256(resolved_config),
            "schema_version": self.adapter.profile.config_version,
            "calibration_hash": stable_sha256(
                resolved_config.get("transforms", [])
            ),
            "point_cloud_frame": native_contract.get("point_cloud_frame"),
            "state_frame": (
                next(iter(state_sources)) if len(state_sources) == 1
                else native_contract.get("state_frame", native_contract.get("state_semantics"))
            ),
            "policy_action_frame": (
                next(iter(action_targets)) if len(action_targets) == 1 else action_semantics
            ),
            "environment_action_frame": (
                next(iter(action_sources)) if len(action_sources) == 1 else action_semantics
            ),
            "augmentation_frame": settings.augmentation_frame,
            "data_operation": resolved_config.get("data_operation"),
        }

    @classmethod
    def from_config(cls, config: Any) -> Optional["DatasetFrameTransform"]:
        settings = FrameAdapterSettings.from_config(config)
        return cls(settings) if settings.enabled else None

    def _flat_training_sample(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if "obs" not in data or "action" not in data:
            raise KeyError("Training sample must contain obs and action")
        obs = data["obs"]
        return {
            "point_cloud": obs["point_cloud"],
            "agent_pos": obs["agent_pos"],
            "action": data["action"],
        }

    def training_fields_to_policy(
        self,
        *,
        point_cloud,
        agent_pos,
        action,
        runtime_context=None,
    ):
        sample = {
            "point_cloud": point_cloud,
            "agent_pos": agent_pos,
            "action": action,
        }
        return self.adapter.training_sample_to_policy_with_metadata(
            sample,
            self._native_metadata,
            runtime_context=runtime_context,
        ).data

    def training_sample_to_policy(self, data, runtime_context=None):
        transformed = self.training_fields_to_policy(
            **self._flat_training_sample(data),
            runtime_context=runtime_context,
        )
        output = dict(data)
        output["obs"] = dict(data["obs"])
        output["obs"]["point_cloud"] = transformed["point_cloud"]
        output["obs"]["agent_pos"] = transformed["agent_pos"]
        output["action"] = transformed["action"]
        return output


def configure_dataset_frame_transform(dataset, config):
    if not hasattr(dataset, "configure_frame_transform"):
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not expose configure_frame_transform"
        )
    dataset.configure_frame_transform(config)
    return dataset
