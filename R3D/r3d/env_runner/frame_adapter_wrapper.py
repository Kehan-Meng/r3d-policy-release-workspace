"""Inference boundary for canonical observations and native environment actions."""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional

from r3d.dataset.transforms.frame_aware_normalizer import normalizer_sha256
from r3d.dataset.transforms.frame_transform import (
    DatasetFrameTransform,
    FrameAdapterSettings,
    _resolve_profile_path,
)
from r3d.model.geometry.benchmark.profile import load_profile_bundle
from r3d.model.geometry.serialization import stable_sha256, to_primitive
from r3d.policy.base_policy import BasePolicy


REQUIRED_FRAME_CHECKPOINT_METADATA_KEYS = frozenset({
    "frame_adapter_enabled",
    "frame_profile",
    "canonical_frame",
    "resolved_frame_config",
    "frame_config_hash",
    "resolved_config_hash",
    "schema_version",
    "calibration_hash",
    "normalizer_hash",
    "train_split_hash",
    "normalizer_frame_config_hash",
    "point_cloud_frame",
    "state_frame",
    "policy_action_frame",
    "environment_action_frame",
    "augmentation_frame",
    "data_operation",
})


def _metadata_mismatch(message: str, *, allow_override: bool) -> None:
    if allow_override:
        warnings.warn(f"Frame configuration override enabled: {message}", RuntimeWarning)
        return
    raise ValueError(message)


def validate_frame_checkpoint_metadata(
    frame_config,
    checkpoint_metadata: Optional[Mapping[str, Any]],
    *,
    normalizer=None,
    require_checkpoint_metadata: bool = False,
    expected_train_split_hash: Optional[str] = None,
):
    settings = FrameAdapterSettings.from_config(frame_config)
    metadata = dict(checkpoint_metadata or {})
    if not settings.enabled:
        if metadata.get("frame_adapter_enabled", False):
            _metadata_mismatch(
                "Checkpoint was trained with a frame adapter, but the current config disables it",
                allow_override=settings.allow_frame_config_override,
            )
        return None
    if not metadata:
        if require_checkpoint_metadata:
            raise ValueError("Frame-enabled loading requires checkpoint frame metadata")
        return None
    if not metadata.get("frame_adapter_enabled", False):
        _metadata_mismatch(
            "Current config enables a frame adapter but the checkpoint records it disabled",
            allow_override=settings.allow_frame_config_override,
        )

    if require_checkpoint_metadata:
        missing = sorted(REQUIRED_FRAME_CHECKPOINT_METADATA_KEYS - set(metadata))
        if missing:
            raise ValueError(
                "Frame-enabled checkpoint metadata is incomplete; missing required keys: "
                + ", ".join(missing)
            )

    bundle = load_profile_bundle(_resolve_profile_path(str(settings.profile_path)))
    current_metadata = DatasetFrameTransform(settings).metadata
    for key in (
        "frame_profile",
        "canonical_frame",
        "schema_version",
        "point_cloud_frame",
        "state_frame",
        "policy_action_frame",
        "environment_action_frame",
        "augmentation_frame",
        "data_operation",
    ):
        checkpoint_value = metadata.get(key)
        current_value = current_metadata.get(key)
        if checkpoint_value != current_value:
            _metadata_mismatch(
                f"Checkpoint/current {key} mismatch: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}",
                allow_override=settings.allow_frame_config_override,
            )

    expected_hash = metadata.get("frame_config_hash")
    if expected_hash != bundle.adapter.profile_hash:
        _metadata_mismatch(
            "Checkpoint/current frame hash mismatch: "
            f"checkpoint={expected_hash!r}, current={bundle.adapter.profile_hash!r}",
            allow_override=settings.allow_frame_config_override,
        )
    resolved_config = to_primitive(bundle.config)
    expected_resolved_hash = metadata.get("resolved_config_hash")
    current_resolved_hash = stable_sha256(resolved_config)
    checkpoint_resolved_config = metadata.get("resolved_frame_config")
    if (
        checkpoint_resolved_config is not None
        and expected_resolved_hash
        and stable_sha256(checkpoint_resolved_config) != expected_resolved_hash
    ):
        _metadata_mismatch(
            "Checkpoint resolved_frame_config does not match its recorded hash",
            allow_override=settings.allow_frame_config_override,
        )
    if expected_resolved_hash and expected_resolved_hash != current_resolved_hash:
        _metadata_mismatch(
            "Checkpoint/current resolved frame config hash mismatch: "
            f"checkpoint={expected_resolved_hash!r}, current={current_resolved_hash!r}",
            allow_override=settings.allow_frame_config_override,
        )
    expected_calibration_hash = metadata.get("calibration_hash")
    current_calibration_hash = stable_sha256(resolved_config.get("transforms", []))
    if expected_calibration_hash and expected_calibration_hash != current_calibration_hash:
        _metadata_mismatch(
            "Checkpoint/current calibration hash mismatch: "
            f"checkpoint={expected_calibration_hash!r}, current={current_calibration_hash!r}",
            allow_override=settings.allow_frame_config_override,
        )
    expected_normalizer_hash = metadata.get("normalizer_hash")
    if require_checkpoint_metadata and normalizer is None:
        raise ValueError(
            "Frame-enabled loading requires the policy normalizer for hash validation"
        )
    if expected_normalizer_hash and normalizer is not None:
        actual_hash = normalizer_sha256(normalizer)
        if actual_hash != expected_normalizer_hash:
            _metadata_mismatch(
                "Checkpoint normalizer hash does not match the loaded policy: "
                f"checkpoint={expected_normalizer_hash!r}, loaded={actual_hash!r}",
                allow_override=settings.allow_frame_config_override,
            )
    normalizer_frame_hash = metadata.get("normalizer_frame_config_hash")
    if normalizer_frame_hash != bundle.adapter.profile_hash:
        _metadata_mismatch(
            "Checkpoint normalizer frame hash does not match the current profile: "
            f"checkpoint={normalizer_frame_hash!r}, current={bundle.adapter.profile_hash!r}",
            allow_override=settings.allow_frame_config_override,
        )
    train_split_hash = metadata.get("train_split_hash")
    if require_checkpoint_metadata and (
        not isinstance(train_split_hash, str) or len(train_split_hash) != 64
    ):
        raise ValueError(
            "Checkpoint train_split_hash must be a 64-character SHA-256 digest"
        )
    if (
        expected_train_split_hash is not None
        and train_split_hash != expected_train_split_hash
    ):
        raise ValueError(
            "Checkpoint/current train split hash mismatch: "
            f"checkpoint={train_split_hash!r}, current={expected_train_split_hash!r}"
        )
    return bundle


class FrameAdaptedPolicy(BasePolicy):
    """Keep the model in policy frame while exposing native actions to runners."""

    def __init__(
        self,
        policy: BasePolicy,
        frame_config,
        *,
        checkpoint_metadata: Optional[Mapping[str, Any]] = None,
        require_checkpoint_metadata: bool = False,
        runtime_context_builder=None,
    ):
        super().__init__()
        self.policy = policy
        self.settings = FrameAdapterSettings.from_config(frame_config)
        if not self.settings.enabled:
            raise ValueError("FrameAdaptedPolicy requires frame_adapter.enabled=true")
        self.profile_path = _resolve_profile_path(str(self.settings.profile_path))
        self.bundle = load_profile_bundle(self.profile_path)
        self.adapter = self.bundle.adapter
        self.runtime_context_builder = runtime_context_builder
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self._validate_checkpoint_metadata(require=require_checkpoint_metadata)

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def _validate_checkpoint_metadata(self, *, require: bool) -> None:
        validate_frame_checkpoint_metadata(
            {
                "enabled": True,
                "profile_path": str(self.profile_path),
                "augmentation_frame": self.settings.augmentation_frame,
                "allow_frame_config_override": self.settings.allow_frame_config_override,
                "normalizer": {
                    "train_split_only": self.settings.normalizer_train_split_only,
                    "chunk_size": self.settings.normalizer_chunk_size,
                },
            },
            self.checkpoint_metadata,
            normalizer=getattr(self.policy, "normalizer", None),
            require_checkpoint_metadata=require,
        )

    def _runtime_context(self, observation):
        if self.runtime_context_builder is None:
            return None
        return self.runtime_context_builder(observation)

    def predict_action(self, obs_dict, **kwargs):
        runtime_context = self._runtime_context(obs_dict)
        policy_obs = self.adapter.observation_to_policy_with_metadata(
            obs_dict,
            self.adapter.native_metadata(),
            runtime_context=runtime_context,
        ).data
        raw_result = self.policy.predict_action(policy_obs, **kwargs)
        if "action" not in raw_result:
            raise KeyError("Wrapped policy result must contain 'action'")

        result = dict(raw_result)
        action_policy = raw_result["action"]
        action_env = self.adapter.action_to_environment_with_metadata(
            action_policy,
            self.adapter.policy_metadata(),
            runtime_context=runtime_context,
        ).data
        result["action_policy"] = action_policy
        result["action_env"] = action_env
        return result

    def reset(self):
        return self.policy.reset()

    def set_normalizer(self, normalizer):
        return self.policy.set_normalizer(normalizer)

    def set_training_progress(self, epoch: int, num_epochs: int):
        return self.policy.set_training_progress(epoch, num_epochs)


def maybe_wrap_policy_for_environment(
    policy,
    frame_config,
    *,
    checkpoint_metadata=None,
    require_checkpoint_metadata=False,
    runtime_context_builder=None,
):
    settings = FrameAdapterSettings.from_config(frame_config)
    metadata = dict(checkpoint_metadata or {})
    if isinstance(policy, FrameAdaptedPolicy):
        if not settings.enabled:
            raise ValueError("Cannot disable a frame adapter around an already wrapped policy")
        requested = load_profile_bundle(
            _resolve_profile_path(str(settings.profile_path))
        ).adapter.profile_hash
        if policy.adapter.profile_hash != requested:
            raise ValueError("Cannot re-wrap a policy with a different frame profile")
        return policy
    if not settings.enabled:
        if metadata.get("frame_adapter_enabled", False):
            _metadata_mismatch(
                "Checkpoint was trained with a frame adapter, but evaluation disabled it",
                allow_override=settings.allow_frame_config_override,
            )
        return policy
    return FrameAdaptedPolicy(
        policy,
        frame_config,
        checkpoint_metadata=metadata,
        require_checkpoint_metadata=require_checkpoint_metadata,
        runtime_context_builder=runtime_context_builder,
    )


def environment_action(action_dict):
    """Select the executable action while keeping legacy policies compatible."""
    return action_dict.get("action_env", action_dict["action"])


def frame_evaluation_metadata(
    policy,
    frame_config,
    checkpoint_metadata: Optional[Mapping[str, Any]],
    *,
    config_source: str,
):
    """Describe the effective evaluation contract without changing behavior."""
    settings = FrameAdapterSettings.from_config(frame_config)
    checkpoint = dict(checkpoint_metadata or {})
    wrapped = isinstance(policy, FrameAdaptedPolicy)
    effective_hash = policy.adapter.profile_hash if wrapped else None
    effective_profile = policy.adapter.profile.name if wrapped else None
    canonical_frame = policy.adapter.profile.canonical_frame if wrapped else None
    resolved_config = to_primitive(policy.bundle.config) if wrapped else None
    checkpoint_enabled = bool(checkpoint.get("frame_adapter_enabled", False))
    checkpoint_hash = checkpoint.get("frame_config_hash")
    checkpoint_resolved_hash = checkpoint.get("resolved_config_hash")
    checkpoint_calibration_hash = checkpoint.get("calibration_hash")

    base_policy = policy.policy if wrapped else policy
    loaded_normalizer = getattr(base_policy, "normalizer", None)
    actual_normalizer_hash = (
        normalizer_sha256(loaded_normalizer) if loaded_normalizer is not None else None
    )
    checkpoint_normalizer_hash = checkpoint.get("normalizer_hash")
    checkpoint_train_split_hash = checkpoint.get("train_split_hash")
    missing_required_keys = sorted(
        REQUIRED_FRAME_CHECKPOINT_METADATA_KEYS - set(checkpoint)
    ) if checkpoint_enabled else []

    mismatch_reasons = []
    if checkpoint_enabled != settings.enabled:
        mismatch_reasons.append("enabled_state")
    if checkpoint_enabled and settings.enabled and checkpoint_hash != effective_hash:
        mismatch_reasons.append("profile_hash")
    if (
        checkpoint_enabled
        and settings.enabled
        and checkpoint_resolved_hash
        and checkpoint_resolved_hash != stable_sha256(resolved_config)
    ):
        mismatch_reasons.append("resolved_config_hash")
    effective_calibration_hash = (
        stable_sha256(resolved_config.get("transforms", [])) if wrapped else None
    )
    if (
        checkpoint_enabled
        and settings.enabled
        and checkpoint_calibration_hash
        and checkpoint_calibration_hash != effective_calibration_hash
    ):
        mismatch_reasons.append("calibration_hash")
    if (
        checkpoint_normalizer_hash
        and actual_normalizer_hash
        and checkpoint_normalizer_hash != actual_normalizer_hash
    ):
        mismatch_reasons.append("normalizer_hash")
    if missing_required_keys:
        mismatch_reasons.append("incomplete_checkpoint_metadata")

    return {
        "config_source": str(config_source),
        "enabled": settings.enabled,
        "profile_path": str(policy.profile_path) if wrapped else settings.profile_path,
        "profile": effective_profile,
        "canonical_frame": canonical_frame,
        "schema_version": policy.adapter.profile.config_version if wrapped else None,
        "resolved_frame_config": resolved_config,
        "resolved_config_hash": stable_sha256(resolved_config) if wrapped else None,
        "calibration_hash": effective_calibration_hash,
        "frame_config_hash": effective_hash,
        "checkpoint_enabled": checkpoint_enabled,
        "checkpoint_frame_config_hash": checkpoint_hash,
        "checkpoint_resolved_config_hash": checkpoint_resolved_hash,
        "checkpoint_calibration_hash": checkpoint_calibration_hash,
        "normalizer_hash": actual_normalizer_hash,
        "checkpoint_normalizer_hash": checkpoint_normalizer_hash,
        "checkpoint_train_split_hash": checkpoint_train_split_hash,
        "checkpoint_metadata_complete": not missing_required_keys,
        "checkpoint_missing_required_keys": missing_required_keys,
        "profile_hash_match": (
            checkpoint_hash == effective_hash
            if checkpoint_enabled and settings.enabled
            else checkpoint_enabled == settings.enabled
        ),
        "normalizer_hash_match": (
            checkpoint_normalizer_hash == actual_normalizer_hash
            if checkpoint_normalizer_hash and actual_normalizer_hash
            else None
        ),
        "allow_frame_config_override": settings.allow_frame_config_override,
        "override_used": bool(mismatch_reasons),
        "override_reasons": mismatch_reasons,
    }


__all__ = [
    "FrameAdaptedPolicy",
    "REQUIRED_FRAME_CHECKPOINT_METADATA_KEYS",
    "environment_action",
    "frame_evaluation_metadata",
    "maybe_wrap_policy_for_environment",
    "validate_frame_checkpoint_metadata",
]
