"""The single high-level canonical-frame adapter implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Tuple

import numpy as np
import torch

from .errors import (
    DoubleTransformError,
    FrameMetadataMismatchError,
    MissingFieldError,
    SchemaDimensionError,
)
from .representations import (
    transform_absolute_pose,
    transform_orientation,
    transform_relative_pose_body,
    transform_relative_pose_spatial,
    transform_twist,
)
from .schema import CanonicalFrameProfile, FieldSpec, SampleSchema, TensorSpec
from .se3 import clone_array, transform_points, transform_vectors
from .serialization import stable_sha256
from .types import ArrayLike
from .validation import is_numpy_array, is_torch_array, require_array


FrameState = Literal["native", "policy"]


@dataclass(frozen=True)
class FrameMetadata:
    profile_name: str
    profile_hash: str
    state: FrameState


@dataclass(frozen=True)
class AdaptationResult:
    data: Any
    metadata: FrameMetadata


def _get_at_path(root: Any, path: Tuple[str, ...]) -> Any:
    if not path:
        return root
    current = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise MissingFieldError(f"Missing sample key {'.'.join(path)!r}")
        current = current[key]
    return current


def _replace_at_path(root: Any, path: Tuple[str, ...], value: Any) -> Any:
    if not path:
        return value
    if not isinstance(root, Mapping):
        raise MissingFieldError(f"Cannot traverse {'.'.join(path)!r} in non-mapping sample")
    key = path[0]
    if key not in root:
        raise MissingFieldError(f"Missing sample key {'.'.join(path)!r}")
    output = dict(root)
    output[key] = _replace_at_path(root[key], path[1:], value)
    return output


class CanonicalFrameAdapter:
    def __init__(self, profile: CanonicalFrameProfile, transform_provider):
        self.profile = profile
        self.transform_provider = transform_provider
        self.profile_hash = stable_sha256(
            {
                "profile": profile,
                "transforms": transform_provider.to_config(),
            }
        )

    def _schema_or_raise(self, schema: Optional[SampleSchema], operation: str) -> SampleSchema:
        if schema is None:
            raise ValueError(f"Profile {self.profile.name!r} has no schema for {operation}")
        return schema

    def _validate_tensor(self, value: Any, spec: TensorSpec) -> ArrayLike:
        require_array(value, name=spec.name)
        if value.ndim < 1 or value.shape[-1] != spec.expected_last_dim:
            raise SchemaDimensionError(
                f"Tensor {spec.name!r} at {'.'.join(spec.key_path)!r} expects last dim "
                f"{spec.expected_last_dim}, got shape {tuple(value.shape)}"
            )
        return value

    def _get_matrix(
        self,
        field: FieldSpec,
        value: ArrayLike,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ):
        if direction == "forward":
            source_frame, target_frame = field.source_frame, field.target_frame
        else:
            source_frame, target_frame = field.target_frame, field.source_frame
        if source_frame is None or target_frame is None:
            raise ValueError(f"Field {field.name!r} has no transform frames")
        transform = self.transform_provider.get_transform(
            target_frame=target_frame,
            source_frame=source_frame,
            runtime_context=runtime_context,
            like=value,
        )
        if field.units is not None and field.kind in (
            "point",
            "absolute_pose",
            "relative_pose_spatial",
        ):
            if field.units != transform.length_unit:
                raise ValueError(
                    f"Field {field.name!r} uses {field.units!r}, transform uses "
                    f"{transform.length_unit!r}"
                )
        return transform.matrix

    def _transform_field(
        self,
        value: ArrayLike,
        field: FieldSpec,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ) -> ArrayLike:
        if field.kind in ("scalar", "joint", "actuator", "passthrough"):
            return clone_array(value)
        if field.kind == "relative_pose_body":
            return transform_relative_pose_body(
                value,
                representation=field.representation,
                quaternion_order=field.quaternion_order,
            )
        if field.kind == "twist_body":
            return transform_twist(value, None, kind="body")

        if field.source_frame == field.target_frame:
            return clone_array(value)
        matrix = self._get_matrix(
            field,
            value,
            direction=direction,
            runtime_context=runtime_context,
        )
        if field.kind == "point":
            transformed = transform_points(value, matrix)
            if field.preserve_zero_sentinel:
                if is_torch_array(value):
                    zero = torch.all(value == 0, dim=-1, keepdim=True)
                    transformed = torch.where(zero, torch.zeros_like(transformed), transformed)
                else:
                    zero = np.all(value == 0, axis=-1, keepdims=True)
                    transformed = np.where(zero, np.zeros_like(transformed), transformed)
            return transformed
        if field.kind in ("vector", "direction"):
            return transform_vectors(value, matrix)
        if field.kind == "orientation":
            return transform_orientation(
                value,
                matrix,
                representation=field.representation,
                quaternion_order=field.quaternion_order,
            )
        if field.kind == "absolute_pose":
            return transform_absolute_pose(
                value,
                matrix,
                representation=field.representation,
                quaternion_order=field.quaternion_order,
            )
        if field.kind == "relative_pose_spatial":
            return transform_relative_pose_spatial(
                value,
                matrix,
                representation=field.representation,
                quaternion_order=field.quaternion_order,
            )
        if field.kind == "twist_spatial":
            return transform_twist(value, matrix, kind="spatial")
        raise ValueError(f"Unhandled field kind: {field.kind!r}")

    def _transform_tensor(
        self,
        value: ArrayLike,
        spec: TensorSpec,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ) -> ArrayLike:
        value = self._validate_tensor(value, spec)
        chunks = []
        for field in sorted(spec.fields, key=lambda item: item.start):
            source = value[..., field.start:field.end]
            chunks.append(
                self._transform_field(
                    source,
                    field,
                    direction=direction,
                    runtime_context=runtime_context,
                )
            )
        if is_torch_array(value):
            return torch.cat(chunks, dim=-1)
        if is_numpy_array(value):
            return np.concatenate(chunks, axis=-1)
        raise TypeError(type(value))

    def _transform_sample(
        self,
        sample: Any,
        schema: SampleSchema,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ) -> Any:
        output = sample
        for tensor_spec in schema.tensors:
            value = _get_at_path(output, tensor_spec.key_path)
            transformed = self._transform_tensor(
                value,
                tensor_spec,
                direction=direction,
                runtime_context=runtime_context,
            )
            output = _replace_at_path(output, tensor_spec.key_path, transformed)
        return output

    def _plain_transform(
        self,
        sample: Any,
        schema: SampleSchema,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ) -> Any:
        if self.profile.require_metadata:
            raise FrameMetadataMismatchError(
                "This profile requires sidecar metadata; use the corresponding *_with_metadata method"
            )
        return self._transform_sample(
            sample,
            schema,
            direction=direction,
            runtime_context=runtime_context,
        )

    def _transform_with_metadata(
        self,
        sample: Any,
        schema: SampleSchema,
        metadata: FrameMetadata,
        *,
        direction: Literal["forward", "inverse"],
        runtime_context: Optional[Mapping[str, Any]],
    ) -> AdaptationResult:
        if metadata.profile_name != self.profile.name or metadata.profile_hash != self.profile_hash:
            raise FrameMetadataMismatchError("Frame metadata profile name/hash does not match adapter")
        expected = "native" if direction == "forward" else "policy"
        target = "policy" if direction == "forward" else "native"
        if metadata.state != expected:
            raise DoubleTransformError(
                f"Cannot apply {direction} transform to data already marked {metadata.state!r}"
            )
        transformed = self._transform_sample(
            sample,
            schema,
            direction=direction,
            runtime_context=runtime_context,
        )
        return AdaptationResult(
            transformed,
            FrameMetadata(self.profile.name, self.profile_hash, target),
        )

    def native_metadata(self) -> FrameMetadata:
        return FrameMetadata(self.profile.name, self.profile_hash, "native")

    def policy_metadata(self) -> FrameMetadata:
        return FrameMetadata(self.profile.name, self.profile_hash, "policy")

    def observation_to_policy(self, observation, runtime_context=None):
        schema = self._schema_or_raise(self.profile.observation_schema, "observation")
        return self._plain_transform(
            observation, schema, direction="forward", runtime_context=runtime_context
        )

    def observation_to_native(self, observation, runtime_context=None):
        """Invert an observation transform for offline validation and replay tools."""
        schema = self._schema_or_raise(self.profile.observation_schema, "observation")
        return self._plain_transform(
            observation, schema, direction="inverse", runtime_context=runtime_context
        )

    def action_to_policy(self, action, runtime_context=None):
        schema = self._schema_or_raise(self.profile.action_schema, "action")
        return self._plain_transform(
            action, schema, direction="forward", runtime_context=runtime_context
        )

    def action_to_environment(self, action, runtime_context=None):
        schema = self._schema_or_raise(self.profile.action_schema, "action")
        return self._plain_transform(
            action, schema, direction="inverse", runtime_context=runtime_context
        )

    def training_sample_to_policy(self, sample, runtime_context=None):
        schema = self._schema_or_raise(self.profile.training_schema, "training sample")
        return self._plain_transform(
            sample, schema, direction="forward", runtime_context=runtime_context
        )

    def training_sample_to_native(self, sample, runtime_context=None):
        """Invert a complete training sample for Phase 2 round-trip validation."""
        schema = self._schema_or_raise(self.profile.training_schema, "training sample")
        return self._plain_transform(
            sample, schema, direction="inverse", runtime_context=runtime_context
        )

    def observation_to_policy_with_metadata(
        self, observation, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.observation_schema, "observation")
        return self._transform_with_metadata(
            observation,
            schema,
            metadata,
            direction="forward",
            runtime_context=runtime_context,
        )

    def observation_to_native_with_metadata(
        self, observation, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.observation_schema, "observation")
        return self._transform_with_metadata(
            observation,
            schema,
            metadata,
            direction="inverse",
            runtime_context=runtime_context,
        )

    def action_to_policy_with_metadata(
        self, action, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.action_schema, "action")
        return self._transform_with_metadata(
            action,
            schema,
            metadata,
            direction="forward",
            runtime_context=runtime_context,
        )

    def action_to_environment_with_metadata(
        self, action, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.action_schema, "action")
        return self._transform_with_metadata(
            action,
            schema,
            metadata,
            direction="inverse",
            runtime_context=runtime_context,
        )

    def training_sample_to_policy_with_metadata(
        self, sample, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.training_schema, "training sample")
        return self._transform_with_metadata(
            sample,
            schema,
            metadata,
            direction="forward",
            runtime_context=runtime_context,
        )

    def training_sample_to_native_with_metadata(
        self, sample, metadata: FrameMetadata, runtime_context=None
    ) -> AdaptationResult:
        schema = self._schema_or_raise(self.profile.training_schema, "training sample")
        return self._transform_with_metadata(
            sample,
            schema,
            metadata,
            direction="inverse",
            runtime_context=runtime_context,
        )

    def validate_sample(self, sample, *, schema: Optional[SampleSchema] = None) -> None:
        schema = schema or self.profile.training_schema or self.profile.observation_schema
        schema = self._schema_or_raise(schema, "validation")
        for tensor_spec in schema.tensors:
            self._validate_tensor(_get_at_path(sample, tensor_spec.key_path), tensor_spec)
