"""Build canonical-frame adapters from plain serializable mappings."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .frame_adapter import CanonicalFrameAdapter
from .providers import (
    EyeInHandTransformProvider,
    RuntimeTransformProvider,
    StaticTransformProvider,
)
from .schema import (
    CanonicalFrameProfile,
    FieldSpec,
    SampleSchema,
    TensorSpec,
)
from .transform_graph import TransformGraph
from .types import Transform


def _field_from_config(config: Mapping[str, Any]) -> FieldSpec:
    data = dict(config)
    if "slice" in data:
        start, end = data.pop("slice")
        data["start"] = start
        data["end"] = end
    return FieldSpec(**data)


def _tensor_from_config(config: Mapping[str, Any]) -> TensorSpec:
    return TensorSpec(
        name=config["name"],
        key_path=config.get("key_path", ""),
        expected_last_dim=int(config["expected_last_dim"]),
        fields=tuple(_field_from_config(field) for field in config["fields"]),
    )


def _schema_from_config(config: Optional[Mapping[str, Any]]) -> Optional[SampleSchema]:
    if config is None:
        return None
    return SampleSchema(
        name=config["name"],
        tensors=tuple(_tensor_from_config(tensor) for tensor in config["tensors"]),
    )


def build_adapter(config: Mapping[str, Any]) -> CanonicalFrameAdapter:
    if int(config.get("config_version", 1)) != 1:
        raise ValueError("Only frame profile config_version=1 is supported")

    named_providers = {}
    deferred_eye_in_hand = []
    graph = TransformGraph()
    for transform_config in config.get("transforms", ()):
        provider_config = transform_config.get("provider", {})
        provider_type = provider_config.get("type")
        name = transform_config["name"]
        if name in named_providers or any(item[0] == name for item in deferred_eye_in_hand):
            raise ValueError(f"Duplicate transform provider name: {name!r}")
        source_frame = transform_config["source_frame"]
        target_frame = transform_config["target_frame"]
        if provider_type == "static":
            matrix = np.asarray(provider_config["matrix"], dtype=np.float64)
            provider = StaticTransformProvider(
                Transform(
                    source_frame,
                    target_frame,
                    matrix,
                    length_unit=provider_config.get("length_unit", "meter"),
                )
            )
        elif provider_type == "runtime":
            provider = RuntimeTransformProvider(
                source_frame=source_frame,
                target_frame=target_frame,
                context_key=provider_config["context_key"],
                length_unit=provider_config.get("length_unit", "meter"),
            )
        elif provider_type == "eye_in_hand":
            deferred_eye_in_hand.append(
                (
                    name,
                    source_frame,
                    target_frame,
                    provider_config,
                    bool(transform_config.get("expose_in_graph", True)),
                )
            )
            continue
        else:
            raise ValueError(f"Unknown transform provider type: {provider_type!r}")
        named_providers[name] = provider
        if bool(transform_config.get("expose_in_graph", True)):
            graph.add_provider(provider)

    for name, source_frame, target_frame, provider_config, expose_in_graph in deferred_eye_in_hand:
        dependency_names = (
            provider_config["base_from_ee"],
            provider_config["ee_from_camera"],
        )
        missing = [dependency for dependency in dependency_names if dependency not in named_providers]
        if missing:
            raise ValueError(
                f"Eye-in-hand transform {name!r} references unknown providers: {missing}"
            )
        provider = EyeInHandTransformProvider(
            base_from_ee_provider=named_providers[dependency_names[0]],
            ee_from_camera_provider=named_providers[dependency_names[1]],
            camera_timestamp_key=provider_config["camera_timestamp_key"],
            robot_timestamp_key=provider_config["robot_timestamp_key"],
            timestamp_tolerance=float(provider_config["timestamp_tolerance"]),
        )
        if provider.source_frame != source_frame or provider.target_frame != target_frame:
            raise ValueError(f"Eye-in-hand transform {name!r} frame declaration is inconsistent")
        named_providers[name] = provider
        if expose_in_graph:
            graph.add_provider(provider)

    schemas = config.get("schemas", {})
    profile = CanonicalFrameProfile(
        name=config["name"],
        canonical_frame=config["canonical_frame"],
        observation_schema=_schema_from_config(schemas.get("observation")),
        action_schema=_schema_from_config(schemas.get("action")),
        training_schema=_schema_from_config(schemas.get("training")),
        config_version=int(config.get("config_version", 1)),
        require_metadata=bool(config.get("require_metadata", False)),
    )
    return CanonicalFrameAdapter(profile, graph)
