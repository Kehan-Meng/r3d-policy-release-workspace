"""Strict native-data contracts and low-coupling semantic decoding helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from ..errors import ProfileContractError, SchemaDimensionError
from ..validation import require_array


@dataclass(frozen=True)
class BenchmarkNativeContract:
    benchmark: str
    task: str
    point_cloud_dim: int
    state_dim: int
    action_dim: int
    point_cloud_frame: str
    state_semantics: str
    action_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clone_array(value):
    require_array(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.clone()
    raise TypeError(type(value))


def _mapping_value(mapping: Mapping[str, Any], key: str):
    if key not in mapping:
        raise ProfileContractError(f"Native sample is missing required key {key!r}")
    return mapping[key]


def _field_config(
    config: Mapping[str, Any], schema_name: str, tensor_name: str, field_name: str
) -> Mapping[str, Any]:
    schemas = config.get("schemas", {})
    schema = schemas.get(schema_name)
    if not isinstance(schema, Mapping):
        raise ProfileContractError(f"Profile is missing schema {schema_name!r}")
    for tensor in schema.get("tensors", ()):
        if tensor.get("name") != tensor_name:
            continue
        for field in tensor.get("fields", ()):
            if field.get("name") == field_name:
                return field
        raise ProfileContractError(
            f"Schema {schema_name!r} tensor {tensor_name!r} is missing field {field_name!r}"
        )
    raise ProfileContractError(
        f"Schema {schema_name!r} is missing tensor {tensor_name!r}"
    )


def expect_field(
    config: Mapping[str, Any],
    schema_name: str,
    tensor_name: str,
    field_name: str,
    **expected,
) -> None:
    field = _field_config(config, schema_name, tensor_name, field_name)
    for key, value in expected.items():
        if field.get(key) != value:
            raise ProfileContractError(
                f"{schema_name}.{tensor_name}.{field_name} expects {key}={value!r}, "
                f"got {field.get(key)!r}"
            )


class BenchmarkSemanticDecoder:
    """Rename native arrays into the stable keys consumed by frame profiles.

    Decoders contain benchmark semantics and shape assertions only. They do not
    perform geometry, normalization, clipping, or environment control.
    """

    contract: BenchmarkNativeContract

    def __init__(self, contract: BenchmarkNativeContract):
        self.contract = contract

    def _validate_array(self, name: str, value, expected_last_dim: int):
        require_array(value, name=name)
        if value.ndim < 1 or value.shape[-1] != expected_last_dim:
            raise SchemaDimensionError(
                f"{self.contract.benchmark}/{self.contract.task} {name} expects last dim "
                f"{expected_last_dim}, got {tuple(value.shape)}"
            )
        return value

    def decode_training_sample(self, native: Mapping[str, Any]) -> dict[str, Any]:
        point_cloud = self._validate_array(
            "point_cloud",
            _mapping_value(native, "point_cloud"),
            self.contract.point_cloud_dim,
        )
        state = self._validate_array(
            "state", _mapping_value(native, "state"), self.contract.state_dim
        )
        action = self._validate_array(
            "action", _mapping_value(native, "action"), self.contract.action_dim
        )
        return {
            "point_cloud": clone_array(point_cloud),
            "agent_pos": clone_array(state),
            "action": clone_array(action),
        }

    def encode_training_sample(self, decoded: Mapping[str, Any]) -> dict[str, Any]:
        point_cloud = self._validate_array(
            "point_cloud",
            _mapping_value(decoded, "point_cloud"),
            self.contract.point_cloud_dim,
        )
        state = self._validate_array(
            "agent_pos", _mapping_value(decoded, "agent_pos"), self.contract.state_dim
        )
        action = self._validate_array(
            "action", _mapping_value(decoded, "action"), self.contract.action_dim
        )
        return {
            "point_cloud": clone_array(point_cloud),
            "state": clone_array(state),
            "action": clone_array(action),
        }

    def decode_observation(self, native: Mapping[str, Any]) -> dict[str, Any]:
        state_key = "agent_pos" if "agent_pos" in native else "state"
        point_cloud = self._validate_array(
            "point_cloud",
            _mapping_value(native, "point_cloud"),
            self.contract.point_cloud_dim,
        )
        state = self._validate_array(
            state_key, _mapping_value(native, state_key), self.contract.state_dim
        )
        return {
            "point_cloud": clone_array(point_cloud),
            "agent_pos": clone_array(state),
        }

    def decode_action(self, native_action):
        action = self._validate_array("action", native_action, self.contract.action_dim)
        return clone_array(action)

    def encode_action(self, decoded_action):
        return self.decode_action(decoded_action)

    def validate_profile_config(self, config: Mapping[str, Any]) -> None:
        benchmark = str(config.get("benchmark", "")).lower()
        if benchmark != self.contract.benchmark:
            raise ProfileContractError(
                f"Profile benchmark {benchmark!r} does not match decoder "
                f"{self.contract.benchmark!r}"
            )
        configured_task = str(config.get("task", ""))
        if self.contract.task != "*" and configured_task != self.contract.task:
            raise ProfileContractError(
                f"Profile task {configured_task!r} does not match decoder task "
                f"{self.contract.task!r}"
            )

        native = config.get("native_contract")
        if not isinstance(native, Mapping):
            raise ProfileContractError("Profile is missing native_contract")
        expected = {
            "point_cloud_dim": self.contract.point_cloud_dim,
            "state_dim": self.contract.state_dim,
            "action_dim": self.contract.action_dim,
            "point_cloud_frame": self.contract.point_cloud_frame,
            "state_semantics": self.contract.state_semantics,
            "action_semantics": self.contract.action_semantics,
        }
        for key, value in expected.items():
            if native.get(key) != value:
                raise ProfileContractError(
                    f"native_contract.{key} expects {value!r}, got {native.get(key)!r}"
                )

    def profile_summary(self) -> dict[str, Any]:
        return self.contract.to_dict()


__all__ = [
    "BenchmarkNativeContract",
    "BenchmarkSemanticDecoder",
    "clone_array",
    "expect_field",
]
