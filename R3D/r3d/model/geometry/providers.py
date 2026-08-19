"""Static and runtime transform providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

from .errors import (
    RuntimeTransformMissingError,
    TimestampMismatchError,
    TransformPathNotFoundError,
    UnsupportedArrayTypeError,
)
from .schema import normalize_key_path
from .se3 import compose_transforms, invert_transform
from .types import ArrayLike, Transform
from .validation import is_numpy_array, is_torch_array


@runtime_checkable
class TransformProvider(Protocol):
    source_frame: str
    target_frame: str

    def get_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
        runtime_context: Optional[Mapping[str, Any]] = None,
        like: Optional[ArrayLike] = None,
    ) -> Transform:
        ...

    def to_config(self) -> Mapping[str, Any]:
        ...


def _nested_get(mapping: Mapping[str, Any], key_path: Tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            raise RuntimeTransformMissingError(
                f"runtime_context is missing {'.'.join(key_path)!r}"
            )
        current = current[key]
    return current


def _coerce_like(value: ArrayLike, like: Optional[ArrayLike]) -> ArrayLike:
    if like is None:
        return value.clone() if is_torch_array(value) else np.array(value, copy=True)
    if is_torch_array(like):
        if is_torch_array(value):
            return value.to(device=like.device, dtype=like.dtype)
        if is_numpy_array(value):
            return torch.as_tensor(value, device=like.device, dtype=like.dtype)
    elif is_numpy_array(like):
        if is_numpy_array(value):
            return np.asarray(value, dtype=like.dtype).copy()
        raise UnsupportedArrayTypeError(
            "A runtime Torch transform cannot be converted to NumPy implicitly"
        )
    raise UnsupportedArrayTypeError(f"Unsupported array type: {type(value)!r}")


class StaticTransformProvider:
    def __init__(self, transform: Transform):
        self.source_frame = transform.source_frame
        self.target_frame = transform.target_frame
        self.length_unit = transform.length_unit
        if is_torch_array(transform.matrix):
            if transform.matrix.requires_grad:
                raise ValueError("Static calibration matrices cannot require gradients")
            canonical = transform.matrix.detach().cpu().to(torch.float64).numpy()
        else:
            canonical = np.asarray(transform.matrix, dtype=np.float64)
        self._forward = np.array(canonical, copy=True)
        self._inverse = np.asarray(
            invert_transform(
                Transform(
                    self.source_frame,
                    self.target_frame,
                    self._forward,
                    length_unit=self.length_unit,
                )
            ).matrix,
            dtype=np.float64,
        )
        self._forward.setflags(write=False)
        self._inverse.setflags(write=False)
        self._torch_cache: dict[tuple[str, torch.dtype, bool], torch.Tensor] = {}
        self._transform_cache: dict[tuple[str, str, str, object], Transform] = {}

    def _matrix_like(self, matrix: np.ndarray, like: Optional[ArrayLike], inverse: bool) -> ArrayLike:
        if like is None:
            return np.array(matrix, copy=True)
        if is_numpy_array(like):
            return np.asarray(matrix, dtype=like.dtype).copy()
        if is_torch_array(like):
            key = (str(like.device), like.dtype, inverse)
            cached = self._torch_cache.get(key)
            if cached is None:
                cached = torch.tensor(matrix, device=like.device, dtype=like.dtype)
                self._torch_cache[key] = cached
            return cached
        raise UnsupportedArrayTypeError(f"Unsupported like value: {type(like)!r}")

    def get_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
        runtime_context: Optional[Mapping[str, Any]] = None,
        like: Optional[ArrayLike] = None,
    ) -> Transform:
        del runtime_context
        if like is None:
            backend_key, dtype_key = "numpy", np.dtype(np.float64).str
        elif is_numpy_array(like):
            backend_key, dtype_key = "numpy", like.dtype.str
        elif is_torch_array(like):
            backend_key, dtype_key = str(like.device), like.dtype
        else:
            raise UnsupportedArrayTypeError(f"Unsupported like value: {type(like)!r}")
        cache_key = (source_frame, target_frame, backend_key, dtype_key)
        cached_transform = self._transform_cache.get(cache_key)
        if cached_transform is not None:
            return cached_transform

        if source_frame == self.source_frame and target_frame == self.target_frame:
            matrix = self._matrix_like(self._forward, like, inverse=False)
        elif source_frame == self.target_frame and target_frame == self.source_frame:
            matrix = self._matrix_like(self._inverse, like, inverse=True)
        else:
            raise TransformPathNotFoundError(
                f"Provider only connects {self.source_frame!r} and {self.target_frame!r}, "
                f"not {source_frame!r} -> {target_frame!r}"
            )
        if is_numpy_array(matrix):
            matrix.setflags(write=False)
        result = Transform(
            source_frame=source_frame,
            target_frame=target_frame,
            matrix=matrix,
            length_unit=self.length_unit,
        )
        self._transform_cache[cache_key] = result
        return result

    def to_config(self) -> Mapping[str, Any]:
        return {
            "type": "static",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "length_unit": self.length_unit,
            "matrix": self._forward.tolist(),
        }


class RuntimeTransformProvider:
    def __init__(
        self,
        *,
        source_frame: str,
        target_frame: str,
        context_key: str | Tuple[str, ...],
        length_unit: str = "meter",
    ):
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.context_key = normalize_key_path(context_key)
        self.length_unit = length_unit
        if not self.context_key:
            raise ValueError("Runtime transform context_key must be non-empty")

    def get_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
        runtime_context: Optional[Mapping[str, Any]] = None,
        like: Optional[ArrayLike] = None,
    ) -> Transform:
        if runtime_context is None:
            raise RuntimeTransformMissingError("runtime_context is required")
        raw_matrix = _nested_get(runtime_context, self.context_key)
        matrix = _coerce_like(raw_matrix, like)
        forward = Transform(
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            matrix=matrix,
            length_unit=self.length_unit,
        )
        if source_frame == self.source_frame and target_frame == self.target_frame:
            return forward
        if source_frame == self.target_frame and target_frame == self.source_frame:
            return invert_transform(forward)
        raise TransformPathNotFoundError(
            f"Runtime provider only connects {self.source_frame!r} and "
            f"{self.target_frame!r}"
        )

    def to_config(self) -> Mapping[str, Any]:
        return {
            "type": "runtime",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "length_unit": self.length_unit,
            "context_key": ".".join(self.context_key),
        }


class EyeInHandTransformProvider:
    """Compose dynamic base<-EE FK with a static EE<-camera calibration."""

    def __init__(
        self,
        *,
        base_from_ee_provider: TransformProvider,
        ee_from_camera_provider: TransformProvider,
        camera_timestamp_key: str | Tuple[str, ...],
        robot_timestamp_key: str | Tuple[str, ...],
        timestamp_tolerance: float,
    ):
        if base_from_ee_provider.source_frame != ee_from_camera_provider.target_frame:
            raise ValueError("Eye-in-hand provider frames do not compose")
        self._base_from_ee = base_from_ee_provider
        self._ee_from_camera = ee_from_camera_provider
        self.source_frame = ee_from_camera_provider.source_frame
        self.target_frame = base_from_ee_provider.target_frame
        self.camera_timestamp_key = normalize_key_path(camera_timestamp_key)
        self.robot_timestamp_key = normalize_key_path(robot_timestamp_key)
        self.timestamp_tolerance = float(timestamp_tolerance)
        if self.timestamp_tolerance < 0:
            raise ValueError("timestamp_tolerance must be non-negative")

    def _validate_timestamps(self, runtime_context: Mapping[str, Any]) -> None:
        camera_time = float(_nested_get(runtime_context, self.camera_timestamp_key))
        robot_time = float(_nested_get(runtime_context, self.robot_timestamp_key))
        if abs(camera_time - robot_time) > self.timestamp_tolerance:
            raise TimestampMismatchError(
                f"Camera/robot timestamp gap {abs(camera_time - robot_time):.6g} "
                f"exceeds tolerance {self.timestamp_tolerance:.6g}"
            )

    def get_transform(
        self,
        *,
        target_frame: str,
        source_frame: str,
        runtime_context: Optional[Mapping[str, Any]] = None,
        like: Optional[ArrayLike] = None,
    ) -> Transform:
        if runtime_context is None:
            raise RuntimeTransformMissingError("runtime_context is required")
        self._validate_timestamps(runtime_context)
        base_from_ee = self._base_from_ee.get_transform(
            target_frame=self._base_from_ee.target_frame,
            source_frame=self._base_from_ee.source_frame,
            runtime_context=runtime_context,
            like=like,
        )
        ee_from_camera = self._ee_from_camera.get_transform(
            target_frame=self._ee_from_camera.target_frame,
            source_frame=self._ee_from_camera.source_frame,
            runtime_context=runtime_context,
            like=like,
        )
        base_from_camera = compose_transforms(base_from_ee, ee_from_camera)
        if source_frame == self.source_frame and target_frame == self.target_frame:
            return base_from_camera
        if source_frame == self.target_frame and target_frame == self.source_frame:
            return invert_transform(base_from_camera)
        raise TransformPathNotFoundError(
            f"Eye-in-hand provider only connects {self.source_frame!r} and "
            f"{self.target_frame!r}"
        )

    def to_config(self) -> Mapping[str, Any]:
        return {
            "type": "eye_in_hand",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "camera_timestamp_key": ".".join(self.camera_timestamp_key),
            "robot_timestamp_key": ".".join(self.robot_timestamp_key),
            "timestamp_tolerance": self.timestamp_tolerance,
            "base_from_ee": self._base_from_ee.to_config(),
            "ee_from_camera": self._ee_from_camera.to_config(),
        }
