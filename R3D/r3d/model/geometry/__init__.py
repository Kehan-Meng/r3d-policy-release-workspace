"""Canonical, benchmark-agnostic coordinate-frame transformations."""

from .errors import *
from .frame_adapter import (
    AdaptationResult,
    CanonicalFrameAdapter,
    FrameMetadata,
)
from .providers import (
    EyeInHandTransformProvider,
    RuntimeTransformProvider,
    StaticTransformProvider,
    TransformProvider,
)
from .registry import build_adapter
from .schema import (
    CanonicalFrameProfile,
    FieldSpec,
    SampleSchema,
    TensorSpec,
)
from .transform_graph import TransformGraph
from .types import ArrayLike, FieldKind, Transform

__all__ = [
    "AdaptationResult",
    "ArrayLike",
    "CanonicalFrameAdapter",
    "CanonicalFrameProfile",
    "EyeInHandTransformProvider",
    "FieldKind",
    "FieldSpec",
    "FrameMetadata",
    "RuntimeTransformProvider",
    "SampleSchema",
    "StaticTransformProvider",
    "TensorSpec",
    "Transform",
    "TransformGraph",
    "TransformProvider",
    "build_adapter",
]
