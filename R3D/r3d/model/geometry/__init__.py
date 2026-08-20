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
from .real_robot import (
    RealRobotPreflightReport,
    RealRobotRuntimeContextBuilder,
    preflight_real_robot_profile,
)
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
    "RealRobotPreflightReport",
    "RealRobotRuntimeContextBuilder",
    "SampleSchema",
    "StaticTransformProvider",
    "TensorSpec",
    "Transform",
    "TransformGraph",
    "TransformProvider",
    "build_adapter",
    "preflight_real_robot_profile",
]
