"""Benchmark semantic decoders layered above the benchmark-agnostic core."""

from .adroit import AdroitSemanticDecoder
from .base import BenchmarkNativeContract, BenchmarkSemanticDecoder
from .maniskill2 import ManiSkill2SemanticDecoder
from .metaworld import MetaWorldSemanticDecoder
from .profile import FrameProfileBundle, load_profile_bundle, load_profile_config


def decoder_from_profile_config(config):
    benchmark = str(config.get("benchmark", "")).lower()
    if benchmark == "metaworld":
        return MetaWorldSemanticDecoder()
    if benchmark == "maniskill2":
        return ManiSkill2SemanticDecoder(str(config.get("task", "")))
    if benchmark == "adroit":
        return AdroitSemanticDecoder(str(config.get("task", "")))
    from ..errors import ProfileContractError

    raise ProfileContractError(f"Unsupported benchmark in frame profile: {benchmark!r}")


__all__ = [
    "AdroitSemanticDecoder",
    "BenchmarkNativeContract",
    "BenchmarkSemanticDecoder",
    "FrameProfileBundle",
    "ManiSkill2SemanticDecoder",
    "MetaWorldSemanticDecoder",
    "decoder_from_profile_config",
    "load_profile_bundle",
    "load_profile_config",
]
