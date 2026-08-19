"""Load a profile, benchmark decoder, and adapter as one validated bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..frame_adapter import CanonicalFrameAdapter
from ..registry import build_adapter
from .base import BenchmarkSemanticDecoder


@dataclass(frozen=True)
class FrameProfileBundle:
    path: Path
    config: Mapping[str, Any]
    decoder: BenchmarkSemanticDecoder
    adapter: CanonicalFrameAdapter


def load_profile_config(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError(f"Frame profile {path} must contain a YAML mapping")
    return config


def load_profile_bundle(path: str | Path) -> FrameProfileBundle:
    path = Path(path).resolve()
    config = load_profile_config(path)
    from . import decoder_from_profile_config

    decoder = decoder_from_profile_config(config)
    decoder.validate_profile_config(config)
    return FrameProfileBundle(path, config, decoder, build_adapter(config))


__all__ = ["FrameProfileBundle", "load_profile_bundle", "load_profile_config"]
