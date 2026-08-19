"""Chunked normalizer fitting over transformed training episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from r3d.model.geometry.serialization import stable_sha256


@dataclass(frozen=True)
class FrameNormalizerSource:
    arrays: Mapping[str, Any]
    episode_ends: Any
    episode_mask: Any
    source_id: str


class _StreamingStats:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        self.total = None
        self.total_square = None
        self.count = 0

    def update(self, value) -> None:
        array = np.asarray(value)
        dim = int(array.shape[-1])
        flat = array.reshape(-1, dim).astype(np.float64, copy=False)
        if flat.size == 0:
            return
        if not np.isfinite(flat).all():
            raise ValueError("Normalizer input contains NaN or Inf")
        minimum = flat.min(axis=0)
        maximum = flat.max(axis=0)
        total = flat.sum(axis=0)
        total_square = np.square(flat).sum(axis=0)
        if self.minimum is None:
            self.minimum = minimum
            self.maximum = maximum
            self.total = total
            self.total_square = total_square
        else:
            self.minimum = np.minimum(self.minimum, minimum)
            self.maximum = np.maximum(self.maximum, maximum)
            self.total += total
            self.total_square += total_square
        self.count += flat.shape[0]

    def build(
        self,
        *,
        mode="limits",
        dtype=torch.float32,
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ) -> SingleFieldLinearNormalizer:
        if self.count == 0:
            raise ValueError("Cannot fit a normalizer from an empty training split")
        mean = self.total / self.count
        if self.count > 1:
            centered_square_sum = self.total_square - self.count * np.square(mean)
            variance = np.maximum(centered_square_sum / (self.count - 1), 0.0)
        else:
            # Match torch.std's undefined single-sample estimate without allowing
            # NaN statistics into a production checkpoint.
            variance = np.zeros_like(mean)
        std = np.sqrt(variance)
        minimum = torch.as_tensor(self.minimum, dtype=dtype)
        maximum = torch.as_tensor(self.maximum, dtype=dtype)
        mean = torch.as_tensor(mean, dtype=dtype)
        std = torch.as_tensor(std, dtype=dtype)

        if mode == "limits":
            if fit_offset:
                input_range = maximum - minimum
                ignored = input_range < range_eps
                input_range[ignored] = output_max - output_min
                scale = (output_max - output_min) / input_range
                offset = output_min - scale * minimum
                offset[ignored] = (output_max + output_min) / 2 - minimum[ignored]
            else:
                output_abs = min(abs(output_min), abs(output_max))
                input_abs = torch.maximum(torch.abs(minimum), torch.abs(maximum))
                ignored = input_abs < range_eps
                input_abs[ignored] = output_abs
                scale = output_abs / input_abs
                offset = torch.zeros_like(mean)
        elif mode == "gaussian":
            ignored = std < range_eps
            safe_std = std.clone()
            safe_std[ignored] = 1
            scale = 1 / safe_std
            offset = -mean * scale if fit_offset else torch.zeros_like(mean)
        else:
            raise ValueError(f"Unsupported normalizer mode: {mode!r}")

        return SingleFieldLinearNormalizer.create_manual(
            scale,
            offset,
            {"min": minimum, "max": maximum, "mean": mean, "std": std},
        )


def _episode_ranges(episode_ends, episode_mask, train_split_only):
    ends = np.asarray(episode_ends, dtype=np.int64)
    mask = np.asarray(episode_mask, dtype=bool)
    if ends.ndim != 1 or mask.shape != ends.shape:
        raise ValueError("episode_ends and episode_mask must be matching 1D arrays")
    start = 0
    ranges = []
    for enabled, end in zip(mask, ends):
        if (enabled or not train_split_only) and int(end) > start:
            ranges.append((start, int(end)))
        start = int(end)
    return ranges


def _identity_normalizer(dim, dtype=torch.float32):
    scale = torch.ones(dim, dtype=dtype)
    offset = torch.zeros(dim, dtype=dtype)
    stats = {
        "min": torch.zeros(dim, dtype=dtype),
        "max": torch.ones(dim, dtype=dtype),
        "mean": torch.zeros(dim, dtype=dtype),
        "std": torch.ones(dim, dtype=dtype),
    }
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stats)


def normalizer_sha256(normalizer: LinearNormalizer) -> str:
    return stable_sha256(normalizer.state_dict())


def fit_frame_aware_normalizer(
    sources: Sequence[FrameNormalizerSource],
    frame_transform,
    *,
    mode="limits",
    dtype=torch.float32,
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
    identity_fields=None,
):
    settings = frame_transform.settings
    stats = {key: _StreamingStats() for key in ("action", "agent_pos", "point_cloud")}
    split_descriptor = []

    for source in sources:
        arrays = source.arrays
        missing = {"action", "agent_pos", "point_cloud"} - set(arrays)
        if missing:
            raise KeyError(f"Normalizer source {source.source_id!r} is missing {sorted(missing)}")
        ranges = _episode_ranges(
            source.episode_ends,
            source.episode_mask,
            settings.normalizer_train_split_only,
        )
        split_descriptor.append({
            "source_id": source.source_id,
            "episode_ends": np.asarray(source.episode_ends, dtype=np.int64),
            "episode_mask": np.asarray(source.episode_mask, dtype=bool),
            "selected_ranges": ranges,
        })
        for range_start, range_end in ranges:
            for start in range(range_start, range_end, settings.normalizer_chunk_size):
                end = min(start + settings.normalizer_chunk_size, range_end)
                transformed = frame_transform.training_fields_to_policy(
                    point_cloud=np.asarray(arrays["point_cloud"][start:end]),
                    agent_pos=np.asarray(arrays["agent_pos"][start:end]),
                    action=np.asarray(arrays["action"][start:end]),
                )
                for key in stats:
                    stats[key].update(transformed[key])

    kwargs = {
        "mode": mode,
        "dtype": dtype,
        "output_max": output_max,
        "output_min": output_min,
        "range_eps": range_eps,
        "fit_offset": fit_offset,
    }
    normalizer = LinearNormalizer()
    for key, field_stats in stats.items():
        normalizer[key] = field_stats.build(**kwargs)
    for key, dim in (identity_fields or {}).items():
        normalizer[key] = _identity_normalizer(int(dim), dtype=dtype)

    metadata = {
        "frame_profile": frame_transform.adapter.profile.name,
        "frame_config_hash": frame_transform.adapter.profile_hash,
        "train_split_only": settings.normalizer_train_split_only,
        "train_split_hash": stable_sha256(split_descriptor),
        "source_count": len(split_descriptor),
        "total_episode_count": int(sum(
            len(item["episode_ends"]) for item in split_descriptor
        )),
        "selected_episode_count": int(sum(
            len(item["selected_ranges"]) for item in split_descriptor
        )),
        "selected_step_count": int(sum(
            end - start
            for item in split_descriptor
            for start, end in item["selected_ranges"]
        )),
        "statistics_hash": normalizer_sha256(normalizer),
        "mode": mode,
    }
    return normalizer, metadata


def fit_policy_array_normalizer(
    sources: Sequence[FrameNormalizerSource],
    *,
    mode="limits",
    dtype=torch.float32,
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    """Fit already-canonical policy arrays using training episodes only."""
    stats = {key: _StreamingStats() for key in ("action", "agent_pos", "point_cloud")}
    split_descriptor = []
    for source in sources:
        missing = set(stats) - set(source.arrays)
        if missing:
            raise KeyError(f"Normalizer source {source.source_id!r} is missing {sorted(missing)}")
        ranges = _episode_ranges(source.episode_ends, source.episode_mask, True)
        split_descriptor.append({
            "source_id": source.source_id,
            "episode_ends": np.asarray(source.episode_ends, dtype=np.int64),
            "episode_mask": np.asarray(source.episode_mask, dtype=bool),
            "selected_ranges": ranges,
        })
        for start, end in ranges:
            for key in stats:
                stats[key].update(np.asarray(source.arrays[key][start:end]))

    kwargs = {
        "mode": mode,
        "dtype": dtype,
        "output_max": output_max,
        "output_min": output_min,
        "range_eps": range_eps,
        "fit_offset": fit_offset,
    }
    normalizer = LinearNormalizer()
    for key, field_stats in stats.items():
        normalizer[key] = field_stats.build(**kwargs)
    metadata = {
        "coordinate_state": "policy",
        "train_split_only": True,
        "train_split_hash": stable_sha256(split_descriptor),
        "source_count": len(split_descriptor),
        "total_episode_count": int(sum(len(x["episode_ends"]) for x in split_descriptor)),
        "selected_episode_count": int(sum(len(x["selected_ranges"]) for x in split_descriptor)),
        "selected_step_count": int(sum(
            end - start
            for item in split_descriptor
            for start, end in item["selected_ranges"]
        )),
        "statistics_hash": normalizer_sha256(normalizer),
        "mode": mode,
    }
    return normalizer, metadata


__all__ = [
    "FrameNormalizerSource",
    "fit_frame_aware_normalizer",
    "fit_policy_array_normalizer",
    "normalizer_sha256",
]
