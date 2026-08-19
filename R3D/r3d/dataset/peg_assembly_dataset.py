"""Zarr dataset adapter for the MuJoCo Franka peg-assembly benchmark.

The collector contract is intentionally small and matches the other R3D
single-task datasets::

    data/point_cloud  float32 [T, 512, 6]
    data/state        float32 [T, 30]
    data/action       float32 [T, 6]
    meta/episode_ends int64   [E]

Additional arrays written by the collector are diagnostic data and are not
loaded into memory by this adapter.
"""

import copy
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from r3d.common.pytorch_util import dict_apply
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.text_dataset import TextInstructionDataset, attach_text_fields
from r3d.model.common.normalizer import LinearNormalizer


RESET_DISTRIBUTION_RANGES = {
    "train": (0.020, 20.0, 30.0, (0.020, 0.050)),
    "id": (0.020, 20.0, 30.0, (0.020, 0.050)),
    "ood": (0.030, 30.0, 45.0, (0.020, 0.070)),
    "aligned": (0.0, 0.0, 0.0, (0.0, 0.0)),
}


class PegAssemblyDataset(BaseDataset):
    """Load successful peg-assembly demonstrations for behavior cloning."""

    POINT_CLOUD_SHAPE = (512, 6)
    STATE_DIM = 30
    ACTION_DIM = 6

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name="peg_assembly",
        text_json_path=None,
        text_command=None,
        return_text=False,
        strict_text_lookup=False,
        instruction_bank=None,
        instruction_aug=None,
        validate_shapes=True,
        expected_shape=None,
        expected_distribution=None,
        normalizer_train_split_only=True,
        normalizer_chunk_size=2048,
    ):
        super().__init__()
        self.zarr_path = str(zarr_path)
        self.task_name = task_name
        self.text_command = (
            text_command or task_name or os.path.basename(self.zarr_path)
        )
        self.return_text = bool(return_text)
        self.normalizer_train_split_only = bool(normalizer_train_split_only)
        self.normalizer_chunk_size = int(normalizer_chunk_size)
        if self.normalizer_chunk_size < 1:
            raise ValueError("normalizer_chunk_size must be at least 1")
        self.text_dataset = (
            TextInstructionDataset(text_json_path, strict=strict_text_lookup)
            if self.return_text
            else None
        )

        self.instruction_bank = instruction_bank
        self.instruction_apply_in_train = True
        self.instruction_apply_in_val = False
        if self.instruction_bank is None and (
            isinstance(instruction_aug, dict) or OmegaConf.is_dict(instruction_aug)
        ):
            if instruction_aug.get("enabled"):
                from r3d.common.instruction_bank import InstructionBank

                self.instruction_bank = InstructionBank(
                    instruction_aug.get("bank_path")
                )
                self.instruction_apply_in_train = instruction_aug.get(
                    "apply_in_train", True
                )
                self.instruction_apply_in_val = instruction_aug.get(
                    "apply_in_val", False
                )

        self.training = True
        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        if validate_shapes:
            self._validate_replay_contract()
        if expected_shape is not None:
            self._validate_episode_shapes(str(expected_shape))
        if expected_distribution is not None:
            self._validate_collection_distribution(str(expected_distribution))

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)

    def _validate_replay_contract(self):
        episode_ends_raw = np.asarray(self.replay_buffer.episode_ends)
        if episode_ends_raw.dtype != np.dtype(np.int64):
            raise TypeError(
                "PegAssemblyDataset meta/episode_ends must be int64, got "
                f"{episode_ends_raw.dtype}"
            )
        episode_ends = episode_ends_raw.astype(np.int64, copy=False)
        if episode_ends.ndim != 1 or episode_ends.size == 0:
            raise ValueError(
                f"PegAssemblyDataset requires at least one complete episode in "
                f"meta/episode_ends, got shape {episode_ends.shape}"
            )
        if episode_ends[0] <= 0 or np.any(np.diff(episode_ends) <= 0):
            raise ValueError(
                "PegAssemblyDataset meta/episode_ends must be positive and "
                f"strictly increasing, got {episode_ends.tolist()}"
            )

        expected = {
            "point_cloud": self.POINT_CLOUD_SHAPE,
            "state": (self.STATE_DIM,),
            "action": (self.ACTION_DIM,),
        }
        for key, trailing_shape in expected.items():
            actual = tuple(self.replay_buffer[key].shape[1:])
            if actual != trailing_shape:
                raise ValueError(
                    f"PegAssemblyDataset expected data/{key} trailing shape "
                    f"{trailing_shape}, got {actual} in {self.zarr_path}"
                )

            values = np.asarray(self.replay_buffer[key])
            if values.dtype != np.dtype(np.float32):
                raise TypeError(
                    f"PegAssemblyDataset data/{key} must be float32, got "
                    f"dtype {values.dtype}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"PegAssemblyDataset data/{key} contains NaN or infinite values"
                )

        actions = np.asarray(self.replay_buffer["action"])
        action_tolerance = 1.0e-4
        if (
            float(actions.min()) < -1.0 - action_tolerance
            or float(actions.max()) > 1.0 + action_tolerance
        ):
            raise ValueError(
                "PegAssemblyDataset expects normalized actions in [-1, 1]; "
                f"observed [{float(actions.min()):.6g}, "
                f"{float(actions.max()):.6g}]"
            )

    def _validate_episode_shapes(self, expected_shape):
        """Reject accidental shape-OOD leakage into behavior-cloning data."""

        root = zarr.open_group(self.zarr_path, mode="r")
        if "meta" not in root or "shape" not in root["meta"]:
            raise ValueError(
                "PegAssemblyDataset expected meta/shape so the square-only "
                f"training protocol can be enforced: {self.zarr_path}"
            )
        raw = np.asarray(root["meta/shape"][:]).reshape(-1)
        actual = {
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in raw
        }
        if actual != {expected_shape}:
            raise ValueError(
                "PegAssemblyDataset shape protocol violation: "
                f"expected only {expected_shape!r}, found {sorted(actual)} in "
                f"{self.zarr_path}"
            )

    def _validate_collection_distribution(self, expected_distribution):
        """Keep wider pose-OOD demonstrations out of the square train split."""

        manifest_path = Path(self.zarr_path) / "generation_metadata.json"
        if not manifest_path.is_file():
            raise ValueError(
                "PegAssemblyDataset expected generation_metadata.json so the "
                f"training reset distribution can be enforced: {self.zarr_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collector = manifest.get("collector", {})
        actual = collector.get("distribution")
        if actual != expected_distribution:
            raise ValueError(
                "PegAssemblyDataset reset-distribution protocol violation: "
                f"expected {expected_distribution!r}, found {actual!r} in "
                f"{self.zarr_path}"
            )
        if expected_distribution == "train" and collector.get(
            "fixed_reset_options"
        ) is not None:
            raise ValueError(
                "PegAssemblyDataset square train data must use independently "
                "sampled resets, not fixed_reset_options"
            )
        if expected_distribution not in RESET_DISTRIBUTION_RANGES:
            raise ValueError(
                f"Unknown PegAssembly reset distribution {expected_distribution!r}"
            )
        expected_env_distribution = (
            "id" if expected_distribution in ("train", "aligned")
            else expected_distribution
        )
        actual_env_distribution = collector.get("env_kwargs", {}).get("distribution")
        if actual_env_distribution != expected_env_distribution:
            raise ValueError(
                "PegAssemblyDataset collector.env_kwargs.distribution mismatch: "
                f"expected {expected_env_distribution!r}, found "
                f"{actual_env_distribution!r}"
            )

        root = zarr.open_group(self.zarr_path, mode="r")
        reset_options = np.asarray(root["meta/reset_options"][:], dtype=np.float64)
        xy_bound, tilt_deg, yaw_deg, z_range = RESET_DISTRIBUTION_RANGES[
            expected_distribution
        ]
        upper = np.array(
            [
                xy_bound,
                xy_bound,
                np.deg2rad(tilt_deg),
                np.deg2rad(tilt_deg),
                np.deg2rad(yaw_deg),
            ]
        )
        tolerance = 2.0e-6
        if np.any(np.abs(reset_options[:, :5]) > upper + tolerance) or np.any(
            reset_options[:, 5] < z_range[0] - tolerance
        ) or np.any(reset_options[:, 5] > z_range[1] + tolerance):
            raise ValueError(
                "PegAssemblyDataset reset options exceed the declared "
                f"{expected_distribution!r} distribution bounds"
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        val_set.training = False
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        if self.frame_transform_enabled:
            from r3d.dataset.transforms.frame_aware_normalizer import (
                FrameNormalizerSource,
            )

            return self._fit_frame_aware_normalizer(
                [
                    FrameNormalizerSource(
                        arrays={
                            "action": self.replay_buffer["action"],
                            "agent_pos": self.replay_buffer["state"],
                            "point_cloud": self.replay_buffer["point_cloud"],
                        },
                        episode_ends=self.replay_buffer.episode_ends,
                        episode_mask=self.train_mask,
                        source_id=self.zarr_path,
                    )
                ],
                mode=mode,
                **kwargs,
            )

        # The stock LinearNormalizer consumes complete arrays, which leaks
        # validation extrema into training.  Reuse the repository's streaming
        # statistics primitive to fit only selected episodes without copying a
        # potentially multi-gigabyte point-cloud array.
        from r3d.dataset.transforms.frame_aware_normalizer import (
            _StreamingStats,
            _episode_ranges,
        )

        last_n_dims = int(kwargs.pop("last_n_dims", 1))
        if last_n_dims != 1:
            raise ValueError(
                "PegAssemblyDataset normalizer expects last_n_dims=1, got "
                f"{last_n_dims}"
            )
        arrays = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        stats = {key: _StreamingStats() for key in arrays}
        ranges = _episode_ranges(
            self.replay_buffer.episode_ends,
            self.train_mask,
            self.normalizer_train_split_only,
        )
        for range_start, range_end in ranges:
            for start in range(
                range_start, range_end, self.normalizer_chunk_size
            ):
                end = min(start + self.normalizer_chunk_size, range_end)
                for key, array in arrays.items():
                    stats[key].update(array[start:end])

        normalizer = LinearNormalizer()
        for key, field_stats in stats.items():
            normalizer[key] = field_stats.build(mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self.replay_buffer["action"], dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(
                    np.float32, copy=False
                ),
                "agent_pos": sample["state"].astype(np.float32, copy=False),
            },
            "action": sample["action"].astype(np.float32, copy=False),
        }
        return self._sample_to_policy_frame(data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return attach_text_fields(
            torch_data,
            self.text_dataset,
            self.text_command,
            enabled=self.return_text,
            instruction_bank=self.instruction_bank,
            is_train=self.training,
            apply_in_train=self.instruction_apply_in_train,
            apply_in_val=self.instruction_apply_in_val,
        )
