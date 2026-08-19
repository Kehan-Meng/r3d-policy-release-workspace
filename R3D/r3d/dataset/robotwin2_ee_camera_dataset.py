"""Dataset loader for replay-validated RoboTwin2 camera-frame EE16 data."""

from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import torch

from r3d.common.pytorch_util import dict_apply
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.transforms.frame_aware_normalizer import (
    FrameNormalizerSource,
    fit_policy_array_normalizer,
)
from r3d.dataset.text_dataset import TextInstructionDataset, attach_text_fields


class Robotwin2EECameraDataset(BaseDataset):
    """Read policy-ready tensors while retaining world arrays in the Zarr."""

    REQUIRED_KEYS = (
        "point_cloud_camera",
        "policy_state30",
        "commanded_ee16_camera",
    )

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.2,
        max_train_episodes=None,
        task_name=None,
        text_command=None,
        text_json_path=None,
        return_text=False,
        strict_text_lookup=False,
    ):
        super().__init__()
        self.zarr_path = str(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            self.zarr_path, keys=list(self.REQUIRED_KEYS)
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask, max_n=max_train_episodes, seed=seed
        )
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.sampler = self._make_sampler(train_mask)
        self.task_name = task_name
        self.text_command = text_command or task_name or os.path.basename(self.zarr_path)
        self.return_text = bool(return_text)
        self.text_dataset = (
            TextInstructionDataset(text_json_path, strict=strict_text_lookup)
            if self.return_text else None
        )
        self.training = True
        self._policy_normalizer_metadata = None

    def _make_sampler(self, episode_mask):
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self):
        dataset = copy.copy(self)
        dataset.train_mask = ~self.train_mask
        dataset.sampler = dataset._make_sampler(dataset.train_mask)
        dataset.training = False
        return dataset

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer, metadata = fit_policy_array_normalizer(
            [FrameNormalizerSource(
                arrays={
                    "action": self.replay_buffer["commanded_ee16_camera"],
                    "agent_pos": self.replay_buffer["policy_state30"],
                    "point_cloud": self.replay_buffer["point_cloud_camera"],
                },
                episode_ends=self.replay_buffer.episode_ends,
                episode_mask=self.train_mask,
                source_id=self.zarr_path,
            )],
            mode=mode,
            **kwargs,
        )
        self._policy_normalizer_metadata = metadata
        return normalizer

    @property
    def policy_normalizer_metadata(self):
        return copy.deepcopy(self._policy_normalizer_metadata)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(index)
        data = {
            "obs": {
                "point_cloud": sample["point_cloud_camera"].astype(np.float32),
                "agent_pos": sample["policy_state30"].astype(np.float32),
            },
            "action": sample["commanded_ee16_camera"].astype(np.float32),
        }
        torch_data = dict_apply(data, torch.from_numpy)
        return attach_text_fields(
            torch_data,
            self.text_dataset,
            self.text_command,
            enabled=self.return_text,
            is_train=self.training,
        )
