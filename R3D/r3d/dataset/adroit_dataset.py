from typing import Dict
import torch
import numpy as np
import copy
import os
from omegaconf import OmegaConf
from r3d.common.pytorch_util import dict_apply
from r3d.common.replay_buffer import ReplayBuffer
from r3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from r3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from r3d.dataset.base_dataset import BaseDataset
from r3d.dataset.text_dataset import TextInstructionDataset, attach_text_fields

class AdroitDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            text_json_path=None,
            text_command=None,
            return_text=False,
            strict_text_lookup=False,
            instruction_bank=None,
            instruction_aug=None,
        ):
        super().__init__()
        self.zarr_path = str(zarr_path)
        self.task_name = task_name
        self.text_command = text_command or task_name or os.path.basename(str(zarr_path))
        self.return_text = return_text
        self.text_dataset = (
            TextInstructionDataset(text_json_path, strict=strict_text_lookup)
            if return_text else None
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
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set.training = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        if self.frame_transform_enabled:
            from r3d.dataset.transforms.frame_aware_normalizer import (
                FrameNormalizerSource,
            )

            return self._fit_frame_aware_normalizer(
                [FrameNormalizerSource(
                    arrays={
                        'action': self.replay_buffer['action'],
                        'agent_pos': self.replay_buffer['state'],
                        'point_cloud': self.replay_buffer['point_cloud'],
                    },
                    episode_ends=self.replay_buffer.episode_ends,
                    episode_mask=self.train_mask,
                    source_id=self.zarr_path,
                )],
                mode=mode,
                **kwargs,
            )
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return self._sample_to_policy_frame(data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data = attach_text_fields(
            torch_data,
            self.text_dataset,
            self.text_command,
            enabled=self.return_text,
            instruction_bank=self.instruction_bank,
            is_train=self.training,
            apply_in_train=self.instruction_apply_in_train,
            apply_in_val=self.instruction_apply_in_val,
        )
        return torch_data
