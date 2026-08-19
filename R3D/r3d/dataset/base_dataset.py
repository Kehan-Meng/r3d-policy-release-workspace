from typing import Dict

import torch
import torch.nn
from r3d.model.common.normalizer import LinearNormalizer


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._frame_transform = None
        self._frame_normalizer_metadata = None

    def configure_frame_transform(self, config) -> None:
        """Install the optional canonical-frame transform before loading samples."""
        from r3d.dataset.transforms.frame_transform import DatasetFrameTransform

        self._frame_transform = DatasetFrameTransform.from_config(config)

    @property
    def frame_transform_enabled(self) -> bool:
        return self._frame_transform is not None

    @property
    def frame_transform_metadata(self):
        if self._frame_transform is None:
            return {
                'frame_adapter_enabled': False,
                'normalizer': self._frame_normalizer_metadata,
            }
        metadata = dict(self._frame_transform.metadata)
        metadata['normalizer'] = self._frame_normalizer_metadata
        return metadata

    def _sample_to_policy_frame(self, data):
        if self._frame_transform is None:
            return data
        return self._frame_transform.training_sample_to_policy(data)

    def _fit_frame_aware_normalizer(self, sources, **kwargs):
        if self._frame_transform is None:
            raise RuntimeError("Frame-aware normalizer requested while adapter is disabled")
        from r3d.dataset.transforms.frame_aware_normalizer import (
            fit_frame_aware_normalizer,
        )

        normalizer, metadata = fit_frame_aware_normalizer(
            sources,
            self._frame_transform,
            **kwargs,
        )
        self._frame_normalizer_metadata = metadata
        return normalizer

    def get_validation_dataset(self) -> 'BaseDataset':
        # return an empty dataset by default
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
