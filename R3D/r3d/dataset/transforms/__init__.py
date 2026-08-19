"""Dataset-side canonical-frame integration."""

from .frame_transform import (
    DatasetFrameTransform,
    FrameAdapterSettings,
    configure_dataset_frame_transform,
)

__all__ = [
    "DatasetFrameTransform",
    "FrameAdapterSettings",
    "configure_dataset_frame_transform",
]
