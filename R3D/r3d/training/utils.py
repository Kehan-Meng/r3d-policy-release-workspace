import datetime
import json

import numpy as np
import torch
import torch.distributed as dist

from r3d.common.pytorch_util import dict_apply


def setup_ddp(rank, world_size):
    """Initialize the distributed training process group."""
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=10),
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def batch_to_device(batch, device):
    """Move tensors to a device while preserving text and other metadata."""
    return dict_apply(
        batch,
        lambda value: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value,
    )


def copy_state_dict_to_cpu(state_dict):
    return {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

