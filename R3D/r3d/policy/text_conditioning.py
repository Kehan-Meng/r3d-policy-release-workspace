from __future__ import annotations

from typing import Optional

import torch


def compute_text_feature(text_encoder, batch_size, *, texts=None, commands=None):
    if text_encoder is None:
        return None
    return text_encoder.forward(batch_size, texts=texts, commands=commands)


def resolve_text_prompts(
    text_encoder,
    batch_size: int,
    *,
    texts=None,
    commands=None,
):
    if text_encoder is not None:
        return text_encoder._resolve_texts(
            batch_size,
            texts=texts,
            commands=commands,
        )

    source = texts if texts is not None else commands
    if source is None:
        resolved = [""]
    elif isinstance(source, str):
        resolved = [source]
    elif isinstance(source, (tuple, list)):
        resolved = list(source)
    else:
        resolved = [str(source)]

    resolved = [str(item) for item in resolved]
    if len(resolved) == batch_size:
        return resolved
    if len(resolved) == 1:
        return resolved * batch_size
    raise ValueError(
        f"text batch size mismatch: got {len(resolved)} prompts "
        f"for batch_size={batch_size}"
    )


def expand_prompts_for_obs_steps(prompts, n_obs_steps: int):
    return [prompt for prompt in prompts for _ in range(n_obs_steps)]


def concat_text_to_global_condition(
    global_cond: torch.Tensor,
    text_feature: Optional[torch.Tensor],
) -> torch.Tensor:
    if text_feature is None:
        return global_cond
    text_feature = text_feature.to(
        device=global_cond.device,
        dtype=global_cond.dtype,
    )
    if global_cond.dim() == 3:
        text_feature = text_feature.unsqueeze(1).expand(
            -1,
            global_cond.shape[1],
            -1,
        )
    return torch.cat([global_cond, text_feature], dim=-1)
