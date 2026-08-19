import torch
import torch.nn.functional as F


def pool_text_tokens(text_tokens, text_pool="cls"):
    """Pool text tokens into one text feature per batch item."""
    if text_tokens.ndim != 3:
        raise ValueError(
            f"text_tokens must have shape [B, M_text, D], got {tuple(text_tokens.shape)}"
        )

    if text_pool == "cls":
        if text_tokens.shape[1] < 1:
            raise ValueError("text_tokens must contain at least one token for cls pooling")
        return text_tokens[:, 0, :]
    if text_pool == "mean":
        if text_tokens.shape[1] < 1:
            raise ValueError("text_tokens must contain at least one token for mean pooling")
        return text_tokens.mean(dim=1)

    raise ValueError(f"Unsupported text_pool: {text_pool}. Expected 'cls' or 'mean'.")


def normalize_heatmap_scores(scores, normalize="minmax", eps=1e-6):
    """Normalize [B, N] similarity scores to heatmap values in [0, 1]."""
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [B, N], got {tuple(scores.shape)}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if normalize == "minmax":
        score_min = scores.amin(dim=1, keepdim=True)
        score_max = scores.amax(dim=1, keepdim=True)
        return (scores - score_min) / (score_max - score_min).clamp_min(eps)
    if normalize == "sigmoid":
        return torch.sigmoid(scores)

    raise ValueError(
        f"Unsupported normalize: {normalize}. Expected 'minmax' or 'sigmoid'."
    )


def build_pseudo_heatmap_from_text(
    patch_tokens,
    text_tokens,
    text_pool="cls",
    normalize="minmax",
    eps=1e-6,
):
    """
    Build a patch-level pseudo affordance heatmap from patch and text tokens.

    Args:
        patch_tokens: Tensor with shape [B, N, D].
        text_tokens: Tensor with shape [B, M_text, D].
        text_pool: "cls" uses the first text token, "mean" averages all text tokens.
        normalize: "minmax" or "sigmoid".
        eps: Numerical epsilon for cosine and min-max normalization.

    Returns:
        Tensor with shape [B, N, 1] and values in [0, 1].
    """
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"patch_tokens must have shape [B, N, D], got {tuple(patch_tokens.shape)}"
        )
    if text_tokens.ndim != 3:
        raise ValueError(
            f"text_tokens must have shape [B, M_text, D], got {tuple(text_tokens.shape)}"
        )
    if patch_tokens.shape[0] != text_tokens.shape[0]:
        raise ValueError(
            "patch_tokens and text_tokens must share the same batch size, got "
            f"{patch_tokens.shape[0]} and {text_tokens.shape[0]}"
        )
    if patch_tokens.shape[-1] != text_tokens.shape[-1]:
        raise ValueError(
            "patch_tokens and text_tokens must share the same feature dimension, got "
            f"{patch_tokens.shape[-1]} and {text_tokens.shape[-1]}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    text_feature = pool_text_tokens(text_tokens, text_pool=text_pool)
    patch_feature = F.normalize(patch_tokens, dim=-1, eps=eps)
    text_feature = F.normalize(text_feature, dim=-1, eps=eps)

    scores = torch.sum(patch_feature * text_feature.unsqueeze(1), dim=-1)
    heatmap = normalize_heatmap_scores(scores, normalize=normalize, eps=eps)
    return heatmap.unsqueeze(-1)
