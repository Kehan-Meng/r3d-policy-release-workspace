from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from termcolor import cprint

from r3d.dataset.text_dataset import TextInstructionDataset


TextInput = Union[str, Sequence[str]]


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder for command-conditioned policies."""

    def __init__(
            self,
            clip_model_name: str = "openai/clip-vit-base-patch32",
            text_json_path: Optional[str] = None,
            task_name: Optional[str] = None,
            text_feat_dim: int = 64,
            max_length: int = 77,
            freeze_clip: bool = True,
            strict_text_lookup: bool = False,
            fallback_to_command: bool = True,
            device: Optional[str] = None):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.task_name = task_name
        self.text_feat_dim = text_feat_dim
        self.max_length = max_length
        self.freeze_clip = freeze_clip
        self.text_dataset = TextInstructionDataset(
            text_json_path=text_json_path,
            strict=strict_text_lookup,
            fallback_to_command=fallback_to_command,
        )

        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required when policy.use_text=true. "
                "Install it with: pip install transformers"
            ) from exc

        cprint(f"[CLIPTextEncoder] loading CLIP model: {clip_model_name}", "cyan")
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_text_model.parameters():
                param.requires_grad = False
            cprint("[CLIPTextEncoder] CLIP text encoder weights frozen", "yellow")

        clip_output_dim = self.clip_text_model.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(clip_output_dim, text_feat_dim),
            nn.LayerNorm(text_feat_dim),
            nn.ReLU(),
        )
        cprint(f"[CLIPTextEncoder] text projection: {clip_output_dim} -> {text_feat_dim}", "cyan")

        if device is not None:
            self.to(device)

    @property
    def output_dim(self) -> int:
        return self.text_feat_dim

    def lookup_text(self, command: Any) -> str:
        return self.text_dataset.lookup(command)

    def set_command(self, command: Any):
        self.task_name = str(command)

    @staticmethod
    def _as_list(value: Optional[Union[Any, Sequence[Any]]]) -> Optional[List[Any]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _expand_to_batch(values: List[str], batch_size: int) -> List[str]:
        if len(values) == batch_size:
            return values
        if len(values) == 1:
            return values * batch_size
        raise ValueError(
            f"text batch size mismatch: got {len(values)} texts for batch_size={batch_size}"
        )

    def _resolve_texts(
            self,
            batch_size: int,
            texts: Optional[TextInput] = None,
            commands: Optional[Union[Any, Sequence[Any]]] = None) -> List[str]:
        text_list = self._as_list(texts)
        if text_list is not None:
            return self._expand_to_batch([str(text) for text in text_list], batch_size)

        command_list = self._as_list(commands)
        if command_list is None and self.task_name is not None:
            command_list = [self.task_name]

        if command_list is None:
            return ["" for _ in range(batch_size)]

        resolved = [self.text_dataset.lookup(command) for command in command_list]
        return self._expand_to_batch(resolved, batch_size)

    def forward(
            self,
            batch_size: int = 1,
            texts: Optional[TextInput] = None,
            commands: Optional[Union[Any, Sequence[Any]]] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        resolved_texts = self._resolve_texts(batch_size, texts=texts, commands=commands)

        if all(text == "" for text in resolved_texts):
            return torch.zeros(batch_size, self.text_feat_dim, device=device)

        tokenized = self.tokenizer(
            resolved_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        clip_outputs = self.clip_text_model(**tokenized)
        pooler_output = clip_outputs.pooler_output
        return self.text_projection(pooler_output)

    def forward_with_tokens(
        self,
        batch_size: int = 1,
        texts: Optional[TextInput] = None,
        commands: Optional[Union[Any, Sequence[Any]]] = None,
    ):
        """一次 CLIP forward，同时返回 pooled 投影特征和 token-level hidden states.

        Returns:
            pooled_proj:        [B, text_feat_dim]  等价于 forward() 的输出
            last_hidden_state:  [B, max_length, 512]   CLIPTextModel last_hidden_state
            attention_mask:     [B, max_length] or None  tokenizer attention_mask
            special_tokens_mask: [B, max_length] or None 1=special token
        """
        device = next(self.parameters()).device
        resolved_texts = self._resolve_texts(batch_size, texts=texts, commands=commands)

        if all(text == "" for text in resolved_texts):
            return (
                torch.zeros(batch_size, self.text_feat_dim, device=device),
                torch.zeros(
                    batch_size,
                    self.max_length,
                    self.clip_text_model.config.hidden_size,
                    device=device,
                ),
                None,
                None,
            )

        tokenized = self.tokenizer(
            resolved_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        attn_mask = tokenized.get("attention_mask", None)
        special_tokens_mask = tokenized.get("special_tokens_mask", None)

        clip_outputs = self.clip_text_model(
            input_ids=tokenized["input_ids"],
            attention_mask=attn_mask,
        )

        pooled_proj_new = self.text_projection(clip_outputs.pooler_output)
        last_hidden = clip_outputs.last_hidden_state

        return pooled_proj_new, last_hidden, attn_mask, special_tokens_mask
