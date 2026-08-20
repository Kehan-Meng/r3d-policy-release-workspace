"""OpenCLIP token-level text encoder for text-guided heatmap prediction."""

from contextlib import nullcontext
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn

class OpenCLIPTokenTextEncoder(nn.Module):
    """Return token features [B, L, D] and EOT features [B, D]."""

    def __init__(
        self,
        model_name: str = "EVA02-E-14-plus",
        pretrained: str = "",
        context_length: int = 77,
        freeze_clip: bool = True,
        drop_special_tokens: bool = True,
        drop_visual: bool = True,
        open_clip_python_path: str = "",
        text_only: bool = True,
        precision: str = "fp32",
        bpe_path: str = "",
        apply_text_projection: bool = True,
        cache_frozen_features: bool = True,
        cache_max_entries: int = 20000,
        cache_dtype: str = "bfloat16",
    ):
        super().__init__()
        if open_clip_python_path:
            warnings.warn(
                "open_clip_python_path is ignored. Install open-clip-torch in "
                "the active environment.",
                DeprecationWarning,
                stacklevel=2,
            )

        from pc_sam.utils.tokenizer import SimpleTokenizer

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenCLIPTokenTextEncoder requires the open_clip package."
            ) from exc

        if text_only:
            self.clip_model = self._build_text_only_open_clip(
                open_clip=open_clip,
                model_name=model_name,
                pretrained=pretrained,
                precision=precision,
            )
        elif hasattr(open_clip, "create_model"):
            self.clip_model = open_clip.create_model(
                model_name=model_name,
                pretrained=pretrained,
                precision=precision,
            )
        else:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
            )
        if drop_visual and hasattr(self.clip_model, "visual"):
            self.clip_model.visual = nn.Identity()
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path) if bpe_path else SimpleTokenizer()
        self.context_length = int(
            getattr(self.clip_model, "context_length", context_length)
            or context_length
        )
        self.freeze_clip = freeze_clip
        self.drop_special_tokens = drop_special_tokens
        self.apply_text_projection = apply_text_projection
        self.cache_frozen_features = bool(cache_frozen_features and freeze_clip)
        self.cache_max_entries = int(cache_max_entries)
        if self.cache_max_entries < 0:
            raise ValueError(
                f"cache_max_entries must be non-negative, got {cache_max_entries}"
            )
        cache_dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        cache_dtype_key = str(cache_dtype).lower()
        if cache_dtype_key not in cache_dtypes:
            raise ValueError(
                "cache_dtype must be one of float16, bfloat16, or float32, "
                f"got {cache_dtype}"
            )
        self.cache_dtype = cache_dtypes[cache_dtype_key]
        # Plain Python state on purpose: cached features must not enter model
        # state_dict/checkpoints. Each DDP rank owns a bounded CPU-side LRU.
        self._feature_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]
        self.text_dim = self._infer_text_width()

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
            self.clip_model.eval()

    @staticmethod
    def _build_text_only_open_clip(open_clip, model_name, pretrained, precision):
        from open_clip.factory import get_model_config, load_state_dict
        from open_clip.model import _build_text_tower, get_cast_dtype

        model_cfg = get_model_config(model_name)
        if model_cfg is None:
            raise RuntimeError(f"OpenCLIP model config not found: {model_name}")

        text_tower = _build_text_tower(
            embed_dim=model_cfg["embed_dim"],
            text_cfg=model_cfg["text_cfg"],
            quick_gelu=model_cfg.get("quick_gelu", False),
            cast_dtype=get_cast_dtype(precision),
        )

        if pretrained:
            try:
                state_dict = load_state_dict(pretrained, device="cpu")
            except TypeError:
                state_dict = load_state_dict(pretrained, map_location="cpu")
            text_state = {}
            text_prefixes = (
                "text_projection",
                "positional_embedding",
                "token_embedding.",
                "transformer.",
                "ln_final.",
            )
            for key, value in state_dict.items():
                clean_key = key
                if clean_key.startswith("text."):
                    clean_key = clean_key[len("text.") :]
                if clean_key.startswith(text_prefixes):
                    text_state[clean_key] = value

            result = text_tower.load_state_dict(text_state, strict=False)
            bad_missing = [
                key for key in result.missing_keys
                if not key.endswith("attn_mask")
            ]
            if bad_missing or result.unexpected_keys:
                raise RuntimeError(
                    "Failed to load OpenCLIP text tower: "
                    f"missing={bad_missing}, unexpected={result.unexpected_keys}"
                )

        return text_tower

    def _text_tower(self):
        if hasattr(self.clip_model, "token_embedding"):
            return self.clip_model
        if hasattr(self.clip_model, "text"):
            return self.clip_model.text
        raise AttributeError("Unsupported OpenCLIP model: no text tower found.")

    def _infer_text_width(self):
        tower = self._text_tower()
        text_projection = getattr(tower, "text_projection", None)
        if self.apply_text_projection and text_projection is not None:
            return int(text_projection.shape[-1])
        if hasattr(tower, "ln_final"):
            return int(tower.ln_final.weight.shape[0])
        if hasattr(tower, "width"):
            return int(tower.width)
        raise AttributeError("Cannot infer OpenCLIP text feature width.")

    @staticmethod
    def _normalize_texts(texts):
        if isinstance(texts, str):
            texts = [[texts]]
        elif isinstance(texts, tuple):
            texts = list(texts)
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], str):
            texts = [[t] for t in texts]

        flat = []
        for i, item in enumerate(texts):
            if not isinstance(item, (list, tuple)) or len(item) != 1:
                raise ValueError(
                    "OpenCLIPTokenTextEncoder expects exactly one text per "
                    f"sample, got sample {i}: {item}"
                )
            flat.append(str(item[0]))
        return flat

    def _encode_token_features(self, tokens: torch.Tensor):
        tower = self._text_tower()
        transformer = tower.transformer
        cast_dtype = transformer.get_cast_dtype()
        seq_len = tokens.shape[1]

        x = tower.token_embedding(tokens).to(cast_dtype)
        attn_mask = tower.attn_mask

        cls_emb = getattr(tower, "cls_emb", None)
        if cls_emb is not None:
            seq_len += 1
            cls_tokens = cls_emb.reshape(1, 1, -1).repeat(x.shape[0], 1, 1)
            x = torch.cat([x, cls_tokens.to(cast_dtype)], dim=1)
            if hasattr(tower, "build_cls_mask"):
                attn_mask = (
                    tower.attn_mask[None, :seq_len, :seq_len]
                    + tower.build_cls_mask(tokens, cast_dtype)[:, :seq_len, :seq_len]
                )

        x = x + tower.positional_embedding[:seq_len].to(cast_dtype)
        # 新版 OpenCLIP 的 Transformer 自己处理 batch_first；旧版则要求 LBD。
        # 通过属性存在性兼容两类实现，避免把 batch 长度误当成序列长度。
        legacy_sequence_first = not hasattr(transformer, "batch_first")
        if legacy_sequence_first:
            x = x.permute(1, 0, 2)
        x = transformer(x, attn_mask=attn_mask)
        if legacy_sequence_first:
            x = x.permute(1, 0, 2)

        if cls_emb is not None:
            x = x[:, :-1]
        x = tower.ln_final(x)
        text_projection = getattr(tower, "text_projection", None)
        if self.apply_text_projection and text_projection is not None:
            x = x @ text_projection.to(device=x.device, dtype=x.dtype)
        return x

    def _active_token_mask(self, tokens: torch.Tensor):
        active = tokens.ne(0)
        if self.drop_special_tokens:
            content = (
                active
                & tokens.ne(self.sot_token)
                & tokens.ne(self.eot_token)
            )
            has_content = content.any(dim=1, keepdim=True)
            active = torch.where(has_content, content, active)
        return active

    def clear_cache(self):
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_info(self):
        total = self._cache_hits + self._cache_misses
        return {
            "enabled": self.cache_frozen_features and self.cache_max_entries > 0,
            "size": len(self._feature_cache),
            "max_entries": self.cache_max_entries,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total else 0.0,
        }

    def _cache_put(self, text, active_indices, active_features, eot_feature):
        if not self.cache_frozen_features or self.cache_max_entries <= 0:
            return
        self._feature_cache[text] = (
            active_indices.detach().to(device="cpu", dtype=torch.int16),
            active_features.detach().to(device="cpu", dtype=self.cache_dtype),
            eot_feature.detach().to(device="cpu", dtype=self.cache_dtype),
        )
        self._feature_cache.move_to_end(text)
        while len(self._feature_cache) > self.cache_max_entries:
            self._feature_cache.popitem(last=False)

    def _forward_uncached(self, flat_texts, device, dtype, return_eot):
        tokens = self.tokenizer(
            flat_texts,
            context_length=self.context_length,
        ).to(device=device, dtype=torch.long)

        context = torch.no_grad() if self.freeze_clip else nullcontext()
        with context:
            token_features = self._encode_token_features(tokens)

        if dtype is not None:
            token_features = token_features.to(dtype=dtype)

        eot_features = None
        if return_eot:
            eot_indices = tokens.argmax(dim=-1)
            eot_features = token_features[
                torch.arange(token_features.shape[0], device=token_features.device),
                eot_indices,
            ]

        active = self._active_token_mask(tokens).to(
            device=token_features.device,
            dtype=token_features.dtype,
        )
        token_features = token_features * active.unsqueeze(-1)
        if return_eot:
            return token_features, eot_features
        return token_features

    def _forward_cached(self, flat_texts, device, dtype, return_eot):
        output_dtype = dtype or next(self.clip_model.parameters()).dtype
        output = torch.zeros(
            len(flat_texts),
            self.context_length,
            self.text_dim,
            device=device,
            dtype=output_dtype,
        )
        eot_output = (
            torch.empty(
                len(flat_texts),
                self.text_dim,
                device=device,
                dtype=output_dtype,
            )
            if return_eot
            else None
        )

        # Encode each cache miss only once, even when a batch contains the same
        # prompt multiple times.
        cached_before = {}
        for text in dict.fromkeys(flat_texts):
            if text in self._feature_cache:
                cached_before[text] = self._feature_cache[text]
                self._feature_cache.move_to_end(text)

        missing_texts = []
        missing_set = set()
        for text in flat_texts:
            if text not in cached_before and text not in missing_set:
                missing_texts.append(text)
                missing_set.add(text)

        fresh = {}
        if missing_texts:
            tokens = self.tokenizer(
                missing_texts,
                context_length=self.context_length,
            ).to(device=device, dtype=torch.long)
            with torch.no_grad():
                token_features = self._encode_token_features(tokens)
            token_features = token_features.to(dtype=output_dtype)
            eot_indices = tokens.argmax(dim=-1)
            eot_features = token_features[
                torch.arange(token_features.shape[0], device=device),
                eot_indices,
            ]
            active_masks = self._active_token_mask(tokens)

            for i, text in enumerate(missing_texts):
                active_indices = torch.nonzero(
                    active_masks[i], as_tuple=False
                ).flatten()
                active_features = token_features[i, active_indices]
                fresh[text] = (active_indices, active_features, eot_features[i])
                self._cache_put(
                    text,
                    active_indices,
                    active_features,
                    eot_features[i],
                )

        cached_batch_indices = []
        cached_token_indices = []
        cached_features = []
        cached_eot_batch_indices = []
        cached_eot_features = []

        for batch_index, text in enumerate(flat_texts):
            if text in fresh:
                active_indices, active_features, eot_feature = fresh[text]
                output[batch_index, active_indices] = active_features
                if return_eot:
                    eot_output[batch_index] = eot_feature
                self._cache_misses += 1
                continue

            active_indices, active_features, eot_feature = cached_before[text]
            num_active = active_indices.numel()
            cached_batch_indices.append(
                torch.full((num_active,), batch_index, dtype=torch.long)
            )
            cached_token_indices.append(active_indices.to(dtype=torch.long))
            cached_features.append(active_features)
            if return_eot:
                cached_eot_batch_indices.append(batch_index)
                cached_eot_features.append(eot_feature)
            self._cache_hits += 1

        # Consolidate all cache hits into one host-to-device copy per tensor.
        if cached_features:
            batch_indices = torch.cat(cached_batch_indices).to(device=device)
            token_indices = torch.cat(cached_token_indices).to(device=device)
            features = torch.cat(cached_features).to(
                device=device,
                dtype=output_dtype,
            )
            output[batch_indices, token_indices] = features
            if return_eot:
                eot_indices = torch.tensor(
                    cached_eot_batch_indices,
                    device=device,
                    dtype=torch.long,
                )
                eot_features = torch.stack(cached_eot_features).to(
                    device=device,
                    dtype=output_dtype,
                )
                eot_output[eot_indices] = eot_features

        if return_eot:
            return output, eot_output
        return output

    def forward(self, texts, device=None, dtype=None, return_eot=False):
        flat_texts = self._normalize_texts(texts)
        if device is None:
            device = next(self.clip_model.parameters()).device
        device = torch.device(device)
        model_device = next(self.clip_model.parameters()).device
        if device.type == "cuda" and device.index is None:
            device = model_device
        if model_device != device:
            raise RuntimeError(
                f"OpenCLIP is on {model_device}, but input is on {device}. "
                "Move the complete model before calling forward."
            )
        if self.freeze_clip:
            self.clip_model.eval()

        if self.cache_frozen_features and self.cache_max_entries > 0:
            return self._forward_cached(flat_texts, device, dtype, return_eot)
        return self._forward_uncached(flat_texts, device, dtype, return_eot)
