"""Download released encoder weights and the official CLIP text model."""

import hashlib
import os
import pathlib
import sys


MODEL_ID = "Lewandovski/twowayca-affordance"
MODEL_FILE = "model.safetensors"
EXPECTED_SHA256 = (
    "76e1daaca15d617288186e48af314250212bb906ae5e4bcea18330323c7d8951"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent
TARGET = REPO_ROOT / "pretrained" / "twowayca-affordance" / MODEL_FILE
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_TARGET = REPO_ROOT / "pretrained" / "clip-vit-base-patch32"
CLIP_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path):
    actual = sha256(path)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
        )


def clip_snapshot_ready(path):
    metadata_ready = all((path / name).is_file() for name in CLIP_REQUIRED_FILES)
    weights_ready = any(
        (path / name).is_file()
        for name in ("pytorch_model.bin", "model.safetensors")
    )
    return metadata_ready and weights_ready


def main():
    if TARGET.is_file():
        verify(TARGET)
        print(f"[weights] ready: {TARGET}")
    else:
        try:
            from modelscope import model_file_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope is required to download the encoder checkpoint. "
                "Install it with `pip install modelscope==1.39.1`."
            ) from exc

        TARGET.parent.mkdir(parents=True, exist_ok=True)
        downloaded = pathlib.Path(
            model_file_download(
                MODEL_ID,
                MODEL_FILE,
                local_dir=str(TARGET.parent),
                token=os.environ.get("MODELSCOPE_API_TOKEN"),
            )
        )
        if downloaded.resolve() != TARGET.resolve():
            raise RuntimeError(
                f"ModelScope downloaded to unexpected path {downloaded}; expected {TARGET}"
            )
        verify(TARGET)
        print(f"[weights] downloaded and verified: {TARGET}")

    if clip_snapshot_ready(CLIP_TARGET):
        print(f"[weights] ready: {CLIP_TARGET}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the policy CLIP model."
        ) from exc
    CLIP_TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=CLIP_MODEL_ID,
        local_dir=str(CLIP_TARGET),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
            "model.safetensors",
        ],
    )
    print(f"[weights] downloaded official CLIP model: {CLIP_TARGET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[weights] error: {exc}", file=sys.stderr)
        raise
