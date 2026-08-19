"""Shared path and environment helpers for the public entry points."""

from __future__ import annotations

import os
import pathlib

from omegaconf import OmegaConf


REPO_ROOT = pathlib.Path(__file__).resolve().parent


def load_experiment_config(path: str | os.PathLike, stack=()):
    """Load an experiment config and resolve local ``extends`` recursively."""
    path = pathlib.Path(path).expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Config extends cycle: {chain}")
    cfg = OmegaConf.load(path)
    if cfg.get("extends") is None:
        return cfg
    base_path = pathlib.Path(str(cfg.extends)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    derived = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    del derived["extends"]
    return OmegaConf.merge(
        load_experiment_config(base_path, (*stack, path)),
        derived,
    )


def resolve_repo_path(value, *, default=None) -> pathlib.Path:
    """Resolve a user path without depending on the repository directory name."""
    if value is None:
        if default is None:
            raise ValueError("A path value or default is required")
        value = default
    path = pathlib.Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def prepend_env_path(env: dict[str, str], name: str, path) -> None:
    path = str(path)
    current = env.get(name, "")
    parts = [item for item in current.split(os.pathsep) if item]
    if path not in parts:
        env[name] = os.pathsep.join([path, *parts])


def configure_native_libraries(env: dict[str, str]) -> None:
    """Honor optional native-library locations without user-specific paths."""
    candidates = []
    explicit_mujoco = env.get("R3D_MUJOCO_LIB_DIR")
    if explicit_mujoco:
        candidates.append(pathlib.Path(explicit_mujoco).expanduser())
    mujoco_home = env.get("MUJOCO_HOME")
    if mujoco_home:
        candidates.append(pathlib.Path(mujoco_home).expanduser() / "bin")
    explicit_nvidia = env.get("R3D_NVIDIA_LIB_DIR")
    if explicit_nvidia:
        candidates.append(pathlib.Path(explicit_nvidia).expanduser())

    for candidate in candidates:
        if candidate.is_dir():
            prepend_env_path(env, "LD_LIBRARY_PATH", candidate.resolve())
