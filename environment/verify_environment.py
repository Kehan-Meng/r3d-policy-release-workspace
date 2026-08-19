"""Check release dependencies without initializing CUDA or simulators."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import pathlib
import sys


CORE_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "open_clip": "open-clip-torch",
    "timm": "timm",
    "einops": "einops",
    "ftfy": "ftfy",
    "zarr": "zarr",
    "numcodecs": "numcodecs",
    "safetensors": "safetensors",
    "pointnet2_ops": "pointnet2-ops",
    "pc_sam": "pointsam-r3d",
    "r3d": "r3d-policy-core",
}

BENCHMARK_MODULES = {
    "adroit": ("gym", "mujoco_py"),
    "maniskill2": ("mani_skill2", "sapien"),
    "metaworld": ("metaworld",),
    "robotwin2": ("sapien", "gymnasium", "mplib", "toppra"),
}


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_modules(modules):
    missing = []
    for module_name, distribution in modules.items():
        available = importlib.util.find_spec(module_name) is not None
        status = "OK" if available else "MISSING"
        version = package_version(distribution) if available else "-"
        print(f"{status:7s} {module_name:20s} {version}")
        if not available:
            missing.append(module_name)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["adroit", "maniskill2", "metaworld", "robotwin2", "all"],
        default=None,
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {pathlib.Path(sys.executable).resolve()}")
    print("\nCore dependencies")
    missing = check_modules(CORE_MODULES)

    if args.benchmark:
        names = BENCHMARK_MODULES if args.benchmark == "all" else {
            args.benchmark: BENCHMARK_MODULES[args.benchmark]
        }
        for benchmark, module_names in names.items():
            print(f"\n{benchmark} dependencies")
            missing.extend(
                check_modules({module: module for module in module_names})
            )

    if missing:
        print("\nMissing: " + ", ".join(sorted(set(missing))))
        raise SystemExit(1)
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
