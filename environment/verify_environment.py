"""Check release dependencies without initializing CUDA or simulators."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
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
    "pytorch3d": "pytorch3d",
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


def check_local_package(module_name: str, expected_root: pathlib.Path):
    module = importlib.import_module(module_name)
    module_file = pathlib.Path(module.__file__).resolve()
    try:
        module_file.relative_to(expected_root.resolve())
    except ValueError:
        print(
            f"WRONG   {module_name:20s} {module_file} "
            f"(expected under {expected_root.resolve()})"
        )
        return [f"{module_name} resolves outside this checkout"]
    print(f"OK      {module_name + ' source':20s} {module_file}")
    return []


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

    repository_root = pathlib.Path(__file__).resolve().parents[1]
    if "r3d" not in missing:
        missing.extend(check_local_package("r3d", repository_root / "R3D"))
    if "pc_sam" not in missing:
        missing.extend(check_local_package("pc_sam", repository_root / "PointSAM"))

    clip_dir = repository_root / "pretrained" / "clip-vit-base-patch32"
    clip_ready = (clip_dir / "config.json").is_file()
    print(f"{'OK' if clip_ready else 'MISSING':7s} {'policy CLIP snapshot':20s} {clip_dir}")
    if not clip_ready:
        missing.append("policy CLIP snapshot (run download_pretrained.py)")

    if args.benchmark:
        names = BENCHMARK_MODULES if args.benchmark == "all" else {
            args.benchmark: BENCHMARK_MODULES[args.benchmark]
        }
        for benchmark, module_names in names.items():
            print(f"\n{benchmark} dependencies")
            missing.extend(
                check_modules({module: module for module in module_names})
            )
            if benchmark == "robotwin2":
                root = pathlib.Path(os.environ.get(
                    "ROBOTWIN2_ROOT",
                    pathlib.Path(__file__).resolve().parents[1] / "third_party" / "robotwin2",
                ))
                runtime_ready = all((root / item).exists() for item in (
                    "envs", "assets", "description", "task_config"
                ))
                print(
                    f"{'OK' if runtime_ready else 'MISSING':7s} "
                    f"{'RoboTwin2 runtime':20s} {root}"
                )
                if not runtime_ready:
                    missing.append("RoboTwin2 runtime/assets")

    if missing:
        print("\nMissing: " + ", ".join(sorted(set(missing))))
        raise SystemExit(1)
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
