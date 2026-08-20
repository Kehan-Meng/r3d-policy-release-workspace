#!/usr/bin/env python3
"""Fail-closed preflight for one real-robot frame profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
R3D_ROOT = PROJECT_ROOT / "R3D"
if str(R3D_ROOT) not in sys.path:
    sys.path.insert(0, str(R3D_ROOT))

from r3d.model.geometry.benchmark.profile import load_profile_config  # noqa: E402
from r3d.model.geometry.real_robot import preflight_real_robot_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate real-robot identity, camera calibration, controller semantics, "
            "timing, safety limits and SE(3) round-trip before deployment."
        )
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--runtime-context",
        type=Path,
        help="YAML/JSON synchronized frame_context sample required by eye-in-hand profiles",
    )
    parser.add_argument("--roundtrip-samples", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.roundtrip_samples <= 0:
        parser.error("--roundtrip-samples must be positive")

    config = load_profile_config(args.profile)
    runtime_context = None
    if args.runtime_context:
        with args.runtime_context.open("r", encoding="utf-8") as stream:
            runtime_context = yaml.safe_load(stream)
        if not isinstance(runtime_context, dict):
            parser.error("--runtime-context must contain a mapping")

    report = preflight_real_robot_profile(
        config,
        runtime_context=runtime_context,
        roundtrip_samples=args.roundtrip_samples,
    ).to_dict()
    report["profile_path"] = str(args.profile.resolve())
    if args.runtime_context:
        report["runtime_context_path"] = str(args.runtime_context.resolve())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
