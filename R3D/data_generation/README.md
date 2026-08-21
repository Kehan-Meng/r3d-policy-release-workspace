# Dataset Preparation

Generated Zarr datasets are local artifacts and are intentionally excluded from
Git. Install the requested benchmark first with
`environment/install_benchmarks.sh`.

## Adroit

Adroit demonstrations are collected by a pretrained VRL3 expert:

```bash
python R3D/data_generation/adroit/generate_expert_zarr.py \
  --env_name door --num_episodes 10 \
  --expert_ckpt_path /path/to/vrl3_door.pt \
  --root_dir R3D/data --not_use_multi_view --use_point_crop
```

The generator is public, but VRL3 expert checkpoints are third-party model
artifacts and are not redistributed by this repository. Set `R3D_VRL3_ROOT`
when the benchmark checkout is outside the default `third_party` directory.

## MetaWorld

MetaWorld uses the benchmark's scripted expert policies and writes R3D Zarr
directly:

```bash
python R3D/data_generation/metaworld/generate_expert_zarr.py \
  --env_name lever-pull --num_episodes 10 \
  --num_points 1024 --root_dir R3D/data
```

## RoboTwin2

The pipeline can collect official episodes, convert legacy HDF5 episodes to
R3D Zarr, validate the result, and generate task/text configs:

```bash
python R3D/data_generation/robotwin2/run_pipeline.py --help
python R3D/data_generation/robotwin2/run_pipeline.py \
  --tasks-json R3D/data_generation/robotwin2/example_tasks.json \
  --collect --convert-missing --collect-gpu 0
```

Set `ROBOTWIN2_ROOT` if the pinned checkout is not under
`third_party/robotwin2`. Current official RoboTwin releases use a newer data
layout; this integration deliberately pins the July 2025 runtime that matches
the published R3D task/action contract.

## ManiSkill2

The implemented camera-EE stage converts an existing native R3D Zarr into the
camera-frame Cartesian control contract:

```bash
python R3D/data_generation/build_maniskill2_camera_ee_zarr.py \
  --task PickCube --source R3D/data/PickCube-v0.zarr
```

The source native Zarr must first be produced from an official ManiSkill2
trajectory. This repository currently does not contain a second, independently
validated official-HDF5-to-native-R3D-Zarr converter; the camera-EE converter
must not be described as that missing stage.
