# Release Workspace

This directory is the staging area for the public release. The outer directory
name is intentionally not part of any import or runtime path contract.

Code will be migrated incrementally from the live research repositories. The
live training repositories, datasets, checkpoints, logs, and experiment runs
must remain untouched while the release tree is assembled and verified.

Planned top-level components:

- `R3D/`: policy, datasets, benchmark environments, runners, and data generation
- `PointSAM/`: the visual encoder package and its public training entry point
- `experiments/`: curated reproducible configurations and result summaries
- `environment/`: unified installation and environment verification files
- `tests/`: focused release tests

## Current Status

The reproducibility slice now includes:

- public `train.py` and `eval.py` entry points;
- the R3D policy core used by the current V5 model;
- the PointSAM two-way cross-attention encoder package;
- package metadata for editable installs;
- path-clean reference training configurations for Adroit, MetaWorld,
  ManiSkill2 Camera-EE, and RoboTwin2;
- observation-centric frame profiles for Adroit, MetaWorld, and ManiSkill2;
- source-built CUDA FPS/PointNet2 operators and a fixed installation order;
- Adroit, MetaWorld, RoboTwin2, and ManiSkill2 Camera-EE data tooling;
- a pinned external RoboTwin2 runtime integration and benchmark installer.

Large benchmark assets remain external by design. Adroit data collection also
requires a third-party VRL3 expert checkpoint, and the ManiSkill2 Camera-EE
builder starts from an existing native R3D Zarr. See
`R3D/data_generation/README.md` for the exact boundary.

## Development Install

From this directory:

```bash
conda env create -f environment/environment.yml
conda activate r3d-release
bash environment/install.sh
bash environment/install_benchmarks.sh all
```

Benchmark runtimes are optional and checked explicitly, for example:

```bash
python environment/verify_environment.py --benchmark maniskill2
```

Large checkpoints are never committed. The released two-way cross-attention
encoder is hosted at
[`Lewandovski/twowayca-affordance`](https://modelscope.cn/models/Lewandovski/twowayca-affordance)
on ModelScope. Download and verify it with:

```bash
pip install modelscope==1.39.1
python download_pretrained.py
```

The script writes the checkpoint to
`pretrained/twowayca-affordance/model.safetensors`, which is the relative path
used by the public experiment configs. It also downloads the official
`openai/clip-vit-base-patch32` snapshot to
`pretrained/clip-vit-base-patch32`. The encoder checkpoint already contains its
own EVA02-E-14-plus OpenCLIP text tower; the separate CLIP-B/32 model supplies
the policy/ATA text features. The ModelScope repository is public and
does not require authentication. `MODELSCOPE_API_TOKEN` remains supported for
authenticated or mirrored deployments. The expected SHA256 is
`76e1daaca15d617288186e48af314250212bb906ae5e4bcea18330323c7d8951`.

## Entry Points

Inspect a generated training command without starting a process:

```bash
python train.py \
  --config experiments/configs/adroit/door_v5_masked_cosine.yaml
```

Start training by adding `--execute`. Evaluation uses the same config:

```bash
python eval.py \
  --config experiments/configs/adroit/door_v5_masked_cosine.yaml \
  --checkpoint outputs/adroit_door_v5_masked_cosine/checkpoints/EPOCH.ckpt \
  --eval-episodes 100 \
  --eval-seed 20260721
```

All relative paths are resolved from this repository root, so renaming the
outer directory does not change imports or runtime paths.

## Observation-Centric Frame Adapter

The frame adapter is configured by a versioned profile under
`R3D/r3d/config/frame_transform/`. It transforms decoded observations and
training actions into the policy frame before normalization, stores the
resolved profile hash in checkpoints, and converts predicted actions back to
the benchmark-native contract during rollout.

The public profiles cover Adroit Door/Hammer/Pen, MetaWorld's `corner2`
camera contract, and ManiSkill2 PickCube/StackCube/PegInsertionSide. The two
fully supported observation-centric control paths are:

- **MetaWorld:** world-frame Cartesian observations/actions are rigidly
  re-expressed in the camera frame and actions are rotated back before
  `env.step()`.
- **ManiSkill2 Camera-EE:** native joint demonstrations are converted offline
  to camera-frame TCP state and 7D target-delta actions. Evaluation converts
  the predicted target pose back through the official
  `pd_ee_target_delta_pose` controller. This is geometrically reversible, but
  it intentionally does not preserve joint-space null-space information.

Build a validated ManiSkill2 Camera-EE dataset with:

```bash
python R3D/data_generation/build_maniskill2_camera_ee_zarr.py \
  --task PickCube \
  --source R3D/data/PickCube-v0.zarr
```

The matching Hydra task configs are `maniskill_oc_PickCube`,
`maniskill_oc_StackCube`, and `maniskill_oc_PegInsertionSide`. Converted data
stores the profile SHA256 and controller contract; training fails closed when
either differs from the selected profile.

## Real-Robot Preflight

Vendor-neutral real-robot frame contracts live under
`R3D/r3d/config/frame_transform/real_robot/`. Templates are provided for a
fixed external camera, an eye-in-hand camera, and a dual-arm fixed-camera
setup. They are intentionally incomplete and cannot pass preflight until the
robot identity, camera calibration, timing, controller semantics, and safety
limits are filled with measured values.

```bash
python preflight_real_robot.py \
  --profile R3D/r3d/config/frame_transform/real_robot/my_robot_v1.yaml \
  --output experiments/results/real_robot/my_robot_preflight.json
```

The frozen policy contract is camera-frame absolute EE pose plus gripper for
state, and camera-frame spatial/left delta pose plus gripper for action. A
joint, body-frame, or velocity controller requires a separate thin controller
codec; changing YAML labels is not sufficient.

Use the OC Door example with:

```bash
python train.py \
  --config experiments/configs/adroit/door_v5_masked_cosine_oc.yaml
```
