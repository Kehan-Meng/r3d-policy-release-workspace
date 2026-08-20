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

The first migration slice is in place:

- public `train.py` and `eval.py` entry points;
- the R3D policy core used by the current V5 model;
- the PointSAM two-way cross-attention encoder package;
- package metadata for editable installs;
- a path-clean Adroit Door reference configuration;
- observation-centric frame profiles for Adroit, MetaWorld, and ManiSkill2;
- a first unified environment specification and dependency checker.

The workspace is not a complete public release yet. RoboTwin assets/runtime,
benchmark data generation, the cleaned PointSAM training entry point, curated
MetaWorld/RoboTwin configs, checkpoint download tooling, and final dependency
installation are still being migrated.

## Development Install

From this directory:

```bash
pip install -e ./PointSAM --no-deps
pip install -e ./R3D --no-deps
python environment/verify_environment.py
```

Benchmark runtimes are optional and checked explicitly, for example:

```bash
python environment/verify_environment.py --benchmark maniskill2
```

Large checkpoints are never committed. Place downloaded weights under
`pretrained/` or set the checkpoint paths in an experiment config.

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

Use the OC Door example with:

```bash
python train.py \
  --config experiments/configs/adroit/door_v5_masked_cosine_oc.yaml
```
