# Real-Robot Frame Profiles

These templates extend the canonical-frame adapter to real hardware without
adding vendor SDK code to the geometry core.

## Templates

- `fixed_camera_cartesian_template_v1.yaml`: one arm, fixed external RGB-D camera.
- `eye_in_hand_cartesian_template_v1.yaml`: one arm, eye-in-hand RGB-D camera.
- `dual_arm_fixed_camera_cartesian_template_v1.yaml`: two arms and one fixed camera.

All templates are intentionally invalid (`status: template_incomplete` and
`readiness: incomplete`). Never replace missing calibration with identity.

## Required conventions

- Every matrix is `T_target_from_source`.
- Length is meters and time is seconds.
- Camera optical axes are OpenCV: `+x right, +y down, +z forward`.
- State Cartesian pose is `xyz + rotation_6d_columns`.
- Action Cartesian delta is spatial/left-composed `xyz + axis_angle`.
- Point cloud, Cartesian state and Cartesian action all target the camera optical frame.
- Gripper scalars pass through unchanged.
- Training targets should use executed pose deltas when controller clipping or saturation exists.

## Preflight

```bash
python preflight_real_robot.py \
  --profile R3D/r3d/config/frame_transform/real_robot/my_robot_v1.yaml \
  --output experiments/results/real_robot/my_robot_preflight.json
```

Eye-in-hand profiles also require one synchronized runtime-context sample:

```yaml
fk:
  T_robot_base_from_tool0: [[...], [...], [...], [0, 0, 0, 1]]
timestamps:
  camera_s: 123.456
  robot_s: 123.460
```

Pass it with `--runtime-context context.yaml`.

Exit code `0` means ready. Exit code `2` means fail closed; read `errors` before
collecting data or moving the robot.

At inference, construct the wrapper with:

```python
runtime_builder = RealRobotRuntimeContextBuilder.from_profile_config(profile_config)
policy = maybe_wrap_policy_for_environment(
    policy,
    frame_config,
    checkpoint_metadata=checkpoint_metadata,
    require_checkpoint_metadata=True,
    runtime_context_builder=runtime_builder if eye_in_hand else None,
)
```

The hardware integration layer is responsible only for producing the native
observation and, for eye-in-hand cameras, the synchronized `frame_context`.
