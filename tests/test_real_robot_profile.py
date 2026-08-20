from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import yaml

from r3d.model.geometry.real_robot import (
    RealRobotRuntimeContextBuilder,
    preflight_real_robot_profile,
)


PROFILE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "R3D/r3d/config/frame_transform/real_robot"
)


def _load(name):
    return yaml.safe_load((PROFILE_ROOT / name).read_text(encoding="utf-8"))


def _fill_common(config, *, dual=False):
    config["name"] = "lab_robot_test_v1"
    config["task"] = "pick_test"
    config["status"] = "ready"
    contract = config["real_robot_contract"]
    contract["readiness"] = "ready"
    contract["hardware"]["robot_model"] = "test_robot"
    contract["hardware"]["robot_serial"] = "TEST-001"
    contract["hardware"]["robot_description_hash"] = "sha256:test-urdf"
    contract["camera"].update(
        serial="CAM-001",
        optical_axis_convention="opencv_x_right_y_down_z_forward",
        resolution=[640, 480],
        intrinsics=[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        distortion_model="none",
        distortion_coefficients=[],
        depth_scale_m_per_unit=0.001,
        depth_registered_to_color=True,
    )
    contract["calibration"].update(
        method="synthetic_unit_test",
        artifact_sha256="sha256:test-calibration",
        stored_transform_convention="T_target_from_source",
        measured_at_utc="2026-08-20T00:00:00Z",
        sample_count=30,
        translation_rmse_m=0.001,
        rotation_rmse_deg=0.2,
        reprojection_rmse_px=0.4,
        independent_point_rmse_m=0.001,
        axis_alignment_min_cosine=0.999,
    )
    contract["timing"].update(
        observation_frequency_hz=30.0,
        control_frequency_hz=10.0,
        max_camera_robot_skew_s=0.01,
        camera_timestamp_source="ptp_camera_clock",
        robot_timestamp_source="ptp_robot_clock",
    )
    contract["controller"]["gripper_semantics"] = "normalized_open_close"
    contract["controller"]["controller_config_hash"] = "sha256:test-controller"
    contract["safety"].update(
        workspace_bounds_m=[[0.1, 1.0], [-1.0, 1.0], [0.05, 1.0]],
        max_translation_step_m=0.02,
        max_rotation_step_rad=0.1,
        watchdog_timeout_s=0.2,
    )
    if dual:
        contract["hardware"]["robot_serial"] = "LEFT-001+RIGHT-001"
    return config


class TestRealRobotProfile(unittest.TestCase):
    def test_unfilled_template_fails_closed(self):
        config = _load("fixed_camera_cartesian_template_v1.yaml")
        report = preflight_real_robot_profile(config)
        self.assertEqual(report.status, "failed")
        self.assertIsNone(report.profile_hash)
        self.assertTrue(any("readiness" in error for error in report.errors))
        self.assertTrue(any("intrinsics" in error for error in report.errors))

    def test_fixed_camera_profile_passes_and_is_reversible(self):
        config = _fill_common(_load("fixed_camera_cartesian_template_v1.yaml"))
        config["transforms"][0]["provider"]["matrix"] = np.eye(4).tolist()
        report = preflight_real_robot_profile(config)
        self.assertEqual(report.status, "passed", report.errors)
        self.assertLess(report.roundtrip["observation_max_abs"], 1e-12)
        self.assertLess(report.roundtrip["action_max_abs"], 1e-12)
        self.assertIsNotNone(report.profile_hash)
        self.assertIsNotNone(report.calibration_hash)

    def test_eye_in_hand_requires_synchronized_context(self):
        config = _fill_common(_load("eye_in_hand_cartesian_template_v1.yaml"))
        config["real_robot_contract"]["timing"]["max_camera_robot_skew_s"] = 0.02
        config["transforms"][1]["provider"]["matrix"] = np.eye(4).tolist()
        missing = preflight_real_robot_profile(config)
        self.assertEqual(missing.status, "failed")
        self.assertTrue(any("runtime_context" in error for error in missing.errors))

        context = {
            "fk": {"T_robot_base_from_tool0": np.eye(4).tolist()},
            "timestamps": {"camera_s": 1.0, "robot_s": 1.005},
        }
        passed = preflight_real_robot_profile(config, runtime_context=context)
        self.assertEqual(passed.status, "passed", passed.errors)

        stale = {
            "fk": {"T_robot_base_from_tool0": np.eye(4).tolist()},
            "timestamps": {"camera_s": 1.0, "robot_s": 1.1},
        }
        rejected = preflight_real_robot_profile(config, runtime_context=stale)
        self.assertEqual(rejected.status, "failed")
        self.assertTrue(any("timestamp" in error.lower() for error in rejected.errors))

    def test_runtime_context_builder_converts_matrix_and_rejects_missing_key(self):
        config = _load("eye_in_hand_cartesian_template_v1.yaml")
        builder = RealRobotRuntimeContextBuilder.from_profile_config(config)
        observation = {
            "frame_context": {
                "fk": {"T_robot_base_from_tool0": np.eye(4).tolist()},
                "timestamps": {"camera_s": 2.0, "robot_s": 2.0},
            }
        }
        context = builder(observation)
        self.assertIsInstance(context["fk"]["T_robot_base_from_tool0"], np.ndarray)
        with self.assertRaises(KeyError):
            builder({"frame_context": {}})

    def test_dual_arm_profile_keeps_two_transform_chains_separate(self):
        config = _fill_common(
            _load("dual_arm_fixed_camera_cartesian_template_v1.yaml"), dual=True
        )
        left = np.eye(4)
        right = np.eye(4)
        right[0, 3] = 1.0
        config["transforms"][0]["provider"]["matrix"] = left.tolist()
        config["transforms"][1]["provider"]["matrix"] = right.tolist()
        report = preflight_real_robot_profile(config)
        self.assertEqual(report.status, "passed", report.errors)
        self.assertEqual(report.information_inventory["arm_count"], 2)
        self.assertLess(report.roundtrip["action_max_abs"], 1e-12)


if __name__ == "__main__":
    unittest.main()
