from __future__ import annotations

from pathlib import Path
import hashlib
import unittest

import numpy as np
import yaml

from r3d.model.geometry.benchmark import load_profile_bundle


PROFILE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "R3D"
    / "r3d"
    / "config"
    / "frame_transform"
)


class TestObservationCentricFrameProfiles(unittest.TestCase):
    def test_public_profiles_load(self):
        expected = {
            "adroit_door_legacy_camera_v1.yaml",
            "adroit_hammer_legacy_camera_v1.yaml",
            "adroit_pen_legacy_camera_v1.yaml",
            "metaworld_corner2_true_camera_v1.yaml",
            "maniskill2_pickcube_base_camera_v1.yaml",
            "maniskill2_stackcube_base_camera_v1.yaml",
            "maniskill2_peginsertionside_base_camera_v1.yaml",
        }
        self.assertEqual({path.name for path in PROFILE_ROOT.glob("*.yaml")}, expected)
        for name in expected:
            with self.subTest(profile=name):
                load_profile_bundle(PROFILE_ROOT / name)

    def test_maniskill_pickcube_roundtrip_and_zero_padding(self):
        bundle = load_profile_bundle(
            PROFILE_ROOT / "maniskill2_pickcube_base_camera_v1.yaml"
        )
        qpos = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
        goal = np.array([0.05, -0.1, 0.25], dtype=np.float64)
        action = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        point_cloud = np.array(
            [
                [0.1, 0.2, 0.3, 10.0, 20.0, 30.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        native = {
            "point_cloud": point_cloud,
            "state": np.concatenate([qpos, goal]),
            "action": action,
        }

        decoded = bundle.decoder.decode_training_sample(native)
        policy = bundle.adapter.training_sample_to_policy_with_metadata(
            decoded, bundle.adapter.native_metadata()
        )
        np.testing.assert_array_equal(policy.data["point_cloud"][1], 0.0)
        np.testing.assert_array_equal(policy.data["point_cloud"][..., 3:], point_cloud[..., 3:])
        np.testing.assert_array_equal(policy.data["agent_pos"][..., :9], qpos)
        np.testing.assert_array_equal(policy.data["action"], action)

        recovered = bundle.adapter.training_sample_to_native_with_metadata(
            policy.data, policy.metadata
        )
        encoded = bundle.decoder.encode_training_sample(recovered.data)
        for key in native:
            np.testing.assert_allclose(encoded[key], native[key], atol=2e-15)

    def test_metaworld_action_roundtrip(self):
        bundle = load_profile_bundle(
            PROFILE_ROOT / "metaworld_corner2_true_camera_v1.yaml"
        )
        action = np.array([0.2, -0.3, 0.4, 1.0], dtype=np.float64)
        policy = bundle.adapter.action_to_policy_with_metadata(
            action, bundle.adapter.native_metadata()
        )
        recovered = bundle.adapter.action_to_environment_with_metadata(
            policy.data, policy.metadata
        )
        np.testing.assert_allclose(recovered.data, action, atol=2e-15)

    def test_maniskill_camera_ee_task_contracts_match_profiles(self):
        expected = {
            "PickCube": (14, "pickcube"),
            "StackCube": (11, "stackcube"),
            "PegInsertionSide": (11, "peginsertionside"),
        }
        task_root = PROFILE_ROOT.parent / "task"
        for task, (state_dim, profile_stem) in expected.items():
            with self.subTest(task=task):
                config = yaml.safe_load(
                    (task_root / f"maniskill_oc_{task}.yaml").read_text()
                )
                dataset = config["dataset"]
                profile = Path(__file__).resolve().parents[1] / dataset["camera_profile_path"]
                digest = hashlib.sha256(profile.read_bytes()).hexdigest()
                self.assertTrue(dataset["camera_ee_contract"])
                self.assertEqual(dataset["camera_profile_sha256"], digest)
                self.assertEqual(config["shape_meta"]["obs"]["agent_pos"]["shape"], [state_dim])
                self.assertEqual(config["shape_meta"]["action"]["shape"], [7])
                self.assertIn(profile_stem, profile.name)

    def test_robotwin_has_no_coordinate_conversion_contract(self):
        root = Path(__file__).resolve().parents[1] / "R3D" / "r3d"
        self.assertFalse(any((root / "model" / "cartesian_policy").glob("*.py")))
        self.assertFalse((root / "dataset" / "robotwin2_ee_camera_dataset.py").exists())


if __name__ == "__main__":
    unittest.main()
