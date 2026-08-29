import inspect
import pathlib
import tempfile
import unittest
from unittest import mock

from r3d.training.checkpointing import CheckpointMixin
from r3d.training.evaluation import WorkspaceEvaluationMixin


class _CheckpointWorkspace(CheckpointMixin):
    def __init__(self, output_dir):
        self._output_dir = str(output_dir)


class TrainingHelpersTest(unittest.TestCase):
    def test_get_checkpoint_path_resolves_regular_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            workspace = _CheckpointWorkspace(root)

            self.assertEqual(
                workspace.get_checkpoint_path("600"),
                root / "checkpoints" / "600.ckpt",
            )

    def test_get_checkpoint_path_selects_highest_scored_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = pathlib.Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "latest.ckpt").touch()
            (checkpoint_dir / "epoch=400-test_mean_score=0.35.ckpt").touch()
            best = checkpoint_dir / "epoch=600-test_mean_score=0.82.ckpt"
            best.touch()
            (checkpoint_dir / "unrelated.ckpt").touch()

            workspace = _CheckpointWorkspace(directory)

            self.assertEqual(workspace.get_checkpoint_path("best"), best)

    def test_get_checkpoint_path_best_requires_scored_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            (pathlib.Path(directory) / "checkpoints").mkdir()
            workspace = _CheckpointWorkspace(directory)

            with self.assertRaisesRegex(
                FileNotFoundError, "No scored checkpoints"
            ):
                workspace.get_checkpoint_path("best")

    def test_get_policy_does_not_accept_unused_cfg_argument(self):
        parameters = inspect.signature(
            WorkspaceEvaluationMixin.get_policy
        ).parameters

        self.assertEqual(list(parameters), ["self", "checkpoint_num"])

    def test_eval_returns_before_initialization_on_non_main_rank(self):
        workspace = object.__new__(WorkspaceEvaluationMixin)

        with mock.patch(
            "r3d.training.evaluation.is_main_process", return_value=False
        ):
            self.assertIsNone(workspace.eval())


if __name__ == "__main__":
    unittest.main()
