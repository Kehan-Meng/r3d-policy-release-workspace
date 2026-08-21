import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "checkpoints",
    "outputs",
    "pretrained",
    "results",
    "runs",
}
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ARTIFACT_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".log",
    ".mp4",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".zarr",
}
PRIVATE_PATTERNS = {
    "private filesystem path": re.compile(
        r"/(?:DATA|home|root|yuan|jigu-haosu-vol)/"
    ),
    "local machine identity": re.compile(
        r"(?:zykh|fiveages|Kehan-Meng|(?<![A-Za-z])wzy(?![A-Za-z]))",
        re.IGNORECASE,
    ),
    "credential placeholder": re.compile(
        r"(?:YOUR_HF_TOKEN|github_pat_|ghp_[A-Za-z0-9]+|sk-[A-Za-z0-9]{16,})"
    ),
}


def release_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if path == pathlib.Path(__file__).resolve():
            continue
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        yield path, relative


class ReleaseHygieneTest(unittest.TestCase):
    def test_no_private_paths_or_credentials(self):
        violations = []
        for path, relative in release_files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PRIVATE_PATTERNS.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{relative}:{line}: {label}")
        self.assertEqual([], violations, "\n".join(violations))

    def test_no_generated_artifacts(self):
        violations = []
        for path, relative in release_files():
            if path.suffix.lower() in ARTIFACT_SUFFIXES:
                violations.append(str(relative))
        self.assertEqual([], violations, "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
