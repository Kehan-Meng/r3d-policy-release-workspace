"""InstructionBank: prompt augmentation for training-time text diversity.

Loads a JSON bank of task_name → list of semantically-equivalent prompts.

Usage (in dataset __getitem__):
    original = bank.get_original(task_name)
    aux_prompt = bank.sample_aux(task_name)
"""

import json
import os
import random
from typing import Dict, List, Optional, Set

from termcolor import cprint


# ---------------------------------------------------------------------------
# Key normalisation helpers — same logic as TextInstructionDataset
# ---------------------------------------------------------------------------

def _normalize_key(command: str) -> str:
    key = command.strip()
    key = key.replace(" ", "_")
    return key


def _candidate_keys(task_name: str) -> List[str]:
    """Generate alternative key forms so that 'lift_pot-demo_randomized-50'
    or 'move-playingcard-away' still match 'lift_pot' / 'move_playingcard_away'.
    """
    key = _normalize_key(task_name)
    base = [key]
    if key.endswith(".zarr"):
        base.append(os.path.basename(key[:-5]))

    suffixes = (
        "-demo_clean-50", "-demo_randomized-50",
        "_demo_clean_50", "_demo_randomized_50",
        "-demo_clean", "-demo_randomized",
        "_demo_clean", "_demo_randomized",
        "-50", "_50",
    )
    seen: Set[str] = set()
    for b in base:
        variants = [b, b.replace("_", "-"), b.replace("-", "_")]
        for v in variants:
            if v not in seen:
                seen.add(v)
                yield v
            for sfx in suffixes:
                if v.endswith(sfx):
                    stripped = v[:-len(sfx)]
                    for sv in (stripped, stripped.replace("_", "-"), stripped.replace("-", "_")):
                        if sv not in seen:
                            seen.add(sv)
                            yield sv


class InstructionBank:
    """Load and sample from a JSON prompt bank.

    Expected JSON format:
        {
          "task_name": {
            "sampling": "uniform",
            "prompts": ["prompt0", "prompt1", ...]
          }
        }

    The first prompt in each list (prompts[0]) is treated as the canonical
    original and is used for eval/inference.
    """

    def __init__(self, bank_path: Optional[str] = None):
        self._bank: Dict[str, dict] = {}
        self._warned_missing: set = set()
        self._bank_path = bank_path

        if bank_path is not None:
            self._load(bank_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return bool(self._bank)

    @property
    def num_tasks(self) -> int:
        return len(self._bank)

    def task_names(self) -> List[str]:
        return sorted(self._bank.keys())

    def prompt_count(self, task_name: str) -> int:
        entry = self._bank.get(task_name)
        return len(entry["prompts"]) if entry else 0

    def get_original(self, task_name: str) -> Optional[str]:
        """Return the canonical (first) prompt for eval/inference."""
        resolved = self._resolve_key(task_name)
        if resolved is not None:
            return self._bank[resolved]["prompts"][0]
        return None

    def sample_aux(
        self,
        task_name: str,
        fallback_text: Optional[str] = None,
        exclude_original: bool = True,
    ) -> Optional[str]:
        """Randomly sample an auxiliary prompt for text reconstruction.

        The policy input should keep using ``get_original``.  This method is
        only for training-only auxiliary objectives.  By default it samples
        from prompts[1:] so the canonical dataset instruction is excluded.
        """
        resolved = self._resolve_key(task_name)
        if resolved is not None:
            prompts: list = self._bank[resolved]["prompts"]
            candidates = prompts[1:] if exclude_original else prompts
            if candidates:
                return candidates[random.randrange(len(candidates))]
            return fallback_text

        if task_name not in self._warned_missing:
            cprint(
                f"[InstructionBank] task {task_name!r} not in bank; "
                f"using fallback text.",
                "yellow",
            )
            self._warned_missing.add(task_name)

        return fallback_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_key(self, task_name: str) -> Optional[str]:
        """Match *task_name* against bank keys using normalised forms."""
        if task_name in self._bank:
            return task_name
        for candidate in _candidate_keys(task_name):
            if candidate in self._bank:
                return candidate
        return None

    def _load(self, bank_path: str):
        path = os.path.expanduser(bank_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"InstructionBank: file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"InstructionBank: root must be an object, got {type(raw)}")

        for task_name, entry in raw.items():
            self._validate_entry(task_name, entry)
            self._bank[str(task_name)] = entry

        cprint(
            f"[InstructionBank] loaded {len(self._bank)} tasks "
            f"({sum(len(e['prompts']) for e in self._bank.values())} prompts) "
            f"from {path}",
            "cyan",
        )
        for task_name in sorted(self._bank):
            cprint(
                f"  [InstructionBank] {task_name}: "
                f"{len(self._bank[task_name]['prompts'])} prompts",
                "cyan",
            )

    @staticmethod
    def _validate_entry(task_name: str, entry):
        if not isinstance(entry, dict):
            raise ValueError(
                f"InstructionBank: {task_name!r} must be an object, got {type(entry)}"
            )
        # sampling mode
        mode = entry.get("sampling", "uniform")
        if mode != "uniform":
            raise ValueError(
                f"InstructionBank: {task_name!r}.sampling must be 'uniform', "
                f"got {mode!r}"
            )
        prompts = entry.get("prompts")
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(
                f"InstructionBank: {task_name!r}.prompts must be a non-empty list"
            )
        for i, p in enumerate(prompts):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(
                    f"InstructionBank: {task_name!r}.prompts[{i}] "
                    f"must be a non-empty string, got {p!r}"
                )
