import json
import os
from typing import Any, Dict, Iterable, Optional

from termcolor import cprint


class TextInstructionDataset:
    """Small JSON-backed command -> instruction lookup."""

    def __init__(
            self,
            text_json_path: Optional[str] = None,
            strict: bool = False,
            fallback_to_command: bool = True):
        self.text_json_path = os.path.expanduser(text_json_path) if text_json_path else None
        self.strict = strict
        self.fallback_to_command = fallback_to_command
        self.instructions = self._load_text_json(self.text_json_path)
        self._warned_missing = set()

    @staticmethod
    def _load_text_json(text_json_path: Optional[str]) -> Dict[str, str]:
        if text_json_path is None:
            return {}

        try:
            with open(text_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            cprint(f"[TextInstructionDataset] text json not found: {text_json_path}", "red")
            return {}
        except json.JSONDecodeError as exc:
            cprint(f"[TextInstructionDataset] failed to parse {text_json_path}: {exc}", "red")
            return {}

        if not isinstance(data, dict):
            raise ValueError(f"text json must contain an object, got {type(data)}")

        instructions = {}
        for key, value in data.items():
            if isinstance(value, str):
                instructions[str(key)] = value
            elif isinstance(value, dict) and isinstance(value.get("text"), str):
                instructions[str(key)] = value["text"]
            elif isinstance(value, dict) and isinstance(value.get("instruction"), str):
                instructions[str(key)] = value["instruction"]
            elif (
                isinstance(value, dict)
                and isinstance(value.get("prompts"), list)
                and value["prompts"]
                and isinstance(value["prompts"][0], str)
            ):
                # Instruction-bank format: prompts[0] is the canonical policy input.
                instructions[str(key)] = value["prompts"][0]
            else:
                raise ValueError(
                    f"text json value for {key!r} must be a string or contain "
                    "text/instruction/non-empty prompts"
                )

        cprint(
            f"[TextInstructionDataset] loaded {len(instructions)} text instructions from {text_json_path}",
            "cyan",
        )
        return instructions

    @staticmethod
    def _normalize_key(command: Any) -> str:
        key = str(command).strip()
        key = key.replace(" ", "_")
        return key

    @staticmethod
    def _candidate_keys(command: Any) -> Iterable[str]:
        key = TextInstructionDataset._normalize_key(command)
        base_keys = [key]
        if key.endswith(".zarr"):
            base_keys.append(os.path.basename(key[:-5]))

        suffixes = (
            "-demo_clean-50",
            "-demo_randomized-50",
            "_demo_clean_50",
            "_demo_randomized_50",
            "-demo_clean",
            "-demo_randomized",
            "_demo_clean",
            "_demo_randomized",
            "-50",
            "_50",
        )
        seen = set()
        for base_key in base_keys:
            variants = [base_key, base_key.replace("_", "-"), base_key.replace("-", "_")]
            for variant in variants:
                if variant not in seen:
                    seen.add(variant)
                    yield variant
                for suffix in suffixes:
                    if variant.endswith(suffix):
                        stripped = variant[:-len(suffix)]
                        for stripped_variant in (
                                stripped,
                                stripped.replace("_", "-"),
                                stripped.replace("-", "_")):
                            if stripped_variant not in seen:
                                seen.add(stripped_variant)
                                yield stripped_variant

    def lookup(self, command: Any, default: Optional[str] = None) -> str:
        if command is None:
            if default is not None:
                return default
            raise ValueError("command must be provided for text lookup")

        for candidate in self._candidate_keys(command):
            if candidate in self.instructions:
                return self.instructions[candidate]

        if self.strict:
            available = list(self.instructions.keys())
            raise KeyError(f"Command {command!r} not found in text json. Available keys: {available}")

        if command not in self._warned_missing and self.instructions:
            cprint(
                f"[TextInstructionDataset] command {command!r} not found; using fallback text",
                "yellow",
            )
            self._warned_missing.add(command)

        if default is not None:
            return default
        if self.fallback_to_command:
            return str(command)
        return ""

    def __contains__(self, command: Any) -> bool:
        return any(candidate in self.instructions for candidate in self._candidate_keys(command))


def attach_text_fields(
        data: Dict[str, Any],
        text_dataset: Optional[TextInstructionDataset],
        command: Any,
        enabled: bool = False,
        instruction_bank=None,
        is_train: bool = True,
        apply_in_train: bool = True,
        apply_in_val: bool = False,
) -> Dict[str, Any]:
    if not enabled:
        return data

    if text_dataset is None:
        text_dataset = TextInstructionDataset()

    command = str(command)
    bank_enabled = instruction_bank is not None and instruction_bank.is_enabled

    # Main policy text is always the canonical/original instruction.
    fallback_text = text_dataset.lookup(command)
    if bank_enabled:
        text = instruction_bank.get_original(command) or fallback_text
    else:
        text = fallback_text

    data["command"] = command
    data["task_name"] = command
    data["text"] = text

    return data
