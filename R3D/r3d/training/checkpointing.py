import copy
import os
import pathlib
import threading

import dill
import torch

from r3d.training.utils import copy_state_dict_to_cpu


class CheckpointMixin:
    """Checkpoint persistence for a training workspace.

    The method names and payload format intentionally match the historical
    ``TrainDP3Workspace`` API so existing checkpoints and evaluation scripts
    remain compatible.
    """

    @property
    def output_dir(self):
        from hydra.core.hydra_config import HydraConfig

        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        print("saved in ", path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(
                "checkpoints", f"{tag}.ckpt"
            )
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
            "frame_adapter": copy.deepcopy(
                self.frame_adapter_checkpoint_metadata
                or {"frame_adapter_enabled": False}
            ),
        }

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    state_dict = value.state_dict()
                    if key == "model" and self.use_ddp and hasattr(value, "module"):
                        state_dict = {
                            name[7:] if name.startswith("module.") else name: tensor
                            for name, tensor in state_dict.items()
                        }
                    if use_thread:
                        state_dict = copy_state_dict_to_cpu(state_dict)
                    payload["state_dicts"][key] = state_dict
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(
                    payload, path.open("wb"), pickle_module=dill
                )
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag:
            return pathlib.Path(self.output_dir).joinpath(
                "checkpoints", f"{tag}.ckpt"
            )
        if tag == "best":
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            best_ckpt = None
            best_score = -1e10
            for ckpt in os.listdir(checkpoint_dir):
                if "latest" in ckpt:
                    continue
                score = float(
                    ckpt.split("test_mean_score=")[1].split(".ckpt")[0]
                )
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return checkpoint_dir.joinpath(best_ckpt)
        raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        self.frame_adapter_checkpoint_metadata = copy.deepcopy(
            payload.get("frame_adapter", {"frame_adapter_enabled": False})
        )

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                try:
                    self.__dict__[key].load_state_dict(value, strict=False, **kwargs)
                except TypeError:
                    self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

        from r3d.env_runner.frame_adapter_wrapper import (
            validate_frame_checkpoint_metadata,
        )

        frame_config = self.cfg.get("frame_adapter", None)
        validate_frame_checkpoint_metadata(
            frame_config,
            self.frame_adapter_checkpoint_metadata,
            normalizer=getattr(self.model, "normalizer", None),
            require_checkpoint_metadata=bool(
                (frame_config or {}).get("enabled", False)
            ),
        )

    def load_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(
            path.open("rb"), pickle_module=dill, map_location="cpu"
        )
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
        )
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """Save the complete workspace for short-term research use."""
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)
