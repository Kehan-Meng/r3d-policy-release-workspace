import copy

import hydra
import wandb
from termcolor import cprint
from omegaconf import OmegaConf

from r3d.env_runner.base_runner import BaseRunner
from r3d.training.utils import is_main_process


class WorkspaceEvaluationMixin:
    """Standalone checkpoint evaluation helpers for the training workspace."""

    def eval(self):
        if not is_main_process():
            return None

        cfg = copy.deepcopy(self.cfg)

        wandb_run = None
        wandb_enabled = str(cfg.logging.get("mode", "online")).lower() != "disabled"
        if wandb_enabled:
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB Eval] group: {cfg.logging.group}_eval", "yellow")
            cprint(f"[WandB Eval] name: {cfg.logging.name}_eval", "yellow")
            cprint("-----------------------------", "yellow")
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                group=f"{cfg.logging.group}_eval",
                name=f"{cfg.logging.name}_eval",
                project="maniskill eval",
                tags=cfg.logging.get("tags", []) + ["evaluation"],
            )
            wandb.config.update(
                {"output_dir": self.output_dir, "eval_mode": True}
            )
        else:
            cprint("[WandB Eval] disabled", "yellow")

        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseRunner)

        epochs_to_eval = list(range(50, 151, 50))
        all_results = {}
        for epoch_tag in epochs_to_eval:
            cprint(f"\n{'=' * 60}", "cyan")
            cprint(f"Evaluating checkpoint: {epoch_tag}", "cyan")
            cprint(f"{'=' * 60}", "cyan")

            checkpoint_path = self.get_checkpoint_path(tag=str(epoch_tag))
            if not checkpoint_path.is_file():
                cprint(f"Checkpoint {checkpoint_path} not found, skipping...", "red")
                continue

            cprint(f"Loading checkpoint {checkpoint_path}", "magenta")
            self.load_checkpoint(path=checkpoint_path)
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()
            policy.cuda()

            from r3d.env_runner.frame_adapter_wrapper import (
                maybe_wrap_policy_for_environment,
            )

            policy = maybe_wrap_policy_for_environment(
                policy,
                cfg.get("frame_adapter", None),
                checkpoint_metadata=self.frame_adapter_checkpoint_metadata,
                require_checkpoint_metadata=bool(
                    (cfg.get("frame_adapter", None) or {}).get("enabled", False)
                ),
            )
            runner_log = env_runner.run(policy)

            cprint(
                f"\n---------------- Eval Results (Epoch {epoch_tag}) --------------",
                "magenta",
            )
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", "magenta")
            all_results[str(epoch_tag)] = runner_log

            if wandb_run is not None:
                wandb_log = {"eval_epoch": epoch_tag}
                wandb_log.update(
                    {
                        f"eval/{key}": value
                        for key, value in runner_log.items()
                        if isinstance(value, (int, float))
                    }
                )
                wandb_run.log(wandb_log, step=epoch_tag)

        cprint(f"\n{'=' * 60}", "green")
        cprint("Evaluation Summary", "green")
        cprint(f"{'=' * 60}", "green")
        for epoch_tag, results in all_results.items():
            cprint(f"\nEpoch {epoch_tag}:", "yellow")
            for key, value in results.items():
                if isinstance(value, float):
                    cprint(f"  {key}: {value:.4f}", "yellow")

        if all_results and wandb_run is not None:
            import pandas as pd

            rows = []
            for epoch_tag, results in all_results.items():
                row = {"epoch": int(epoch_tag)}
                row.update(
                    {
                        key: value
                        for key, value in results.items()
                        if isinstance(value, (int, float))
                    }
                )
                rows.append(row)
            wandb_run.log(
                {"eval_summary_table": wandb.Table(dataframe=pd.DataFrame(rows))}
            )
            cprint("\nLogged evaluation summary to wandb", "green")

        if wandb_run is not None:
            wandb_run.finish()
            cprint("\nWandB run finished", "green")

    def get_policy(self, checkpoint_num=3000):
        cfg = copy.deepcopy(self.cfg)
        checkpoint_path = self.get_checkpoint_path(tag=str(checkpoint_num))
        assert checkpoint_path.is_file(), f"ckpt file doesn't exist, {checkpoint_path}"

        cprint(f"Resuming from checkpoint {checkpoint_path}", "magenta")
        self.load_checkpoint(path=checkpoint_path)
        policy = self.ema_model if cfg.training.use_ema else self.model
        policy.eval()
        policy.cuda()

        from r3d.env_runner.frame_adapter_wrapper import (
            maybe_wrap_policy_for_environment,
        )

        return maybe_wrap_policy_for_environment(
            policy,
            cfg.get("frame_adapter", None),
            checkpoint_metadata=self.frame_adapter_checkpoint_metadata,
            require_checkpoint_metadata=bool(
                (cfg.get("frame_adapter", None) or {}).get("enabled", False)
            ),
        )
