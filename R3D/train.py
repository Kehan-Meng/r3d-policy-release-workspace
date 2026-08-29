import os
import copy
import random
import time
import pathlib
import json
import re
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from torch.utils.data import DataLoader
import wandb
import tqdm
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf
from r3d.policy.dp3 import DP3
from r3d.dataset.base_dataset import BaseDataset
from r3d.env_runner.base_runner import BaseRunner
from r3d.common.checkpoint_util import TopKCheckpointManager
from r3d.common.pytorch_util import optimizer_to
from r3d.model.diffusion.ema_model import EMAModel
from r3d.model.common.lr_scheduler import get_scheduler
from r3d.training.checkpointing import CheckpointMixin
from r3d.training.evaluation import WorkspaceEvaluationMixin
from r3d.training.utils import (
    batch_to_device,
    cleanup_ddp,
    is_main_process,
    json_safe,
    setup_ddp,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace(CheckpointMixin, WorkspaceEvaluationMixin):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        self.frame_adapter_checkpoint_metadata = None

        # DDP setup
        self.use_ddp = cfg.training.get('use_ddp', False)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0)) if self.use_ddp else 0
        self.world_size = int(os.environ.get('WORLD_SIZE', 1)) if self.use_ddp else 1

        self.use_bfloat16 = cfg.training.get('use_bfloat16', False)
        self.autocast_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        
        if self.use_ddp:
            setup_ddp(self.local_rank, self.world_size)
            if is_main_process():
                print(f"DDP initialized: rank {self.local_rank}/{self.world_size}")
        if self.use_bfloat16 and is_main_process():
            cprint(f"[Training] Using bfloat16 mixed precision training", "green")
        
        # set seed (add rank to seed for different random states across processes)
        seed = cfg.training.seed + (self.local_rank if self.use_ddp else 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        # Optional low-coupling controls for fine-tuning the point-cloud encoder.
        # Defaults preserve all existing experiments exactly.
        self.encoder_lr_scale = float(cfg.training.get('encoder_lr_scale', 1.0))
        self.encoder_unfreeze_epoch = int(cfg.training.get('encoder_unfreeze_epoch', 0))
        self._delayed_encoder_params = []
        encoder_module = getattr(getattr(self.model, 'obs_encoder', None), 'extractor', None)
        encoder_param_ids = set()
        if encoder_module is not None:
            encoder_param_ids = {id(p) for p in encoder_module.parameters()}
            if self.encoder_unfreeze_epoch > 0:
                self._delayed_encoder_params = [
                    p for p in encoder_module.parameters() if p.requires_grad
                ]
                for p in self._delayed_encoder_params:
                    p.requires_grad_(False)
                cprint(
                    f"[Training] Delayed point encoder fine-tuning until epoch "
                    f"{self.encoder_unfreeze_epoch} ({len(self._delayed_encoder_params)} tensors)",
                    "cyan",
                )

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        if encoder_module is not None and self.encoder_lr_scale != 1.0:
            base_lr = float(cfg.optimizer.lr)
            encoder_params = [p for p in self.model.parameters() if id(p) in encoder_param_ids]
            other_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer,
                params=[
                    {'params': other_params},
                    {'params': encoder_params, 'lr': base_lr * self.encoder_lr_scale},
                ],
                _convert_='all',
            )
            cprint(
                f"[Training] Point encoder LR scale={self.encoder_lr_scale:g}: "
                f"{base_lr:g} -> {base_lr * self.encoder_lr_scale:g}",
                "cyan",
            )
        else:
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if not lastest_ckpt_path.is_file():
                checkpoint_dir = pathlib.Path(self.output_dir) / 'checkpoints'
                epoch_checkpoints = []
                for candidate in checkpoint_dir.glob('epoch_*.ckpt'):
                    match = re.search(r'epoch_(\d+)', candidate.name)
                    if match is not None:
                        epoch_checkpoints.append((int(match.group(1)), candidate))
                if epoch_checkpoints:
                    _, lastest_ckpt_path = max(epoch_checkpoints, key=lambda item: item[0])
                    print(f"latest.ckpt not found; using highest epoch checkpoint {lastest_ckpt_path}")
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, exclude_keys=('optimizer',))
                # Increment epoch since checkpoint contains the completed epoch
                # We should start training from the next epoch
                print(f"Checkpoint loaded: epoch {self.epoch} completed")
                self.epoch += 1
                self.global_step += 1
                print(f"Will resume training from epoch {self.epoch}")

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        from r3d.dataset.transforms.frame_transform import (
            configure_dataset_frame_transform,
        )
        configure_dataset_frame_transform(dataset, cfg.get('frame_adapter', None))
        
        # Configure data loaders with DDP support
        train_sampler = None
        val_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.local_rank)
            val_dataset = dataset.get_validation_dataset()
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.local_rank)

            # Remove shuffle from dataloader config when using DistributedSampler
            train_dataloader_cfg = dict(cfg.dataloader)
            train_dataloader_cfg['shuffle'] = False
            train_dataloader_cfg['batch_size'] = cfg.dataloader['batch_size'] // self.world_size
            val_dataloader_cfg = dict(cfg.val_dataloader)
            val_dataloader_cfg['shuffle'] = False
            val_dataloader_cfg['batch_size'] = cfg.val_dataloader['batch_size'] // self.world_size

            if is_main_process():
                print(f"Rank {self.local_rank}: Train batch size = {train_dataloader_cfg['batch_size']}")
                print(f"Rank {self.local_rank}: Val batch size = {val_dataloader_cfg['batch_size']}")

            train_dataloader = DataLoader(dataset, sampler=train_sampler, **train_dataloader_cfg)
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **val_dataloader_cfg)
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()

        loaded_frame_metadata = copy.deepcopy(
            self.frame_adapter_checkpoint_metadata
        )
        frame_metadata = dataset.frame_transform_metadata
        if frame_metadata.get('frame_adapter_enabled', False):
            from r3d.dataset.transforms.frame_aware_normalizer import (
                normalizer_sha256,
            )
            from r3d.env_runner.frame_adapter_wrapper import (
                validate_frame_checkpoint_metadata,
            )
            normalizer_metadata = frame_metadata.get('normalizer')
            if not normalizer_metadata:
                raise RuntimeError(
                    "Frame adapter is enabled, but the dataset did not fit a "
                    "frame-aware normalizer"
                )
            frame_metadata = dict(frame_metadata)
            frame_metadata['normalizer_hash'] = normalizer_sha256(normalizer)
            frame_metadata['train_split_hash'] = normalizer_metadata['train_split_hash']
            frame_metadata['normalizer_frame_config_hash'] = normalizer_metadata[
                'frame_config_hash'
            ]
            if cfg.training.resume and loaded_frame_metadata:
                validate_frame_checkpoint_metadata(
                    cfg.get('frame_adapter', None),
                    loaded_frame_metadata,
                    normalizer=normalizer,
                    require_checkpoint_metadata=True,
                    expected_train_split_hash=frame_metadata['train_split_hash'],
                )
        self.frame_adapter_checkpoint_metadata = frame_metadata

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # Ensure initial_lr is set for fresh optimizers (e.g. after skipping
        # optimizer state during resume due to checkpoint compatibility issues).
        for pg in self.optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            
        # configure env
        env_runner = None
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        # configure logging (only on main process)
        wandb_run = None
        wandb_enabled = str(cfg.logging.get('mode', 'online')).lower() != 'disabled'
        if is_main_process() and wandb_enabled:
            # cfg.logging.name = str(cfg.task.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )
        elif is_main_process():
            cprint("[WandB] disabled; using local JSON logs only", "yellow")

        # configure checkpoint
        topk_kwargs = OmegaConf.to_container(cfg.checkpoint.topk, resolve=True)
        if topk_kwargs.get("monitor_key") == "train_loss":
            topk_kwargs["format_str"] = "epoch_{epoch:04d}-train_loss_{train_loss:.6f}.ckpt"
        elif topk_kwargs.get("monitor_key") == "val_loss":
            topk_kwargs["format_str"] = "epoch_{epoch:04d}-val_loss_{val_loss:.6f}.ckpt"
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **topk_kwargs
        )

        # device transfer
        if self.use_ddp:
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device(cfg.training.device)
        
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Wrap model with DDP
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            # EMA model doesn't need DDP wrapping

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        # num_epochs is the target final epoch, including resumed training.
        # For example, resuming an epoch-800 checkpoint with num_epochs=3001
        # runs epochs 801..3000 instead of another 3001 epochs.
        remaining_epochs = max(0, int(cfg.training.num_epochs) - int(self.epoch))
        for local_epoch_idx in range(remaining_epochs):
            if self._delayed_encoder_params and self.epoch == self.encoder_unfreeze_epoch:
                for p in self._delayed_encoder_params:
                    p.requires_grad_(True)
                cprint(
                    f"[Training] Unfroze point encoder fine-tuning parameters at epoch {self.epoch}",
                    "cyan",
                )
            policy_for_schedule = self.model.module if self.use_ddp else self.model
            policy_for_schedule.set_training_progress(
                epoch=self.epoch,
                num_epochs=int(cfg.training.num_epochs),
            )
            # Set epoch for DistributedSampler
            if self.use_ddp and train_sampler is not None:
                train_sampler.set_epoch(self.epoch)

            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            
            # Only show progress bar on main process
            if is_main_process():
                tepoch = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec)
            else:
                tepoch = train_dataloader
            
            for batch_idx, batch in enumerate(tepoch):
                t1 = time.time()
                # device transfer
                batch = batch_to_device(batch, device)

                if train_sampling_batch is None:
                    train_sampling_batch = batch
            
                # compute loss
                t1_1 = time.time()
                with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                    if self.use_ddp:
                        raw_loss, loss_dict = self.model.module.compute_loss(batch)
                    else:
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()
                
                t1_2 = time.time()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                t1_3 = time.time()
                # update ema
                if cfg.training.use_ema:
                    if self.use_ddp:
                        # For DDP, update EMA with the underlying model (without DDP wrapper)
                        ema.step(self.model.module)
                    else:
                        ema.step(self.model)
                t1_4 = time.time()
                # logging
                raw_loss_cpu = raw_loss.item()
                if is_main_process():
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'train_step_time_sec': t1_4 - t1,
                    'compute_loss_time_sec': t1_2 - t1_1,
                    'optimizer_step_time_sec': t1_3 - t1_2,
                    'ema_step_time_sec': t1_4 - t1_3,
                }
                if torch.cuda.is_available():
                    step_log['gpu_mem_allocated_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    step_log['gpu_mem_reserved_mb'] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                t1_5 = time.time()
                step_log.update(loss_dict)
                t2 = time.time()
                
                if verbose and is_main_process():
                    print(f"total one step time: {t2-t1:.3f}")
                    print(f" compute loss time: {t1_2-t1_1:.3f}")
                    print(f" step optimizer time: {t1_3-t1_2:.3f}")
                    print(f" update ema time: {t1_4-t1_3:.3f}")
                    print(f" logging time: {t1_5-t1_4:.3f}")

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                        # log of last step is combined with validation and rollout
                    if is_main_process() and wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model.module if self.use_ddp else self.model
            eval_use_ema = cfg.training.get('eval_use_ema', False)
            if cfg.training.use_ema and eval_use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout (only on main process)
            rollout_milestones = {
                int(epoch) for epoch in cfg.training.get('rollout_milestone_epochs', [])
            }
            periodic_rollout = (
                (self.epoch % cfg.training.rollout_every) == 0 and self.epoch != 0
            )
            milestone_rollout = self.epoch in rollout_milestones
            if (periodic_rollout or milestone_rollout) and RUN_ROLLOUT and env_runner is not None and is_main_process():
                t3 = time.time()
                task_config = getattr(cfg, 'setting', None)
                default_eval_episodes = getattr(env_runner, 'eval_episodes', None)
                if milestone_rollout:
                    requested_eval_episodes = cfg.training.get(
                        'rollout_milestone_eval_episodes', default_eval_episodes
                    )
                else:
                    requested_eval_episodes = cfg.training.get(
                        'rollout_eval_episodes', default_eval_episodes
                    )
                if requested_eval_episodes is not None and default_eval_episodes is not None:
                    env_runner.eval_episodes = int(requested_eval_episodes)
                try:
                    from r3d.env_runner.frame_adapter_wrapper import (
                        maybe_wrap_policy_for_environment,
                    )
                    rollout_policy = maybe_wrap_policy_for_environment(
                        policy,
                        cfg.get('frame_adapter', None),
                        checkpoint_metadata=self.frame_adapter_checkpoint_metadata,
                        require_checkpoint_metadata=False,
                    )
                    runner_log = env_runner.run(
                        rollout_policy, self.epoch, task_config=task_config
                    )
                    t4 = time.time()
                    step_log.update(runner_log)
                except Exception as e:
                    cprint(f"[Rollout Error] epoch={self.epoch}: {e}", 'red')
                    import traceback
                    traceback.print_exc()
                finally:
                    if default_eval_episodes is not None:
                        env_runner.eval_episodes = default_eval_episodes

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    
                    if is_main_process():
                        val_tepoch = tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec)
                    else:
                        val_tepoch = val_dataloader
                    
                    for batch_idx, batch in enumerate(val_tepoch):
                        batch = batch_to_device(batch, device)
                        with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                            if self.use_ddp:
                                loss, loss_dict = self.model.module.compute_loss(batch)
                            else:
                                loss, loss_dict = self.model.compute_loss(batch)
                        val_losses.append(loss)
                        if (cfg.training.max_val_steps is not None) \
                            and batch_idx >= (cfg.training.max_val_steps-1):
                            break
                    
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()

                        # Synchronize validation loss across all processes if using DDP
                        if self.use_ddp:
                            val_loss_tensor = torch.tensor(val_loss, device=device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            val_loss = (val_loss_tensor / self.world_size).item()

                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = batch_to_device(train_sampling_batch, device)
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_bfloat16):
                        predict_kwargs = {}
                        if 'text' in batch:
                            predict_kwargs['text'] = batch['text']
                        if 'command' in batch:
                            predict_kwargs['command'] = batch['command']
                        result = policy.predict_action(obs_dict, **predict_kwargs)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint (only save on main process)
            checkpoint_milestones = {
                int(epoch) for epoch in cfg.training.get('checkpoint_milestone_epochs', [])
            }
            checkpoint_start_epoch = int(cfg.training.get('checkpoint_start_epoch', 0))
            periodic_checkpoint = (
                self.epoch >= checkpoint_start_epoch
                and (self.epoch % cfg.training.checkpoint_every) == 0
            )
            if (periodic_checkpoint or self.epoch in checkpoint_milestones) and cfg.checkpoint.save_ckpt and is_main_process():
                save_path = topk_manager.get_ckpt_path(step_log)
                if save_path is not None:
                    self.save_checkpoint(save_path)

            # Synchronize all processes after checkpoint saving
            if self.use_ddp:
                dist.barrier()
                
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if is_main_process() and wandb_run is not None:
                wandb_run.log(step_log, step=self.global_step)
            if is_main_process():
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(json_safe(step_log), ensure_ascii=False) + "\n")
            self.global_step += 1
            self.epoch += 1
            del step_log

        # Clean up DDP
        if self.use_ddp:
            cleanup_ddp()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'r3d', 'config'))
)
def main(cfg):
    # Handle DDP environment variables
    if cfg.training.get('use_ddp', False):
        # Set local rank from environment variable
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        cfg.training.local_rank = local_rank

        # Adjust device setting for DDP
        if cfg.training.device == "cuda":
            cfg.training.device = f"cuda:{local_rank}"

    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
