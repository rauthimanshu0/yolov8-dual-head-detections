# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics BaseTrainer — dual‑head friendly training loop
Location: /ultralytics/engine/trainer.py
=======================================

Role
- Provides the generic training engine used by task‑specific trainers (detect/segment/pose/obb/classify).
- Manages configuration, devices, data loading, distributed training, optimization, logging, saving, and validation.
- Designed to be dual‑head aware via extensible hooks and label preprocessing.

Key features
- Configuration and environment:
  - Loads args (cfg + overrides), seeds RNGs, selects device, creates run directory and weights/ paths.
  - Saves args.yaml; maintains results.csv and plot artifacts via callbacks.

- Distributed and AMP:
  - Auto DDP launcher with generate_ddp_command() for multi‑GPU; NCCL/Gloo fallback.
  - Mixed precision via torch GradScaler; broadcasts AMP enablement across ranks.

- Model and dataset lifecycle:
  - setup_model() loads cfg/weights (safe loaders), constructs the model via task‑specific get_model().
  - get_dataset() supports detect/segment/pose/obb/classify; resolves data yaml path.
  - set_model_attributes() is task‑overridable (e.g., to inject class names or axis maps).

- Optimizer & scheduler:
  - build_optimizer() groups parameters (bias/norm/other) and supports Adam/AdamW/SGD/RMSProp/etc.
  - Warmup and cosine/linear LR scheduling; gradient clipping and accumulation.

- Training loop:
  - Warmup steps, forward/backward with autocast; optimizer stepping on schedule with accumulate.
  - Mosaic close scheduling; multi‑scale resizing; per‑epoch EMA updates (optional).
  - Periodic validation; fitness/early stopping; time‑based stop; per‑rank logging.

- Checkpointing:
  - save_model() writes last/best/periodic .pt with:
      ema or model module, optimizer (FP16 converted), args, metrics, results CSV snapshot, version metadata.
    Note: avoids dtype casting on modules to prevent recursion issues in ._apply.
  - resume_training() restores optimizer/EMA/epoch safely (accepts EMA as module or state_dict).

Dual‑head specifics
- Label preprocessing (for DualDetect + v8DetectionLoss):
  - BaseTrainer.preprocess_batch() normalizes labels so the loss always finds cls1 and cls2:
      - If batch has split labels already: ensures dtype matches bboxes for concatenation.
      - If batch["cls"] is (N,2): splits to cls1/cls2.
      - If batch["cls"] is combined indices: infers num_cls2 from data['names_head2'] or model head.num_cls2
        and splits via cls1 = cls // num_cls2, cls2 = cls % num_cls2 (cast to bboxes dtype).
  - This makes dual‑head training work even when datasets provide joint labels only.

- Metrics CSV key normalization:
  - save_metrics() rewrites common keys for clearer dashboards:
      - "metrics/*(B)" → "metrics/*(joint)"
      - "h1/*" → "Taxonomy Dual/*"
      - "h2/*" → "Substrate Dual/*"
      - Legacy "species/*" → "Taxonomy/*", "stage/*" → "substrate/*"
  - Prevents collisions and preserves the first seen value on rename conflicts (with a warning).

Validation and plotting
- validator is injected by task‑specific trainers (e.g., DualHeadDetectionValidator for detect) and called each epoch/final.
- on final_eval(), validates best.pt with current plot settings and updates callbacks.
- Plot hooks (plot_training_samples, plot_training_labels, plot_metrics) are implemented in task trainers.

Callbacks and extensibility
- Rich callback map (on_train_start/end, on_pretrain_routine_start/end, on_train_epoch_start/end, on_batch_end, etc.).
- Task trainers override:
  - get_model(), get_validator(), get_dataloader(), build_dataset(),
  - set_model_attributes(), label_loss_items(), plot_*(), progress_string().

Safety and housekeeping
- Robust cleanup of CUDA/Metal caches; deterministic toggle, DDP cleanup, CSV header management, LR logging to CSV.
- AutoBatch support (checks optimal batch given image size and AMP).
- Freezes selected layers (by index and DFL head by default), keeping BN stats frozen when requested.

Usage
- Inherit from BaseTrainer to implement a task:
    class DetectionTrainer(BaseTrainer):
        def get_model(...): ...
        def get_validator(...): ...
        def get_dataloader(...): ...
        def build_dataset(...): ...
        ...
  Dual‑head flows plug into this base via:
  - preprocess_batch() for cls1/cls2,
  - task validator (axis metrics),
  - and task trainer plotting/logging (priority panels).
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


class BaseTrainer:
    """
    A base class for creating trainers.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:
            world_size = 0
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning("'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'")
                self.args.batch = 16

            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp.int(), src=0)
        self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # Batch size
        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))

            # EMA optional (disabled unless args.ema True) and resilient
            self.ema = None
            try:
                if getattr(self.args, "ema", False):
                    self.ema = ModelEMA(self.model)
            except Exception as e:
                LOGGER.warning(f"EMA disabled due to exception during init: {e}")
                self.ema = None

            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                if self.ema:
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            # Early Stopping
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )

    def _get_memory(self, fraction=False):
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float = None):
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        import io as _io_test  # keep lazy import pattern intact but test import works
        import pandas as pd
        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata.

        Important change: avoid dtype casting (.half()/.float()) on modules to prevent RecursionError in Module._apply.
        We save the EMA/model module as-is (FP32) to keep Ultralytics loaders compatible and avoid deep recursion.
        """
        import io

        # Prepare module to save (EMA preferred if available), without dtype casting
        if self.ema and getattr(self.ema, "ema", None) is not None:
            try:
                ema_model_to_save = deepcopy(self.ema.ema)
            except Exception as e:
                LOGGER.warning(f"Deepcopy EMA failed ({e}); falling back to current model.")
                ema_model_to_save = deepcopy(self.model)
            ema_updates = int(getattr(self.ema, "updates", 0))
        else:
            ema_model_to_save = deepcopy(self.model)
            ema_updates = 0

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # follow Ultralytics convention: inference uses 'ema' if present
                "ema": ema_model_to_save,  # keep as Module for compatibility with AutoBackend/strip_optimizer
                "updates": ema_updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)

    def get_dataset(self):
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {"detect", "segment", "pose", "obb"}:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        if isinstance(self.model, torch.nn.Module):
            return
        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
        return ckpt

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Ensure batch has cls1/cls2 for DualDetect training, even if the dataset only provided:
          - combined 'cls' (N,) or (N,1), or
          - split 'cls' (N,2)

        We compute cls1/cls2 and cast them to the same dtype as bboxes so concatenation in loss works.
        """
        import torch

        b = batch

        # Preferred dtype for labels to match loss cat(...): use bboxes dtype if available
        target_dtype = (
            b["bboxes"].dtype
            if isinstance(b.get("bboxes", None), torch.Tensor)
            else torch.float32
        )

        # Already present? Then just normalize dtype and return
        if "cls1" in b and "cls2" in b:
            if isinstance(b["cls1"], torch.Tensor):
                b["cls1"] = b["cls1"].to(dtype=target_dtype)
            if isinstance(b["cls2"], torch.Tensor):
                b["cls2"] = b["cls2"].to(dtype=target_dtype)
            return b

        c = b.get("cls", None)
        if not isinstance(c, torch.Tensor):
            # Nothing we can do here; downstream will error if loss expects cls1/cls2
            return b

        # Case 1: split labels already present in cls as (N,2)
        if c.ndim == 2 and c.shape[1] == 2:
            b["cls1"] = c[:, 0].view(-1, 1).to(dtype=target_dtype)
            b["cls2"] = c[:, 1].view(-1, 1).to(dtype=target_dtype)
            return b

        # Case 2: combined class ids -> split using num_cls2
        # Determine num_cls2 (stage axis) from data yaml first, then model head as fallback
        n2 = None
        try:
            nh2 = self.data.get("names_head2", {}) if isinstance(self.data, dict) else {}
            if isinstance(nh2, dict) and len(nh2) > 0:
                n2 = int(len(nh2))
        except Exception:
            pass
        if n2 is None:
            try:
                head = self.model.module.model[-1] if hasattr(self.model, "module") else self.model.model[-1]
                n2 = int(getattr(head, "num_cls2"))
            except Exception:
                n2 = None

        if n2:
            cf = c.view(-1).to(dtype=torch.long)
            b["cls1"] = (cf // n2).view(-1, 1).to(dtype=target_dtype)
            b["cls2"] = (cf %  n2).view(-1, 1).to(dtype=target_dtype)
        # else: leave as-is; loss will raise with a clearer error if cls1/cls2 are still missing

        return b

    def validate(self):
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ""

    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def save_metrics(self, metrics):
        # Rename metric keys just before CSV write (output-only transformation)
        def rename_key(k: str) -> str:
            # Exact replacements first
            exact = {
                "metrics/precision(B)": "metrics/precision(joint)",
                "metrics/recall(B)": "metrics/recall(joint)",
                "metrics/mAP50(B)": "metrics/mAP50(joint)",
                "metrics/mAP50-95(B)": "metrics/mAP50-95(joint)",
            }
            if k in exact:
                return exact[k]

            # Prefix-based / segment replacements
            # species -> Genus
            if k.startswith("species/"):
                return k.replace("species/", "Genus/", 1)
            # stage -> substrat
            if k.startswith("stage/"):
                return k.replace("stage/", "substrate/", 1)
            # h1 -> Genus Dual
            if k.startswith("h1/"):
                return k.replace("h1/", "Genus Dual/", 1)
            # h2 -> Substrate Dual
            if k.startswith("h2/"):
                return k.replace("h2/", "Substrate Dual/", 1)

            return k

        renamed_metrics = {}
        for k, v in metrics.items():
            new_k = rename_key(k)
            # Prevent accidental collisions (keep first, warn on conflict)
            if new_k in renamed_metrics and new_k != k:
                LOGGER.warning(f"Metric key collision after renaming: '{k}' -> '{new_k}' (keeping first value).")
                continue
            renamed_metrics[new_k] = v

        keys, vals = list(renamed_metrics.keys()), list(renamed_metrics.values())
        n = len(renamed_metrics) + 2
        header_needed = not self.csv.exists()
        if header_needed:
            header = ["epoch", "time"] + keys
            with open(self.csv, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")

        t = time.time() - self.train_time_start
        row = [self.epoch + 1, t] + vals
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(",".join(f"{x:.6g}" if isinstance(x, (int, float)) else str(x) for x in row) + "\n")

    def plot_metrics(self):
        pass

    def on_plot(self, name, data=None):
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)
                for k in ("imgsz", "batch", "device", "close_mosaic"):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            # Accept either a Module or a state_dict in 'ema'
            ema_src = ckpt["ema"]
            try:
                state_dict = ema_src.state_dict() if hasattr(ema_src, "state_dict") else ema_src
                self.ema.ema.load_state_dict(state_dict)
                self.ema.updates = ckpt.get("updates", 0)
            except Exception as e:
                LOGGER.warning(f"Failed to load EMA from checkpoint: {e}")
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1].append(param)
                else:
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer