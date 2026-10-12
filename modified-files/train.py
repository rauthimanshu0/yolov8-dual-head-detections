# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Dual‑Head–aware DetectionTrainer (Taxonomy & State)
Location: /ultralytics/models/yolo/detect/train.py
==================================================

Purpose
- Training entrypoint for YOLO detection with first‑class support for dual‑axis labeling (e.g., Taxonomy & State).
- Extends BaseTrainer and wires in dataset building, dual‑head preprocessing, validator, logging, and plots.

Key behaviors and dual‑head specifics
- Dataset & DataLoader:
  - build_dataset/build_dataloader mirror Ultralytics defaults (rect for val; shuffle rules; workers scaling).
  - Uses model stride for grid size and passes args/data through build_yolo_dataset.

- Preprocess:
  - Calls BaseTrainer.preprocess_batch first to preserve/restore split labels (cls1/cls2) built upstream.
  - Normalizes images to float [0,1] and supports multi_scale resizing snapped to model stride.

- Model attributes and metadata:
  - Sets model.nc, model.names, and model.args from data/args.
  - Defines custom loss_names to match the logging vector returned by v8DetectionLoss:
      ["bbox", "dfl", "regression", "cls_Taxonomy", "cls_substrate", "overall"]
  - Propagates per‑axis names to the model and head (names_head1, names_head2) and, when possible, sets
    head.num_cls1/num_cls2 for downstream decoding and analysis.

- Model factory:
  - get_model returns a DetectionModel (constructed with cfg/nc/ch), and optionally loads weights.

- Validation with dual‑axis metrics:
  - get_validator returns DualHeadDetectionValidator with plots enabled by default.
  - Attaches names_head1/names_head2 to the validator if present; pre‑initializes with set_model()
    so axis metric keys (h1/*, h2/*) are registered before results.csv is created.

- Loss logging and progress:
  - label_loss_items maps the 6‑component loss vector to train/* keys in the same order as v8DetectionLoss logging.
  - progress_string formats a row with GPU mem + 6 custom loss columns + instances and size.

- Visualization:
  - plot_training_samples uses plot_images with names=self.data["names"] so combined GT labels like "Species - Stage"
    render cleanly (and extras are supported when present).
  - plot_metrics calls plot_results_priority to always render four priority panels:
      Loss (bbox, dfl, cls_Taxonomy, cls_substrate, overall),
      Precision (Taxonomy Dual/ Substrate Dual),
      Recall (Taxonomy Dual/ Substrate Dual),
      mAP50 (Taxonomy Dual/ Substrate Dual),
    and optionally includes legacy results.png.
  - plot_training_labels uses plot_labels with dataset names for class histograms and box stats.

- Auto batch sizing:
  - Estimates a safe maximum objects per image (×4 for mosaic) to compute an optimal batch size.

Why this matters
- Keeps the standard YOLO training surface while enabling dual‑head losses, axis‑wise validation/plots, and clear
  logging that separates regression vs per‑axis classification terms.
"""

import math
import random
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection including dataset building, data loading, preprocessing, and model configuration.
    """

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        use_rect = (mode == "val") or bool(getattr(self.args, "rect", False) and mode == "train")
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=use_rect, stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch: Dict) -> Dict:
        """
        Preprocess a batch of images by scaling and converting to float.
        Also preserves dual-head label splitting (cls1/cls2) from BaseTrainer.
        """
        # Keep dual-head cls1/cls2 reconstruction from BaseTrainer
        batch = super().preprocess_batch(batch)

        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        super().set_model_attributes()
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

        # NEW: set loss_names for logging the 4 custom components produced by v8DetectionLoss logging vector
        # Order must match logging_items in loss.py
        self.loss_names = ["bbox", "dfl", "regression", "cls_genus", "cls_substrate", "overall"]

        # Propagate split-head metadata
        nh1 = self.data.get("names_head1", {})
        nh2 = self.data.get("names_head2", {})
        try:
            nh1 = {int(k): str(v) for k, v in nh1.items()} if isinstance(nh1, dict) else nh1
            nh2 = {int(k): str(v) for k, v in nh2.items()} if isinstance(nh2, dict) else nh2
        except Exception:
            pass
        setattr(self.model, "names_head1", nh1)
        setattr(self.model, "names_head2", nh2)
        n1 = len(nh1) if isinstance(nh1, dict) else 0
        n2 = len(nh2) if isinstance(nh2, dict) else 0
        try:
            head = self.model.model[-1]
            setattr(head, "num_cls1", n1)
            setattr(head, "num_cls2", n2)
        except Exception:
            pass

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        from ultralytics.engine.dual_head_validator import DualHeadDetectionValidator

        args = copy(self.args)
        args.plots = getattr(self.args, "plots", True)
        v = DualHeadDetectionValidator(dataloader=self.test_loader, save_dir=self.save_dir, args=args)

        # Provide split names to the validator explicitly
        try:
            nh1 = getattr(self.model, "names_head1", None)
            nh2 = getattr(self.model, "names_head2", None)
            if isinstance(nh1, dict) and len(nh1):
                setattr(v, "names_head1", nh1)
            if isinstance(nh2, dict) and len(nh2):
                setattr(v, "names_head2", nh2)
        except Exception:
            pass

        # Pre-initialize with current model so metric keys (h1/*, h2/*) register before results.csv header is written
        try:
            v.set_model(self.model)
        except Exception as e:
            LOGGER.warning(f"Validator set_model pre-init failed (continuing): {e}")
        return v

    def label_loss_items(self, loss_items: Optional[List[float]] = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Adds names=self.data["names"] so GT labels render nicely ("Species - Stage").
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            names=self.data.get("names", {}),  # optional; cosmetic
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """
        Plot the standard results (optional) plus the priority panels requested:
          - Loss (bbox, dfl, cls_genus, cls_substrate, overall)
          - Precision (Genus Dual, Substrate Dual)
          - Recall (Genus Dual, Substrate Dual)
          - mAP50 (Genus Dual, Substrate Dual)
        Then fill extra panels with other metrics if space remains.
        """
        from ultralytics.utils.plotting import plot_results_priority
        # include_original=True if you still want the legacy results.png; set False to skip.
        plot_results_priority(
            file=self.csv,
            save_dir=self.save_dir,
            smoothing=0.0,          # change to e.g. 0.1 for light smoothing
            total_panels=8,          # at least 4; increase if you want more extra metrics
            max_cols=4,
            dpi=220,
            include_original=True,   # produce both results.png and results_priority.png
            on_plot=self.on_plot
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)