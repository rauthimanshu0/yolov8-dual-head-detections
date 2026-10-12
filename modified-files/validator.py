# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validation engine — dual‑head aware per‑axis confusion matrices + optional latent analysis
Location: /ultralytics/engine/validator.py
=======================================================

Overview
- Provides a flexible BaseValidator used by task validators.
- Adds first‑class support for Dual‑Head detection (two axes, e.g., Taxonomy/State) by:
  - Building per‑axis confusion matrices from joint predictions.
  - Inferring or using provided mappings between joint classes and per‑axis ids.
- Integrates an optional latent space collector controlled by environment variables to analyze learned embeddings during validation without affecting core metrics.

Key capabilities
1) Dual‑Head metadata discovery
   - Reads head sizes from model head attributes (num_cls1, num_cls2) when available.
   - Falls back to dataset metadata:
       data['names_head1'], data['names_head2'] (counts and names)
     and constructs a combined joint class map data['names'] of size C1*C2 when needed.
   - If neither is available, attempts to infer class counts from Nx2 GT labels in the val loader.

2) Joint ↔ per‑axis mapping (sparse‑aware)
   - Preferred: data['joint_to_heads'] as a list of [h1, h2] per joint index j.
   - Alternate: data['heads_to_joint'] as a dict "h1_h2" -> j; converted into dense joint→head tensors.
   - Fallback: arithmetic mapping using sizes: h1 = j // C2, h2 = j % C2.

3) Per‑axis confusion matrices
   - Initializes ConfusionMatrix instances for head‑1 and head‑2 using names_head1/names_head2 (or synthetic names).
   - Updates per image by:
       - Selecting the best GT xyxy in pixel space (several candidate interpretations) to maximize IoU with predictions.
       - Mapping ground‑truth joint labels to per‑axis GT: (gt_h1, gt_h2) via joint→head mapping.
       - Decoding predicted axes from joint predictions:
           • If a mapping exists → direct lookup
           • Else infer a decode order once ("h1_first" vs "h2_first") by comparing match counts, then use:
             h1_first: p_h1 = j // C2, p_h2 = j % C2
             h2_first: p_h1 = j % C1, p_h2 = j // C1
   - Saves both raw and normalized per‑axis confusion matrices under save_dir/{h1,h2}/.

4) Two‑stage NMS for dual‑head models
   - postprocess() calls non_max_suppression with two_stage_object=True and object_score_mode="product",
     preserving extras to enable dual‑head‑friendly filtering, and species‑aware duplicate cleanup.

5) Latent space analysis (opt‑in)
   - Enable by setting LATENT=1 in environment.
   - Collects labels (joint ids; constructs from Nx2 when necessary) and leverages LatentCollector
     for epoch‑wise capture and visualization (configurable via LATENT_* env vars).
   - Injects derived latent metrics into the returned stats dict at the end of validation.

6) Training vs standalone validation
   - When called from a trainer:
       • Uses EMA model if present, disables half, and toggles plots only near stopping/final epoch.
   - Standalone:
       • Uses AutoBackend, honors half precision, sets batch/rect appropriately, warms up model,
         builds dataloader and dataset metadata.

7) Hooks and extensibility
   - The following remain intentionally abstract for task‑specific validators to implement:
       update_metrics(), finalize_metrics(), get_stats(), print_results(), plot_val_samples(), plot_predictions(),
       pred_to_json(), eval_json(), get_dataloader(), build_dataset().
   - Provides helper methods for device alignment, GT interpretation, IoU‑based matching, and per‑axis CM updates.

Inputs and expectations
- Model head (if available) exposes num_cls1/num_cls2 for axis cardinalities.
- Dataset dict may include:
    names_head1: {id: name}, names_head2: {id: name}
    joint_to_heads: List[[h1_id, h2_id] for joint_id]
    or heads_to_joint: {"h1_h2": joint_id}
- Predictions returned to postprocess() follow standard layout:
    [x1,y1,x2,y2, conf, joint_cls, extras...] per detection (extras preserved by NMS).
- Ground‑truth labels may be:
    • combined joint ids (N,), or
    • per‑axis pairs (N, 2) → converted to combined using C2.

Performance and numerics
- Uses vectorized PyTorch ops for mapping, IoU, and matching where possible.
- Robust dtype/device alignment for batches and model.
- Debug toggles:
    PER_HEAD_CM_DEBUG, PER_HEAD_CM_DUMP for verbose logging and artifact inspection.

Backwards compatibility
- If dual‑head metadata is not present (single‑head models), per‑axis logic remains dormant.
- Latent analysis is entirely opt‑in and imposes no overhead when disabled.

"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile, non_max_suppression
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils.metrics import ConfusionMatrix, box_iou

# Latent collector (new)
try:
    from ultralytics.engine.latent_utils import LatentCollector
except Exception:
    # If relative import style
    from .latent_utils import LatentCollector  # type: ignore

_PER_HEAD_CM_DEBUG = os.getenv("PER_HEAD_CM_DEBUG", "0") not in {"", "0", "false", "False"}
_PER_HEAD_CM_DUMP = os.getenv("PER_HEAD_CM_DUMP", "0") not in {"", "0", "false", "False"}


class BaseValidator:
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.01 if self.args.task == "obb" else 0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # Dual-head info
        self._dual_num_cls1: Optional[int] = None
        self._dual_num_cls2: Optional[int] = None

        # Per-head CMs
        self.cm_h1: Optional[ConfusionMatrix] = None
        self.cm_h2: Optional[ConfusionMatrix] = None
        self._per_head_cm_inited = False

        # joint->heads mapping (sparse-aware)
        self._j2h1: Optional[torch.Tensor] = None
        self._j2h2: Optional[torch.Tensor] = None

        # Dense fallback decode order
        self._decode_order: Optional[str] = None

        # Latent (will be created lazily in __call__)
        self._latent: Optional[LatentCollector] = None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = False

            ema_obj = getattr(trainer, "ema", None)
            ema_model = getattr(ema_obj, "ema", None) if ema_obj is not None else None
            model = ema_model or trainer.model

            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)

            try:
                model.eval()
            except RecursionError:
                LOGGER.warning("RecursionError in model.eval(); continuing.")

            self._maybe_set_dual_info_and_names_from_model(model)
            self._maybe_build_joint_to_heads_map()
            if self._dual_num_cls1 is None or self._dual_num_cls2 is None:
                try:
                    if self.dataloader is not None:
                        self._infer_dual_from_loader_and_apply_names(model)
                        self._maybe_build_joint_to_heads_map()
                except Exception as e:
                    LOGGER.warning(f"Could not infer dual-head sizes from loader (train val): {e}")

        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("Validating an untrained model YAML → expect 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            try:
                model.eval()
            except RecursionError:
                LOGGER.warning("RecursionError in model.eval() (standalone val).")
            try:
                model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))
            except Exception:
                pass

            self._maybe_set_dual_info_and_names_from_model(model, fallback_to_data=True)
            self._maybe_build_joint_to_heads_map()
            if self._dual_num_cls1 is None or self._dual_num_cls2 is None:
                try:
                    self._infer_dual_from_loader_and_apply_names(model)
                    self._maybe_build_joint_to_heads_map()
                except Exception as e:
                    LOGGER.warning(f"Could not infer dual-head sizes (standalone val): {e}")

        # Latent collector lazy init
        if self._latent is None:
            try:
                total_epochs = getattr(trainer, "epochs", None) if trainer is not None else 1
                self._latent = LatentCollector(de_parallel(model), self.save_dir, total_epochs)
            except Exception as e:
                LOGGER.warning(f"[Latent] initialization failed: {e}")
                self._latent = None

        self.run_callbacks("on_val_start")
        dt = (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        # Latent schedule start
        if self._latent and self._latent.enabled:
            epochs_total = getattr(trainer, "epochs", 1) if trainer is not None else 1
            self._latent.build_schedule(epochs_total)
            current_epoch = getattr(trainer, "epoch", 0) if trainer is not None else 0
            self._latent.start_epoch(current_epoch)

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            with dt[0]:
                batch = self.preprocess(batch)
                batch = self._to_device_and_align(batch, model)

            with dt[1]:
                img = batch["img"]
                preds = model(img, augment=augment)

            # ---- LATENT CAPTURE (added) ----
            if self._latent and self._latent.enabled and self._latent.active:
                joint_targets = None
                if "cls_flat" in batch:
                    joint_targets = batch["cls_flat"]
                elif "cls" in batch and torch.is_tensor(batch["cls"]) and batch["cls"].ndim == 1:
                    joint_targets = batch["cls"]
                elif "cls" in batch and torch.is_tensor(batch["cls"]) and batch["cls"].ndim == 2 and batch["cls"].shape[1] == 2:
                    # combine (species, stage)
                    n2 = self._dual_num_cls2
                    if n2 is None:
                        try:
                            n2 = int(batch["cls"][:, 1].max().item()) + 1
                            self._dual_num_cls2 = n2
                        except Exception:
                            n2 = 1
                    joint_targets = (batch["cls"][:, 0].long() * int(n2) + batch["cls"][:, 1].long()).view(-1)
                if joint_targets is not None:
                    try:
                        self._latent.add_batch(joint_targets, num_cls2=self._dual_num_cls2)
                    except Exception as e:
                        LOGGER.warning(f"[Latent] add_batch error: {e}")
            # ---- END LATENT CAPTURE ----

            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            metrics_batch = batch
            if "cls_flat" in batch:
                metrics_batch = dict(batch)
                metrics_batch["cls"] = batch["cls_flat"]

            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, metrics_batch)

            if not self._per_head_cm_inited and self._dual_num_cls1 and self._dual_num_cls2:
                self._init_per_head_cms()

            if self.cm_h1 is not None and self.cm_h2 is not None:
                try:
                    self._update_dual_head_confusion_matrices(preds, metrics_batch)
                except Exception as e:
                    LOGGER.warning(f"[dual-conf-matrix] update error: {e}")

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(metrics_batch, batch_i)
                self.plot_predictions(metrics_batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        self._save_per_head_confusion_matrices()

        # ---- LATENT FINALIZE (added) ----
        if self._latent and self._latent.enabled and self._latent.active:
            try:
                epoch_idx = getattr(trainer, "epoch", 0) if trainer is not None else 0
                # Joint names
                names_joint = self.names if isinstance(self.names, dict) else {}
                # Axis names if present in data
                names_h1 = {}
                names_h2 = {}
                if isinstance(self.data, dict):
                    nh1 = self.data.get("names_head1", {})
                    nh2 = self.data.get("names_head2", {})
                    if isinstance(nh1, dict):
                        names_h1 = {int(k): str(v) for k, v in nh1.items()}
                    if isinstance(nh2, dict):
                        names_h2 = {int(k): str(v) for k, v in nh2.items()}
                latent_metrics = self._latent.stop_epoch_and_analyze(epoch_idx, names_joint, names_h1, names_h2)
                stats.update({k: float(v) for k, v in latent_metrics.items()})
            except Exception as e:
                LOGGER.warning(f"[Latent] finalize failed: {e}")
        # ---- END LATENT FINALIZE ----

        if self.training:
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def preprocess(self, batch: Dict[str, Any]):
        return batch

    # ---------- joint→heads mapping ----------
    def _maybe_build_joint_to_heads_map(self):
        self._j2h1 = None
        self._j2h2 = None
        try:
            if not isinstance(self.data, dict):
                return
            j2h = self.data.get("joint_to_heads", None)
            if isinstance(j2h, list) and len(j2h) > 0 and all(isinstance(t, (list, tuple)) and len(t) == 2 for t in j2h):
                arr = np.asarray(j2h, dtype=np.int64)
                self._j2h1 = torch.as_tensor(arr[:, 0], dtype=torch.long, device=self.device)
                self._j2h2 = torch.as_tensor(arr[:, 1], dtype=torch.long, device=self.device)
                if _PER_HEAD_CM_DEBUG:
                    LOGGER.info(f"[PerHeadCM] Using joint_to_heads mapping (nc={len(j2h)}).")
                return
            htj = self.data.get("heads_to_joint", None)
            if isinstance(htj, dict) and len(htj) > 0:
                pairs = []
                max_joint = -1
                for k, v in htj.items():
                    try:
                        h1, h2 = k.split("_")
                        h1, h2 = int(h1), int(h2)
                        j = int(v)
                    except Exception:
                        continue
                    pairs.append((j, h1, h2))
                    if j > max_joint:
                        max_joint = j
                if pairs and max_joint >= 0:
                    j2h1 = torch.full((max_joint + 1,), -1, dtype=torch.long, device=self.device)
                    j2h2 = torch.full((max_joint + 1,), -1, dtype=torch.long, device=self.device)
                    for j, h1, h2 in pairs:
                        j2h1[j] = h1
                        j2h2[j] = h2
                    self._j2h1, self._j2h2 = j2h1, j2h2
                    if _PER_HEAD_CM_DEBUG:
                        LOGGER.info(f"[PerHeadCM] Built mapping from heads_to_joint (len={max_joint+1}).")
        except Exception as e:
            LOGGER.warning(f"[PerHeadCM] Failed building joint→heads mapping: {e}")

    def _map_joint_ids_to_axes(self, joint_ids: torch.Tensor, n1: int, n2: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if joint_ids.numel() == 0:
            z = torch.zeros((0,), dtype=torch.long, device=joint_ids.device)
            return z, z
        if isinstance(self._j2h1, torch.Tensor) and isinstance(self._j2h2, torch.Tensor):
            max_ok = min(self._j2h1.numel(), self._j2h2.numel()) - 1
            ids = joint_ids.clamp(min=0, max=max_ok)
            a1 = self._j2h1[ids]
            a2 = self._j2h2[ids]
            a1 = torch.where(a1 >= 0, a1, torch.zeros_like(a1))
            a2 = torch.where(a2 >= 0, a2, torch.zeros_like(a2))
            return a1.long(), a2.long()
        a1 = joint_ids // n2
        a2 = joint_ids % n2
        return a1.long(), a2.long()
    # -----------------------------------------

    def _infer_dual_from_loader_and_apply_names(self, model=None):
        if self.dataloader is None:
            return
        max1, max2 = -1, -1
        for b in self.dataloader:
            cls = b.get("cls", None)
            if cls is None:
                continue
            if isinstance(cls, np.ndarray):
                cls = torch.from_numpy(cls)
            if torch.is_tensor(cls) and cls.ndim == 2 and cls.shape[1] == 2:
                if cls.numel() == 0:
                    continue
                m1 = int(cls[:, 0].max().item()) if cls[:, 0].numel() else -1
                m2 = int(cls[:, 1].max().item()) if cls[:, 1].numel() else -1
                max1 = max(max1, m1)
                max2 = max(max2, m2)
        if max1 >= 0 and max2 >= 0:
            self._dual_num_cls1 = (self._dual_num_cls1 or 0)
            self._dual_num_cls2 = (self._dual_num_cls2 or 0)
            self._dual_num_cls1 = max(self._dual_num_cls1, max1 + 1)
            self._dual_num_cls2 = max(self._dual_num_cls2, max2 + 1)

            n1, n2 = int(self._dual_num_cls1), int(self._dual_num_cls2)
            total = n1 * n2
            names_ok = isinstance(self.data, dict) and isinstance(self.data.get("names"), dict) and len(self.data["names"]) == total
            if not names_ok:
                nh1 = self.data.get("names_head1", {}) if isinstance(self.data, dict) else {}
                nh2 = self.data.get("names_head2", {}) if isinstance(self.data, dict) else {}

                def sorted_vals(dic: Dict[str, str]) -> List[str]:
                    keys = list(dic.keys())
                    try:
                        keys = sorted(keys, key=lambda k: int(k))
                    except Exception:
                        keys = sorted(keys)
                    return [str(dic[k]) for k in keys]

                if isinstance(nh1, dict) and len(nh1) == n1 and isinstance(nh2, dict) and len(nh2) == n2:
                    l1 = sorted_vals(nh1)
                    l2 = sorted_vals(nh2)
                else:
                    l1 = [f"h1_{i}" for i in range(n1)]
                    l2 = [f"h2_{j}" for j in range(n2)]

                combined = {i * n2 + j: f"{l1[i]} - {l2[j]}" for i in range(n1) for j in range(n2)}
                try:
                    self.data["names"] = combined
                    self.data["nc"] = total
                except Exception:
                    pass
                try:
                    if hasattr(model, "names"):
                        model.names = combined
                except Exception:
                    pass

    def _to_device_and_align(self, batch: Dict[str, Any], model=None) -> Dict[str, Any]:
        dev = self.device
        use_half = bool(self.args.half)
        img_dtype = torch.float16 if use_half else torch.float32

        model_dtype = None
        if model is not None and hasattr(model, "parameters"):
            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = None

        new_batch: Dict[str, Any] = {}
        cls_tensor: Optional[torch.Tensor] = None
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if torch.is_tensor(v):
                v = v.to(dev, non_blocking=True)
                if k == "img":
                    target_dtype = model_dtype or img_dtype
                    if v.dtype != target_dtype:
                        v = v.to(target_dtype, non_blocking=True)
                if k == "cls":
                    cls_tensor = v
                new_batch[k] = v
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                new_batch[k] = type(v)(t.to(dev, non_blocking=True) for t in v)
            else:
                new_batch[k] = v

        if cls_tensor is not None and cls_tensor.ndim == 2 and cls_tensor.shape[1] == 2:
            n2 = self._dual_num_cls2
            if n2 is None and isinstance(self.data, dict):
                nh2 = self.data.get("names_head2", {})
                if isinstance(nh2, dict) and len(nh2) > 0:
                    n2 = len(nh2)
            if n2 is None and cls_tensor.numel():
                n2 = int(cls_tensor[:, 1].max().item()) + 1
                self._dual_num_cls2 = n2
            if n2 is not None:
                cls1 = cls_tensor[:, 0].long()
                cls2 = cls_tensor[:, 1].long()
                new_batch["cls_flat"] = (cls1 * int(n2) + cls2).view(-1)
            else:
                LOGGER.warning("Dual-head Nx2 labels but num_cls2 unresolved; confusion matrices may fail.")
        return new_batch

    def _maybe_set_dual_info_and_names_from_model(self, model, fallback_to_data: bool = False):
        num_cls1 = None
        num_cls2 = None
        try:
            m = de_parallel(model)
            head_list = getattr(m, "model", None)
            if head_list:
                last = head_list[-1]
                num_cls1 = getattr(last, "num_cls1", None)
                num_cls2 = getattr(last, "num_cls2", None)
        except Exception:
            pass

        if (num_cls1 is None or num_cls2 is None) and fallback_to_data and isinstance(self.data, dict):
            nh1 = self.data.get("names_head1", {})
            nh2 = self.data.get("names_head2", {})
            if isinstance(nh1, dict) and len(nh1) and isinstance(nh2, dict) and len(nh2):
                num_cls1 = len(nh1)
                num_cls2 = len(nh2)

        self._dual_num_cls1 = num_cls1
        self._dual_num_cls2 = num_cls2

        if isinstance(self.data, dict) and num_cls1 and num_cls2:
            total = int(num_cls1) * int(num_cls2)
            nh1 = self.data.get("names_head1", {})
            nh2 = self.data.get("names_head2", {})

            def sorted_vals(dic: Dict[str, str]) -> List[str]:
                keys = list(dic.keys())
                try:
                    keys = sorted(keys, key=lambda k: int(k))
                except Exception:
                    keys = sorted(keys)
                return [str(dic[k]) for k in keys]

            if isinstance(nh1, dict) and len(nh1) == num_cls1 and isinstance(nh2, dict) and len(nh2) == num_cls2:
                l1 = sorted_vals(nh1)
                l2 = sorted_vals(nh2)
            else:
                l1 = [f"h1_{i}" for i in range(int(num_cls1))]
                l2 = [f"h2_{j}" for j in range(int(num_cls2))]
            combined = {i * int(num_cls2) + j: f"{l1[i]} - {l2[j]}" for i in range(int(num_cls1)) for j in range(int(num_cls2))}
            try:
                self.data["names"] = combined
                self.data["nc"] = total
            except Exception:
                pass
            try:
                if hasattr(model, "names"):
                    model.names = combined
            except Exception:
                pass

    def _joint_nc(self) -> int:
        if self._dual_num_cls1 and self._dual_num_cls2:
            return int(self._dual_num_cls1) * int(self._dual_num_cls2)
        if isinstance(self.data, dict):
            if isinstance(self.data.get("names"), dict):
                return len(self.data["names"])
            if isinstance(self.data.get("nc"), (int, np.integer)):
                return int(self.data["nc"])
        return 0

    def postprocess(self, preds):
        task = str(getattr(self.args, "task", "detect"))
        if task not in {"detect", "obb"}:
            return preds
        y = preds[0] if isinstance(preds, (list, tuple)) else preds
        joint_nc = self._joint_nc()
        rotated = task == "obb"
        dets = non_max_suppression(
            y,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=joint_nc,
            rotated=rotated,
            # Two-stage object-based NMS settings (unchanged existing functionality)
            two_stage_object=True,
            stage1_conf_thres=self.args.conf,
            stage1_iou_thres=self.args.iou,
            stage2_iou_thres=self.args.iou,
            object_score_mode="product",
            species_cleanup=True,
            species_col_in_extras=0,
            num_cls2_for_species=None,
        )
        return dets

    def init_metrics(self, model):
        if self._dual_num_cls1 and self._dual_num_cls2 and not self._per_head_cm_inited:
            self._init_per_head_cms()

    def _init_per_head_cms(self):
        def _extract_names(key, count, prefix):
            if isinstance(self.data, dict):
                src = self.data.get(key)
                if isinstance(src, dict) and len(src) == count:
                    try:
                        return [str(src[i]) for i in range(count)]
                    except Exception:
                        try:
                            return [str(v) for _, v in sorted(src.items(), key=lambda kv: int(kv[0]))]
                        except Exception:
                            return [str(v) for v in src.values()]
            return [f"{prefix}_{i}" for i in range(count)]

        try:
            names_h1 = _extract_names("names_head1", int(self._dual_num_cls1), "h1")
            names_h2 = _extract_names("names_head2", int(self._dual_num_cls2), "h2")
            self.cm_h1 = ConfusionMatrix(names=names_h1, task="detect")
            self.cm_h2 = ConfusionMatrix(names=names_h2, task="detect")
            self._per_head_cm_inited = True
            if _PER_HEAD_CM_DEBUG:
                LOGGER.info(f"[PerHeadCM] Initialized h1({len(names_h1)}) h2({len(names_h2)})")
        except Exception as e:
            LOGGER.warning(f"Failed initializing per-head confusion matrices: {e}")
            self.cm_h1 = self.cm_h2 = None

    # NOTE: The following methods were stubs in your original file; they remain unchanged to preserve behavior.
    def update_metrics(self, preds, batch):
        pass

    def finalize_metrics(self):
        pass

    def get_stats(self):
        return {}

    def print_results(self):
        pass

    def get_desc(self):
        return "Validating"

    @property
    def metric_keys(self):
        return []

    def match_predictions(self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False):
        device = iou.device
        if self.iouv is None or len(self.iouv) == 0:
            self.iouv = torch.linspace(0.5, 0.95, 10, device=device)
        else:
            self.iouv = self.iouv.to(device)

        npreds = int(pred_classes.numel())
        correct = torch.zeros(npreds, len(self.iouv), dtype=torch.bool, device=device)

        ngt = int(true_classes.numel())
        if npreds == 0 or ngt == 0 or iou.numel() == 0:
            return correct

        pc = pred_classes.to(torch.long).view(-1)
        tc = true_classes.to(torch.long).view(-1)

        for cls in torch.unique(tc).tolist():
            gt_mask = (tc == cls)
            pr_mask = (pc == cls)
            if not gt_mask.any() or not pr_mask.any():
                continue

            gt_idx = torch.where(gt_mask)[0]
            pr_idx = torch.where(pr_mask)[0]
            iou_sub = iou.index_select(0, gt_idx).index_select(1, pr_idx)
            if iou_sub.numel() == 0:
                continue

            for j, thr in enumerate(self.iouv):
                matches = torch.nonzero(iou_sub >= thr, as_tuple=False)
                if matches.numel() == 0:
                    continue
                ious = iou_sub[matches[:, 0], matches[:, 1]]
                order = torch.argsort(ious, descending=True)
                gt_taken = set()
                pr_taken = set()
                for k in order.tolist():
                    g = int(matches[k, 0].item())
                    p = int(matches[k, 1].item())
                    if g in gt_taken or p in pr_taken:
                        continue
                    gt_taken.add(g)
                    pr_taken.add(p)
                    correct[pr_idx[p], j] = True

        return correct

    # -------- GT interpretation helpers --------
    def _gt_xyxy_candidates_pix(self, gt_raw_xy: torch.Tensor, W: int, H: int) -> Dict[str, torch.Tensor]:
        dev = gt_raw_xy.device
        z = torch.zeros((0, 4), device=dev)
        if gt_raw_xy is None or gt_raw_xy.numel() == 0:
            return {"norm_xywh": z, "norm_xyxy": z, "pix_xywh": z, "pix_xyxy": z}

        def clamp_xyxy(t: torch.Tensor) -> torch.Tensor:
            if t.numel() == 0:
                return t
            x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            x1_, x2_ = torch.minimum(x1, x2), torch.maximum(x1, x2)
            y1_, y2_ = torch.minimum(y1, y2), torch.maximum(y1, y2)
            return torch.stack([x1_, y1_, x2_, y2_], 1)

        x, y, w, h = gt_raw_xy[:, 0] * W, gt_raw_xy[:, 1] * H, gt_raw_xy[:, 2] * W, gt_raw_xy[:, 3] * H
        a = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
        b = gt_raw_xy.clone()
        b[:, [0, 2]] *= W
        b[:, [1, 3]] *= H
        c = torch.stack(
            [gt_raw_xy[:, 0] - gt_raw_xy[:, 2] / 2, gt_raw_xy[:, 1] - gt_raw_xy[:, 3] / 2,
             gt_raw_xy[:, 0] + gt_raw_xy[:, 2] / 2, gt_raw_xy[:, 1] + gt_raw_xy[:, 3] / 2], 1
        )
        d = gt_raw_xy
        return {"norm_xywh": clamp_xyxy(a), "norm_xyxy": clamp_xyxy(b), "pix_xywh": clamp_xyxy(c), "pix_xyxy": clamp_xyxy(d)}

    def _select_best_gt_xyxy_pix(
        self, gt_raw_xy: torch.Tensor, pred_xyxy_pix: torch.Tensor, W: int, H: int, dbg_prefix: str = ""
    ) -> Tuple[str, torch.Tensor, float]:
        candidates = self._gt_xyxy_candidates_pix(gt_raw_xy, W, H)

        def score_iou(iou: torch.Tensor) -> float:
            if not torch.is_tensor(iou) or iou.numel() == 0:
                return 0.0
            row_max = iou.max(dim=1).values if iou.shape[0] > 0 else torch.zeros(0, device=iou.device)
            col_max = iou.max(dim=0).values if iou.shape[1] > 0 else torch.zeros(0, device=iou.device)
            med_row = float(row_max.median().item()) if row_max.numel() else 0.0
            med_col = float(col_max.median().item()) if col_max.numel() else 0.0
            return max(med_row, med_col)

        best_name, best_xyxy, best_score = "norm_xywh", candidates["norm_xywh"], 0.0
        scores: Dict[str, float] = {}
        for name, gt_xyxy_pix in candidates.items():
            if gt_xyxy_pix.numel() == 0 or pred_xyxy_pix.numel() == 0:
                s = 0.0
            else:
                iou = box_iou(gt_xyxy_pix, pred_xyxy_pix)
                s = score_iou(iou)
            scores[name] = s
            if s > best_score:
                best_name, best_xyxy, best_score = name, gt_xyxy_pix, s

        if _PER_HEAD_CM_DEBUG and dbg_prefix:
            n_gt = int(candidates["norm_xywh"].shape[0])
            n_pr = int(pred_xyxy_pix.shape[0]) if torch.is_tensor(pred_xyxy_pix) else 0
            LOGGER.info(f"[PerHeadCM] dbg {dbg_prefix}: n_gt={n_gt}, n_pred={n_pr}, scores={scores}")
        return best_name, best_xyxy, best_score

    def _update_dual_head_confusion_matrices(self, preds, batch: Dict[str, Any]):
        if self.cm_h1 is None or self.cm_h2 is None:
            return
        if self._dual_num_cls1 is None or self._dual_num_cls2 is None:
            return

        def _coerce_det(item) -> Optional[torch.Tensor]:
            if item is None:
                return None
            if torch.is_tensor(item):
                return item
            if isinstance(item, dict):
                b = item.get("bboxes", None)
                c = item.get("conf", None)
                k = item.get("cls", None)
                if b is None or c is None or k is None:
                    return None
                if not (torch.is_tensor(b) and torch.is_tensor(c) and torch.is_tensor(k)):
                    return None
                if b.numel() == 0:
                    return None
                c = c.view(-1, 1)
                k = k.view(-1, 1)
                try:
                    return torch.cat([b, c, k], dim=1)
                except Exception:
                    return None
            return None

        imgs = batch.get("img")
        gt_cls_comb = batch.get("cls")
        gt_boxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        if imgs is None or gt_cls_comb is None or gt_boxes is None or batch_idx is None:
            return

        if isinstance(batch_idx, list):
            batch_idx = torch.as_tensor(batch_idx, device=gt_boxes.device)

        if torch.is_tensor(gt_cls_comb) and gt_cls_comb.ndim == 2 and gt_cls_comb.shape[1] == 2:
            n2tmp = int(self._dual_num_cls2)
            gt_cls_comb = (gt_cls_comb[:, 0].long() * n2tmp + gt_cls_comb[:, 1].long()).view(-1)

        n1 = int(self._dual_num_cls1)
        n2 = int(self._dual_num_cls2)
        bs = imgs.shape[0]
        _, _, H, W = imgs.shape

        have_map = isinstance(self._j2h1, torch.Tensor) and isinstance(self._j2h2, torch.Tensor)

        if self._decode_order is None and not have_map:
            total_matches_a = 0
            total_matches_b = 0
            sample_bs = min(bs, 4)
            for i in range(sample_bs):
                m = (batch_idx == i)
                gt_inds = torch.where(m)[0] if torch.is_tensor(m) else []
                if gt_inds.numel() == 0:
                    continue
                gcls = gt_cls_comb[gt_inds].long()
                det_raw = preds[i] if i < len(preds) else None
                det = _coerce_det(det_raw)
                if det is None or det.shape[0] == 0:
                    continue
                pboxes_pix = det[:, :4]
                _, gxyxy_pix, _ = self._select_best_gt_xyxy_pix(
                    gt_boxes[gt_inds], pboxes_pix, W, H, dbg_prefix="order_infer"
                )
                g_h1 = (gcls // n2)
                g_h2 = (gcls % n2)
                c = det[:, 5].long()
                p_h1_a = c // n2
                p_h2_a = c % n2
                p_h1_b = c % n1
                p_h2_b = c // n1
                iou = box_iou(gxyxy_pix, pboxes_pix) if gxyxy_pix.numel() and pboxes_pix.numel() else torch.zeros(0)
                thr = float(self.args.iou or 0.45)
                def count_matches(pred_axis: torch.Tensor, gt_axis: torch.Tensor) -> int:
                    if iou.numel() == 0:
                        return 0
                    g_idx, p_idx = torch.where(iou >= thr)
                    if g_idx.numel() == 0:
                        return 0
                    ious = iou[g_idx, p_idx]
                    order = torch.argsort(ious, descending=True)
                    g_idx = g_idx[order]
                    p_idx = p_idx[order]
                    matched_g, matched_p = set(), set()
                    cnt = 0
                    for gi, pi in zip(g_idx.tolist(), p_idx.tolist()):
                        if gi in matched_g or pi in matched_p:
                            continue
                        if int(gt_axis[gi].item()) == int(pred_axis[pi].item()):
                            matched_g.add(gi)
                            matched_p.add(pi)
                            cnt += 1
                    return cnt
                total_matches_a += count_matches(p_h1_a, g_h1) + count_matches(p_h2_a, g_h2)
                total_matches_b += count_matches(p_h1_b, g_h1) + count_matches(p_h2_b, g_h2)
            self._decode_order = "h2_first" if total_matches_b > total_matches_a else "h1_first"
            if _PER_HEAD_CM_DEBUG:
                LOGGER.info(
                    f"[PerHeadCM] decode order inferred: {self._decode_order} "
                    f"(h1_first={total_matches_a}, h2_first={total_matches_b})"
                )

        for i in range(bs):
            m = (batch_idx == i)
            gt_inds = torch.where(m)[0] if torch.is_tensor(m) else []
            det_raw = preds[i] if i < len(preds) else None
            det = _coerce_det(det_raw)
            if det is None or det.shape[0] == 0:
                det_boxes = torch.zeros((0, 4), device=imgs.device)
                det_conf = torch.zeros((0,), device=imgs.device)
                p_joint = torch.zeros((0,), dtype=torch.long, device=imgs.device)
            else:
                det_boxes = det[:, :4]
                det_conf = det[:, 4]
                p_joint = det[:, 5].long()
            if gt_inds.numel():
                gcls_joint = gt_cls_comb[gt_inds].long()
                _, gxyxy_pix, _ = self._select_best_gt_xyxy_pix(
                    gt_boxes[gt_inds], det_boxes, W, H, dbg_prefix=""
                )
                gcls1, gcls2 = self._map_joint_ids_to_axes(gcls_joint, n1, n2)
                labels_h1 = {"cls": gcls1, "bboxes": gxyxy_pix}
                labels_h2 = {"cls": gcls2, "bboxes": gxyxy_pix}
            else:
                z4 = torch.zeros((0, 4), device=imgs.device)
                zc = torch.zeros((0,), dtype=torch.long, device=imgs.device)
                labels_h1 = {"cls": zc, "bboxes": z4}
                labels_h2 = {"cls": zc, "bboxes": z4}
            if p_joint.numel() == 0:
                pcls1 = torch.zeros((0,), dtype=torch.long, device=imgs.device)
                pcls2 = torch.zeros((0,), dtype=torch.long, device=imgs.device)
            else:
                if have_map:
                    pcls1, pcls2 = self._map_joint_ids_to_axes(p_joint, n1, n2)
                else:
                    if self._decode_order == "h2_first":
                        pcls1 = p_joint % n1
                        pcls2 = p_joint // n1
                    else:
                        pcls1 = p_joint // n2
                        pcls2 = p_joint % n2
            det_dict_h1 = {"cls": pcls1, "conf": det_conf, "bboxes": det_boxes}
            det_dict_h2 = {"cls": pcls2, "conf": det_conf, "bboxes": det_boxes}
            try:
                self.cm_h1.process_batch(det_dict_h1, labels_h1, conf=0.0, iou_thres=self.args.iou)
                self.cm_h2.process_batch(det_dict_h2, labels_h2, conf=0.0, iou_thres=self.args.iou)
            except Exception as e:
                LOGGER.warning(f"[dual-conf-matrix] image {i} failed: {e}")

    def _save_per_head_confusion_matrices(self):
        if self.cm_h1 is None or self.cm_h2 is None:
            return
        for tag, cm in (("h1", self.cm_h1), ("h2", self.cm_h2)):
            out_dir = self.save_dir / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                cm.plot(normalize=False, save_dir=str(out_dir))
            except Exception as e:
                LOGGER.warning(f"[PerHeadCM] raw plot failed ({tag}): {e}")
            try:
                cm.plot(normalize=True, save_dir=str(out_dir))
            except Exception as e:
                LOGGER.warning(f"[PerHeadCM] normalized plot failed ({tag}): {e}")
            if _PER_HEAD_CM_DEBUG:
                pngs = list(out_dir.glob("*.png"))
                LOGGER.info(f"[PerHeadCM] {tag} PNGs: {[p.name for p in pngs]}")

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        raise NotImplementedError

    def build_dataset(self, img_path):
        raise NotImplementedError

    def plot_val_samples(self, batch, ni):
        pass

    def plot_predictions(self, batch, preds, ni):
        pass

    def pred_to_json(self, preds, batch):
        pass

    def eval_json(self, stats):
        return stats