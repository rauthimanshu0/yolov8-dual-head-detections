# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validation engine with guarded eval(), robust preprocessing, dual-head (Nx2) class support for metrics,
pre-scan inference of dual-head cardinalities from the dataloader, and a concrete class-aware IoU matcher.

Now extended to produce per-head (h1 and h2) confusion matrices in addition to any combined metrics.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile, non_max_suppression
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils.metrics import ConfusionMatrix  


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
        self.confusion_matrix = None  # existing (combined) matrix if used by derived classes
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

        # Dual-head info (optional)
        self._dual_num_cls1: Optional[int] = None
        self._dual_num_cls2: Optional[int] = None

        # Per-head confusion matrices (initialized lazily in init_metrics)
        self.cm_h1: Optional[ConfusionMatrix] = None
        self.cm_h2: Optional[ConfusionMatrix] = None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)

        if self.training:
            self.device = trainer.device
            self.data = trainer.data

            # Keep validation FP32 during training to avoid Half input vs Float weights issues
            self.args.half = False

            # Prefer EMA if available; else use training model
            ema_obj = getattr(trainer, "ema", None)
            ema_model = getattr(ema_obj, "ema", None) if ema_obj is not None else None
            model = ema_model or trainer.model

            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)

            # Guarded eval
            try:
                model.eval()
            except RecursionError:
                LOGGER.warning("RecursionError in model.eval(); proceeding without switching to eval() for validation.")

            # Derive dual-head info and combined names for metrics from model or data
            self._maybe_set_dual_info_and_names_from_model(model)

            # If still unknown and we already have a dataloader, infer from dataloader labels
            if self._dual_num_cls1 is None or self._dual_num_cls2 is None:
                try:
                    if self.dataloader is not None:
                        self._infer_dual_from_loader_and_apply_names(model)
                except Exception as e:
                    LOGGER.warning(f"Could not infer dual-head sizes from loader during training validation: {e}")

        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
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

            # Load dataset info
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

            # Guarded eval for non-training path as well
            try:
                model.eval()
            except RecursionError:
                LOGGER.warning("RecursionError in model.eval() (standalone val); continuing in current mode.")
            try:
                model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))
            except Exception:
                pass

            # Infer dual-head info from model or dataset
            self._maybe_set_dual_info_and_names_from_model(model, fallback_to_data=True)
            if self._dual_num_cls1 is None or self._dual_num_cls2 is None:
                try:
                    self._infer_dual_from_loader_and_apply_names(model)
                except Exception as e:
                    LOGGER.warning(f"Could not infer dual-head sizes from loader during final eval: {e}")

        self.run_callbacks("on_val_start")
        dt = (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        # MAIN LOOP
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            with dt[0]:
                batch = self.preprocess(batch)
                batch = self._to_device_and_align(batch, model)

            with dt[1]:
                img = batch["img"]
                preds = model(img, augment=augment)

            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # For metrics, ensure cls is 1D if dual-head labels were provided (handled in _to_device_and_align)
            metrics_batch = batch
            if "cls_flat" in batch:
                metrics_batch = dict(batch)
                metrics_batch["cls"] = batch["cls_flat"]

            with dt[3]:
                preds = self.postprocess(preds)

            # Standard metrics update (implemented by derived classes)
            self.update_metrics(preds, metrics_batch)

            # ------------------ NEW: Per-head confusion matrix updates ------------------
            # We only attempt if dual-head sizes are known and ConfusionMatrix objects initialized.
            if self.cm_h1 is not None and self.cm_h2 is not None and self._dual_num_cls1 and self._dual_num_cls2:
                try:
                    self._update_dual_head_confusion_matrices(preds, batch)
                except Exception as e:
                    LOGGER.warning(f"[dual-conf-matrix] Skipping per-head confusion update due to error: {e}")
            # ---------------------------------------------------------------------------

            if self.args.plots and batch_i < 3:
                self.plot_val_samples(metrics_batch, batch_i)
                self.plot_predictions(metrics_batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        # Plot per-head confusion matrices
        if (self.cm_h1 is not None and self.cm_h2 is not None) and (self.args.plots or not self.training):
            try:
                (self.save_dir / "h1").mkdir(parents=True, exist_ok=True)
                (self.save_dir / "h2").mkdir(parents=True, exist_ok=True)
                self.cm_h1.plot(save_dir=self.save_dir / "h1")
                self.cm_h1.plot(save_dir=self.save_dir / "h1", normalize=True)
                self.cm_h2.plot(save_dir=self.save_dir / "h2")
                self.cm_h2.plot(save_dir=self.save_dir / "h2", normalize=True)
            except Exception as e:
                LOGGER.warning(f"Failed plotting per-head confusion matrices: {e}")

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
            if self._dual_num_cls1 is None:
                self._dual_num_cls1 = max1 + 1
            else:
                self._dual_num_cls1 = max(self._dual_num_cls1, max1 + 1)
            if self._dual_num_cls2 is None:
                self._dual_num_cls2 = max2 + 1
            else:
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

        # If dual-head class labels exist in Nx2 form, compute flattened labels for metrics only
        if cls_tensor is not None and cls_tensor.ndim == 2 and cls_tensor.shape[1] == 2:
            n2 = self._dual_num_cls2
            if n2 is None and isinstance(self.data, dict):
                nh2 = self.data.get("names_head2", {})
                if isinstance(nh2, dict) and len(nh2) > 0:
                    n2 = len(nh2)
            if n2 is None:
                if cls_tensor.numel():
                    n2 = int(cls_tensor[:, 1].max().item()) + 1
                    self._dual_num_cls2 = n2
            if n2 is not None:
                cls1 = cls_tensor[:, 0].to(torch.long)
                cls2 = cls_tensor[:, 1].to(torch.long)
                new_batch["cls_flat"] = (cls1 * int(n2) + cls2).view(-1)
            else:
                LOGGER.warning("Dual-head cls Nx2 detected but num_cls2 is unknown; metrics may fail.")

        return new_batch

    def _maybe_set_dual_info_and_names_from_model(self, model, fallback_to_data: bool = False):
        num_cls1 = None
        num_cls2 = None
        try:
            m = de_parallel(model)
            head_list = getattr(m, "model", None)
            if head_list is not None:
                try:
                    last = head_list[-1]
                    num_cls1 = getattr(last, "num_cls1", None)
                    num_cls2 = getattr(last, "num_cls2", None)
                except Exception:
                    pass
        except Exception:
            pass

        if (num_cls1 is None or num_cls2 is None) and fallback_to_data and isinstance(self.data, dict):
            nh1 = self.data.get("names_head1", {})
            nh2 = self.data.get("names_head2", {})
            if isinstance(nh1, dict) and isinstance(nh2, dict) and len(nh1) > 0 and len(nh2) > 0:
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

            if isinstance(nh1, dict) and isinstance(nh2, dict) and len(nh1) == num_cls1 and len(nh2) == num_cls2:
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
        try:
            if self._dual_num_cls1 and self._dual_num_cls2:
                return int(self._dual_num_cls1) * int(self._dual_num_cls2)
        except Exception:
            pass
        try:
            if isinstance(self.data, dict):
                if isinstance(self.data.get("names"), dict) and len(self.data["names"]) > 0:
                    return len(self.data["names"])
                if isinstance(self.data.get("nc"), (int, np.integer)) and self.data["nc"] > 0:
                    return int(self.data["nc"])
        except Exception:
            pass
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
        )
        return dets

    # ---------------- Metrics Lifecycle Stubs (extend in derived classes) ----------------
    def init_metrics(self, model):
        """
        Initialize (or re-initialize) per-head confusion matrices if dual-head info is known.
        Uses the ConfusionMatrix(names=..., task='detect') signature from metrics.py.
        """
        if self._dual_num_cls1 and self._dual_num_cls2:
            # Build per-head names
            def _extract_names(key, count, prefix):
                src = None
                if isinstance(self.data, dict):
                    src = self.data.get(key, None)
                if isinstance(src, dict) and len(src) == count:
                    # ensure ordered by integer key
                    try:
                        return [str(src[i]) for i in range(count)]
                    except Exception:
                        # fallback: sort by key
                        return [str(v) for _, v in sorted(src.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])]
                return [f"{prefix}_{i}" for i in range(count)]

            names_h1 = _extract_names("names_head1", int(self._dual_num_cls1), "h1")
            names_h2 = _extract_names("names_head2", int(self._dual_num_cls2), "h2")

            try:
                self.cm_h1 = ConfusionMatrix(names=names_h1, task="detect")
                self.cm_h2 = ConfusionMatrix(names=names_h2, task="detect")
            except Exception as e:
                LOGGER.warning(f"Failed to initialize dual-head confusion matrices: {e}")
                self.cm_h1, self.cm_h2 = None, None

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
    # ------------------------------------------------------------------------------

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

    def _update_dual_head_confusion_matrices(self, preds, batch: Dict[str, Any]):
        """
        Update per-head confusion matrices using current batch predictions and labels.

        preds: list of per-image tensors (N_i, 6+extras) after NMS  -> [x1,y1,x2,y2,conf,combined_cls,(extras...)]
        batch keys required: 'img', 'batch_idx', 'cls' (combined or flattened), 'bboxes' (xywh normalized)
        """
        if (self.cm_h1 is None or self.cm_h2 is None or
                self._dual_num_cls1 is None or self._dual_num_cls2 is None):
            return

        imgs = batch.get("img", None)
        batch_idx = batch.get("batch_idx", None)
        gt_combined = batch.get("cls", None)
        gt_boxes = batch.get("bboxes", None)

        if imgs is None or batch_idx is None or gt_combined is None or gt_boxes is None:
            return

        if isinstance(gt_combined, torch.Tensor) and gt_combined.ndim > 1 and gt_combined.shape[1] == 1:
            gt_combined = gt_combined.view(-1)

        device = gt_combined.device
        n2 = int(self._dual_num_cls2)
        bs = imgs.shape[0]
        _, _, H, W = imgs.shape

        # Iterate each image in batch
        for i in range(bs):
            # Slice GT for image i
            if isinstance(batch_idx, list):
                batch_idx = torch.as_tensor(batch_idx, device=device)
            mask_i = (batch_idx == i)
            gt_idx_i = torch.where(mask_i)[0]

            if gt_idx_i.numel():
                # Combined -> (cls1, cls2)
                gt_c_flat = gt_combined[gt_idx_i].long()
                gt_c1 = gt_c_flat // n2
                gt_c2 = gt_c_flat % n2
                gt_xywh = gt_boxes[gt_idx_i]
                # Convert normalized xywh to pixel xyxy
                x, y, w, h = gt_xywh[:, 0], gt_xywh[:, 1], gt_xywh[:, 2], gt_xywh[:, 3]
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # Build label dicts expected by ConfusionMatrix
                labels_h1 = {
                    "cls": gt_c1.to(torch.long),
                    "bboxes": torch.stack([x1, y1, x2, y2], dim=1).to(device)
                }
                labels_h2 = {
                    "cls": gt_c2.to(torch.long),
                    "bboxes": torch.stack([x1, y1, x2, y2], dim=1).to(device)
                }
            else:
                labels_h1 = {"cls": torch.zeros(0, dtype=torch.long, device=device),
                             "bboxes": torch.zeros((0, 4), device=device)}
                labels_h2 = {"cls": torch.zeros(0, dtype=torch.long, device=device),
                             "bboxes": torch.zeros((0, 4), device=device)}

            # Predictions for image i
            det_i = preds[i] if i < len(preds) else None
            if det_i is None or not isinstance(det_i, torch.Tensor) or det_i.numel() == 0:
                det_dict_h1 = {"cls": torch.zeros(0, dtype=torch.long, device=device),
                               "conf": torch.zeros(0, device=device),
                               "bboxes": torch.zeros((0, 4), device=device)}
                det_dict_h2 = {"cls": torch.zeros(0, dtype=torch.long, device=device),
                               "conf": torch.zeros(0, device=device),
                               "bboxes": torch.zeros((0, 4), device=device)}
            else:
                # det_i: [x1,y1,x2,y2,conf,combined_cls,...]
                det_cls_comb = det_i[:, 5].long()
                det_c1 = det_cls_comb // n2
                det_c2 = det_cls_comb % n2
                det_bboxes = det_i[:, :4]
                det_conf = det_i[:, 4]

                det_dict_h1 = {"cls": det_c1, "conf": det_conf, "bboxes": det_bboxes}
                det_dict_h2 = {"cls": det_c2, "conf": det_conf, "bboxes": det_bboxes}

            try:
                # Use the ConfusionMatrix API (detections dict, labels dict)
                self.cm_h1.process_batch(det_dict_h1, labels_h1, conf=self.args.conf, iou_thres=self.args.iou)
                self.cm_h2.process_batch(det_dict_h2, labels_h2, conf=self.args.conf, iou_thres=self.args.iou)
            except Exception as e:
                LOGGER.warning(f"[dual-conf-matrix] per-head update failed for image {i}: {e}")
                continue

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
