from __future__ import annotations
"""
Dual‑Head Detection Validator (Taxonomy & State) — sparse‑grid aware
Location: /ultralytics/engine/dual_head_validator.py
===============================================================

Purpose
- Drop‑in validator for dual‑axis detection models that predict a joint class id
  encoding two heads (h1=Taxonomy, h2=State).
- Extends Ultralytics DetectionValidator and augments it with per‑axis decoding,
  metrics, confusion matrices, PR curves, and optional sparse‑grid remapping.

What’s different vs the stock validator
- Axis decoding from joint class ids:
  - Uses data['joint_to_heads']: List[[h1,h2]] (length==nc) when provided.
  - Otherwise infers split order (h1_first vs h2_first) via a small matching probe.
- Sparse joint grid support (incomplete Cartesian label space):
  - Dense→sparse remap built from data['joint_to_heads'] and (num_head1, num_head2).
  - Alternative filter mode using data['heads_to_joint'] to mask invalid pairs.
  - Works on Results.boxes.cls in place; keeps bboxes/conf synced.
- Axis‑wise metrics:
  - For each head: mAP50‑95, mAP50, mean precision, mean recall (keys: h1/*, h2/*).
  - Per‑class arrays stored and exported; metrics merged into results_dict.
- Per‑class artifacts (per epoch):
  - {h1,h2}_per_class.csv with precision, recall, AP50, AP50‑95 per class.
  - Confusion matrices and PR curves per head (plots enabled via args.plots).
- Prediction format robustness:
  - Accepts Ultralytics Results objects, dicts, or raw tensors/ndarrays.
  - Greedy IoU matching per head with configurable IoU vector (0.50:0.95).
- Latent space analysis hook:
  - Integrates with BaseValidator latent capture (enabled externally), no custom
    capture logic needed here.
- Standalone summary:
  - When called outside training, writes metrics_summary.csv with joint/axis
    aggregates for quick inspection.

Expected data entries (optional but recommended)
- data['joint_to_heads']: list[(h1,h2)] mapping each joint class id → axis ids.
- data['heads_to_joint']: dict{"h1_h2": joint_id} to filter to valid pairs.
- data['num_head1'], data['num_head2']: explicit axis cardinalities.
- data['names_head1'], data['names_head2']: axis label dictionaries (id→name).

Notes
- The model can also carry names_head1/names_head2; otherwise numeric class ids
  are used in tables/plots.
- Combined‑index training is handled elsewhere; this validator only decodes and
  scores per axis while preserving stock joint metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import os
import csv
from pathlib import Path

import torch
import numpy as np

try:
    from ultralytics.engine.validator import DetectionValidator  # Preferred
except Exception:
    from ultralytics.models.yolo.detect.val import DetectionValidator  # Fallback

from ultralytics.utils import LOGGER, ops
from ultralytics.utils import metrics as ul_metrics
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -------------------------------------------------------------------------
# Helper dataclasses & utilities
# -------------------------------------------------------------------------
@dataclass
class AxisStats:
    correct: List[np.ndarray]
    conf: List[np.ndarray]
    pcls: List[np.ndarray]
    tcls: List[np.ndarray]
    cm: Optional[ConfusionMatrix]


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float().cpu()
    b = b.float().cpu()
    if hasattr(ul_metrics, "box_iou"):
        try:
            return ul_metrics.box_iou(a, b).cpu()
        except Exception:
            pass
    Na, Nb = a.shape[0], b.shape[0]
    if Na == 0 or Nb == 0:
        return torch.zeros((Na, Nb), dtype=torch.float32, device="cpu")
    a_x1, a_y1, a_x2, a_y2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    b_x1, b_y1, b_x2, b_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = torch.maximum(a_x1, b_x1)
    inter_y1 = torch.maximum(a_y1, b_y1)
    inter_x2 = torch.minimum(a_x2, b_x2)
    inter_y2 = torch.minimum(a_y2, b_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1).clamp(min=0) * (a_y2 - a_y1).clamp(min=0)
    area_b = (b_x2 - b_x1).clamp(min=0) * (b_y2 - b_y1).clamp(min=0)
    union = area_a + area_b - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou.float().cpu()


def _safe_fname(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_")).strip() or "cls"


def _monotone_pr(recall: np.ndarray, precision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    if r.ndim != 1 or p.ndim != 1 or r.size != p.size or r.size == 0:
        return r, p
    idx = np.argsort(r)
    r = r[idx]
    p = p[idx]
    p = np.maximum.accumulate(p[::-1])[::-1]
    r = np.clip(r, 0.0, 1.0)
    p = np.clip(p, 0.0, 1.0)
    return r, p


# -------------------------------------------------------------------------
# Main Validator Class
# -------------------------------------------------------------------------
class DualHeadDetectionValidator(DetectionValidator):
    """
    Dual-head validator with latent analysis integration (latent capture handled in BaseValidator logic).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Axis counts
        self.num_cls1: Optional[int] = None
        self.num_cls2: Optional[int] = None

        # Ensure iou vector on CPU (device will be set later)
        self.iouv = torch.linspace(0.5, 0.95, 10)

        # Axis stats container
        self.axis: Dict[str, AxisStats] = {}

        # Joint -> axis mapping
        self._map_j2h1: Optional[torch.Tensor] = None
        self._map_j2h2: Optional[torch.Tensor] = None

        # Sparse mapping artifacts
        self._sparse_filter_enabled: bool = False
        self._sparse_heads_to_joint: Optional[Dict[str, int]] = None
        self._sparse_mode: bool = False
        self._dense_to_sparse: Optional[torch.Tensor] = None
        self._dense_axis_sizes: Tuple[int, int] = (0, 0)

        # Misc
        self._split_order: Optional[str] = None
        self._extra_results: Dict[str, float] = {}
        self._per_class: Dict[str, Dict[str, np.ndarray]] = {}
        self._current_epoch: Optional[int] = None
        self._dual_debug = os.getenv("DUAL_DEBUG", "0") not in {"", "0", "false", "False"}
        self._pc_written_epochs = {"h1": set(), "h2": set()}

    # ------------------------------------------------------------------
    # Initialization / dataset mapping
    # ------------------------------------------------------------------
    def _load_joint_mapping(self):
        if not isinstance(getattr(self, "data", None), dict):
            return
        nc = self.data.get("nc", None)
        if not isinstance(nc, int) or nc <= 0:
            return
        j2heads = self.data.get("joint_to_heads", None)
        if isinstance(j2heads, list) and len(j2heads) == nc:
            try:
                arr = np.asarray(j2heads, dtype=np.int64)
                self._map_j2h1 = torch.as_tensor(arr[:, 0], dtype=torch.long)
                self._map_j2h2 = torch.as_tensor(arr[:, 1], dtype=torch.long)
                if self._dual_debug:
                    LOGGER.info("Dual-head: using 'joint_to_heads' mapping.")
            except Exception as e:
                LOGGER.warning(f"Dual-head: parsing joint_to_heads failed: {e}")
        htj = self.data.get("heads_to_joint", None)
        if isinstance(htj, dict) and len(htj):
            try:
                self._sparse_heads_to_joint = {str(k): int(v) for k, v in htj.items()}
                self._sparse_filter_enabled = True
                if self._dual_debug:
                    LOGGER.info(f"Dual-head: heads_to_joint present ({len(self._sparse_heads_to_joint)})")
            except Exception as e:
                LOGGER.warning(f"Dual-head: parsing heads_to_joint failed: {e}")
        maybe_n1 = self.data.get("num_head1", None)
        maybe_n2 = self.data.get("num_head2", None)
        if isinstance(maybe_n1, int) and maybe_n1 > 0:
            self.num_cls1 = maybe_n1
        if isinstance(maybe_n2, int) and maybe_n2 > 0:
            self.num_cls2 = maybe_n2
        self._init_sparse_remap()

    def _init_sparse_remap(self):
        if self._map_j2h1 is None or self._map_j2h2 is None:
            return
        if self.num_cls1 is None or self.num_cls2 is None:
            return
        observed_nc = int(self._map_j2h1.shape[0])
        full_grid = int(self.num_cls1 * self.num_cls2)
        self._dense_axis_sizes = (self.num_cls1, self.num_cls2)
        if observed_nc >= full_grid:
            self._sparse_mode = False
            if self._dual_debug:
                LOGGER.info("Dual-head: grid dense, no remap.")
            return
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        dense_to_sparse = torch.full((full_grid,), -1, dtype=torch.long, device=device)
        joint_to_heads = self.data.get("joint_to_heads", None)
        if not isinstance(joint_to_heads, list) or len(joint_to_heads) != observed_nc:
            return
        for sparse_id, (c1, c2) in enumerate(joint_to_heads):
            if not (0 <= c1 < self.num_cls1) or not (0 <= c2 < self.num_cls2):
                raise ValueError(f"joint_to_heads entry {(c1, c2)} out of bounds")
            idx = c1 * self.num_cls2 + c2
            if dense_to_sparse[idx] != -1:
                raise ValueError("Duplicate dense index in joint_to_heads")
            dense_to_sparse[idx] = sparse_id
        if (dense_to_sparse >= 0).sum().item() != observed_nc:
            raise RuntimeError("Remap mismatch count.")
        self._dense_to_sparse = dense_to_sparse
        self._sparse_mode = True
        LOGGER.info(
            f"Sparse mapping active: observed={observed_nc}, full={full_grid}, unused={full_grid - observed_nc}"
        )

    def _ensure_axis_initialized(self):
        if "h1" in self.axis and "h2" in self.axis:
            return
        n1, n2 = self.num_cls1, self.num_cls2
        if n1 is None or n2 is None:
            if isinstance(self.data, dict):
                nh1 = self.data.get("names_head1")
                nh2 = self.data.get("names_head2")
                if isinstance(nh1, dict):
                    n1 = n1 or len(nh1)
                if isinstance(nh2, dict):
                    n2 = n2 or len(nh2)
            if (n1 is None or n2 is None) and isinstance(self.names, dict):
                nc = len(self.names)
                for cand in (5, 6, 4, 3, 2):
                    if nc % cand == 0:
                        n2 = n2 or cand
                        n1 = n1 or nc // cand
                        break
                if n1 is None or n2 is None:
                    n1 = n1 or 1
                    n2 = n2 or nc
        self.num_cls1, self.num_cls2 = int(n1), int(n2)
        self.axis = {
            "h1": AxisStats(correct=[], conf=[], pcls=[], tcls=[],
                            cm=ConfusionMatrix(names=[str(i) for i in range(self.num_cls1)], task="detect") if self.args.plots else None),
            "h2": AxisStats(correct=[], conf=[], pcls=[], tcls=[],
                            cm=ConfusionMatrix(names=[str(i) for i in range(self.num_cls2)], task="detect") if self.args.plots else None),
        }
        self._load_joint_mapping()

    # ------------------------------------------------------------------
    # Model & metric initialization
    # ------------------------------------------------------------------
    def set_model(self, model=None):
        try:
            super().set_model(model)
        except Exception:
            if model is not None:
                self.model = model
        self._ensure_axis_initialized()
        LOGGER.info(
            f"Dual-head validator ready: head1={self.num_cls1}, head2={self.num_cls2}, sparse={self._sparse_mode}"
        )

    def init_metrics(self, model):
        super().init_metrics(model)
        self._ensure_axis_initialized()
        self._extra_results.clear()
        self._per_class.clear()
        self._pc_written_epochs = {"h1": set(), "h2": set()}
        for ax in ("h1", "h2"):
            st = self.axis[ax]
            st.correct.clear()
            st.conf.clear()
            st.pcls.clear()
            st.tcls.clear()
            if st.cm and hasattr(st.cm, "reset"):
                try:
                    st.cm.reset()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Sparse remap
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_sparse_remap_inplace(self, preds):
        if not self._sparse_mode or self._dense_to_sparse is None:
            return preds
        map_tensor = self._dense_to_sparse
        dense_len = map_tensor.numel()
        for r in preds:
            boxes = getattr(r, "boxes", None)
            if boxes is None or not hasattr(boxes, "cls") or boxes.cls is None:
                continue
            cls_raw = boxes.cls.long()
            in_range = cls_raw < dense_len
            cls_raw = cls_raw[in_range]
            keep_mask = in_range
            if cls_raw.numel() == 0:
                boxes.data = boxes.data[:0]
                boxes.cls = boxes.cls[:0]
                continue
            sparse_ids = map_tensor[cls_raw]
            valid = sparse_ids >= 0
            final_mask = keep_mask.clone()
            final_mask[keep_mask] = valid
            if final_mask.sum() == 0:
                boxes.data = boxes.data[:0]
                boxes.cls = boxes.cls[:0]
                continue
            if hasattr(boxes, "data"):
                boxes.data = boxes.data[final_mask]
            for attr in ("xyxy", "conf", "id"):
                if hasattr(boxes, attr):
                    t = getattr(boxes, attr)
                    if isinstance(t, torch.Tensor) and t.shape[0] == final_mask.shape[0]:
                        setattr(boxes, attr, t[final_mask])
            boxes.cls = sparse_ids[valid].to(boxes.cls.dtype)
        return preds

    def _filter_sparse_predictions(self, preds):
        if self._sparse_mode and self._dense_to_sparse is not None:
            return self._apply_sparse_remap_inplace(preds)
        if not self._sparse_filter_enabled or not isinstance(self._sparse_heads_to_joint, dict):
            return preds
        c2 = self.num_cls2 or 1
        def remap_tensor_cls(t_cls: torch.Tensor):
            h1 = t_cls // c2
            h2 = t_cls % c2
            keys = [f"{int(a)}_{int(b)}" for a, b in zip(h1.tolist(), h2.tolist())]
            keep_mask_list = [k in self._sparse_heads_to_joint for k in keys]
            keep_mask = torch.tensor(keep_mask_list, device=t_cls.device, dtype=torch.bool)
            new_cls_vals = [self._sparse_heads_to_joint[k] for k in keys if k in self._sparse_heads_to_joint]
            new_cls_t = torch.as_tensor(new_cls_vals, device=t_cls.device, dtype=torch.long)
            return keep_mask, new_cls_t
        for p in preds:
            try:
                if hasattr(p, "boxes") and hasattr(p.boxes, "cls"):
                    t_cls = p.boxes.cls.long()
                    keep_mask, new_cls = remap_tensor_cls(t_cls)
                    if keep_mask.shape[0]:
                        if hasattr(p.boxes, "data"):
                            p.boxes.data = p.boxes.data[keep_mask]
                        for attr in ("xyxy", "conf"):
                            if hasattr(p.boxes, attr):
                                t = getattr(p.boxes, attr)
                                if isinstance(t, torch.Tensor) and t.shape[0] == keep_mask.shape[0]:
                                    setattr(p.boxes, attr, t[keep_mask])
                    p.boxes.cls = new_cls
            except Exception as e:
                if self._dual_debug:
                    LOGGER.warning(f"Dual-head sparse filter error: {e}")
        return preds

    # ------------------------------------------------------------------
    # Axis splitting helpers
    # ------------------------------------------------------------------
    def _split_ids(self, ids: torch.Tensor, order: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._map_j2h1 is not None and self._map_j2h2 is not None:
            ids_cpu = ids.detach().cpu().long()
            return self._map_j2h1[ids_cpu], self._map_j2h2[ids_cpu]
        if order == "h2_first":
            h2 = ids // self.num_cls1
            h1 = ids % self.num_cls1
        else:
            h1 = ids // self.num_cls2
            h2 = ids % self.num_cls2
        return h1, h2

    # ------------------------------------------------------------------
    # Prediction extraction utilities
    # ------------------------------------------------------------------
    def _extract_preds(self, preds_i: Any):
        if preds_i is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), None, None, None
        if hasattr(preds_i, "boxes"):
            b = preds_i.boxes
            xyxy = b.xyxy
            conf = getattr(b, "conf", None)
            cls = getattr(b, "cls", None)
            xyxy_np = xyxy.detach().cpu().numpy() if isinstance(xyxy, torch.Tensor) else np.asarray(xyxy)
            conf_np = conf.detach().cpu().numpy().reshape(-1) if isinstance(conf, torch.Tensor) else \
                (np.asarray(conf).reshape(-1) if conf is not None else np.zeros((xyxy_np.shape[0],), np.float32))
            cls_t = cls.long().detach().cpu() if isinstance(cls, torch.Tensor) else \
                (torch.as_tensor(cls, dtype=torch.long) if cls is not None else torch.zeros((xyxy_np.shape[0],), dtype=torch.long))
            return xyxy_np, conf_np, cls_t, None, None
        if isinstance(preds_i, dict):
            xyxy = None
            conf = None
            cls = None
            if "boxes" in preds_i and isinstance(preds_i["boxes"], dict):
                boxes = preds_i["boxes"]
                xyxy = boxes.get("xyxy", boxes.get("bboxes", boxes.get("xyxyn", None)))
                conf = boxes.get("conf", preds_i.get("conf", None))
                cls = boxes.get("cls", preds_i.get("cls", None))
            else:
                for k in ("bboxes", "xyxy", "boxes"):
                    if k in preds_i:
                        xyxy = preds_i.get(k)
                        break
                conf = preds_i.get("conf", None)
                cls = preds_i.get("cls", None)
            if isinstance(xyxy, torch.Tensor):
                xyxy_np = xyxy.detach().cpu().numpy()
            elif xyxy is not None:
                xyxy_np = np.asarray(xyxy)
            else:
                xyxy_np = np.zeros((0, 4), np.float32)
            if isinstance(conf, torch.Tensor):
                conf_np = conf.detach().cpu().numpy().reshape(-1)
            elif conf is not None:
                conf_np = np.asarray(conf).reshape(-1)
            else:
                conf_np = np.zeros((xyxy_np.shape[0],), np.float32)
            if isinstance(cls, torch.Tensor):
                cls_t = cls.long().detach().cpu()
            elif cls is not None:
                cls_t = torch.as_tensor(cls, dtype=torch.long)
            else:
                cls_t = torch.zeros((xyxy_np.shape[0],), dtype=torch.long)
            n = int(xyxy_np.shape[0])
            if conf_np.shape[0] != n:
                conf_np = conf_np[:n] if conf_np.shape[0] > n else np.pad(conf_np, (0, n - conf_np.shape[0]), constant_values=0.0)
            if cls_t.numel() != n:
                cls_t = cls_t[:n] if cls_t.numel() > n else torch.cat([cls_t, torch.zeros((n - cls_t.numel(),), dtype=torch.long)], 0)
            return xyxy_np, conf_np, cls_t, None, None
        if isinstance(preds_i, torch.Tensor):
            t = preds_i.detach().cpu()
            return (
                t[:, :4].numpy(),
                t[:, 4].numpy().reshape(-1),
                t[:, 5].long(),
                (t[:, 6].long() if t.shape[1] >= 7 else None),
                (t[:, 7].long() if t.shape[1] >= 8 else None),
            )
        if isinstance(preds_i, np.ndarray):
            a = preds_i
            return (
                a[:, :4],
                a[:, 4].reshape(-1),
                torch.as_tensor(a[:, 5], dtype=torch.long),
                (torch.as_tensor(a[:, 6], dtype=torch.long) if a.shape[1] >= 7 else None),
                (torch.as_tensor(a[:, 7], dtype=torch.long) if a.shape[1] >= 8 else None),
            )
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), None, None, None

    # ------------------------------------------------------------------
    # Per-image axis stats
    # ------------------------------------------------------------------
    def _per_image_axis_stats(self, preds_i, labels_i, img_wh):
        w, h = img_wh
        if labels_i.numel():
            t_cls_joint = labels_i[:, 0].long().cpu()
            t_xywh = labels_i[:, 1:5]
            t_xyxy = ops.xywh2xyxy(t_xywh)
            t_xyxy[:, [0, 2]] *= w
            t_xyxy[:, [1, 3]] *= h
            t_xyxy = t_xyxy.detach().cpu().numpy()
        else:
            t_cls_joint = torch.zeros((0,), dtype=torch.long)
            t_xyxy = np.zeros((0, 4), dtype=np.float32)
        p_xyxy, p_conf, p_cls_joint, p_h1, p_h2 = self._extract_preds(preds_i)
        if self._map_j2h1 is None and self._split_order is None and p_cls_joint is not None and t_cls_joint.numel():
            iouv_np = self.iouv.detach().cpu().numpy()
            thr = np.array([iouv_np[0]], dtype=np.float32)
            h1a, h2a = self._split_ids(p_cls_joint, "h1_first")
            t_h1a, t_h2a = self._split_ids(t_cls_joint, "h1_first")
            corr1 = (
                self._greedy_axis_matches(p_xyxy, p_conf, h1a.numpy(), t_xyxy, t_h1a.numpy(), thr).sum()
                + self._greedy_axis_matches(p_xyxy, p_conf, h2a.numpy(), t_xyxy, t_h2a.numpy(), thr).sum()
            )
            h1b, h2b = self._split_ids(p_cls_joint, "h2_first")
            t_h1b, t_h2b = self._split_ids(t_cls_joint, "h2_first")
            corr2 = (
                self._greedy_axis_matches(p_xyxy, p_conf, h1b.numpy(), t_xyxy, t_h1b.numpy(), thr).sum()
                + self._greedy_axis_matches(p_xyxy, p_conf, h2b.numpy(), t_xyxy, t_h2b.numpy(), thr).sum()
            )
            self._split_order = "h2_first" if corr2 > corr1 else "h1_first"
            if self._dual_debug:
                LOGGER.info(f"Dual-head: split_order={self._split_order} (h1_first={int(corr1)}, h2_first={int(corr2)})")
        if p_cls_joint is None:
            p_h1_np = np.zeros((0,), dtype=np.int64)
            p_h2_np = np.zeros((0,), dtype=np.int64)
        else:
            if p_h1 is None or p_h2 is None:
                h1_, h2_ = self._split_ids(p_cls_joint, self._split_order or "h1_first")
            else:
                h1_, h2_ = p_h1, p_h2
            p_h1_np = h1_.numpy()
            p_h2_np = h2_.numpy()
        if t_cls_joint.numel():
            t_h1, t_h2 = self._split_ids(t_cls_joint, self._split_order or "h1_first")
            t_h1_np = t_h1.numpy()
            t_h2_np = t_h2.numpy()
        else:
            t_h1_np = np.zeros((0,), dtype=np.int64)
            t_h2_np = np.zeros((0,), dtype=np.int64)
        iouv_np = self.iouv.detach().cpu().numpy()
        correct_h1 = self._greedy_axis_matches(p_xyxy, p_conf, p_h1_np, t_xyxy, t_h1_np, iouv_np)
        correct_h2 = self._greedy_axis_matches(p_xyxy, p_conf, p_h2_np, t_xyxy, t_h2_np, iouv_np)
        if self.args.plots:
            iou05 = float(iouv_np[0])
            if self.axis["h1"].cm is not None:
                det = {"cls": torch.from_numpy(p_h1_np), "conf": torch.from_numpy(p_conf), "bboxes": torch.from_numpy(p_xyxy)}
                tgt = {"cls": torch.from_numpy(t_h1_np), "bboxes": torch.from_numpy(t_xyxy)}
                self.axis["h1"].cm.process_batch(det, tgt, conf=0.25, iou_thres=iou05)
            if self.axis["h2"].cm is not None:
                det = {"cls": torch.from_numpy(p_h2_np), "conf": torch.from_numpy(p_conf), "bboxes": torch.from_numpy(p_xyxy)}
                tgt = {"cls": torch.from_numpy(t_h2_np), "bboxes": torch.from_numpy(t_xyxy)}
                self.axis["h2"].cm.process_batch(det, tgt, conf=0.25, iou_thres=iou05)
        return {"h1": (correct_h1, p_conf, p_h1_np, t_h1_np), "h2": (correct_h2, p_conf, p_h2_np, t_h2_np)}

    def _greedy_axis_matches(self, p_boxes, p_conf, p_cls_axis, t_boxes, t_cls_axis, iouv):
        n_preds = p_boxes.shape[0]
        correct = np.zeros((n_preds, iouv.size), dtype=bool)
        if t_boxes.shape[0] == 0 or n_preds == 0:
            return correct
        iou = _box_iou(torch.from_numpy(p_boxes), torch.from_numpy(t_boxes)).numpy()
        order = np.argsort(-p_conf.reshape(-1))
        for t_idx, thr in enumerate(iouv):
            matched_t = set()
            for j in order:
                candidates = np.where(
                    (t_cls_axis == p_cls_axis[j]) & (~np.isin(np.arange(t_boxes.shape[0]), list(matched_t)))
                )[0]
                if candidates.size == 0:
                    continue
                ious = iou[j, candidates]
                k_rel = int(np.argmax(ious))
                if ious[k_rel] >= thr:
                    k = int(candidates[k_rel])
                    correct[j, t_idx] = True
                    matched_t.add(k)
        return correct

    def _looks_like_results_sequence(self, preds, nb: int) -> bool:
        if not isinstance(preds, (list, tuple)) or len(preds) == 0:
            return False
        p0 = preds[0]
        return hasattr(p0, "boxes") or (isinstance(p0, dict) and ("boxes" in p0 or "xyxy" in p0 or "bboxes" in p0))

    # ------------------------------------------------------------------
    # Metrics update
    # ------------------------------------------------------------------
    def update_metrics(self, preds, batch):
        self._ensure_axis_initialized()
        if isinstance(preds, (list, tuple)) and self._sparse_mode:
            self._apply_sparse_remap_inplace(preds)
        elif isinstance(preds, (list, tuple)) and self._sparse_filter_enabled:
            preds = self._filter_sparse_predictions(preds)
        super().update_metrics(preds, batch)
        ims = batch["img"]
        h, w = int(ims.shape[2]), int(ims.shape[3])
        nb = ims.shape[0]
        if self._looks_like_results_sequence(preds, nb):
            items = preds
        else:
            try:
                items = self.postprocess(preds)
            except Exception:
                items = preds if isinstance(preds, (list, tuple)) else []
        bidx = batch.get("batch_idx", None)
        cls = batch.get("cls", None)
        bboxes = batch.get("bboxes", None)
        if isinstance(bidx, (list, np.ndarray)):
            bidx = torch.as_tensor(bidx, device=self.device)
        if isinstance(cls, (list, np.ndarray)):
            cls = torch.as_tensor(cls, device=self.device)
        if isinstance(bboxes, (list, np.ndarray)):
            bboxes = torch.as_tensor(bboxes, device=self.device)
        for si in range(nb):
            preds_i = items[si] if isinstance(items, (list, tuple)) and si < len(items) else None
            if bidx is not None and cls is not None and bboxes is not None:
                mask = (bidx == si)
                if isinstance(mask, torch.Tensor) and mask.any():
                    li_cls = cls[mask].view(-1, 1)
                    li_box = bboxes[mask]
                    labels_i = torch.cat((li_cls, li_box), 1)
                else:
                    labels_i = torch.zeros((0, 5), device=self.device)
            else:
                labels_i = torch.zeros((0, 5), device=self.device)
            axis_stats = self._per_image_axis_stats(preds_i, labels_i, (w, h))
            for ax in ("h1", "h2"):
                corr, conf, pcls, tcls = axis_stats[ax]
                self.axis[ax].correct.append(corr)
                self.axis[ax].conf.append(conf.reshape(-1))
                self.axis[ax].pcls.append(pcls.reshape(-1))
                self.axis[ax].tcls.append(tcls.reshape(-1))

    # ------------------------------------------------------------------
    # Axis naming helpers
    # ------------------------------------------------------------------
    def _axis_names(self, axis: str) -> Dict[int, str]:
        nm = getattr(self.model, "names_head1" if axis == "h1" else "names_head2", None)
        if isinstance(nm, dict) and len(nm):
            return {int(k): str(v) for k, v in nm.items()}
        if isinstance(self.data, dict):
            key = "names_head1" if axis == "h1" else "names_head2"
            nm = self.data.get(key, {})
            if isinstance(nm, dict) and len(nm):
                return {int(k): str(v) for k, v in nm.items()}
        nc = self.num_cls1 if axis == "h1" else self.num_cls2
        return {i: str(i) for i in range(int(nc or 0))}

    # ------------------------------------------------------------------
    # Save per-class tables & PR curves
    # ------------------------------------------------------------------
    def _save_per_class_tables(self, epoch: int):
        for ax in ("h1", "h2"):
            pc = self._per_class.get(ax)
            if not pc or epoch in self._pc_written_epochs.get(ax, set()):
                continue
            p, r, ap, ap_class = pc.get("p"), pc.get("r"), pc.get("ap"), pc.get("ap_class")
            if not (isinstance(p, np.ndarray) and p.size and isinstance(ap, np.ndarray) and ap.ndim == 2):
                continue
            names = self._axis_names(ax)
            rows = []
            for i, cid in enumerate(ap_class.astype(int)):
                rows.append(
                    {
                        "epoch": epoch,
                        "class_id": int(cid),
                        "class_name": names.get(int(cid), str(cid)),
                        "precision": float(p[i]) if i < p.shape[0] else 0.0,
                        "recall": float(r[i]) if i < r.shape[0] else 0.0,
                        "AP50": float(ap[i, 0]),
                        "AP50_95": float(ap[i].mean()),
                    }
                )
            out_csv = Path(self.save_dir) / f"{ax}_per_class.csv"
            write_header = not out_csv.exists()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "class_id", "class_name", "precision", "recall", "AP50", "AP50_95"],
                )
                if write_header:
                    w.writeheader()
                w.writerows(rows)
            p_curve = pc.get("p_curve", None)
            r_curve = pc.get("r_curve", None)
            if plt is not None and isinstance(p_curve, np.ndarray) and isinstance(r_curve, np.ndarray):
                pr_dir = Path(self.save_dir) / ax / "pr_curves"
                pr_dir.mkdir(parents=True, exist_ok=True)
                for i, cid in enumerate(ap_class.astype(int)):
                    try:
                        pr_p = np.asarray(p_curve[i], dtype=float)
                        pr_r = np.asarray(r_curve[i], dtype=float)
                        if pr_p.size and pr_r.size and pr_p.shape == pr_r.shape:
                            r_m, p_m = _monotone_pr(pr_r, pr_p)
                            cls_name = _safe_fname(names.get(int(cid), str(cid)))
                            fig = plt.figure(figsize=(4, 3))
                            plt.plot(r_m, p_m, label=f"{names.get(int(cid), str(cid))}", lw=1.5)
                            plt.xlabel("Recall")
                            plt.ylabel("Precision")
                            plt.title(f"{ax.upper()} PR: {names.get(int(cid), str(cid))}")
                            plt.grid(True, ls="--", alpha=0.4)
                            plt.xlim(0, 1)
                            plt.ylim(0, 1)
                            fig.tight_layout()
                            fig.savefig(pr_dir / f"{cid:02d}_{cls_name}.png", dpi=200)
                            plt.close(fig)
                    except Exception as e:
                        if self._dual_debug:
                            LOGGER.warning(f"Dual-head: PR plot failed for {ax} class {int(cid)}: {e}")
            self._pc_written_epochs.setdefault(ax, set()).add(epoch)
            st = self.axis.get(ax)
            if self.args.plots and st and st.cm is not None:
                (Path(self.save_dir) / ax).mkdir(parents=True, exist_ok=True)
                st.cm.names = [names.get(i, str(i)) for i in range(len(st.cm.names))]
                try:
                    st.cm.plot(save_dir=str(Path(self.save_dir) / ax))
                except Exception as e:
                    if self._dual_debug:
                        LOGGER.warning(f"Dual-head: confusion matrix plot failed for {ax}: {e}")

    # ------------------------------------------------------------------
    # Public stats API
    # ------------------------------------------------------------------
    def get_stats(self):
        self._ensure_axis_initialized()
        metrics = super().get_stats()
        out: Dict[str, float] = {}
        for ax in ("h1", "h2"):
            stats = self.axis.get(ax)
            if not stats or not stats.conf:
                out[f"{ax}/mAP50-95"] = 0.0
                out[f"{ax}/mAP50"] = 0.0
                out[f"{ax}/precision"] = 0.0
                out[f"{ax}/recall"] = 0.0
                self._per_class[ax] = {"p": np.array([]), "r": np.array([]), "ap": np.zeros((0, 10)), "ap_class": np.array([])}
                continue
            correct = np.concatenate(stats.correct, axis=0) if stats.correct else np.zeros((0, self.iouv.numel()))
            conf = np.concatenate(stats.conf, axis=0) if stats.conf else np.zeros((0,))
            pcls = np.concatenate(stats.pcls, axis=0) if stats.pcls else np.zeros((0,))
            tcls = np.concatenate(stats.tcls, axis=0) if stats.tcls else np.zeros((0,))
            n_pred = int(correct.shape[0])
            names_map = self._axis_names(ax)
            if conf.size and tcls.size and pcls.size and n_pred and (conf.size == n_pred == pcls.size):
                res = ap_per_class(tp=correct, conf=conf, pred_cls=pcls, target_cls=tcls, plot=False, save_dir=None, names=names_map)
                tp, fp, p, r, f1, ap, ap_class, *rest = res
                p_curve = r_curve = None
                try:
                    if len(rest) >= 3:
                        p_curve = rest[1]
                        r_curve = rest[2]
                except Exception:
                    pass
                out[f"{ax}/mAP50-95"] = float(ap.mean()) if ap.ndim == 2 else 0.0
                out[f"{ax}/mAP50"] = float(ap[:, 0].mean()) if ap.ndim == 2 else 0.0
                out[f"{ax}/precision"] = float(p.mean()) if p.size else 0.0
                out[f"{ax}/recall"] = float(r.mean()) if r.size else 0.0
                self._per_class[ax] = {
                    "p": p, "r": r, "ap": ap, "ap_class": ap_class,
                    "p_curve": np.asarray(p_curve, dtype=object) if isinstance(p_curve, (np.ndarray, list)) else np.array([]),
                    "r_curve": np.asarray(r_curve, dtype=object) if isinstance(r_curve, (np.ndarray, list)) else np.array([]),
                }
            else:
                if self._dual_debug and (conf.size or pcls.size or n_pred):
                    LOGGER.warning(
                        f"Dual-head axis stats mismatch {ax}: n_pred={n_pred}, conf={conf.size}, pcls={pcls.size}, tcls={tcls.size}"
                    )
                out[f"{ax}/mAP50-95"] = 0.0
                out[f"{ax}/mAP50"] = 0.0
                out[f"{ax}/precision"] = 0.0
                out[f"{ax}/recall"] = 0.0
                self._per_class[ax] = {"p": np.array([]), "r": np.array([]), "ap": np.zeros((0, 10)), "ap_class": np.array([])}
        metrics.update(out)
        rd = getattr(self, "metrics", None)
        rd = getattr(rd, "results_dict", None) if rd is not None else None
        if isinstance(rd, dict):
            rd.update({k: float(v) for k, v in out.items()})
        self._extra_results = {k: float(v) for k, v in out.items()}
        return metrics

    # --- NEW helper: write summary metrics CSV (aggregate metrics only) ---
    def _write_summary_metrics_csv(self, results: Dict[str, float], filename: str = "metrics_summary.csv"):
        try:
            summary_order = [
                # joint (possible naming variants)
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95",
                # axis
                "h1/precision", "h1/recall", "h1/mAP50", "h1/mAP50-95",
                "h2/precision", "h2/recall", "h2/mAP50", "h2/mAP50-95"
            ]
            rows = [{"metric": k, "value": results[k]} for k in summary_order if k in results]
            if not rows:
                if self._dual_debug:
                    LOGGER.info("Dual-head: no summary metrics to write.")
                return
            fp = Path(self.save_dir) / filename
            with fp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["metric", "value"])
                w.writeheader()
                w.writerows(rows)
            if self._dual_debug:
                LOGGER.info(f"Wrote metrics summary CSV: {fp}")
        except Exception as e:
            LOGGER.warning(f"Dual-head: failed writing metrics summary CSV: {e}")

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            try:
                self._current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            except Exception:
                self._current_epoch = None
        else:
            self._current_epoch = 0  # force pseudo epoch for standalone test

        results = super().__call__(trainer, model)

        extra = self._extra_results or {k: v for k, v in self.get_stats().items() if k.startswith(("h1/", "h2/"))}
        results.update({k: float(v) for k, v in extra.items()})

        if self._current_epoch is not None:
            try:
                self._save_per_class_tables(self._current_epoch)
            except Exception as e:
                if self._dual_debug:
                    LOGGER.warning(f"Dual-head: per-class table save failed: {e}")

        # NEW: write aggregate metrics summary CSV only for standalone test (or also for training final if you want)
        if trainer is None:
            self._write_summary_metrics_csv(results)

        if trainer is None:
            self._current_epoch = None

        if self._dual_debug:
            subset = {k: results[k] for k in results if k.startswith(("h1/", "h2/"))}
            LOGGER.info(f"Dual-head axis metrics: {subset}")
        return results