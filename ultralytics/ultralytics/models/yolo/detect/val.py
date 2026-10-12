# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Extended to:
    - Keep the standard combined-class (species+stage) metrics as-is.
    - If model outputs 'extra' per detection (from DualDetect), compute per-head metrics:
      species-only and stage-only mAP/PR.
    - Preserve extras through NMS and show dual scores in plots via plot_images.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        # Ensure an on_plot attribute exists for all Ultralytics versions
        self.on_plot = getattr(self, "on_plot", None)

        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in {"batch_idx", "cls", "bboxes"}:
            batch[k] = batch[k].to(self.device)
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation and per-head (species/stage) metrics.
        """
        self.model = model  # IMPORTANT: keep a reference for _prepare_batch() inference of num_cls2

        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics = DetMetrics()
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=list(model.names.values()))

        # Per-head metrics (optional; requires dataset to provide names_head1/names_head2 and model to output 'extra')
        self.nh1 = self.data.get("names_head1", {}) if isinstance(self.data, dict) else {}
        self.nh2 = self.data.get("names_head2", {}) if isinstance(self.data, dict) else {}
        self.species_metrics = DetMetrics()
        self.stage_metrics = DetMetrics()
        if self.nh1:
            self.species_metrics.names = {int(k): str(v) for k, v in self.nh1.items()}
        if self.nh2:
            self.stage_metrics.names = {int(k): str(v) for k, v in self.nh2.items()}

        # Optional: allow dropping Unknown stage from stage-only metrics (not from combined metrics)
        self.drop_unknown = bool(self.data.get("drop_unknown", False)) if isinstance(self.data, dict) else False
        self.unknown_stage_id = None
        if self.drop_unknown and self.nh2:
            for k, v in self.nh2.items():
                if str(v) == "Unknown":
                    self.unknown_stage_id = int(k)
                    break

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        # cosmetic fix: remove trailing parenthesis after mAP50-95
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P)", "R", "mAP50", "mAP50-95")

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.
        Keeps 'extra' channels if present (after the 6 standard columns).
        """
        outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=int(self.nc),  # important: use combined class count
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        out_list = []
        for x in outputs:
            if x.numel() == 0:
                out_list.append(
                    {
                        "bboxes": torch.zeros((0, 4), device=self.device),
                        "conf": torch.zeros((0,), device=self.device),
                        "cls": torch.zeros((0,), device=self.device),
                        "extra": torch.zeros((0, 0), device=self.device),
                    }
                )
                continue
            out_list.append(
                {
                    "bboxes": x[:, :4],
                    "conf": x[:, 4],
                    "cls": x[:, 5],
                    "extra": x[:, 6:] if x.shape[1] > 6 else x.new_zeros((x.shape[0], 0)),
                }
            )
        return out_list

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.
        - Supports dual-head GT cls tensors of shape (N, 2) by flattening to 1-D id if needed.
        - Returns keys expected by _prepare_pred/_process_batch: 'bboxes', 'imgsz', 'ratio_pad'.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx]
        bboxes = batch["bboxes"][idx]  # normalized xywh
        ori_shape = batch["ori_shape"][si]
        imgsz = tuple(batch["img"].shape[2:])
        ratio_pad = batch.get("ratio_pad", None)
        if isinstance(ratio_pad, (list, tuple)) or torch.is_tensor(ratio_pad):
            ratio_pad = ratio_pad[si]

        if torch.is_tensor(cls):
            if cls.ndim == 0:
                cls = cls.view(1)
            elif cls.ndim == 2 and cls.shape[1] == 1:
                cls = cls.squeeze(-1)
            elif cls.ndim == 2 and cls.shape[1] == 2:
                # Try to infer num_cls2 (stage count) from several sources
                n2 = None
                # 1) Prefer validator-cached names_head2 if available
                try:
                    if hasattr(self, "nh2") and isinstance(self.nh2, dict) and len(self.nh2) > 0:
                        n2 = len(self.nh2)
                except Exception:
                    pass
                # 2) Fall back to dataset dict if present
                if n2 is None:
                    try:
                        nh2 = self.data.get("names_head2", {})
                        if isinstance(nh2, dict) and len(nh2) > 0:
                            n2 = len(nh2)
                    except Exception:
                        pass
                # 3) Infer from label tensor itself
                if n2 is None and cls.numel():
                    try:
                        n2 = int(cls[:, 1].max().item()) + 1
                    except Exception:
                        n2 = None
                # 4) Fall back to model head attribute, robust to DDP-wrapped models
                if n2 is None:
                    try:
                        root = getattr(self.model, "module", self.model)
                        head_list = getattr(root, "model", None)
                        if head_list is not None and len(head_list):
                            n2 = int(getattr(head_list[-1], "num_cls2"))
                        else:
                            # Some models may expose num_cls2 at root
                            n2 = int(getattr(root, "num_cls2"))
                    except Exception as e:
                        raise RuntimeError(f"Validator could not infer num_cls2 for dual-head labels: {e}")
                # Flatten (species, stage) -> combined class id
                cls = (cls[:, 0].long() * int(n2) + cls[:, 1].long()).view(-1)

        if torch.is_tensor(cls) and cls.numel() > 0:
            bboxes = ops.xywh2xyxy(bboxes) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bboxes, ori_shape, ratio_pad=ratio_pad)

        return {"cls": cls, "bboxes": bboxes, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.
        """
        cls = pred["cls"]
        if self.args.single_cls:
            cls *= 0
        bboxes = ops.scale_boxes(
            pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        out = {"bboxes": bboxes, "conf": pred["conf"], "cls": cls}
        if "extra" in pred:
            out["extra"] = pred["extra"]
        return out

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.
        Computes:
          - Combined metrics (strict: species+stage must both be correct)
          - Species-only metrics (per-head) if GT cls1 and extras exist
          - Stage-only metrics (per-head) if GT cls2 and extras exist
        """
        # Combined metrics (existing behavior)
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            if no_pred:
                continue

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

        # Per-head metrics (optional)
        gt_cls1 = batch.get("cls1", None)
        gt_cls2 = batch.get("cls2", None)
        if (gt_cls1 is None and gt_cls2 is None) or not len(preds):
            return

        for si, pred in enumerate(preds):
            if "extra" not in pred or pred["extra"].numel() == 0:
                continue

            # Prepare batch slice
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred, pbatch)
            idx = (batch["batch_idx"] == si)

            # Species metrics
            if gt_cls1 is not None and gt_cls1.numel():
                gt1 = gt_cls1[idx].view(-1).to(torch.long).to(self.device)
                pred_species = pred["extra"][:, 0].to(torch.long) if pred["extra"].numel() else pred["extra"].new_zeros((0,), dtype=torch.long)
                if len(gt1) and len(pred_species):
                    self.species_metrics.update_stats(
                        {
                            **self._process_batch(
                                {"cls": pred_species, "bboxes": predn["bboxes"]},
                                {"cls": gt1, "bboxes": pbatch["bboxes"]},
                            ),
                            "target_cls": gt1.detach().cpu().numpy(),
                            "target_img": np.unique(gt1.detach().cpu().numpy()),  # added to satisfy DetMetrics keys
                            "pred_cls": pred_species.detach().cpu().numpy(),
                            "conf": pred["conf"].detach().cpu().numpy(),
                        }
                    )

            # Stage metrics
            if gt_cls2 is not None and gt_cls2.numel():
                gt2 = gt_cls2[idx].view(-1).to(torch.long).to(self.device)
                if len(gt2):
                    pred_stage = pred["extra"][:, 1].to(torch.long) if pred["extra"].numel() else pred["extra"].new_zeros((0,), dtype=torch.long)
                    if self.unknown_stage_id is not None:
                        # filter GT with Unknown stage (combined metrics are unaffected)
                        mask = gt2 != self.unknown_stage_id
                        gt2_eval = gt2[mask]
                        bboxes_eval = pbatch["bboxes"][mask]
                    else:
                        gt2_eval = gt2
                        bboxes_eval = pbatch["bboxes"]
                    if len(gt2_eval) and len(pred_stage):
                        self.stage_metrics.update_stats(
                            {
                                **self._process_batch(
                                    {"cls": pred_stage, "bboxes": predn["bboxes"]},
                                    {"cls": gt2_eval, "bboxes": bboxes_eval},
                                ),
                                "target_cls": gt2_eval.detach().cpu().numpy(),
                                "target_img": np.unique(gt2_eval.detach().cpu().numpy()),  # added to satisfy DetMetrics keys
                                "pred_cls": pred_stage.detach().cpu().numpy(),
                                "conf": pred["conf"].detach().cpu().numpy(),
                            }
                        )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix, and process per-head summaries."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

        # Per-head summaries (no-op if nothing was added)
        if hasattr(self, "species_metrics") and getattr(self.species_metrics, "stats", None):
            if any(len(v) for v in self.species_metrics.stats.values()):
                self.species_metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        if hasattr(self, "stage_metrics") and getattr(self.stage_metrics, "stats", None):
            if any(len(v) for v in self.stage_metrics.stats.values()):
                self.stage_metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate and return metrics statistics.
        Includes combined results and per-head summaries.
        """
        # Combined metrics
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)

        # Per-head metrics (compute only if there is data)
        if hasattr(self, "species_metrics") and getattr(self.species_metrics, "stats", None):
            if any(len(v) for v in self.species_metrics.stats.values()):
                self.species_metrics.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
        if hasattr(self, "stage_metrics") and getattr(self.stage_metrics, "stats", None):
            if any(len(v) for v in self.stage_metrics.stats.values()):
                self.stage_metrics.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)

        # IMPORTANT: read results BEFORE clearing stats
        results = self.metrics.results_dict
        self.metrics.clear_stats()

        # Append per-head summary keys (0 if never updated/processed)
        results.update(
            {
                "species/mAP50": float(getattr(self.species_metrics.box, "map50", 0.0)),
                "species/mAP50-95": float(getattr(self.species_metrics.box, "map", 0.0)),
                "stage/mAP50": float(getattr(self.stage_metrics.box, "map50", 0.0)),
                "stage/mAP50-95": float(getattr(self.stage_metrics.box, "map", 0.0)),
            }
        )
        return results

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.
        Keeps any 'extra' channels for plot_images to show per-head scores.
        """
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions

        # Determine unified extra width across images and pad as needed so cat() won't fail
        extra_dim = 0
        if len(preds) and "extra" in preds[0]:
            extra_dim = max((p["extra"].shape[1] if "extra" in p else 0) for p in preds)

        def get_tensor(p: Dict[str, torch.Tensor], k: str, limit: int):
            t = p[k][:limit]
            if k == "extra" and extra_dim > 0:
                # If this image has no 'extra' or zero-width, pad to extra_dim
                if t.shape[1] == 0:
                    t = t.new_zeros((t.shape[0], extra_dim))
            return t

        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([get_tensor(x, k, max_det) for x in preds], dim=0) for k in keys}

        # Convert to xywh and NORMALIZE by image size expected by plot_images (i.e., batch["img"])
        if batched_preds["bboxes"].numel():
            batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])
            # images are (B, 3, H, W); normalize x,w by W and y,h by H
            H, W = batch["img"].shape[2], batch["img"].shape[3]
            gn = torch.tensor([W, H, W, H], dtype=batched_preds["bboxes"].dtype, device=batched_preds["bboxes"].device)
            batched_preds["bboxes"][:, [0, 2]] /= gn[[0, 2]]  # x, w
            batched_preds["bboxes"][:, [1, 3]] /= gn[[1, 3]]  # y, h

        # Attach per-head name maps if available so plot_images can render class1/class2 lines nicely
        if isinstance(getattr(self, "nh1", {}), dict) and len(self.nh1):
            batched_preds["names_head1"] = {int(k): str(v) for k, v in self.nh1.items()}
        if isinstance(getattr(self, "nh2", {}), dict) and len(self.nh2):
            batched_preds["names_head2"] = {int(k): str(v) for k, v in self.nh2.items()}

        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], filename: str) -> None:
        """
        Serialize YOLO predictions to COCO json format.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: Dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: Union[str, List[str]] = "bbox",
        suffix: Union[str, List[str]] = "Box",
    ) -> Dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats