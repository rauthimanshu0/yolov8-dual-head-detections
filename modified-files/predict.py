from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

"""
Dual‑Head–aware DetectionPredictor (Taxonomy & State)
Location: /ultralytics/models/yolo/detect/predict.py
===================================================

Role
- Wraps BasePredictor for detection with dual‑head compatibility. Preserves and uses the head extras emitted by DualDetect
  to improve postprocessing and visualization.

What’s different for dual‑head
- Two‑stage, object‑based NMS enabled by default:
  - Stage‑1 objectness comes from per‑axis probs in extras (product of p_Taxonomy and p_state).
  - Stage‑2 per‑species cleanup removes duplicate boxes among the same species.
  - Keeps extras aligned with surviving detections and returns kept anchor indices when features are being saved.
- Extras preserved into Results:
  - Passes the full [x1,y1,x2,y2, conf, cls, extras…] tensor to Results so downstream plotting/UI can show two‑line
    labels like "Taxonomy p" and "State p".
  - Attaches names_head1 and names_head2 to each Results object if the model exposes them.

Key behaviors
- non_max_suppression(...) is called with:
  - nc=len(self.model.names) to correctly parse joint class scores and preserve extras.
  - two_stage_object=True, object_score_mode="product", species_cleanup=True (configurable by args).
  - rotated path if task == "obb" and end‑to‑end support via model.end2end.
- Feature capture (optional):
  - If self._feats is set, requests return_idxs from NMS, aggregates backbone feature maps to per‑anchor vectors,
    and indexes them with the kept anchor indices to attach per‑object embeddings to Results (r.feats).

I/O and scaling
- Converts batched torch images to NumPy for Results.
- Scales predicted boxes back to original image size with ops.scale_boxes before constructing Results.

Outcome
- Out‑of‑the‑box predictor that respects dual‑head outputs, improves NMS via per‑axis probs, and surfaces per‑axis
  names/probabilities to downstream consumers without breaking single‑head or OBB/e2e flows.
"""

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),  # preserve extras for DualDetect
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
            # Enable Two-stage object-based NMS by default for prediction
            two_stage_object=True,
            stage1_conf_thres=self.args.conf,
            stage1_iou_thres=self.args.iou,
            stage2_iou_thres=self.args.iou,
            object_score_mode="product",
            species_cleanup=True,
            species_col_in_extras=0,
            num_cls2_for_species=None,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def get_obj_feats(self, feat_maps, idxs):
        import torch
        s = min([x.shape[1] for x in feat_maps])  # find smallest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]

    def construct_results(self, preds, img, orig_imgs):
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        # pred is shape (N, K) and may be empty. Scale only if it has boxes.
        if pred.numel():
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            r = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)  # pass ALL columns incl. extras
        else:
            r = Results(orig_img, path=img_path, names=self.model.names, boxes=None)

        # Best-effort: attach split names if model exposes them
        for attr in ("names_head1", "names_head2"):
            val = getattr(self.model, attr, None)
            if isinstance(val, dict) and len(val):
                setattr(r, attr, {int(k): str(v) for k, v in val.items()})
        return r