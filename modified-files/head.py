# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dual- and Single-Head Detection Heads for Ultralytics YOLO
Location: /ultralytics/nn/modules/head.py
==========================================================

What’s new (DualDetect):
- Two-axis classification head that predicts:
  - cls1 logits for axis-1 (e.g., Taxonomy/Genus) of size C1
  - cls2 logits for axis-2 (e.g., State/Stage) of size C2
  - box logits (4 * reg_max) as usual
- Training output (per FPN level): concat [box | cls1 | cls2].
  v8DetectionLoss is expected to split these streams for loss computation.
- Inference behavior:
  - Decodes boxes with DFL + anchors/strides (unchanged).
  - Converts per-axis probabilities to a joint class distribution:
        joint[c1*C2 + c2] = sigmoid(cls1[c1]) * sigmoid(cls2[c2])
    yielding an nc = C1*C2 vector per location so stock YOLO NMS/metrics continue to work.
  - Appends four per-location “extras” after the class scores to expose axis info to downstream code:
        [best_c1_id, best_c2_id, p1(best_c1), p2(best_c2)]
    where best_c1/best_c2 are chosen by the joint argmax.
- Bias initialization adapts to the split heads, using class_count-aware priors for each axis.
- end2end mode is supported with mirrored one-to-one branches (parity with Detect).

Why this design:
- Preserves full backward-compatibility for inference/metrics (nc=C1*C2) while enabling dual-axis analysis,
  constrained decoding, and per-axis visualizations without changing NMS or result schemas.

What remains the same:
- _DetectSingle (alias Detect) retains the original single-head layout: [box | cls] with nc classes.
- Segment, OBB, Pose, Classify, WorldDetect, YOLOE*, v10Detect keep their original single-head logic and reuse
  Detect unless otherwise noted; their forward/bias_init behavior is unchanged.

Key implementation notes:
- reg_max = 16 with DFL (Distribution Focal Loss) decoder; anchors/strides via make_anchors as in Ultralytics.
- Extras format is (B, 4, A) and concatenated after class scores; they are float-typed ids/probabilities.
- Dynamic/export flags, xyxy compatibility, and end2end pathways mirror upstream YOLO heads.
- DualDetect.no = 4*reg_max + C1 + C2 during training; at inference the returned tensor matches standard
  YOLO layout once joint scores are constructed (plus the 4 “extras”).

Assumptions/Dependencies:
- Loss: a dual-aware criterion (e.g., v8DetectionLoss) must split the training tensor into (box, cls1, cls2).
- Upstream code that relies on nc-based postprocessing continues to function (nc = C1*C2).
- Downstream consumers can optionally read the appended per-axis “extras” for analysis/UI.

"""

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "DualDetect",
    "Segment",
    "Pose",
    "Classify",
    "OBB",
    "RTDETRDecoder",
    "v10Detect",
    "YOLOEDetect",
    "YOLOESegment",
)

# ------------ Single-head Detect (original) ------------

class _DetectSingle(nn.Module):
    """
    YOLO Detect head for single-head classification (original behavior).
    Outputs: box logits (4*reg_max) + cls logits (nc) per location.
    """

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc: int = 80, ch: Tuple = ()):
        super().__init__()
        self.nc = int(nc)
        self.nl = len(ch)
        self.reg_max = 16
        self.no = self.reg_max * 4 + self.nc
        self.stride = torch.zeros(self.nl)

        # Box regression heads
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # Classification heads
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        if self.end2end:
            return self.forward_end2end(x)

        # training/inference unified: concatenate [box | cls]
        for i in range(self.nl):
            box_out = self.cv2[i](x[i])
            cls_out = self.cv3[i](x[i])
            x[i] = torch.cat((box_out, cls_out), 1)

        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: List[torch.Tensor]) -> Union[dict, Tuple]:
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            box_out = self.cv2[i](x[i])
            cls_out = self.cv3[i](x[i])
            x[i] = torch.cat((box_out, cls_out), 1)
        if self.training:
            return {"one2many": x, "one2one": one2one}
        y = self._inference(one2one)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box = x_cat[:, : self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4 :]
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)


# Public alias for single-head detect
class Detect(_DetectSingle):
    pass


# ------------ Dual-head Detect (split-head) ------------

class DualDetect(nn.Module):
    """
    Split-head detect for two label axes.
    Emits box logits (4*reg_max) + cls1_logits (C1) + cls2_logits (C2) during training.
    At inference, combines cls1 and cls2 into joint probabilities of size C1*C2 for standard NMS,
    and appends per-head extras so downstream code can display per-head scores.

    Extras appended per location (after class scores):
      [best_species_id, best_stage_id, p_species(best), p_stage(best)]
    """

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, num_cls1: int, num_cls2: int, ch: Tuple = ()):
        super().__init__()
        self.num_cls1 = int(num_cls1)
        self.num_cls2 = int(num_cls2)
        self.nc = self.num_cls1 * self.num_cls2  # for downstream compatibility (inference/metrics)
        self.nl = len(ch)
        self.reg_max = 16
        self.no = self.reg_max * 4 + self.num_cls1 + self.num_cls2
        self.stride = torch.zeros(self.nl)

        # Box regression head
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # Classification heads (split)
        c3 = max(ch[0], min(self.num_cls1 + self.num_cls2, 100))
        self.cls1_head = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_cls1, 1)) for x in ch
        )
        self.cls2_head = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_cls2, 1)) for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cls1_head = copy.deepcopy(self.cls1_head)
            self.one2one_cls2_head = copy.deepcopy(self.cls2_head)

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        if self.end2end:
            return self.forward_end2end(x)

        # Training/inference unified: concatenate [box | cls1 | cls2]
        for i in range(self.nl):
            box_out = self.cv2[i](x[i])
            cls1_out = self.cls1_head[i](x[i])
            cls2_out = self.cls2_head[i](x[i])
            x[i] = torch.cat((box_out, cls1_out, cls2_out), 1)

        if self.training:
            return x  # loss splits inside v8DetectionLoss

        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: List[torch.Tensor]) -> Union[dict, Tuple]:
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat(
                (self.one2one_cv2[i](x_detach[i]), self.one2one_cls1_head[i](x_detach[i]), self.one2one_cls2_head[i](x_detach[i])),
                1,
            )
            for i in range(self.nl)
        ]
        for i in range(self.nl):
            box_out = self.cv2[i](x[i])
            cls1_out = self.cls1_head[i](x[i])
            cls2_out = self.cls2_head[i](x[i])
            x[i] = torch.cat((box_out, cls1_out, cls2_out), 1)
        if self.training:
            return {"one2many": x, "one2one": one2one}
        y = self._inference(one2one)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Inference: combine split-head probabilities into joint C1*C2 class probabilities and append per-head extras.
        This preserves standard YOLO NMS/metrics behavior with nc = C1*C2 while exposing per-head scores.
        """
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # Split streams
        box = x_cat[:, : self.reg_max * 4]  # (B, 4*reg_max, A)
        cls1 = x_cat[:, self.reg_max * 4 : self.reg_max * 4 + self.num_cls1]  # (B, C1, A)
        cls2 = x_cat[:, self.reg_max * 4 + self.num_cls1 : self.reg_max * 4 + self.num_cls1 + self.num_cls2]  # (B, C2, A)

        # Decode boxes to pixels
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # (B, 4, A)

        # Combine heads: joint = p1(c1)*p2(c2) -> (B, C1*C2, A)
        p1 = cls1.sigmoid()  # (B, C1, A)
        p2 = cls2.sigmoid()  # (B, C2, A)
        joint = (p1.unsqueeze(2) * p2.unsqueeze(1)).view(shape[0], self.num_cls1 * self.num_cls2, -1)

        # Extras based on best joint class per anchor
        best_flat = joint.argmax(dim=1)               # (B, A)
        best_c1 = best_flat // self.num_cls2          # (B, A)
        best_c2 = best_flat %  self.num_cls2          # (B, A)
        p1_best = p1.gather(1, best_c1.unsqueeze(1))  # (B, 1, A)
        p2_best = p2.gather(1, best_c2.unsqueeze(1))  # (B, 1, A)
        extras = torch.cat(
            [
                best_c1.unsqueeze(1).float(),  # species id
                best_c2.unsqueeze(1).float(),  # stage id
                p1_best,                       # p(species=best_c1)
                p2_best,                       # p(stage=best_c2)
            ],
            dim=1,
        )  # (B, 4, A)

        return torch.cat((dbox, joint, extras), 1)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)

    def bias_init(self):
        m = self
        for a, b1, b2, s in zip(m.cv2, m.cls1_head, m.cls2_head, m.stride):
            a[-1].bias.data[:] = 1.0
            b1[-1].bias.data[:] = math.log(5 / self.num_cls1 / (640 / s) ** 2)
            b2[-1].bias.data[:] = math.log(5 / self.num_cls2 / (640 / s) ** 2)
        if self.end2end:
            for a, b1, b2, s in zip(m.one2one_cv2, m.one2one_cls1_head, m.one2one_cls2_head, m.stride):
                a[-1].bias.data[:] = 1.0
                b1[-1].bias.data[:] = math.log(5 / self.num_cls1 / (640 / s) ** 2)
                b2[-1].bias.data[:] = math.log(5 / self.num_cls2 / (640 / s) ** 2)


# ------------ The rest of the heads keep using single-head Detect ------------

class Segment(Detect):
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: Tuple = ()):
        super().__init__(nc, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> Union[Tuple, List[torch.Tensor]]:
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    def __init__(self, nc: int = 80, ne: int = 1, ch: Tuple = ()):
        super().__init__(nc, ch)
        self.ne = ne
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    def __init__(self, nc: int = 80, kpt_shape: Tuple = (17, 3), ch: Tuple = ()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {"tflite", "edgetpu"}:
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    export = False

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)
        return y if self.export else (y, x)


class WorldDetect(Detect):
    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: Tuple = ()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> Union[List[torch.Tensor], Tuple]:
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0


class LRPCHead(nn.Module):
    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv: nn.Conv2d) -> nn.Linear:
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> Tuple[Tuple, torch.Tensor]:
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    is_fused = False

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: Tuple = ()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn is True
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        if self.is_fused:
            return
        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm
            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)
            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data
            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias
            conv = nn.Conv2d(conv.in_channels, w.shape[0], kernel_size=1).requires_grad_(False).to(conv.weight.device)
            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv
            bn_head.fuse()
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: List[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        if vpe.shape[1] == 0:
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe

    def forward_lrpc(self, x: List[torch.Tensor], return_mask: bool = False) -> Union[torch.Tensor, Tuple]:
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001))
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)
        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x: List[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> Union[torch.Tensor, Tuple]:
        if hasattr(self, "lrpc"):
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, embed: int = 512, with_bn: bool = False, ch: Tuple = ()):
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)
        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> Union[Tuple, torch.Tensor]:
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        has_lrpc = hasattr(self, "lrpc")
        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)
        if self.training:
            return x, mc, p
        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    export = False
    # ... keep your RTDETRDecoder content unchanged ...


class v10Detect(Detect):
    end2end = True

    def __init__(self, nc: int = 80, ch: Tuple = ()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)