# Dual‑Head YOLO (Detection) + Classification — Extended Ultralytics Implementation

This repository adds a dual‑head (two‑axis) detection capability to Ultralytics YOLO while keeping standard classification fully supported. It also includes quality‑of‑life improvements around metrics, plotting, NMS, and validation.

- Detection: Dual‑head model for two categorical axes (e.g., Species/Genus and State/Substrate). Joint classes are still supported for compatibility (mAP, PR curves, NMS), with additional per‑axis metrics and confusion matrices.
- Classification: Standard top‑1 / top‑5 training, validation, and inference, with plotting and export utilities.

The modifications remain backward compatible with Ultralytics’ CLI and Python APIs.

<img width="1280" height="685" alt="yolo_dual_examples" src="https://github.com/user-attachments/assets/1d0c069e-f920-4423-8ec3-8c16e66b7944" />

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Repository Highlights](#repository-highlights)
- [Concepts: Dual‑Head Detection](#concepts-dualhead-detection)
- [Datasets](#datasets)
  - [Detection (Dual‑Head)](#detection-dualhead)
  - [Classification](#classification)
- [Model Configuration](#model-configuration)
- [Training](#training)
  - [Detection (Dual‑Head)](#training-detection-dualhead)
  - [Classification](#training-classification)
- [Validation & Metrics](#validation--metrics)
- [Inference & Visualization](#inference--visualization)
- [Results & Logging](#results--logging)
- [Advanced Topics](#advanced-topics)
  - [Two‑Stage Object‑Based NMS](#twostage-objectbased-nms)
  - [Joint↔Axes Mapping](#jointaxes-mapping)
  - [Latent Space Analysis (Optional)](#latent-space-analysis-optional)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Features

- Dual‑head detection head (`DualDetect`) that predicts:
  - Box distribution (DFL) + IoU loss
  - Classification on two separate axes (C1 and C2)
  - Joint class compatibility for standard YOLO pipelines
- Per‑axis validation:
  - mAP/PR/F1 curves per axis (e.g., Taxonomy, Substrate)
  - Per‑axis confusion matrices (optional, auto‑built)
- Two‑stage, object‑based NMS (optional, enabled by default for predict/val):
  - Stage‑1: class‑agnostic NMS using an “objectness” built from per‑axis probabilities
  - Stage‑2: per‑species cleanup (removes duplicates within same axis‑1 class)
- Visualization upgrades:
  - Two‑line box labels show per‑axis names and probabilities
  - Priority results panel (`results_priority.png`) focusing on dual‑head metrics
- Classification kept intact:
  - Standard top‑1 / top‑5 metrics, plotting, and exports

---

## Installation

1) Create/activate a Python environment (Python 3.8+ recommended).
2) Install Ultralytics and dependencies:
```bash
pip install ultralytics
# If you’re developing locally, install in editable mode inside this repo:
# pip install -e .
```
3) Optional extras for COCO mAP evaluation:
```bash
pip install faster-coco-eval>=1.6.7
```

GPU with CUDA is recommended for training. CPU/MPS are supported with reduced performance.

---

## Repository Highlights

Key files modified/added for dual‑head detection:

- utils
  - `ultralytics/utils/loss.py`: v8DetectionLoss extended for dual‑head (two assigners, per‑axis cls losses).
  - `ultralytics/utils/metrics.py`: Axis‑agnostic AP computation and plotting reused for per‑axis metrics.
  - `ultralytics/utils/ops.py`: Two‑stage object‑based NMS, extras preservation.
  - `ultralytics/utils/plotting.py`: Two‑line labels, priority results figure.
- engine
  - `ultralytics/engine/results.py`: Boxes can carry “extras”; plotting uses per‑axis info.
  - `ultralytics/engine/trainer.py`: Dual‑head label preprocessing, CSV metric key renames.
  - `ultralytics/engine/validator.py`: Base validator with dual‑head CM framework + optional latent analysis.
- models
  - `ultralytics/nn/tasks.py`: DualDetect integrated across model parsing, loss routing, and metadata.
  - `ultralytics/models/yolo/detect/predict.py`: Two‑stage NMS by default, extras‑aware Results.
  - `ultralytics/models/yolo/detect/train.py`: Dual‑head trainer, validator wiring, priority plots.
  - `ultralytics/models/yolo/detect/val.py`: Dual‑head validation with per‑axis metrics and plotting.
- config
  - `ultralytics/cfg/models/v8/yolov8.yaml`: Example with `DualDetect` [C1, C2] head.

---

## Concepts: Dual‑Head Detection

A single detection uses two categorical axes:

- Axis‑1 (C1): e.g., Species/Taxonomy
- Axis‑2 (C2): e.g., State/Substrate

The joint class space is C1×C2. We preserve joint compatibility for standard YOLO workflows (NMS, mAP), and enrich validation/visualization with per‑axis metrics and labels.

During inference, each detection row is:
```
[x1, y1, x2, y2, conf, cls_joint, extras...]
```
Where `extras` typically are:
```
[axis1_id, axis2_id, p_axis1, p_axis2]
```

This enables:
- Two‑line labels: “<Axis‑1 name> p1” on line 1, “<Axis‑2 name> p2” on line 2
- Objectness score from per‑axis probabilities for class‑agnostic NMS
- Per‑axis metrics/plots

---

## Datasets

### Detection (Dual‑Head)

You may provide labels as:

- Combined joint ids: `cls` is (N,) with values in `[0..C1*C2-1]`
- Split pairs: `cls` is (N, 2) with `[cls1, cls2]` where `0..C1-1` and `0..C2-1`

In the dataset YAML, add (recommended for human‑readable outputs):

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

# Channel count (3) and nc can be inferred/overridden; nc may be C1*C2 when using joint names.
nc: 75  # e.g., 15 * 5
names: {}  # optional joint names; auto-built from head names if omitted

# Per-axis names (recommended)
names_head1:
  0: "SpeciesA"
  1: "SpeciesB"
  ...
  14: "SpeciesO"   # C1=15

names_head2:
  0: "Fresh"
  1: "Worn"
  2: "Damaged"
  3: "Pristine"
  4: "Unknown"     # C2=5

# Optional: exclude "Unknown" from stage-only metrics (combined metrics unaffected)
drop_unknown: true

# Optional mapping(s)
# 1) joint index -> axes ids (sparse-aware)
# joint_to_heads:
#   - [h1_id, h2_id]   # for j=0
#   - [h1_id, h2_id]   # for j=1
#   ...
#
# 2) axes -> joint index (alternative)
# heads_to_joint:
#   "0_0": 0
#   "0_1": 1
#   ...
```

Notes:
- If `names_head1/2` are provided, the validator/trainer will auto-build a `names` mapping of size `C1*C2`: `"{h1} - {h2}"`.
- If you only have joint names, training still works. Per‑axis metrics/labels will fallback to generic names.

### Classification

Use a standard ImageNet‑style folder structure:

```
dataset/
  train/
    class_0/
      img1.jpg
      ...
    class_1/
      ...
  val/
    class_0/
    class_1/
```

And a simple data YAML (optionally; CLI can infer from paths):

```yaml
path: /path/to/dataset
train: train
val: val
names:
  0: "cat"
  1: "dog"
  ...
```

---

## Model Configuration

An example YOLOv8 config with a dual‑head:

```yaml
# ultralytics/cfg/models/v8/yolov8.yaml
# ...
head:
  # ...
  - [[15, 18, 21], 1, DualDetect, [15, 5]]  # C1=15 (axis-1), C2=5 (axis-2)
```

- Joint class space is `15*5=75`.
- You can switch back to single‑head by replacing `DualDetect` with `Detect` and setting `nc`.

---

<img width="587" height="807" alt="dual_head_yolo_architecture2" src="https://github.com/user-attachments/assets/feadf0c8-5a7f-4cff-ad84-e50f1d8f5247" />

## Training

### Training (Detection, Dual‑Head)

CLI:
```bash
# Train from YAML model that uses DualDetect
yolo train task=detect model=ultralytics/cfg/models/v8/yolov8.yaml data=path/to/data.yaml imgsz=640 epochs=100 batch=16

# Resume
yolo train resume model=path/to/last.pt
```

Python:
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
model.train(data="path/to/data.yaml", imgsz=640, epochs=100, batch=16)
```

Notes:
- You can provide GT labels as `(N,2)` or combined joint ids `(N,)`. The trainer will:
  - If `(N,2)`: split into `cls1`/`cls2` for loss, and flatten to joint ids for combined metrics if needed.
  - If `(N,)`: split into `cls1`/`cls2` using `C2` inferred from `names_head2` or the model head.
- Losses logged per step (order matches results CSV and priority plot):
  - `train/bbox`, `train/dfl`, `train/regression` (= bbox+dfl), `train/cls_genus`, `train/cls_substrate`, `train/overall`

Multi‑GPU:
```bash
# device can be "0,1,2,3" etc.
yolo train task=detect model=... data=... device=0,1 batch=32
```

### Training (Classification)

CLI:
```bash
# Minimal classification training
yolo train task=classify model=yolo11n-cls.yaml data=path/to/cls_data.yaml epochs=50 imgsz=224 batch=64
```

Python:
```python
from ultralytics import YOLO
model = YOLO("yolo11n-cls.yaml")
model.train(data="path/to/cls_data.yaml", imgsz=224, epochs=50, batch=64)
```

Outputs:
- Top‑1 / Top‑5 accuracy
- Plots (`results.png`), logs, and checkpoints under `runs/classify/...`

---

## Validation & Metrics

Detection (Dual‑Head):
- Combined metrics (joint ids) produced as usual: Precision/Recall/mAP50/mAP50‑95
- Per‑axis metrics:
  - Head‑1 and head‑2 DetMetrics (PR/F1/Recall curves, AP per class)
  - Optional “Unknown” filtering on head‑2 (e.g., Substrate) via `drop_unknown: true`
- Confusion matrices:
  - Combined: filtered to only classes present in the validation set for cleaner visuals
  - Per‑axis: saved under `runs/.../{h1,h2}/confusion_matrix*.png`

Classification:
- Top‑1 and Top‑5 accuracy reported and plotted.

CLI:
```bash
# Detect
yolo val task=detect model=path/to/best.pt data=path/to/data.yaml imgsz=640

# Classify
yolo val task=classify model=path/to/best-cls.pt data=path/to/cls_data.yaml imgsz=224
```

---

## Inference & Visualization

Detection (Dual‑Head):

CLI:
```bash
yolo predict task=detect model=path/to/best.pt source=path/to/images imgsz=640 conf=0.25
```

Python:
```python
from ultralytics import YOLO
model = YOLO("path/to/best.pt")
results = model("path/to/image.jpg", imgsz=640, conf=0.25)
for r in results:
    # r.boxes.extra may contain [axis1_id, axis2_id, p_axis1, p_axis2]
    im = r.plot()  # two-line labels auto-rendered if extras and head names are present
```

Highlights:
- Two‑stage object‑based NMS is enabled by default for predict/val.
- Two‑line labels show per‑axis names and probabilities if available.
- Results are saved to `runs/detect/predict.../` with annotated images.

Classification:

CLI:
```bash
yolo predict task=classify model=path/to/best-cls.pt source=dir_or_file imgsz=224
```

Python:
```python
from ultralytics import YOLO
model = YOLO("path/to/best-cls.pt")
results = model("img.jpg", imgsz=224)
print(results[0].probs.top1, results[0].probs.top5)
```

---

## Results & Logging

- CSV: `runs/.../results.csv`
  - For detect, metric keys are renamed for readability:
    - Joint metrics become `metrics/*(joint)` (from `metrics/*(B)`)
    - Per‑axis metric series appear as `Genus Dual/*`, `Substrate Dual/*`
- Plots:
  - `results.png`: legacy plot
  - `results_priority.png`: mandatory panels for dual‑head workflows:
    - Losses: bbox, dfl, cls_genus, cls_substrate, overall
    - Precision (Genus, Substrate)
    - Recall (Genus, Substrate)
    - mAP50 (Genus, Substrate)
  - Per‑axis PR/F1/Precision/Recall curves saved when `plots=True`
- Confusion Matrices:
  - Combined: normalized and raw
  - Head‑wise: saved under `{save_dir}/h1/` and `{save_dir}/h2/`

---

## Advanced Topics

### Two‑Stage Object‑Based NMS

Enabled in predict/val:

- Stage‑1: Class‑agnostic NMS with an objectness score derived from per‑axis probabilities, e.g.:
  - `product` (default): `p_axis1 * p_axis2`
  - `geomean`, `min`, `max_joint` (fallbacks)
- Stage‑2: Per‑axis‑1 cleanup (e.g., per‑species), removing near‑duplicates within the same species.
- Tunables (via `non_max_suppression` / CLI args):
  - `stage1_conf_thres` (defaults to `conf`), `stage1_iou_thres` (defaults to `iou`)
  - `stage2_iou_thres`
  - `object_score_mode` in {product, geomean, min, max_joint}
  - `species_cleanup` bool
  - `species_col_in_extras` (index of axis‑1 id in extras)
  - `num_cls2_for_species` (fallback derivation if extras lack axis‑1 id)

### Joint↔Axes Mapping

- Preferred: Provide `joint_to_heads` (list) or `heads_to_joint` (dict) in your dataset YAML.
- Fallback arithmetic:
  - If `C1` and `C2` are known: `h1 = j // C2`, `h2 = j % C2`
  - An inference step can flip axes if needed (heuristic “h1_first” vs “h2_first”).

### Latent Space Analysis (Optional)

Opt‑in via env var:
```bash
# enable
export LATENT=1
# optional knobs:
# LATENT_LAYER, LATENT_EPOCHS, LATENT_PERIOD, etc.
```
Purpose:
- Collect joint labels per batch (reconstructed from Nx2 if needed) and analyze latent embeddings over epochs.
- Outputs are summarized into validation stats and saved under the run directory.

---

## Troubleshooting

- Shape errors in training:
  - Ensure detection labels are either `(N,)` joint ids or `(N,2)` pairs consistently per sample.
  - If using `(N,2)`, make sure `names_head2` (C2) count matches your labels; the trainer infers `C2` from data/model.
- “Unknown” filtering has no effect:
  - Confirm `drop_unknown: true` and that `names_head2` contains an exact `"Unknown"` entry.
- No two‑line labels:
  - Ensure model returns `extras` and dataset provides `names_head1/2`. Without these, labels fallback to joint text.
- Java/CUDA OOM:
  - Reduce `batch`, lower `imgsz`, or enable gradient accumulation by increasing `nbs` relative to `batch`.

---

## License

AGPL‑3.0 — consistent with Ultralytics license. See https://ultralytics.com/license for details.

If you use this work in research, please cite Ultralytics YOLO and any additional datasets/models accordingly.

---
