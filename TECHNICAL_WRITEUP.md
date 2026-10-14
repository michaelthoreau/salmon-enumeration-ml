# Salmon Enumeration from Aerial Video — Technical Writeup

## 1. Overview

This system counts sockeye salmon in aerial/underwater video footage using a DETR-based object detector with temporal context, followed by multi-object tracking and line-crossing enumeration.

**Pipeline:** Annotation → Training → Tiled Inference → NMS → Tracking → Line-Crossing Count → Annotated Video

---

## 2. Model Architecture

### 2.1 Base Model

- **Architecture:** DETR (DEtection TRansformer) — `facebook/detr-resnet-50` from HuggingFace Transformers
- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Object Queries:** 100 learnable queries
- **Classes:** 1 (salmon)
- **Input Resolution:** 512 × 512 pixels

### 2.2 Temporal Fusion ("Early" Mode)

The key architectural modification is **early temporal fusion** — concatenating context frames along the channel dimension before the backbone:

```
Current frame:    3 channels
Context frame 1:  3 channels  (t - 20 frames)
Context frame 2:  3 channels  (t - 40 frames)
─────────────────────────────────
Concatenated:     9 channels  →  ResNet-50 (modified first conv)
```

**Implementation:**
- The first `Conv2d` layer of ResNet-50 is expanded from 3 → 9 input channels
- New channel weights initialized as 0.1× the original weights (preserves pretrained behavior)
- Context frames are loaded at `max(0, frame_t - n × stride)` for n = 1..N
- Photos use zero context after normalization; training drops all context channels on 20% of samples. Failed video context fetches repeat the target frame.
- **Best model config:** N = 2 context frames, stride = 20 (i.e., frames 20 and 40 frames in the past)

**Why "early" fusion?**
- Simplest modification to the pretrained backbone
- No additional parameters beyond the expanded first conv
- Temporal information is available to all subsequent layers
- Used by the selected model; an alternative "transformer" fusion mode is implemented, but no controlled comparison is reported here

---

## 3. Training

### 3.1 Dataset

| Metric | Value |
|--------|-------|
| Source | Sockeye API (`salmon.projectkiwi.io`) — point-annotated photos and video frames |
| Positive tasks before crowding filter | **1,106** |
| Retained training tasks | **854** |
| Retained validation tasks | **153** (all video) |
| Excluded crowded tasks | **97 train / 2 validation** |
| Annotation format | Point annotations: `[[x, y], ...]` per frame |
| Point → Box conversion | Fixed 50×50 px box centered on each point |

**Data loading:**
- Frames fetched on-demand from the API and cached to disk (`/tmp/frame_cache/`)
- Flight-level 90/10 split with `seed=42`; task counts need not follow that ratio. Fully empty tasks are excluded before splitting.
- Tasks with more than 20% of GT points having a neighbor strictly closer than 50 model-input pixels are excluded after splitting, from both training and validation.
- `SameFrameBatchSampler`: each batch contains independently augmented crops from the same task. Temporal context is supplied within each sample, not between batch items; 854 training tasks means 854 steps per epoch.

### 3.2 Augmentation

| Augmentation | Details |
|-------------|---------|
| Random crop | 512×512 crop from task region; up to 15 attempts to include ≥1 point (not guaranteed); small regions are resized |
| Horizontal flip | 50% probability |
| Color jitter | 50% probability (brightness, contrast, saturation) |
| Context dropout | All context channels zeroed after normalization on 20% of training samples |

### 3.3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 5-epoch unfrozen fine-tune; selected checkpoint after the first epoch |
| Batch size | 16 |
| Image size | 512 × 512 |
| Box size | 50 px |
| Num queries | 100 |
| Score threshold | 0.25 |
| Context frames | 2 |
| Context stride | 20 |
| Temporal fusion | Early (channel concat) |
| Initialization | Best frozen salmon fine-tune; original base `facebook/detr-resnet-50` |
| Optimizer | AdamW |
| LR (backbone) | 2e-6 |
| LR (transformer / heads) | 2e-5 |
| Weight decay | 1e-4 |
| Warmup steps | 200 (LinearLR, 0.01 → 1.0) |
| Decay steps | 4,070 (CosineAnnealingLR, η_min = 0) |
| Precision / workers | CUDA AMP / 8 data-loader workers |
| Gradient clipping | max_norm = 0.1 |
| Loss | DETR Hungarian matching; weighted total L_CE + 5 L_bbox (L1) + 2 L_GIoU |

### 3.4 Training Results

| Metric | Selected Checkpoint | Epoch |
|--------|-----------|-------|
| Val Count MAE | **23.02** | 1 |
| Val Count RMSE | **31.45** | 1 |
| Val Point AP | **0.807** | 1 |
| Val Point Recall | **0.873** | 1 |

**Validation protocol:** Full-task tiled inference (512px tiles, 128px overlap) → predictions above confidence 0.25 → IoU NMS 0.5 and distance NMS 50px → compare to ground-truth points. Point metrics use one-to-one matching within 25 model-input pixels. AP integrates the confidence-ranked precision–recall curve only above the 0.25 cutoff, averaged equally across tasks; reported recall is maximum matching recall at that cutoff, not COCO AR.

Validation loss is evaluated separately on deterministic native-scale center crops (resizing only small task regions), using DETR's weighted total rather than summing diagnostics. Empty center crops can make this loss unrepresentative of full-task detection quality.

> **Note:** Epoch 1 here is the first completed epoch (console epoch 0). The selected unfrozen fine-tune has an average absolute count error of about 23 fish per validation task; this is not a per-task bound. Results cover 153 retained video tasks, exclude the crowded subset, and do not establish performance on photos or an independent test set. This historical checkpoint was selected by count MAE; the current trainer selects new `best.pth` checkpoints by weighted validation loss.

The released inference-only weights and provenance are in [release/salmon-detr/model.pth](release/salmon-detr/model.pth) and [release/salmon-detr/model_info.json](release/salmon-detr/model_info.json).

---

## 4. Inference Pipeline

### 4.1 Tiled Detection

```
┌─────────────────────────────────────────────────┐
│  Full Frame (e.g., 1920×1080)                   │
│                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Tile 1│ │Tile 2│ │Tile 3│  ← 512×512 tiles  │
│  │      │ │      │ │      │     128px overlap  │
│  └──────┘ └──────┘ └──────┘                    │
│  ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Tile 4│ │Tile 5│ │Tile 6│                    │
│  └──────┘ └──────┘ └──────┘                    │
│  ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Tile 7│ │Tile 8│ │Tile 9│                    │
│  └──────┘ └──────┘ └──────┘                    │
└─────────────────────────────────────────────────┘
```

- **Tile size:** 512 × 512 px
- **Overlap:** 128 px (stride = 384 px)
- **Batching:** Demo tiles processed in sub-batches of 8 by default; configurable
- **Context frames:** `t`, `max(0, t-20)`, and `max(0, t-40)` tiled identically and concatenated; photos use zero context
- **Frame skipping:** Demo detects every third source frame by default, while decoding intervening frames for temporal context

### 4.2 Non-Maximum Suppression (Two-Stage)

1. **IoU NMS** (threshold = 0.5): Standard box-based NMS to remove overlapping detections across tile boundaries
2. **Point-distance NMS** (demo / pipeline default = 45 px): Removes detections whose centers are strictly closer than 45 px (handles near-duplicates that IoU NMS misses)

The demo uses the pipeline's detection and tracking confidence default of 0.35.
The historical validation results above used confidence 0.25 and distance NMS
50 px; they are not measurements at the current demo defaults.

### 4.3 Multi-Object Tracking (BYTETracker)

- **Algorithm:** BYTETracker (Byte-level Tracking)
- **Motion model:** 8-state Kalman filter (constant velocity)
  - State: `[x, y, aspect_ratio, height, vx, vy, va, vh]`
  - Measurement: `[x, y, aspect_ratio, height]`
- **Matching:** Hungarian assignment on IoU costs between predicted and detected boxes, with high- and low-confidence association stages
- **Track lifecycle:** Activate → Track → Lost → Removed
- **Outlier rejection:** IQR-based velocity outlier rejection

### 4.4 Line-Crossing Enumeration

- A horizontal line is placed at a configurable position (default: 50% of frame height)
- Each track's position relative to the line is tracked per frame
- **Count logic:**
  - Track transitions from above → below: `count_down += 1`
  - Track transitions from below → above: `count_up += 1`
  - Net count = `count_down - count_up` (or vice versa depending on flow direction)

  **Fuzzy estimate:** Alongside actual crossings, tracking integrates a rough fish
  flux: moving-average active fish count × mean directional vertical velocity ÷
  source-frame height. Separate up/down rates are integrated over elapsed source
  video time, so skipped frames do not shorten the interval. The JSON metric
  `metrics.fuzzy_line_counts` retains cumulative up/down, net (down minus up), total
  crossings, and moving-average fish count. The overlay shows estimates to one
  decimal beside actual counts, using the panel's up-minus-down net convention.

  This assumes roughly uniform fish density across the frame and representative
  tracked motion. It is not an independent count or a unique-fish estimate: camera
  motion, uneven density, missed detections, and track failures can bias it. Missing
  velocity contributes zero rate; motion-history warmup is not extrapolated backward.

### 4.5 Video Rendering

The final annotated video includes:
- **Centroid dots** with motion arrows for each tracked fish
- **Count graph** (raw + moving average, window=30)
- **Motion panel** (speed, track count, direction compass)
- **Line-crossing panel** (up/down/net counts)
- Output height: 1080px by default, preserving aspect ratio
- Output frame rate: source FPS divided by inference stride (e.g., 30 → 10 FPS at stride 3), preserving playback speed. Detection progress counts source frames, including skipped frames, not just model-evaluated frames.

---

## 5. System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Sockeye API │────▶│   Training   │────▶│  Best Model  │
│ (854 tasks)  │     │ (fine-tune)  │     │  (epoch 1)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Photo /     │────▶│  Video Input │────▶│  Tiled DETR  │
│  Video Source│     │  (MP4)       │     │  + NMS       │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Annotated   │◀────│  Line-Cross  │◀────│  BYTETracker │
│  Output Video│     │  Counting    │     │  Tracking    │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 6. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Point annotations → fixed boxes | Simpler annotation workflow; 50px box captures fish scale |
| Early temporal fusion | Minimal modification; extra parameters only in the expanded first convolution |
| 2 context frames @ stride 20 | Covers 40 frames of history (~1.6s at 25fps, ~1.33s at 30fps) |
| Tiled inference | Handles arbitrary frame sizes; 512px matches training resolution |
| Two-stage NMS | IoU NMS handles tile overlap; distance NMS suppresses nearby duplicates, but can also suppress distinct neighboring fish |
| BYTETracker | Motion prediction and two-stage association maintain track continuity; no SORT benchmark reported |
| Line-crossing plus frame counts | Provides directional counts alongside per-frame counts; reliability depends on track continuity |

---

## 7. Reproduction

```bash
# Install inference dependencies and obtain Git LFS model weights
git lfs install
git lfs pull
pip install -r requirements-inference.txt

# Training additionally needs Sockeye credentials and training dependencies
pip install mlflow tensorboard requests
# Repeat the selected model's final fine-tune, if its frozen parent checkpoint is available
python train_detr_api.py \
  --resume=runs/detr_clean_labels_frozen_10ep/best.pth \
  --fresh_finetune=True --freeze_backbone=False \
  --save_path=runs/detr_clean_labels_unfrozen_reproduction \
  --num_context_frames=2 --context_stride=20 \
  --epochs=5 --batch_size=16 --workers=8 --amp=True \
  --lr=0.00002 --lr_backbone=0.000002 \
  --warmup_steps=200 --decay_steps=4070 \
  --max_gt_crowding=0.20 --crowding_distance=50 \
  --context_dropout=0.20 \
  --use_video_data=True --use_ortho_data=False

# Full video inference pipeline; output beside input, stride 3 by default
python demo.py input.mp4 --start-sec 0 --duration-sec 30

# Still-image inference; annotated PNG and JSON beside input
python demo.py input.jpg
```

The frozen parent checkpoint and training dataset are not included in the release.
Training pulls current API labels, so this recipe does not guarantee exact reproduction
of the historical checkpoint. The demo requires neither API credentials nor MLflow.

---

## 8. Limitations & Future Work

- **Count MAE ~23** on retained video validation tasks: This is an average, not a guarantee; missed fish and false detections can cancel in aggregate counts.
- **Crowded scenes excluded:** Metrics do not cover tasks above the 20% GT crowding cutoff. Empty tasks are also excluded by the trainer, so these results do not measure empty-scene false-positive behavior.
- **Single-class detection:** No distinction between species or size classes.
- **Fixed box size (50px):** May not generalize to different camera altitudes/zoom levels.
- **No independent test results:** Validation uses held-out flights and has been used repeatedly for model selection; photo performance is not established by these video-only metrics.
- **Future directions:**
  - Explore density-map approaches for high-density scenes (legacy IOCFormer code is deprecated)
  - Adaptive box size based on camera altitude
  - Multi-scale detection
  - Additional diverse aerial imagery, verified negative scenes, and an untouched flight-level test set
