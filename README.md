# Salmon Enumeration ML

Salmon counting from aerial/underwater imagery using DETR, adapted to work with JSON point annotations. DETR is the only maintained model architecture; older model implementations and all Modal integrations are archived in [deprecated/](deprecated/README.md).

---

## Setup

```bash
conda activate salmon-ml
pip install -r requirements.txt   # if present, otherwise deps are torch, torchvision, flask, fire, tqdm, scipy, opencv-python
```

---

### Annotation JSON format

```json
{
  "filename": "abc123.jpg",
  "width": 1920,
  "height": 1080,
  "points": [[x, y], ...]
}
```

---

## Training

There are two entry points for training: the new **API-based Sockeye Trainer** (which pulls data directly from the Sockeye labeling server) and the **Local File Trainer** (legacy).

### 1. API-based Training (Core DETR)

To train the DETR model on task annotations fetched directly from the Sockeye API, run `train_detr_api.py`. Make sure your `.env` contains the correct `SOCKEYE_API_URL` and `SOCKEYE_API_KEY` before starting.

```bash
# Core standard run (starts at batch_size=64, falls back on OOM, defaults to 1 context frame)
python train_detr_api.py

# DETR training with 2 context frames, a stride of 10, and custom learning rate
python train_detr_api.py \
  --num_context_frames=2 \
  --context_stride=10 \
  --lr=1e-4

# Resume from checkpoint
python train_detr_api.py --resume=runs/exp_detr_cf2_cs10_lr0.0001/best.pth
```

#### Core Arguments for `train_detr_api.py`

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `64` | Initial batch size. Will auto-halve and retry if CUDA OOM is encountered. |
| `--num_context_frames` | `1` | Number of context frames (N) to append before the target frame (0 to disable). |
| `--context_stride` | `5` | Index separation (P) between context frames (e.g. 5 means load frame $F$, $F-5$, $F-10$, ...). |
| `--temporal_fusion` | `early` | How context frames are merged (`early` channels concatenation or `transformer` layers). |
| `--min_batches` | `500` | Minimum batches per epoch. Samples dataset with replacement if smaller to limit validation overhead. |
| `--epochs` | auto | Number of epochs to train. Defaults to `(warmup_steps + decay_steps) / min_batches` (~22 epochs). |
| `--lr` | `1e-4` | Learning rate for Transformer blocks. |
| `--lr_backbone` | `1e-5` | Learning rate for ResNet backbone. |
| `--save_path` | auto | Target folder under `runs/` for model checkpoints. |
| `--resume` | — | Path to `.pth` file or run directory to load model and resume training from. |
| `--max_gt_crowding` | `0.20` | Exclude tasks with more than this fraction of crowded GT points from training and validation. Set to `1` to disable. |
| `--crowding_distance` | `50` | Strict GT neighbor distance in model-input pixels used for task crowding. |
| `--context_dropout` | `0.20` | Probability of dropping all context channels for a training sample; set to `0` to disable. |

Temporal training retains the fixed channel layout (target RGB plus context RGB).
For 20% of training samples, all context channels are zeroed after normalization,
while the target and GT remain unchanged. Validation never uses context dropout.
Photos at frame zero continue to repeat the same frame; failed context fetches
emit a warning and repeat the target in training collation and tiled validation.
`context_policy.json` records this policy in the run directory. No availability
mask is added to the model, so existing checkpoints retain their architecture.
Other inference entry points are unchanged; their missing-context policies must
be checked separately before deploying a model trained with this policy.

Crowded GT points have another GT point strictly closer than the crowding distance.
Task crowding is measured in the audit's coordinate system: native pixels when both
crop dimensions are at least the input size, otherwise coordinates are scaled to
the square model input. Filtering happens after the flight split, before sampling.
Tasks above 20% crowding are excluded entirely, not relabeled as empty background.
All targets in retained tasks remain unchanged; training loss does not apply NMS.
Exclusions and retained split sizes are written to `crowding_exclusions.json` in
the run directory. Validation counts and loss exclude these tasks too, so metrics
are for the solvable subset and are not directly comparable to older all-task runs.
Use the full-dataset audit separately to monitor excluded scenes.

Use a new run name for this changed data policy. For an existing checkpoint, use
`--fresh_finetune=True` with `--resume` to start a new optimizer/scheduler run rather
than silently continuing the previous experiment.

### Training Speed Profiling

Training metrics are sent to MLflow in one request per reporting step. Use
`--log_every=10` to reduce remote logging overhead further. A conservative local
configuration is:

```bash
python train_detr_api.py \
  --num_context_frames=2 --context_stride=20 --epochs=100 \
  --workers=4 --log_every=10
```

To measure speed without validation or checkpoint writes:

```bash
python train_detr_api.py \
  --num_context_frames=2 --context_stride=20 \
  --benchmark_batches=128 --profile_warmup=16 \
  --workers=4 --log_every=10 \
  --mlflow=speed_test --save_path=runs/speed_test
```

`--profile=True` enables synchronized CUDA phase timing during a normal run.
Summaries are written as `speed_epoch_<epoch>.json` in the run directory and
logged to TensorBoard/MLflow under `speed/`. Validation and checkpoint wall times
are logged separately. Profiling adds synchronization overhead; leave it off for
ordinary training.

`--amp=True` enables optional float16 training; `--freeze_backbone=True` freezes
the backbone. Validation uses the same AMP setting and batch size as training,
including batched tiles for counting and the crop-based loss pass. Validation
retains deterministic preprocessing and evaluation mode, without gradients.
Validation loss uses fixed native-resolution center crops for large tasks, rather
than shrinking a whole task into the model input; small tasks are still resized.
The logged `val/loss` is DETR's weighted total (`out.loss`), not the sum of raw
components, which incorrectly included the non-loss `cardinality_error` diagnostic.
Component losses are logged separately and averaged with batch-size weights.
`val/ap` and `val/ar` evaluate full-task tiled detections with one-to-one point
matching at 25 model-input pixels and distance NMS at `--crowding_distance` (50
by default). These are task-mean, score-cutoff-truncated point AP and maximum
recall, not COCO box metrics. Small-task GT is scaled to the same coordinates as
predictions. Each validation writes `validation_debug_<epoch>.json` with task
metrics, loss batches, input shapes, and counts of targets exceeding query capacity.
The corrected validation loss and NMS counts are not comparable to earlier logs.
Neither optimization is enabled by default. See [TRAINING_SPEED.md](TRAINING_SPEED.md)
for measured results, AMP caveats, and small-dataset recommendations.

---

### 2. Local File Training (DETR)

To train on local JPEGs and JSON annotation files in the `data/` folder, run `train_detr.py`:

```bash
# Default run (auto-names runs/exp0, exp1, …)
python train_detr.py

# Common overrides
python -u train_detr.py \
  --data_root=data/ \
  --epochs=1500 \
  --batch_size=8 \
  --lr=1e-5 \
  --save_path=runs/exp1
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `data/` | Root directory containing `train/`, `val/`, `test/` subfolders. |
| `--epochs` | `1500` | Total epochs to train. |
| `--batch_size` | `8` | Training batch size. |
| `--lr` | `1e-5` | Learning rate (all params). |
| `--save_path` | auto | Output directory for checkpoints & logs |
| `--resume` | — | Path to `.pth` checkpoint or run directory |
| `--limit_batches` | `0` | Cap batches per epoch (debug) |
| `--overfit` | `False` | Disable regularisation for overfitting tests |

Logs and TensorBoard events are written to the run directory.

```bash
tensorboard --logdir runs/
```

### Shared MLflow tracking

Training scripts (`train_detr_api.py`, etc.) also log params/metrics/artifacts to MLflow.
Run the tracking server locally; no tunnel, public hostname, or ngrok token is needed:

```bash
# .env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python mlflow_server.py   # stores metadata in ./mlflow.db (sqlite), artifacts in ./mlruns
                          # serves http://127.0.0.1:5000
```

Then point any run at it:

```bash
python train_detr_api.py --mlflow_tracking_uri=http://127.0.0.1:5000 ...
# or
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

View runs at http://127.0.0.1:5000 in a browser. Checkpoint uploads stay on this
machine. Existing SQLite metadata and artifacts are retained.
All Modal integrations are deprecated; use the local DETR training entry points above.

---

## Folder structure

| Path | Description |
|---|---|
| `train_detr_api.py` | DETR training from Sockeye API annotations |
| `train_detr.py` | DETR training from local JPEGs and JSON annotations |
| `dataset_api.py` | Shared API dataset and frame caching |
| `eval_detr_api.py` | DETR evaluation |
| `infer_video_detr.py` | Tiled DETR video inference, tracking, and rendering |
| `tracker.py` | Tracking and line-crossing enumeration |
| `mlflow_server.py` | MLflow tracking server |
| `debug_overlay.py` | Annotation debugging overlays |
| `models/` | Saved DETR checkpoints and provenance metadata |
| `deprecated/` | Archived CNN/IOCFormer code, legacy annotation server, helpers, and all Modal integrations |
| `data/input/images/` | Images to run inference on |
| `data/input/videos/` | Videos to run inference on |
| `data/train/raw/images/` | Raw/unlabeled images queued for labelling |
| `data/train/images/` | Labelled training images (*.jpg) |
| `data/train/annotations/` | Training point annotations (uuid.json) |
| `data/val/images/` | Validation images |
| `data/val/annotations/` | Validation annotations |
| `data/test/images/` | Test images |
| `data/test/annotations/` | Test annotations |
| `runs/expN/` | Training outputs: best.pth, train.log, debug_overlays/ |
| `debug/` | Inference overlay debug images |

---

## Deprecated Modal Integrations

Modal training, inference, evaluation, and the prediction API are preserved in
[deprecated/](deprecated/README.md) for historical reference and are no longer maintained.
The active workflows use local DETR training, `eval_detr_api.py`, and `infer_video_detr.py`.

Local API training and evaluation still accept `MODAL_TILE_SERVER_URL` and
`MODAL_TILE_TOKEN` for backward compatibility with existing HTTP tile servers.
These environment variables do not require the Modal SDK.

---

## Evaluation Results

### Model Comparison
Below is a comparison of performance on the validation task set (105 tasks) between the best context model and the best model without context:

| Model Config | Run ID | Count MAE | Count RMSE | Max Abs Error | AP @ 0.10 | AP @ 0.25 | AP @ 0.50 | AP @ 0.75 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **With Context** (N=2, Stride=20) | `2a4ca563` | 13.2667 | 17.6141 | 63 | 0.7474 | 0.6216 | 0.2412 | 0.0443 |
| **Without Context** (N=0) | `00412b53` | 8.3143 | 12.2081 | 50 | 0.7576 | 0.6102 | 0.1909 | 0.0116 |

---

### DETR with Context (Best Model)
- **Model Directory:** `models/best_with_context/`
- **Run ID:** `2a4ca563725447bf92a921a35f678f76`
- **Context Config:** `num_context_frames: 2`, `context_stride: 20`

#### Count Metrics (Score Threshold = 0.25)
- **Mean Absolute Error (MAE):** 13.2667
- **Root Mean Squared Error (RMSE):** 17.6141
- **Max Absolute Error:** 63

#### Detection Metrics (NMS Centroid Distance = 40.0px)
| IoU Thresh | Precision | Recall | F1-Score | AP | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.10** | 0.5547 | 0.8211 | 0.6621 | 0.7474 | 2286 | 1835 | 498 |
| **0.25** | 0.5050 | 0.7475 | 0.6028 | 0.6216 | 2081 | 2040 | 703 |
| **0.50** | 0.3142 | 0.4652 | 0.3751 | 0.2412 | 1295 | 2826 | 1489 |
| **0.75** | 0.1284 | 0.1900 | 0.1532 | 0.0443 | 529 | 3592 | 2255 |

---

### DETR without Context (Best Model)
- **Model Directory:** `models/best_without_context/`
- **Run ID:** `00412b537a4942619cf7b8086f3720f8`
- **Context Config:** `num_context_frames: 0`, `context_stride: 0`

#### Count Metrics (Score Threshold = 0.25)
- **Mean Absolute Error (MAE):** 8.3143
- **Root Mean Squared Error (RMSE):** 12.2081
- **Max Absolute Error:** 50

#### Detection Metrics (NMS Centroid Distance = 40.0px)
| IoU Thresh | Precision | Recall | F1-Score | AP | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.10** | 0.6682 | 0.7812 | 0.7203 | 0.7576 | 2175 | 1080 | 609 |
| **0.25** | 0.5948 | 0.6954 | 0.6412 | 0.6102 | 1936 | 1319 | 848 |
| **0.50** | 0.3250 | 0.3800 | 0.3504 | 0.1909 | 1058 | 2197 | 1726 |
| **0.75** | 0.0768 | 0.0898 | 0.0828 | 0.0116 | 250 | 3005 | 2534 |



