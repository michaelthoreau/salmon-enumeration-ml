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
To share one tracking server across machines, host it locally and tunnel it with ngrok:

```bash
# .env
NGROK_AUTHTOKEN=xxxxxxxx   # from https://dashboard.ngrok.com
# NGROK_DOMAIN=your-reserved-domain.ngrok-free.app   # optional, for a stable URL

python mlflow_server.py   # stores metadata in ./mlflow.db (sqlite), artifacts in ./mlruns
                          # prints a public https://...ngrok-free.app URL
```

Then point any run at it:

```bash
python train_detr_api.py --mlflow_tracking_uri=https://xxxx.ngrok-free.app ...
# or
export MLFLOW_TRACKING_URI=https://xxxx.ngrok-free.app
```

View runs at the same ngrok URL in a browser (it serves the MLflow UI too).
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



