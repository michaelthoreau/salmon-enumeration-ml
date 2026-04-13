# Salmon Enumeration ML

Salmon counting from aerial/underwater imagery using [IOCFormer](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Indiscernible_Object_Counting_in_Underwater_Scenes_CVPR_2023_paper.pdf) (Conditional DETR + density-map branch), adapted to work with JSON point annotations.

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

```bash
# Default run (auto-names runs/exp0, exp1, …)
python train.py

# Common overrides
python -u train.py \
  --data_root=data/ \
  --epochs=1500 \
  --batch_size=8 \
  --lr=1e-5 \
  --lr_step=1200 \
  --save_path=runs/exp1

# Resume from checkpoint
python -u train.py --resume=runs/exp0/best.pth

```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `data/` | Root containing `train/`, `val/`, `test/` |
| `--epochs` | `1500` | Total epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `1e-5` | Learning rate (all params) |
| `--lr_step` | `1200` | Epoch where LR drops ×0.1 |
| `--save_path` | auto | Output directory for checkpoints & logs |
| `--resume` | — | Path to `.pth` checkpoint or run directory |
| `--limit_batches` | `0` | Cap batches per epoch (debug) |
| `--overfit` | `False` | Disable regularisation for overfitting tests |

Logs and TensorBoard events are written to the run directory.

```bash
tensorboard --logdir runs/
```

---

## Folder structure

| Path | Description |
|---|---|
| `annotate.py` | Annotation viewer/editor + inference UI (Flask) |
| `train.py` | Single-GPU training entry point |
| `train_distributed.py` | Multi-GPU distributed training (legacy) |
| `dataset.py` | Dataset / DataLoader - JSON annotations to tensors |
| `image.py` | load_data: reads JPEG + JSON annotation into kpoint map |
| `ot_loss.py` | Optimal-transport loss |
| `bregman_pytorch.py` | Sinkhorn/Bregman solver used by OT loss |
| `utils.py` | Logging and checkpointing helpers |
| `test.py` | Standalone evaluation script |
| `video_demo.py` | Video inference demo |
| `Networks/CDETR/` | Conditional DETR model (backbone, transformer, matcher) |
| `util/` | box_ops.py, misc.py |
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
