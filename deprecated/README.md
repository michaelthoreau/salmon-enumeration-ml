# Deprecated Code

This directory preserves historical implementations. DETR is the only maintained
model architecture, and all Modal integrations are no longer maintained.

| Path | Archived Functionality |
|---|---|
| `train_cnn_api.py` | CNN heatmap model and API-based trainer |
| `train_iocformer_api.py` | IOCFormer API-based trainer |
| `Networks/CDETR/` | IOCFormer / Conditional DETR architecture |
| `util/` | Utilities used by the archived CDETR architecture |
| `bregman_pytorch.py` | Legacy optimal-transport solver |
| `config.py`, `image.py`, `utils.py` | Legacy configuration, data loading, and helpers |
| `annotate.py` | Legacy annotation server with CDETR-only inference |
| `train_modal.py` | Modal training launcher for all model types, including DETR |
| `infer_modal.py` | Parallel GPU video inference on Modal |
| `eval_modal.py` | Remote DETR evaluation on Modal |
| `predict_modal.py` | Modal prediction API and deployment configuration |
| `.modalignore` | Modal upload exclusions |

The files are retained for reference, not as supported entry points. Their original
imports, relative paths, and deployment configuration have been preserved; moving
them here does not guarantee they remain runnable. Some depend on active root-level
code such as `dataset_api.py` or `train_detr_api.py`.

Use `train_detr_api.py` or `train_detr.py` at the repository root for training.
`infer_video_detr.py` and `eval_detr_api.py` remain active for local inference and
evaluation. No active Python entry point requires the Modal SDK.

The local API trainer and evaluator retain `MODAL_TILE_SERVER_URL` and
`MODAL_TILE_TOKEN` as compatible environment names for HTTP tile-server access.
Archiving this code does not stop or delete any already-deployed Modal apps,
volumes, or other cloud resources.

No checkpoints, datasets, or MLflow records were deleted or moved in this change.