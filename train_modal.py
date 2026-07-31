import modal
from modal import FilePatternMatcher
import os

# 1. Define Modal App and Container Image
app = modal.App("salmon-detr-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "tqdm",
        "tensorboard",
        "mlflow",
        "pillow",
        "numpy",
        "requests",
        "python-dotenv",
        "fire",
        "timm",
        "scipy",
        "opencv-python-headless"
    )
    .add_local_dir(
        ".",
        remote_path="/root/salmon-ml",
        ignore=~FilePatternMatcher("**/*.py")
    )
)

# 2. Persisted Volumes and Shared Storage
runs_volume = modal.Volume.from_name("salmon-runs-volume", create_if_missing=True)
cache_nfs = modal.NetworkFileSystem.from_name("salmon-cache-nfs", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/root/salmon-ml/runs": runs_volume,
    },
    network_file_systems={
        "/tmp/frame_cache": cache_nfs,
    },
    secrets=[modal.Secret.from_dotenv()],
    timeout=28800, # 8-hour timeout limit — raised after a run hit the previous 4h ceiling
                   # mid-epoch due to unusually slow data-fetch latency that session
)
def run_remote_training(model_type, kwargs):
    import sys
    sys.path.append("/root/salmon-ml")
    os.chdir("/root/salmon-ml")
    
    # Import main from selected training script
    if model_type == "cnn":
        from train_cnn_api import main as train_main
    elif model_type == "iocformer":
        from train_iocformer_api import main as train_main
    else:
        from train_detr_api import main as train_main
    
    # Execute the training pipeline with remote volume inputs and parameters
    train_main(**kwargs)
    
    # Commit volume changes to make sure logs and weights persist to Modal storage
    runs_volume.commit()

@app.local_entrypoint()
def main(
    model: str = "detr",
    num_context_frames: int = 1,
    context_stride: int = 5,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    warmup_steps: int = 100,
    decay_steps: int = 5000,
    batch_size: int = 8,
    name: str = None,
    save_path: str = None,
    temporal_fusion: str = "early",
    epochs: int = 20,
    val_every: int = 1,
    min_batches: int = 500,
    limit_batches: int = 0,
    workers: int = 8,
    verbose: bool = False,
    mlflow_tracking_uri: str = None,
    resume: str = None,
    fresh_finetune: bool = False,
    use_video_data: bool = True,
    use_ortho_data: bool = True,
    tile_server_url: str = None,
    tile_server_token: str = None,
    ortho_stitch_radius: int = 1,
):
    # save_path wins if given explicitly (e.g. to keep writing into an existing run's
    # folder); otherwise derive from --name or auto-generate as before.
    if save_path:
        run_name = save_path
    elif name:
        run_name = f"runs/{name}"
    else:
        suffix = f"_{temporal_fusion}" if (model == "detr" and num_context_frames > 0) else ""
        run_name = f"runs/exp_{model}_cf{num_context_frames}_cs{context_stride}{suffix}_lr{lr}"

    kwargs = {
        "num_context_frames": num_context_frames,
        "context_stride": context_stride,
        "lr": lr,
        "lr_backbone": lr_backbone,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "batch_size": batch_size,
        "save_path": run_name,
        "workers": workers,
        "verbose": verbose,
        "val_every": val_every,
        "use_video_data": use_video_data,
        "use_ortho_data": use_ortho_data,
        "ortho_stitch_radius": ortho_stitch_radius,
        "fresh_finetune": fresh_finetune,
    }
    if model == "detr":
        kwargs["temporal_fusion"] = temporal_fusion

    if epochs is not None:
        kwargs["epochs"] = epochs
    if min_batches is not None:
        kwargs["min_batches"] = min_batches
    if limit_batches > 0:
        kwargs["limit_batches"] = limit_batches
    # Falls back to MLFLOW_TRACKING_URI in .env (propagated as a Modal secret) if not passed here.
    if mlflow_tracking_uri is not None:
        kwargs["mlflow_tracking_uri"] = mlflow_tracking_uri
    if resume is not None:
        kwargs["resume"] = resume
    # Both fall back to MODAL_TILE_SERVER_URL/MODAL_TILE_TOKEN in .env (propagated as a
    # Modal secret) if not passed here — same as tile_server_url/token elsewhere.
    if tile_server_url is not None:
        kwargs["tile_server_url"] = tile_server_url
    if tile_server_token is not None:
        kwargs["tile_server_token"] = tile_server_token

    print(f"Launching remote {model} training on Modal (saving logs to {run_name})...")
    run_handle = run_remote_training.spawn(model, kwargs)
    print(f"Task spawned successfully! Training is running detached on Modal's infrastructure —")
    print(f"it will keep running even if this terminal disconnects or your machine loses network.")
    print(f"  App ID:  {run_handle.object_id}")
    print(f"  To view logs:  modal app logs {run_handle.object_id}")
