import modal
from modal import FilePatternMatcher
import os

# Detached Modal wrapper for eval_detr_api.py, mirroring train_modal.py — lets evaluation
# runs survive a local disconnect the same way training runs already do via --detach.
app = modal.App("salmon-detr-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "tqdm",
        "pillow",
        "numpy",
        "requests",
        "python-dotenv",
        "fire",
        "timm",
    )
    .add_local_dir(
        ".",
        remote_path="/root/salmon-ml",
        ignore=~FilePatternMatcher("**/*.py")
    )
)

# Same persistent volume/cache as training — read-only from eval's perspective (loads
# checkpoints from here, never writes).
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
    timeout=28800, # 8-hour ceiling — matches train_modal.py after the earlier 4h timeout issue
)
def run_remote_eval(kwargs):
    import sys
    sys.path.append("/root/salmon-ml")
    os.chdir("/root/salmon-ml")

    from eval_detr_api import evaluate as eval_main
    eval_main(**kwargs)

@app.local_entrypoint()
def main(
    checkpoint: str = None,
    use_video_data: bool = True,
    use_ortho_data: bool = False,
    num_context_frames: int = 0,
    context_stride: int = 0,
    temporal_fusion: str = "early",
    flight_ids: str = None,
    score_thresh: float = 0.25,
    nms_dist: float = 40.0,
    iou_thresholds: str = "0.1,0.25,0.5,0.75",
    ortho_label: str = "fish",
):
    if not checkpoint:
        raise ValueError("--checkpoint is required")

    kwargs = {
        "checkpoint": checkpoint,
        "use_video_data": use_video_data,
        "use_ortho_data": use_ortho_data,
        "num_context_frames": num_context_frames,
        "context_stride": context_stride,
        "temporal_fusion": temporal_fusion,
        "score_thresh": score_thresh,
        "nms_dist": nms_dist,
        "iou_thresholds": iou_thresholds,
        "ortho_label": ortho_label,
    }
    if flight_ids is not None:
        kwargs["flight_ids"] = flight_ids

    print(f"Launching remote evaluation on Modal for checkpoint: {checkpoint}...")
    run_handle = run_remote_eval.spawn(kwargs)
    print(f"Task spawned successfully! Evaluation is running detached on Modal's infrastructure —")
    print(f"it will keep running even if this terminal disconnects or your machine loses network.")
    print(f"  App ID:  {run_handle.object_id}")
    print(f"  To view logs:  modal app logs {run_handle.object_id}")
