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
        "pillow",
        "numpy",
        "requests",
        "python-dotenv",
        "fire",
        "timm",
        "scipy"
    )
    .add_local_dir(
        ".",
        remote_path="/root/salmon-ml",
        ignore=~FilePatternMatcher("**/*.py")
    )
)

# 2. Persisted Volumes
runs_volume = modal.Volume.from_name("salmon-runs-volume", create_if_missing=True)
cache_volume = modal.Volume.from_name("salmon-cache-volume", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S", # Default GPU, can be overridden with local_entrypoint main CLI argument
    volumes={
        "/root/salmon-ml/runs": runs_volume,
        "/root/salmon-ml/frame_cache": cache_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400, # 4-hour timeout limit
)
def run_remote_training(kwargs):
    import sys
    sys.path.append("/root/salmon-ml")
    os.chdir("/root/salmon-ml")
    
    # Import main from training script
    from train_detr_api import main as train_main
    
    # Execute the training pipeline with remote volume inputs and parameters
    train_main(**kwargs)
    
    # Commit volume changes to make sure logs and weights persist to Modal storage
    runs_volume.commit()
    cache_volume.commit()

@app.function(
    image=image,
    volumes={"/root/salmon-ml/runs": runs_volume},
)
@modal.web_server(6006)
def run_tensorboard():
    import subprocess
    import threading
    import time
    
    print("Starting TensorBoard server on port 6006...", flush=True)
    cmd = "tensorboard --logdir /root/salmon-ml/runs --port 6006 --bind_all"
    subprocess.Popen(cmd, shell=True)
    
    # Run the volume reloader in a background thread so the function can return immediately
    def reloader():
        while True:
            time.sleep(15)
            try:
                runs_volume.reload()
            except Exception:
                pass
                
    threading.Thread(target=reloader, daemon=True).start()
    print("TensorBoard started, returning control to Modal.", flush=True)

@app.local_entrypoint()
def main(
    num_context_frames: int = 1,
    context_stride: int = 5,
    lr: float = 1e-4,
    warmup_steps: int = 100,
    decay_steps: int = 5000,
    batch_size: int = 8,
    name: str = None,
    temporal_fusion: str = "early",
):
    if name:
        run_name = f"runs/{name}"
    else:
        run_name = f"runs/exp_cf{num_context_frames}_cs{context_stride}_lr{lr}_{temporal_fusion}"
        
    kwargs = {
        "num_context_frames": num_context_frames,
        "context_stride": context_stride,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "batch_size": batch_size,
        "save_path": run_name,
        "temporal_fusion": temporal_fusion,
    }
    
    print(f"Launching remote training on Modal (saving logs to {run_name})...")
    run_remote_training.remote(kwargs)
