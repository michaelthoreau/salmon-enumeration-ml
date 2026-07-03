"""
Train salmon model on Modal with S3 data and TensorBoard.

Setup
-----
1. Create a Modal secret named "aws-secret" in the Modal dashboard
   (https://modal.com/secrets) with these keys:
       AWS_ACCESS_KEY_ID
       AWS_SECRET_ACCESS_KEY
       AWS_REGION          (e.g. "us-east-1")

   The IAM user/role needs s3:GetObject, s3:ListBucket on
   salmon-enumeration-ml-dev.

2. Run training:
       modal run train_modal.py
   or with overrides:
       modal run train_modal.py --epochs 100 --batch-size 4

3. Live TensorBoard (separate terminal):
       modal serve train_modal.py::tensorboard_server
   Modal prints a HTTPS URL — open it in your browser.
   The server reads from the same Volume the trainer writes to.

4. Download a checkpoint:
       modal volume get salmon-runs exp0/best.pth ./best.pth
"""

import subprocess

import modal

app = modal.App("salmon-training")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121"
        " --extra-index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "tqdm",
        "pillow",
        "scipy",
        "opencv-python-headless",
        "h5py",
        "tensorboard",
        "numpy",
        "fire",
        "pynvml",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=["runs", "data", "data_big", "debug", ".git", "__pycache__", "node_modules", "app"],
    )
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
BUCKET_NAME   = "salmon-enumeration-ml-dev"
BUCKET_PREFIX = "test/"   # mounts only this prefix as the root of /data
DATA_PATH     = "/data"
RUNS_PATH     = "/runs"

# Volume for checkpoints + TensorBoard event files.
# Using a Volume (not the S3 mount) because SummaryWriter opens files in
# append mode, which CloudBucketMount does not support.
runs_volume = modal.Volume.from_name("salmon-runs", create_if_missing=True)

aws_secret = modal.Secret.from_name("aws-secret")

# ---------------------------------------------------------------------------
# Data uploader — run with: modal run train_modal.py::upload_data
# Uploads data/train and data/val (images + annotations) to the S3 bucket.
# ---------------------------------------------------------------------------
@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("boto3")
        .add_local_dir("./data", remote_path="/local_data", ignore=["raw", "unlabelled", "__MACOSX"], copy=True)
    ),
    secrets=[aws_secret],
    timeout=600,
)
def upload_data():
    import boto3, os

    s3 = boto3.client("s3")
    uploaded = 0
    for root, _dirs, files in os.walk("/local_data"):
        for fname in files:
            local_path = os.path.join(root, fname)
            # e.g. /local_data/train/images/foo.jpg -> test/train/images/foo.jpg
            rel = os.path.relpath(local_path, "/local_data")
            s3_key = BUCKET_PREFIX + rel
            print(f"Uploading {rel} -> s3://{BUCKET_NAME}/{s3_key}")
            s3.upload_file(local_path, BUCKET_NAME, s3_key)
            uploaded += 1
    print(f"\nDone. Uploaded {uploaded} files.")


# ---------------------------------------------------------------------------
# Bucket inspector — run with: modal run train_modal.py::ls_bucket
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[aws_secret],
    volumes={
        DATA_PATH: modal.CloudBucketMount(
            bucket_name=BUCKET_NAME,
            key_prefix=BUCKET_PREFIX,
            secret=aws_secret,
            read_only=True,
        ),
    },
)
def ls_bucket(depth: int = 3):
    import os
    for root, dirs, files in os.walk(DATA_PATH):
        level = root.replace(DATA_PATH, "").count(os.sep)
        if level >= depth:
            dirs.clear()
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root) or DATA_PATH}/")
        for f in files[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  ... ({len(files)} files total)")


@app.local_entrypoint()
def ls():
    ls_bucket.remote()


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 12,          # 12 h hard cap
    secrets=[aws_secret],
    volumes={
        DATA_PATH: modal.CloudBucketMount(
            bucket_name=BUCKET_NAME,
            key_prefix=BUCKET_PREFIX,
            secret=aws_secret,
            read_only=True,
        ),
        RUNS_PATH: runs_volume,
    },
)
def train(
    epochs: int       = 1500,
    batch_size: int   = 32,
    lr: float         = 1e-5,
    save_path: str    = "exp0",
    resume: str       = None,
    min_batches: int  = 0,
    limit_batches: int = 0,
):
    import os, sys

    # Make project source importable
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Delay import until we're inside the container with the right sys.path
    import train as train_mod  # noqa: PLC0415

    full_save_path = os.path.join(RUNS_PATH, save_path)

    train_mod.main(
        save_path=full_save_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        data_root=DATA_PATH,
        resume=os.path.join(RUNS_PATH, resume) if resume else None,
        min_batches=min_batches,
        limit_batches=limit_batches,
    )

    # Flush volume writes so they're visible to other containers
    runs_volume.commit()


@app.local_entrypoint()
def main(
    epochs: int       = 1500,
    batch_size: int   = 8,
    lr: float         = 1e-5,
    save_path: str    = "exp0",
    resume: str       = None,
    min_batches: int  = 0,
    limit_batches: int = 0,
):
    train.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path=save_path,
        resume=resume,
        min_batches=min_batches,
        limit_batches=limit_batches,
    )


# ---------------------------------------------------------------------------
# TensorBoard web server
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={RUNS_PATH: runs_volume},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(6006)
def tensorboard_server():
    """Serve TensorBoard from the runs Volume. Start with: modal serve train_modal.py::tensorboard_server"""
    subprocess.Popen(
        [
            "tensorboard",
            "--logdir", RUNS_PATH,
            "--host", "0.0.0.0",
            "--port", "6006",
            "--reload_interval", "30",
        ]
    )
