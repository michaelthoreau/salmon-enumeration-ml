"""
Orchestrate DETR salmon box detection on Modal.com with parallel GPU workers,
then perform tracking and rendering locally.

Setup
-----
1. Install Modal locally:
       pip install modal
2. Configure Modal:
       modal setup
3. Run the pipeline:
       modal run infer_modal.py \
           --checkpoint runs/exp0/best.pth \
           --video data/input/videos/DJI_20260630093119_0001_V.mp4 \
           --num-workers 4 \
           --detect-height 1080 \
           --infer-every 1
"""

import os
import sys
from pathlib import Path
import modal

app = modal.App("salmon-inference")

# Define the container image configuration
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
        "numpy",
        "fire",
        "transformers",
        "timm",
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        # Recursive ignore lambda ensures NO environment, log, or data files are uploaded
        ignore=lambda p: (
            p.name in ["runs", "data", "data_big", "debug", ".git", ".venv", ".gemini", "node_modules", "app", "__pycache__"]
            or any(d in p.parts for d in ["runs", "data", "data_big", "debug", ".git", ".venv", ".gemini", "node_modules", "app", "__pycache__"])
            or p.suffix in [".pyc", ".pyo", ".mp4", ".avi", ".mov", ".jpg", ".png", ".zip", ".tar", ".gz"]
        ),
    )
)

# Persistent volume for checkpoints and videos
volume = modal.Volume.from_name("salmon-assets", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/assets": volume},
    timeout=1800,
)
def detect_chunk(
    checkpoint_rel: str,
    video_rel: str,
    start_sec: float,
    duration_sec: float | None,
    img_size: int,
    score_thresh: float,
    iou_thresh: float,
    num_queries: int,
    companion_mode: str | None,
    batch_frames: int,
    tile_batch_size: int,
    infer_every: int,
    detect_height: int | None
) -> dict:
    """
    GPU Worker running DETR inference on a segment slice in the cloud container.
    """
    import sys
    sys.path.append("/app")
    from infer_video_detr import Pipeline
    
    checkpoint_path = f"/assets/{checkpoint_rel}"
    video_path = f"/assets/{video_rel}"
    
    pipeline = Pipeline()
    res = pipeline.detect(
        checkpoint=checkpoint_path,
        video=video_path,
        output_json=None,
        start_sec=start_sec,
        duration_sec=duration_sec,
        img_size=img_size,
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        num_queries=num_queries,
        companion_mode=companion_mode,
        batch_frames=batch_frames,
        tile_batch_size=tile_batch_size,
        infer_every=infer_every,
        detect_height=detect_height
    )
    return res


@app.local_entrypoint()
def main(
    checkpoint: str = "runs/exp0/best.pth",
    video: str = "data/input/videos/DJI_20260630093119_0001_V.mp4",
    output_video: str = "out_modal_tracked.mp4",
    output_json: str = "detections_modal.json",
    num_workers: int = 4,
    detect_height: int = 1080,
    infer_every: int = 1,
    img_size: int = 512,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.5,
    num_queries: int = 100,
    companion_mode: str = None,
    batch_frames: int = 1,
    tile_batch_size: int = 16,
    track_thresh: float = 0.4,
    min_box_score: float = 0.1,
    ma_window: int = 30,
    traverse_seconds: float = 25.0,
    graph_height: int = 280,
    graph_width: int = 700,
    dot_radius: int = 6,
    render_height: int = 1080
):
    import cv2
    import json
    
    # Import from local project directory
    sys.path.append(str(Path(__file__).parent))
    from infer_video_detr import Pipeline

    # 1. Inspect local video structure to calculate durations
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open local video: {video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    duration_sec = total_frames / fps
    chunk_duration = duration_sec / num_workers

    print(f"Video duration: {duration_sec:.2f}s | Splitting into {num_workers} slices...")

    # 2. Upload only the specific checkpoint and video to the Volume
    checkpoint_rel = f"checkpoints/{Path(checkpoint).name}"
    video_rel = f"videos/{Path(video).name}"
    
    print(f"Uploading local checkpoint and video to Modal Volume 'salmon-assets'...")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(checkpoint, checkpoint_rel)
        batch.put_file(video, video_rel)
    print("Upload complete!")

    # 3. Package worker arguments
    tasks = []
    for i in range(num_workers):
        start_sec = i * chunk_duration
        dur = chunk_duration if i < num_workers - 1 else duration_sec - (i * chunk_duration)
        tasks.append((
            checkpoint_rel, video_rel, start_sec, dur, img_size,
            score_thresh, iou_thresh, num_queries, companion_mode,
            batch_frames, tile_batch_size, infer_every, detect_height
        ))

    print(f"Launching {num_workers} parallel GPU workers (T4) on Modal...")
    # detect_chunk.starmap maps argument tuples across cloud workers concurrently
    results = list(detect_chunk.starmap(tasks))
    print("All Modal detection tasks complete! Merging results...")

    # 4. Merge detection chunk logs
    merged_detections = {
        "video_path": video,
        "fps": fps,
        "width": results[0]["width"],
        "height": results[0]["height"],
        "detections": {}
    }
    for res in results:
        merged_detections["detections"].update(res["detections"])

    with open(output_json, 'w') as f:
        json.dump(merged_detections, f)
    print(f"Saved merged detections to {output_json}")

    # 5. Local CPU Tracking (Stage 2)
    print("\n=== STAGE 2: LOCAL TRACKING ===")
    pipeline = Pipeline()
    track_data = pipeline.track(
        detection_jsons=merged_detections,
        output_track_json=None,
        track_thresh=track_thresh,
        min_box_score=min_box_score,
        ma_window=ma_window
    )

    # 6. Local Rendering (Stage 3)
    print("\n=== STAGE 3: LOCAL RENDERING ===")
    pipeline.render(
        track_json=track_data,
        video=video,
        output_video=output_video,
        start_sec=0.0,
        duration_sec=None,
        graph_height=graph_height,
        graph_width=graph_width,
        dot_radius=dot_radius,
        traverse_seconds=traverse_seconds,
        ma_window=ma_window,
        render_height=render_height
    )
    print(f"\nPipeline run complete! Annotated video output: {output_video}")
