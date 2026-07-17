import os
import io
import re
import glob
import math
import time
import urllib.parse
import hashlib
from types import SimpleNamespace
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

# 1. Define Modal App and Image configuration
app = modal.App("salmon-prediction-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .run_commands(
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121"
        " --extra-index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers",
        "tqdm",
        "pillow",
        "numpy",
        "requests",
        "python-dotenv",
        "timm",
        "scipy",
        "opencv-python-headless",
        "fastapi",
        "av",
    )
    .add_local_dir(
        ".",
        remote_path="/root/salmon-ml",
        ignore=~modal.FilePatternMatcher("**/*.py")
    )
    .add_local_file(
        "models/best.pth",
        remote_path="/root/salmon-ml/models/best.pth"
    )
    .add_local_file(
        "models/model_info.json",
        remote_path="/root/salmon-ml/models/model_info.json"
    )
)

# Import heavy libraries in global scope so they are captured in the memory snapshot.
# This eliminates the >20,000 file-read operations torch requires on every cold start.
with image.imports():
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.ops import nms as tv_nms
    import cv2
    import numpy as np

# IMAGENET mean and std for normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_CHECKPOINT = "/root/salmon-ml/models/best.pth"

# ---------------------------------------------------------------------------
# Helper functions for Model Configuration and Loading
# ---------------------------------------------------------------------------

def _expand_first_conv_6ch(model, in_channels: int = 6):
    """Expand first Conv2d channel count to in_channels."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            old = module
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight.clone()
                for start in range(3, in_channels, 3):
                    end = min(start + 3, in_channels)
                    new_conv.weight[:, start:end] = old.weight[:, :end - start].clone() * 0.1
                if old.bias is not None:
                    new_conv.bias.copy_(old.bias)
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv.to(old.weight.device))
            print(f'  Expanded first conv ({name}): 3 → {in_channels} channels.')
            return

def detect_num_context_frames(state_dict):
    """Detects the number of context frames from the shape of the first conv layer."""
    in_channels = 3
    # Look for the conv1 layer (both CDETR/IOCFormer and DETR names)
    for k, v in state_dict.items():
        if "conv1.weight" in k and len(v.shape) == 4:
            in_channels = v.shape[1]
            break
    num_context_frames = (in_channels // 3) - 1
    return max(0, num_context_frames)

def detect_context_stride(checkpoint_path: str):
    """Reads stride from a model_info.json sidecar if present, then falls back to path heuristics."""
    import json
    # Check for a model_info.json sidecar sitting next to the checkpoint
    sidecar_path = os.path.join(os.path.dirname(checkpoint_path), "model_info.json")
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                info = json.load(f)
            stride = info.get("context_stride")
            if stride is not None:
                print(f"Loaded context_stride={stride} from {sidecar_path}")
                return int(stride)
        except Exception:
            pass
    # Fall back to path heuristics
    if "2848702fb616448ba202bc7c34673da4" in checkpoint_path:
        return 20
    match = re.search(r"cs(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return 10  # default fallback

def detect_model_type(state_dict):
    """Differentiates between CNN, CDETR/IOCFormer, and standard DETR by checking keys."""
    if any("up5" in k for k in state_dict.keys()) or any("fusion" in k for k in state_dict.keys()):
        return "cnn"
    if any("layers_density" in k for k in state_dict.keys()):
        return "iocformer"
    return "detr"

def find_latest_checkpoint(runs_dir="/root/salmon-ml/runs"):
    """Scans the runs folder to locate the latest best.pth checkpoint."""
    checkpoints = glob.glob(os.path.join(runs_dir, "**", "best.pth"), recursive=True)
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(runs_dir, "best.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best.pth checkpoint found in {runs_dir}")
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def resolve_checkpoint(checkpoint_req: str) -> str:
    """Resolves checkpoint request, downloading from MLflow if it starts with mlflow-artifacts: or runs:/."""
    if not checkpoint_req:
        return find_latest_checkpoint()
        
    if checkpoint_req.startswith("mlflow-artifacts:") or checkpoint_req.startswith("runs:/"):
        # Create a persistent cache directory on the runs volume to avoid downloading on cold starts
        match = re.search(r"([0-9a-fA-F]{32})", checkpoint_req)
        run_id = match.group(1) if match else "unknown"
        filename = os.path.basename(checkpoint_req)
        
        local_cache_dir = f"/root/salmon-ml/runs/mlflow_cache/{run_id}"
        local_cache_path = os.path.join(local_cache_dir, filename)
        
        if os.path.exists(local_cache_path):
            print(f"Using cached MLflow checkpoint from persistent volume: {local_cache_path}")
            return local_cache_path
            
        print(f"Downloading checkpoint from MLflow to persistent volume: {checkpoint_req}")
        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        os.makedirs(local_cache_dir, exist_ok=True)
        try:
            local_path = mlflow.artifacts.download_artifacts(checkpoint_req, dst_path=local_cache_dir)
            print(f"Downloaded MLflow artifact to: {local_path}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint from MLflow ({checkpoint_req}): {e}")
            
    if not os.path.isabs(checkpoint_req) and not checkpoint_req.startswith("/root/salmon-ml"):
        return os.path.join("/root/salmon-ml/runs", checkpoint_req)
    return checkpoint_req

def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Loads DETR or IOCFormer dynamically based on the checkpoint structure."""
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', {}))
    if not state_dict:
        raise ValueError("Could not find state dict in checkpoint.")

    model_type = detect_model_type(state_dict)
    num_context_frames = detect_num_context_frames(state_dict)
    context_stride = detect_context_stride(checkpoint_path)

    print(f"Detected parameters: type={model_type}, context_frames={num_context_frames}, context_stride={context_stride}")

    if model_type == "cnn":
        from train_cnn_api import LightweightCNNWithLateFusion
        model = LightweightCNNWithLateFusion(num_context_frames=num_context_frames)
        model.load_state_dict(state_dict)
    elif model_type == "iocformer":
        from train_iocformer_api import build_model
        
        cfg = SimpleNamespace(
            backbone='resnet50',
            dilation=True,
            lr_backbone=1e-5,
            position_embedding='sine',
            masks=False,
            frozen_weights=None,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            pre_norm=False,
            transformer_flag='merge3',
            dm_count=True,
            branch_merge=True,
            branch_merge_way=2,
            channel_point=3,
            only_dm=False,
            with_weights=False,
            aux_loss=True,
            cls_loss_coef=2,
            point_loss_coef=5,
            giou_loss_coef=2,
            mask_loss_coef=1,
            dice_loss_coef=1,
            set_cost_class=2,
            set_cost_point=5,
            set_cost_giou=2,
            focal_alpha=0.25,
            dataset_file='crowd_data',
            device='cuda',
        )
        
        # Override queries if saved in checkpoint
        if 'query_embed.weight' in state_dict:
            cfg.num_queries = state_dict['query_embed.weight'].shape[0]

        model, _, _ = build_model(cfg)

        if num_context_frames > 0:
            from Networks.CDETR.backbone import expand_first_conv
            expand_first_conv(model.backbone, in_channels=3 * (1 + num_context_frames))
            
        model.load_state_dict(state_dict)
    else:
        # Standard DETR
        from transformers import DetrConfig, DetrForObjectDetection
        
        num_queries = 100
        if 'query_embed.weight' in state_dict:
            num_queries = state_dict['query_embed.weight'].shape[0]
            
        config = DetrConfig(
            num_labels=1,
            num_queries=num_queries,
        )
        model = DetrForObjectDetection(config)
        
        # Check temporal fusion mode
        temporal_fusion = "early"
        if any("temporal_embeddings" in k for k in state_dict.keys()):
            temporal_fusion = "transformer"
            
        if num_context_frames > 0:
            if temporal_fusion == "early":
                _expand_first_conv_6ch(model, in_channels=3 * (1 + num_context_frames))
            elif temporal_fusion == "transformer":
                from train_detr_api import DetrModelWithTemporalFusion
                model.model = DetrModelWithTemporalFusion(model.model, num_context_frames)

        # Map state dict keys dynamically to handle mismatches
        model_keys = set(model.state_dict().keys())
        new_state_dict = {}
        for k, v in state_dict.items():
            if k not in model_keys:
                if "model.backbone.model." in k:
                    k_alt = k.replace("model.backbone.model.", "model.backbone.conv_encoder.model.", 1)
                    if k_alt in model_keys: k = k_alt
                elif "model.backbone.conv_encoder.model." in k:
                    k_alt = k.replace("model.backbone.conv_encoder.model.", "model.backbone.model.", 1)
                    if k_alt in model_keys: k = k_alt
                
                if "self_attn.o_proj." in k:
                    k_alt = k.replace("self_attn.o_proj.", "self_attn.out_proj.", 1)
                    if k_alt in model_keys: k = k_alt
                elif "self_attn.out_proj." in k:
                    k_alt = k.replace("self_attn.out_proj.", "self_attn.o_proj.", 1)
                    if k_alt in model_keys: k = k_alt
                    
                if "encoder_attn.o_proj." in k:
                    k_alt = k.replace("encoder_attn.o_proj.", "encoder_attn.out_proj.", 1)
                    if k_alt in model_keys: k = k_alt
                elif "encoder_attn.out_proj." in k:
                    k_alt = k.replace("encoder_attn.out_proj.", "encoder_attn.o_proj.", 1)
                    if k_alt in model_keys: k = k_alt
                
                if ".mlp.fc1." in k:
                    k_alt = k.replace(".mlp.fc1.", ".fc1.", 1)
                    if k_alt in model_keys: k = k_alt
                elif ".fc1." in k:
                    k_alt = k.replace(".fc1.", ".mlp.fc1.", 1)
                    if k_alt in model_keys: k = k_alt
                
                if ".mlp.fc2." in k:
                    k_alt = k.replace(".mlp.fc2.", ".fc2.", 1)
                    if k_alt in model_keys: k = k_alt
                elif ".fc2." in k:
                    k_alt = k.replace(".fc2.", ".mlp.fc2.", 1)
                    if k_alt in model_keys: k = k_alt
                    
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    if device == "cuda":
        model = model.cuda()
    return model, model_type, num_context_frames, context_stride

def extract_frame_from_url(presigned_url: str, frame_num: int, quality: int = 95):
    """Extracts a single frame JPEG from a presigned URL using PyAV and returns it as a PIL Image."""
    import av
    # 1. Open the video stream from the presigned S3 URL
    container = av.open(presigned_url)
    if not container.streams.video:
        raise ValueError(f"No video streams found in the source: {presigned_url}")
    stream = container.streams.video[0]
    
    # 2. Calculate the target timestamp based on frame index & stream parameters
    fps = float(stream.average_rate) if stream.average_rate else 29.97
    time_base = float(stream.time_base) if stream.time_base else 1 / 90000
    target_sec = frame_num / fps
    target_ts = int(target_sec / time_base)
    
    # 3. Seek to the keyframe preceding or at the target timestamp
    container.seek(target_ts, stream=stream, backward=True)
    
    # 4. Decode forward until reaching the exact target timestamp
    result_frame = None
    for frame in container.decode(video=0):
        if frame.pts is not None and frame.pts >= target_ts:
            result_frame = frame
            break
            
    if not result_frame:
        raise ValueError(f"Frame {frame_num} not found in the video stream.")
        
    # 5. Convert the frame to a PIL Image (RGB format)
    pil_img = result_frame.to_image()
    
    # 6. (Optional) Match API JPEG compression/quality
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------------------------
# Sockeye API Frame Download Helper
# ---------------------------------------------------------------------------

def download_frame_from_sockeye(api_url: str, api_key: str, file_id: str, frame_num: int):
    """Downloads a single frame JPEG from Sockeye API and returns it as PIL Image."""
    import requests
    url = f"{api_url.rstrip('/')}/files/{file_id}/frame/{frame_num}?quality=95"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error_info = "No attempts made"
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 200:
                return Image.open(io.BytesIO(res.content)).convert("RGB")
            
            last_error_info = f"HTTP status {res.status_code}: {res.text[:120]}"
            if res.status_code == 429:
                time.sleep(1.0)
            else:
                time.sleep(0.5)
        except Exception as e:
            last_error_info = f"Request Exception: {str(e)}"
            time.sleep(1.0)
            
    raise RuntimeError(f"Failed to download frame {frame_num} for file {file_id} from Sockeye API after 3 attempts. Last error: {last_error_info}")

# ---------------------------------------------------------------------------
# Tiling Helper Functions
# ---------------------------------------------------------------------------

def _make_tiles(pil_img, S: int, companion_crops, num_context_frames: int):
    """Generates non-overlapping tiles, mapping back parameters to scale them appropriately."""
    _normalize_fn = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    def _tile_tensor(main_crop, comp_crops=None):
        t = _normalize_fn(T.functional.to_tensor(main_crop))
        if num_context_frames > 0:
            tensors_to_cat = [t]
            for c_crop in (comp_crops or []):
                if c_crop is not None:
                    t_n = _normalize_fn(T.functional.to_tensor(c_crop))
                else:
                    t_n = torch.zeros_like(t)
                tensors_to_cat.append(t_n)
            expected_count = num_context_frames
            while len(tensors_to_cat) < 1 + expected_count:
                tensors_to_cat.append(torch.zeros_like(t))
            t = torch.cat(tensors_to_cat, dim=0)
        return t

    W, H = pil_img.size
    if W < S or H < S:
        pil_r  = pil_img.resize((S, S), Image.BILINEAR)
        comp_rs = [c.resize((S, S), Image.BILINEAR) if c else None for c in (companion_crops or [])]
        return [(_tile_tensor(pil_r, comp_rs), 0, 0, W, H)]
        
    def origins(dim):
        os_ = list(range(0, dim - S + 1, S))
        if not os_ or os_[-1] + S < dim:
            os_.append(dim - S)
        return sorted(set(os_))
        
    tiles = []
    for y0 in origins(H):
        for x0 in origins(W):
            crop = pil_img.crop((x0, y0, x0 + S, y0 + S))
            comp_crops = [c.crop((x0, y0, x0 + S, y0 + S)) if c else None for c in (companion_crops or [])]
            tiles.append((_tile_tensor(crop, comp_crops), x0, y0, W, H))
    return tiles

# ---------------------------------------------------------------------------
# Web Mercator tile math helpers
# ---------------------------------------------------------------------------

def _latlon_to_tile(z: int, lat: float, lng: float):
    """Convert lat/lng to tile (x, y) at zoom z."""
    n = 2 ** z
    x = int((lng + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y

def _pixel_to_latlon(col: float, row: float, z: int, x_min: int, y_min: int, tile_size: int = 256):
    """Convert pixel (col, row) in stitched image → (lng, lat)."""
    n = 2 ** z
    tx = x_min + col / tile_size
    ty = y_min + row / tile_size
    lng = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lng, lat

# ---------------------------------------------------------------------------
# Modal Class-based Web Server
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_dotenv()],
    timeout=600,
)
class SalmonInferenceServer:
    @modal.enter(snap=True)
    def load_model_cpu(self):
        """Load model weights into CPU RAM. This runs before the memory snapshot is taken,
        so the deserialized weights are captured in the snapshot and don't need to be
        re-read from disk on subsequent cold starts."""
        self.cached_models = {}
        try:
            checkpoint_path = resolve_checkpoint(DEFAULT_CHECKPOINT)
            model_info = load_model_from_checkpoint(checkpoint_path, device="cpu")
            self.cached_models[checkpoint_path] = model_info
            print(f"Model loaded to CPU RAM and ready for snapshot.")
        except Exception as e:
            print(f"Pre-caching to CPU failed: {e}")

    @modal.enter(snap=False)
    def setup_cuda(self):
        """After snapshot restore, re-import torch (to reinitialise CUDA state) and
        move all cached models from CPU RAM to GPU. This is fast (~1-2s) compared
        to the full load-from-disk path."""
        import torch  # re-import to reinitialise CUDA availability after snapshot restore
        for path, (model, *rest) in list(self.cached_models.items()):
            model.cuda()
            print(f"Moved model to CUDA: {path}")

    def get_model(self, checkpoint_path: str):
        """Loads and retrieves a model, caching it in memory. On-demand loads go to CUDA directly."""
        if checkpoint_path not in self.cached_models:
            model_info = load_model_from_checkpoint(checkpoint_path, device="cuda")
            self.cached_models[checkpoint_path] = model_info
        return self.cached_models[checkpoint_path]

    def _infer_image(self, pil_img, model, model_type, num_context_frames,
                     score_thresh, iou_thresh, companion_crops=None, S=512):
        """Tile a PIL image, run inference, apply NMS. Returns pixel-space points."""
        tiles = _make_tiles(pil_img, S, companion_crops or [], num_context_frames)
        all_boxes_xyxy = []
        all_scores = []
        box_size = 50

        with torch.no_grad():
            for tile_t, tx, ty, orig_W, orig_H in tiles:
                pv = tile_t.unsqueeze(0).cuda()
                pm = torch.ones(1, S, S, dtype=torch.long).cuda()

                if model_type == "iocformer":
                    out_all = model(pv)
                    d6 = out_all[0] if isinstance(out_all, list) else out_all
                    probs = d6['pred_logits'].sigmoid()[0, :, 1].cpu()
                    pts = d6['pred_points'][0].cpu()
                    keep = probs > score_thresh
                    if keep.any():
                        p_kept = pts[keep]; s_kept = probs[keep]
                        if orig_W < S or orig_H < S:
                            cx = p_kept[:, 0] * orig_W; cy = p_kept[:, 1] * orig_H
                            w = torch.full_like(cx, box_size * (orig_W / S)); h = torch.full_like(cy, box_size * (orig_H / S))
                        else:
                            cx = tx + p_kept[:, 0] * S; cy = ty + p_kept[:, 1] * S
                            w = torch.full_like(cx, box_size); h = torch.full_like(cy, box_size)
                        all_boxes_xyxy.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                        all_scores.append(s_kept)

                elif model_type == "cnn":
                    from train_cnn_api import extract_peaks
                    out_heatmap = model(pv).cpu()[0]
                    peaks = extract_peaks(out_heatmap.unsqueeze(0), threshold=score_thresh)[0]
                    if peaks:
                        pts = torch.tensor([[p[0], p[1]] for p in peaks], dtype=torch.float)
                        s_kept = torch.tensor([p[2] for p in peaks], dtype=torch.float)
                        if orig_W < S or orig_H < S:
                            cx = pts[:, 0] * (orig_W / S); cy = pts[:, 1] * (orig_H / S)
                            w = torch.full_like(cx, box_size * (orig_W / S)); h = torch.full_like(cy, box_size * (orig_H / S))
                        else:
                            cx = tx + pts[:, 0]; cy = ty + pts[:, 1]
                            w = torch.full_like(cx, box_size); h = torch.full_like(cy, box_size)
                        all_boxes_xyxy.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                        all_scores.append(s_kept)

                else:  # detr
                    out = model(pixel_values=pv, pixel_mask=pm)
                    probs = out.logits.cpu()[0].softmax(-1)[:, 0]
                    b = out.pred_boxes.cpu()[0]
                    keep = probs > score_thresh
                    if keep.any():
                        b = b[keep]; s = probs[keep]
                        if orig_W < S or orig_H < S:
                            cx = b[:, 0] * orig_W; cy = b[:, 1] * orig_H
                            w = torch.full_like(cx, box_size * (orig_W / S)); h = torch.full_like(cy, box_size * (orig_H / S))
                        else:
                            cx = tx + b[:, 0] * S; cy = ty + b[:, 1] * S
                            w = torch.full_like(cx, box_size); h = torch.full_like(cy, box_size)
                        all_boxes_xyxy.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                        all_scores.append(s)

        points = []
        if all_boxes_xyxy:
            boxes_cat = torch.cat(all_boxes_xyxy, 0)
            scores_cat = torch.cat(all_scores, 0)
            kept = tv_nms(boxes_cat, scores_cat, iou_threshold=iou_thresh)
            W, H = pil_img.size
            for idx in kept.tolist():
                bx1, by1, bx2, by2 = boxes_cat[idx].tolist()
                cx = max(0.0, min(float(W), (bx1 + bx2) / 2.0))
                cy = max(0.0, min(float(H), (by1 + by2) / 2.0))
                points.append({"x": cx, "y": cy, "score": round(scores_cat[idx].item(), 4)})
        return points

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import requests
        
        web_app = FastAPI(title="Salmon Prediction API")

        # CORS Support for projectkiwi.io and localhost
        origins = [
            "https://salmon.projectkiwi.io",
            "http://localhost",
            "http://127.0.0.1",
        ]
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_origin_regex="http://localhost(:\\d+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/health")
        def health():
            return {"status": "ok", "cached_models": list(self.cached_models.keys())}

        @web_app.post("/predict")
        def predict(data: dict):
            # Parse parameters
            task = data.get("task", {})
            
            # Prioritise direct keys, then fall back to Sockeye task object keys
            file_id = data.get("file_id") or task.get("fileId")
            frame_idx = data.get("frame_idx") or task.get("frame")
            
            # Bounding box crop coords (optional crop area)
            x1 = data.get("x1") if data.get("x1") is not None else task.get("x1", 0)
            y1 = data.get("y1") if data.get("y1") is not None else task.get("y1", 0)
            x2 = data.get("x2") if data.get("x2") is not None else task.get("x2", 0)
            y2 = data.get("y2") if data.get("y2") is not None else task.get("y2", 0)
            
            presigned_url = (
                data.get("presigned_url") or 
                task.get("presignedUrl") or 
                task.get("presigned_url") or 
                data.get("video_url") or 
                task.get("videoUrl") or 
                task.get("video_url")
            )
            score_thresh = data.get("score_thresh", 0.25)
            iou_thresh = data.get("iou_thresh", 0.5)
            checkpoint_req = data.get("checkpoint_path")
            
            # Resolve Sockeye API details
            api_url = data.get("api_url") or os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1")
            api_key = data.get("api_key") or os.environ.get("SOCKEYE_API_KEY")
            
            if not file_id and not presigned_url:
                raise HTTPException(status_code=400, detail="Missing file_id (or task) or presigned_url/video_url in request payload.")
            if frame_idx is None:
                raise HTTPException(status_code=400, detail="Missing frame_idx (or task frame number).")

            # 1. Resolve and get the model
            try:
                checkpoint_path = resolve_checkpoint(checkpoint_req or DEFAULT_CHECKPOINT)
                model, model_type, num_context_frames, context_stride = self.get_model(checkpoint_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

            # 2. Extract frame inputs
            target_img = None
            context_imgs = []
            
            target_idx = int(frame_idx)
            needed_indices = [target_idx]
            if num_context_frames > 0:
                for i in range(1, num_context_frames + 1):
                    needed_indices.append(max(0, target_idx - i * context_stride))

            is_remote_url = bool(presigned_url and (presigned_url.startswith("http://") or presigned_url.startswith("https://")))

            if is_remote_url:
                # Mode A: Extract frames from remote presigned S3 URL using PyAV (concurrent)
                try:
                    unique_indices = sorted(set(needed_indices))
                    print(f"Extracting {len(unique_indices)} frame(s) concurrently from presigned URL: {unique_indices}")
                    downloaded_frames = {}
                    def _fetch_url(idx):
                        return idx, extract_frame_from_url(presigned_url, idx)
                    with ThreadPoolExecutor(max_workers=len(unique_indices)) as ex:
                        futures = {ex.submit(_fetch_url, idx): idx for idx in unique_indices}
                        for f in as_completed(futures):
                            idx, img = f.result()
                            downloaded_frames[idx] = img
                    target_img = downloaded_frames[target_idx]
                    for i in range(1, num_context_frames + 1):
                        ctx_idx = max(0, target_idx - i * context_stride)
                        context_imgs.append(downloaded_frames[ctx_idx])
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to extract frames from presigned URL: {str(e)}")
            elif file_id:
                # Mode B: Fetch JPEGs directly from Sockeye API (concurrent)
                if not api_key:
                    raise HTTPException(status_code=400, detail="SOCKEYE_API_KEY is required to fetch frames from Sockeye API.")
                try:
                    unique_indices = sorted(set(needed_indices))
                    print(f"Downloading {len(unique_indices)} frame(s) concurrently for file {file_id} from Sockeye: {unique_indices}")
                    downloaded_frames = {}
                    def _fetch_sockeye(idx):
                        return idx, download_frame_from_sockeye(api_url, api_key, file_id, idx)
                    with ThreadPoolExecutor(max_workers=len(unique_indices)) as ex:
                        futures = {ex.submit(_fetch_sockeye, idx): idx for idx in unique_indices}
                        for f in as_completed(futures):
                            idx, img = f.result()
                            downloaded_frames[idx] = img
                    target_img = downloaded_frames[target_idx]
                    for i in range(1, num_context_frames + 1):
                        ctx_idx = max(0, target_idx - i * context_stride)
                        context_imgs.append(downloaded_frames[ctx_idx])
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to download frames from Sockeye API: {str(e)}")
            else:
                # Mode C: OpenCV Video seek path (local file only)
                try:
                    local_video_path = presigned_url
                    cap = cv2.VideoCapture(local_video_path)
                    if not cap.isOpened():
                        raise ValueError(f"Could not open video: {local_video_path}")
                    
                    fetched_frames = {}
                    for idx in sorted(set(needed_indices)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, f_bgr = cap.read()
                        if not ret:
                            print(f"Warning: Failed to fetch frame {idx}. Falling back to zero frame.")
                            if fetched_frames:
                                h, w = next(iter(fetched_frames.values())).shape[:2]
                            else:
                                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                            fetched_frames[idx] = np.zeros((h, w, 3), dtype=np.uint8)
                        else:
                            fetched_frames[idx] = f_bgr
                            
                    cap.release()
                    
                    # Convert fetched BGR frames to RGB PIL Images
                    target_img = Image.fromarray(cv2.cvtColor(fetched_frames[target_idx], cv2.COLOR_BGR2RGB))
                    for i in range(1, num_context_frames + 1):
                        ctx_idx = max(0, target_idx - i * context_stride)
                        ctx_pil = Image.fromarray(cv2.cvtColor(fetched_frames[ctx_idx], cv2.COLOR_BGR2RGB))
                        context_imgs.append(ctx_pil)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to fetch video frames: {str(e)}")

            # 3. Setup prediction crop region
            has_crop = (x2 > x1 and y2 > y1)
            if has_crop:
                pil_img = target_img.crop((x1, y1, x2, y2))
                companion_crops = [img.crop((x1, y1, x2, y2)) for img in context_imgs]
            else:
                pil_img = target_img
                companion_crops = context_imgs
                
            W_crop, H_crop = pil_img.size

            # 4-6. Tile, infer, NMS
            raw_pts = self._infer_image(pil_img, model, model_type, num_context_frames,
                                        score_thresh, iou_thresh, companion_crops)
            points_list = []
            for p in raw_pts:
                cx_crop = max(0.0, min(float(W_crop), p["x"]))
                cy_crop = max(0.0, min(float(H_crop), p["y"]))
                points_list.append({
                    "x": round(x1 + cx_crop, 2),
                    "y": round(y1 + cy_crop, 2),
                    "crop_x": round(cx_crop, 2),
                    "crop_y": round(cy_crop, 2),
                    "score": p["score"],
                })

            return {
                "points": points_list,
                "count": len(points_list),
                "model_parameters": {
                    "model_type": model_type,
                    "num_context_frames": num_context_frames,
                    "context_stride": context_stride,
                    "checkpoint_used": checkpoint_path
                }
            }

        @web_app.post("/predict_ortho")
        def predict_ortho(data: dict):
            import requests as http_requests

            bounds = data.get("bounds")  # [minlng, minlat, maxlng, maxlat]
            zoom = int(data.get("zoom", 22))
            score_thresh = float(data.get("score_thresh", 0.25))
            iou_thresh = float(data.get("iou_thresh", 0.5))
            checkpoint_req = data.get("checkpoint_path")

            tile_server_url = data.get("tile_server_url")
            tile_server_token = data.get("tile_server_token")
            ortho_key = data.get("ortho_key")
            bucket = data.get("bucket", "")

            if not bounds or len(bounds) != 4:
                raise HTTPException(status_code=400, detail="Missing bounds [minlng,minlat,maxlng,maxlat]")
            if not tile_server_url or not tile_server_token or not ortho_key:
                raise HTTPException(status_code=400, detail="Missing tile_server_url, tile_server_token, or ortho_key")

            checkpoint_path = resolve_checkpoint(checkpoint_req or DEFAULT_CHECKPOINT)
            model, model_type, num_context_frames, _ = self.get_model(checkpoint_path)

            minlng, minlat, maxlng, maxlat = bounds

            # Use native zoom, cap at 2000 tiles (processed individually, not stitched)
            tile_count = 9999
            for z in range(zoom, max(zoom - 6, 14) - 1, -1):
                x_min, y_min = _latlon_to_tile(z, maxlat, minlng)
                x_max, y_max = _latlon_to_tile(z, minlat, maxlng)
                tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
                if tile_count <= 2000:
                    zoom = z
                    break

            print(f"[predict_ortho] zoom={zoom}, tiles={tile_count} ({x_min}-{x_max}, {y_min}-{y_max})")

            TILE_SIZE = 256
            INFER_SIZE = 512  # resize each tile to 512×512 for model input

            # Fetch all tiles in parallel
            session = http_requests.Session()
            session.headers.update({"Authorization": f"Bearer {tile_server_token}"})

            def fetch_tile(tx, ty):
                url = (
                    f"{tile_server_url.rstrip('/')}/tiles/{zoom}/{tx}/{ty}"
                    f"?key={urllib.parse.quote(ortho_key, safe='')}"
                    f"&bucket={urllib.parse.quote(bucket, safe='')}"
                    f"&tilesize={TILE_SIZE}"
                )
                try:
                    resp = session.get(url, timeout=15)
                    if resp.status_code == 200:
                        return tx, ty, Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    print(f"[predict_ortho] tile {zoom}/{tx}/{ty} error: {e}")
                return tx, ty, None

            coords = [(tx, ty) for tx in range(x_min, x_max + 1) for ty in range(y_min, y_max + 1)]
            tile_images = {}
            with ThreadPoolExecutor(max_workers=16) as ex:
                for tx, ty, img in ex.map(lambda c: fetch_tile(*c), coords):
                    if img is not None:
                        tile_images[(tx, ty)] = img

            print(f"[predict_ortho] fetched {len(tile_images)}/{tile_count} tiles")

            # Run inference on each tile individually (resized to INFER_SIZE×INFER_SIZE)
            # Fish at zoom 22 are ~20px in 256-px tiles → ~40px after 2× resize = model-friendly scale
            _normalize_fn = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            geo_points = []

            with torch.no_grad():
                for (tx, ty), tile_img in tile_images.items():
                    tile_resized = tile_img.resize((INFER_SIZE, INFER_SIZE), Image.BILINEAR)
                    pv = _normalize_fn(T.functional.to_tensor(tile_resized)).unsqueeze(0).cuda()
                    pm = torch.ones(1, INFER_SIZE, INFER_SIZE, dtype=torch.long).cuda()

                    if model_type == "detr":
                        out = model(pixel_values=pv, pixel_mask=pm)
                        probs = out.logits.cpu()[0].softmax(-1)[:, 0]
                        b = out.pred_boxes.cpu()[0]
                        keep = probs > score_thresh
                        if keep.any():
                            b_k = b[keep]
                            s_k = probs[keep]
                            # NMS in tile pixel space (256-px coords)
                            bsize = 10.0
                            cx_px = b_k[:, 0] * TILE_SIZE
                            cy_px = b_k[:, 1] * TILE_SIZE
                            boxes_xyxy = torch.stack([cx_px - bsize, cy_px - bsize,
                                                      cx_px + bsize, cy_px + bsize], 1)
                            kept_idx = tv_nms(boxes_xyxy, s_k, iou_threshold=iou_thresh)
                            for idx in kept_idx.tolist():
                                col = b_k[idx, 0].item() * TILE_SIZE
                                row = b_k[idx, 1].item() * TILE_SIZE
                                lng, lat = _pixel_to_latlon(col, row, zoom, tx, ty, TILE_SIZE)
                                geo_points.append({"lng": round(lng, 8), "lat": round(lat, 8),
                                                   "score": round(s_k[idx].item(), 4)})

            print(f"[predict_ortho] {len(geo_points)} detections at score_thresh={score_thresh}")

            return {
                "points": geo_points,
                "count": len(geo_points),
                "model_parameters": {
                    "model_type": model_type,
                    "num_context_frames": num_context_frames,
                    "zoom_used": zoom,
                    "tile_count": tile_count,
                    "tiles_fetched": len(tile_images),
                    "checkpoint_used": checkpoint_path,
                },
            }

        return web_app
