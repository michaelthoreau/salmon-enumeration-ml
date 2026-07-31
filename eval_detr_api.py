#!/usr/bin/env python3
"""
DETR salmon box detector - Inference and Evaluation script.

Runs inference with a given DETR checkpoint (.pth) against the validation task set
from the Sockeye API, evaluates detection and count metrics.

Features:
- Centroid distance-based NMS (default 40 pixels suppression radius).
- Evaluation of Precision, Recall, F1-Score, and AP at different IoU thresholds.
- Average RMSE/MAE metrics per image crop.
- Support for context frames and temporal fusion models (early or transformer).
- Support for loading model config and checkpoint from a model_info.json or MLflow.
"""

import os
import io
import math
import time
import random
import warnings
import json
import hashlib
import urllib.parse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import fire
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrConfig

warnings.filterwarnings('ignore')

def _load_dotenv():
    # Load .env file in the same directory as this script if it exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

_load_dotenv()


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 1   # fish
ORTHO_TILE_SIZE = 256

# ---------------------------------------------------------------------------
# Centroid NMS
# ---------------------------------------------------------------------------

def centroid_nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, dist_thresh: float = 40.0) -> torch.Tensor:
    """
    Non-maximum suppression based on centroid distance instead of IoU.
    Suppresses any box whose centroid is within dist_thresh pixels of a higher-scoring box.
    """
    if boxes_xyxy.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)

    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    centroids = torch.stack([cx, cy], dim=1)  # (N, 2)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        # Compute distance from the selected centroid to all other remaining centroids
        dists = torch.norm(centroids[order[1:]] - centroids[i], dim=1)
        # Keep indices where distance >= dist_thresh
        mask = dists >= dist_thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)

# ---------------------------------------------------------------------------
# Bounding Box IoU
# ---------------------------------------------------------------------------

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union

# ---------------------------------------------------------------------------
# Matching and AP computation
# ---------------------------------------------------------------------------

def match_predictions_to_gt(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_thresh: float
) -> Tuple[List[bool], List[bool], int]:
    """
    Greedy bipartite matching of predictions to ground truth boxes.
    Returns:
        tp: list of length len(pred_boxes), indicating if each prediction is a True Positive
        fp: list of length len(pred_boxes), indicating if each prediction is a False Positive
        fn_count: number of unmatched ground truth boxes
    """
    if not pred_boxes:
        return [], [], len(gt_boxes)

    # Sort predictions by score descending
    order = np.argsort(pred_scores)[::-1]
    
    tp = [False] * len(pred_boxes)
    fp = [False] * len(pred_boxes)
    gt_matched = [False] * len(gt_boxes)

    for idx in order:
        box = pred_boxes[idx]
        best_iou = -1.0
        best_gt_idx = -1

        for g_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[g_idx]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh:
            tp[idx] = True
            gt_matched[best_gt_idx] = True
        else:
            fp[idx] = True

    fn_count = sum(1 for m in gt_matched if not m)
    return tp, fp, fn_count

def compute_ap(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    """Computes Average Precision (AP) using standard all-point integration."""
    if total_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1.0)

    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# ---------------------------------------------------------------------------
# Temporal Fusion architecture (mirrors training)
# ---------------------------------------------------------------------------

from transformers.models.detr.modeling_detr import DetrModelOutput

class DetrModelWithTemporalFusion(nn.Module):
    def __init__(self, original_model, num_context_frames):
        super().__init__()
        self.config = original_model.config
        self.backbone = original_model.backbone
        self.input_projection = original_model.input_projection
        self.position_embedding = original_model.position_embedding
        self.encoder = original_model.encoder
        self.decoder = original_model.decoder
        self.query_position_embeddings = original_model.query_position_embeddings
        self.num_context_frames = num_context_frames
        
        self.temporal_embeddings = nn.Parameter(
            torch.zeros(1 + num_context_frames, 1, self.config.d_model)
        )
        nn.init.normal_(self.temporal_embeddings, std=0.02)
        
    def forward(
        self,
        pixel_values=None,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        **kwargs
    ):
        if inputs_embeds is not None or self.num_context_frames == 0:
            if inputs_embeds is None:
                batch_size, num_channels, height, width = pixel_values.shape
                device = pixel_values.device
                if pixel_mask is None:
                    pixel_mask = torch.ones(((batch_size, height, width)), device=device)
                vision_features = self.backbone(pixel_values, pixel_mask)
                feature_map, mask = vision_features[-1]
                projected_feature_map = self.input_projection(feature_map)
                flattened_features = projected_feature_map.flatten(2).transpose(1, 2)
                spatial_position_embeddings = (
                    self.position_embedding(shape=feature_map.shape, device=device, dtype=pixel_values.dtype, mask=mask)
                    .flatten(2)
                    .transpose(1, 2)
                )
                flattened_mask = mask.flatten(1)
            else:
                batch_size = inputs_embeds.shape[0]
                device = inputs_embeds.device
                flattened_features = inputs_embeds
                seq_len = inputs_embeds.shape[1]
                feat_dim = int(seq_len**0.5)
                spatial_position_embeddings = (
                    self.position_embedding(
                        shape=torch.Size([batch_size, self.config.d_model, feat_dim, feat_dim]),
                        device=device,
                        dtype=inputs_embeds.dtype,
                    )
                    .flatten(2)
                    .transpose(1, 2)
                )
                if pixel_mask is not None:
                    mask = nn.functional.interpolate(pixel_mask[None].float(), size=(feat_dim, feat_dim)).to(torch.bool)[0]
                    flattened_mask = mask.flatten(1)
                else:
                    flattened_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        else:
            batch_size, num_channels, height, width = pixel_values.shape
            device = pixel_values.device
            
            groups = 1 + self.num_context_frames
            chunks = torch.chunk(pixel_values, groups, dim=1)
            
            pixel_values_batched = torch.cat(chunks, dim=0)
            
            if pixel_mask is None:
                pixel_mask = torch.ones(((batch_size, height, width)), device=device)
            pixel_mask_batched = pixel_mask.repeat(groups, 1, 1)
            
            vision_features = self.backbone(pixel_values_batched, pixel_mask_batched)
            feature_map, mask = vision_features[-1]
            
            projected_feature_map = self.input_projection(feature_map)
            flattened_features_all = projected_feature_map.flatten(2).transpose(1, 2)
            
            feat_splits = torch.chunk(flattened_features_all, groups, dim=0)
            
            feat_with_time = []
            for i, split in enumerate(feat_splits):
                feat_with_time.append(split + self.temporal_embeddings[i])
                
            flattened_features = torch.cat(feat_with_time, dim=1)
            
            spatial_pos_all = (
                self.position_embedding(shape=feature_map.shape, device=device, dtype=pixel_values.dtype, mask=mask)
                .flatten(2)
                .transpose(1, 2)
            )
            pos_splits = torch.chunk(spatial_pos_all, groups, dim=0)
            spatial_position_embeddings = torch.cat(pos_splits, dim=1)
            
            mask_flat_all = mask.flatten(1)
            mask_splits = torch.chunk(mask_flat_all, groups, dim=0)
            flattened_mask = torch.cat(mask_splits, dim=1)
            
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                spatial_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )

        object_queries_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        if decoder_inputs_embeds is not None:
            queries = decoder_inputs_embeds
        else:
            queries = torch.zeros_like(object_queries_position_embeddings)

        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=decoder_attention_mask,
            spatial_position_embeddings=spatial_position_embeddings,
            object_queries_position_embeddings=object_queries_position_embeddings,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=flattened_mask,
            **kwargs,
        )

        return DetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )

def _expand_first_conv(model, in_channels: int):
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
            print(f'  Expanded first conv ({name}): 3 -> {in_channels} channels.', flush=True)
            return

# ---------------------------------------------------------------------------
# Tiling Helper (reused from validate in train_detr_api.py)
# ---------------------------------------------------------------------------

def _make_tiles(pil_img: Image.Image, S: int, companion_pils: List[Image.Image], num_context_frames: int) -> List[Tuple[torch.Tensor, int, int, int, int]]:
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
        pil_r = pil_img.resize((S, S), Image.BILINEAR)
        comp_rs = [c.resize((S, S), Image.BILINEAR) if c else None for c in (companion_pils or [])]
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
            comp_crops = [c.crop((x0, y0, x0 + S, y0 + S)) if c else None for c in (companion_pils or [])]
            tiles.append((_tile_tensor(crop, comp_crops), x0, y0, W, H))
    return tiles

# ---------------------------------------------------------------------------
# Cache & Fetch API Frames
# ---------------------------------------------------------------------------

def get_frame(api_url: str, api_key: str, file_id: str, frame_num: int, cache_dir: str) -> Image.Image:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_id}_{frame_num:06d}.jpg")
    
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass
                
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{api_url}/files/{file_id}/frame/{frame_num}?quality=95"
    img_bytes = None
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, stream=True, timeout=15)
            if res.status_code == 200:
                img_bytes = res.content
                break
            elif res.status_code == 429:
                time.sleep(random.uniform(1.0, 3.0))
            else:
                time.sleep(random.uniform(0.5, 1.5))
        except Exception:
            time.sleep(random.uniform(1.0, 3.0))
            
    if img_bytes is None:
        raise RuntimeError(f"Error downloading frame {frame_num} for file {file_id}")
        
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    try:
        tmp_path = cache_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass
    return img

def get_ortho_tile_raw(
    tile_server_url: str, tile_server_token: str,
    ortho_key: str, bucket: str, z: int, tx: int, ty: int, cache_dir: str
) -> Image.Image:
    """Downloads and caches a single ortho tile, from the same tile server /predict_ortho
    uses at inference time. Mirrors SockeyeApiDataset._get_ortho_tile_raw in dataset_api.py."""
    os.makedirs(cache_dir, exist_ok=True)
    key_hash = hashlib.md5(ortho_key.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"ortho_{key_hash}_{z}_{tx}_{ty}.jpg")

    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass

    url = (
        f"{tile_server_url.rstrip('/')}/tiles/{z}/{tx}/{ty}"
        f"?key={urllib.parse.quote(ortho_key, safe='')}"
        f"&bucket={urllib.parse.quote(bucket, safe='')}"
        f"&tilesize={ORTHO_TILE_SIZE}"
    )
    headers = {"Authorization": f"Bearer {tile_server_token}"}
    img_bytes = None
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 200:
                img_bytes = res.content
                break
            elif res.status_code == 429:
                time.sleep(random.uniform(1.0, 3.0))
            else:
                time.sleep(random.uniform(0.5, 1.5))
        except Exception:
            time.sleep(random.uniform(1.0, 3.0))

    if img_bytes is None:
        # Neighboring tile may legitimately fall outside the ortho's coverage.
        return Image.new("RGB", (ORTHO_TILE_SIZE, ORTHO_TILE_SIZE), (0, 0, 0))

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    try:
        tmp_path = cache_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass
    return img

def get_ortho_frame(
    task: dict, tile_server_url: str, tile_server_token: str, cache_dir: str, stitch_radius: int = 1
) -> Image.Image:
    """Fetches and stitches the (2*radius+1)^2 tile neighborhood around an ortho task's
    center tile. Mirrors SockeyeApiDataset._get_ortho_tiles_stitched in dataset_api.py —
    keep the stitch_radius in sync with whatever the model being evaluated was trained with."""
    z, cx, cy = task["tileZ"], task["tileX"], task["tileY"]
    ortho_key, bucket = task["orthoKey"], task.get("bucket", "")

    r = stitch_radius
    n = 2 * r + 1
    canvas = Image.new("RGB", (n * ORTHO_TILE_SIZE, n * ORTHO_TILE_SIZE))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            tile_img = get_ortho_tile_raw(
                tile_server_url, tile_server_token, ortho_key, bucket, z, cx + dx, cy + dy, cache_dir
            )
            canvas.paste(tile_img, ((dx + r) * ORTHO_TILE_SIZE, (dy + r) * ORTHO_TILE_SIZE))
    return canvas

# ---------------------------------------------------------------------------
# MLflow Artifact Downloader
# ---------------------------------------------------------------------------

def download_mlflow_checkpoint(checkpoint_uri: str, run_id: str = None) -> str:
    """Downloads checkpoint from local MLflow sqlite database, or locates it locally if available."""
    # First, try to resolve the path locally from the relative mlflow directory
    # URI example: mlflow-artifacts:/1/00412b537a4942619cf7b8086f3720f8/artifacts/best.pth
    if checkpoint_uri.startswith("mlflow-artifacts:"):
        path_part = checkpoint_uri.replace("mlflow-artifacts:", "").lstrip("/")
        # Try finding it in mlruns directory
        local_candidate = os.path.join("mlruns", path_part)
        if os.path.exists(local_candidate):
            print(f"Found checkpoint locally at: {local_candidate}", flush=True)
            return local_candidate
            
    import mlflow
    
    # Set default local tracking URI if not set
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        if os.path.exists("mlflow.db"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        else:
            print("Warning: mlflow.db not found. Using MLflow default tracking.")

    # Resolve run ID if none provided
    if not run_id:
        parts = checkpoint_uri.split("/")
        for part in parts:
            if len(part) == 32:
                run_id = part
                break
                
    if not run_id:
        raise ValueError(f"Could not parse run_id from checkpoint URI: {checkpoint_uri}")

    artifact_name = checkpoint_uri.split("/")[-1]
    
    print(f"Resolving MLflow tracking URI: {mlflow.get_tracking_uri()}", flush=True)
    print(f"Downloading checkpoint '{artifact_name}' for run '{run_id}'...", flush=True)
    
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_name,
        dst_path="./mlflow_downloads"
    )
    return local_path


# ---------------------------------------------------------------------------
# Main Evaluation Command
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: str = None,
    model_info: str = None,
    api_url: str = None,
    api_key: str = None,
    cache_dir: str = None,
    img_size: int = 512,
    box_size: int = 50,
    num_queries: int = 100,
    num_context_frames: int = None,
    context_stride: int = None,
    temporal_fusion: str = "early",
    score_thresh: float = 0.25,
    nms_dist: float = 40.0,
    iou_thresholds: str = "0.1,0.25,0.5,0.75",
    device: str = "cuda",
    use_video_data: bool = True,
    use_ortho_data: bool = False,
    tile_server_url: str = None,
    tile_server_token: str = None,
    ortho_stitch_radius: int = 1,
    flight_ids: str = None,
):
    """
    Evaluates DETR model predictions on Sockeye validation task set.

    Args:
        checkpoint: Path to DETR .pth file or mlflow-artifacts URI.
        model_info: Optional path to model_info.json (loads checkpoint, num_context_frames, context_stride from it).
        api_url: Base endpoint URL of Sockeye API.
        api_key: Sockeye Authorization API Token.
        cache_dir: Directory where downloaded video frames are cached. Defaults to './frame_cache' or '/tmp/frame_cache'.
        img_size: Input dimension of model tiles.
        box_size: Bounding box size (width/height in pixels) to construct around points.
        num_queries: Number of DETR decoder queries.
        num_context_frames: Number of context frames (0 to disable). Overridden by model_info if specified.
        context_stride: Frame step size between context frames. Overridden by model_info if specified.
        temporal_fusion: Mode of temporal integration ('early' or 'transformer').
        score_thresh: Threshold to filter out low-confidence predictions during evaluation.
        nms_dist: Centroid NMS distance suppression radius in pixels.
        iou_thresholds: Comma-separated list of standard IoU thresholds for evaluation (e.g. '0.1,0.25,0.5,0.75').
        device: PyTorch running device ('cuda' or 'cpu').
        use_video_data: Include ground truth from /api/v1/labeling-tasks (raw video/photo crops).
        use_ortho_data: Include ground truth from /api/v1/ortho-tasks (real orthomosaic annotations).
            Requires tile_server_url/tile_server_token (directly or via .env).
        tile_server_url: Base URL of the ortho tile server (same one /predict_ortho uses).
        tile_server_token: Bearer token for the tile server.
        ortho_stitch_radius: Neighboring-tile radius stitched around each ortho task's center
            tile — MUST match whatever the checkpoint being evaluated was trained with.
        flight_ids: Comma-separated list of flightIds to evaluate on. When set, this REPLACES
            the random train/val split entirely — all matching tasks from these flights are
            used as-is, no split applied. This is how to fairly compare two checkpoints that
            were trained with different random splits: pick one fixed, explicit set of flights
            that neither model trained on, and pass it here for both evaluation runs.
    """
    # 1. Resolve arguments from model_info.json if provided
    if model_info:
        model_info_path = Path(model_info)
        if not model_info_path.exists():
            raise FileNotFoundError(f"model_info file not found: {model_info}")
        with open(model_info_path, 'r') as f:
            info = json.load(f)
        
        run_id = info.get("run_id")
        checkpoint_val = checkpoint if checkpoint is not None else info.get("checkpoint")
        if num_context_frames is None:
            num_context_frames = info.get("num_context_frames")
        if context_stride is None:
            context_stride = info.get("context_stride")
        
        print(f"Loaded config from model_info.json: run_id={run_id}, context_frames={num_context_frames}, context_stride={context_stride}", flush=True)
    else:
        checkpoint_val = checkpoint
        run_id = None

    if num_context_frames is None:
        num_context_frames = 0
    if context_stride is None:
        context_stride = 0

    if not checkpoint_val:
        raise ValueError("Must specify either --checkpoint or --model_info path containing checkpoint details.")

    # Resolve caching dir
    if cache_dir is None:
        if os.path.exists("./frame_cache"):
            cache_dir = "./frame_cache"
        else:
            cache_dir = "/tmp/frame_cache"

    # 2. Resolve MLflow checkpoint if needed
    if checkpoint_val.startswith("mlflow-artifacts:"):
        checkpoint_val = download_mlflow_checkpoint(checkpoint_val, run_id)
        
    print(f"Using resolved checkpoint: {checkpoint_val}", flush=True)

    # 3. Handle Sockeye API URL & Key
    if api_url is None:
        api_url = os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1")
    if api_key is None:
        api_key = os.environ.get("SOCKEYE_API_KEY")

    if not api_key:
        raise ValueError("Sockeye API Key is required. Set SOCKEYE_API_KEY in environment or pass --api_key.")

    if tile_server_url is None:
        tile_server_url = os.environ.get("MODAL_TILE_SERVER_URL")
    if tile_server_token is None:
        tile_server_token = os.environ.get("MODAL_TILE_TOKEN")
    if use_ortho_data and not (tile_server_url and tile_server_token):
        raise ValueError(
            "use_ortho_data=True requires tile_server_url/tile_server_token "
            "(set MODAL_TILE_SERVER_URL/MODAL_TILE_TOKEN in .env or pass explicitly)."
        )
    if not use_video_data and not use_ortho_data:
        raise ValueError("At least one of use_video_data / use_ortho_data must be True.")

    # Parse IoU thresholds
    iou_thresh_list = [float(t.strip()) for t in iou_thresholds.split(",")]
    
    # Set PyTorch running device
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Running evaluation on device: {device_obj}", flush=True)

    # 4. Fetch ground truth from Sockeye API
    headers = {"Authorization": f"Bearer {api_key}"}
    limit = 500
    tasks = []

    if use_video_data:
        print("Fetching labeling tasks from Sockeye API...", flush=True)
        offset = 0
        while True:
            url = f"{api_url}/labeling-tasks?completed=true&limit={limit}&offset={offset}"
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                raise RuntimeError(f"Failed to fetch tasks: {res.status_code} - {res.text}")
            data = res.json()
            if not data:
                break
            tasks.extend(data)
            offset += limit
        tasks = [t for t in tasks if len(t.get("points", [])) > 0]
        print(f"Loaded {len(tasks)} completed labeling tasks with annotations.", flush=True)

    if use_ortho_data:
        print("Fetching ortho tasks from Sockeye API...", flush=True)
        ortho_tasks = []
        offset = 0
        while True:
            url = f"{api_url}/ortho-tasks?limit={limit}&offset={offset}"
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                raise RuntimeError(f"Failed to fetch ortho tasks: {res.status_code} - {res.text}")
            data = res.json()
            if not data:
                break
            ortho_tasks.extend(data)
            offset += limit
        ortho_tasks = [t for t in ortho_tasks if len(t.get("points", [])) > 0]
        print(f"Loaded {len(ortho_tasks)} ortho tasks with annotations.", flush=True)
        tasks.extend(ortho_tasks)

    if flight_ids:
        # Explicit, fixed flight set — bypasses the random split entirely. This is how to
        # fairly compare two checkpoints trained with different random splits: pick one set
        # of flights neither model trained on, and pass the same --flight_ids to both runs.
        flight_id_set = {f.strip() for f in flight_ids.split(",") if f.strip()}
        val_tasks = [t for t in tasks if t.get("flightId") in flight_id_set]
        print(
            f"Evaluating on explicit flight set ({len(flight_id_set)} flights): "
            f"{len(val_tasks)} tasks.", flush=True
        )
    else:
        # Reproduce the flight-aware validation split from train_detr_api.py (seed=42) —
        # groups by flightId so a flight's data can't be split across train/val.
        random.seed(42)
        tasks_by_flight: Dict[str, list] = {}
        for t in tasks:
            fid = t.get("flightId") or f"__no_flight__{id(t)}"
            tasks_by_flight.setdefault(fid, []).append(t)
        flight_id_list = list(tasks_by_flight.keys())
        random.shuffle(flight_id_list)
        split_idx = max(1, int(len(flight_id_list) * 0.9))
        val_flight_ids = flight_id_list[split_idx:]
        val_tasks = [t for fid in val_flight_ids for t in tasks_by_flight[fid]]
        print(
            f"Evaluating on the flight-aware validation split: {len(val_flight_ids)} flights, "
            f"{len(val_tasks)} tasks.", flush=True
        )

    # 5. Initialize Model
    print("Building DETR model...", flush=True)
    config = DetrConfig(
        num_labels=NUM_CLASSES,
        num_queries=num_queries,
    )
    model = DetrForObjectDetection(config)

    if num_context_frames > 0:
        if temporal_fusion == "early":
            _expand_first_conv(model, in_channels=3 * (1 + num_context_frames))
        elif temporal_fusion == "transformer":
            model.model = DetrModelWithTemporalFusion(model.model, num_context_frames)
            print("Swapped internal DetrModel to DetrModelWithTemporalFusion.", flush=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_val, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device_obj)
    model.eval()

    # 6. Evaluation Loop
    # Keep track of metrics
    all_abs_errors = []
    all_squared_errors = []
    origin_abs_errors = {'video': [], 'ortho': []}
    
    # Store predictions & GT globally for AP and other stats
    # For each IoU threshold, we accumulate TP / FP / FN across the entire set
    global_predictions_by_iou = {iou: [] for iou in iou_thresh_list} # items will be (score, is_tp)
    global_total_gt = {iou: 0 for iou in iou_thresh_list}
    
    # Summarized counters for F1 at score_thresh
    fixed_tp_by_iou = {iou: 0 for iou in iou_thresh_list}
    fixed_fp_by_iou = {iou: 0 for iou in iou_thresh_list}
    fixed_fn_by_iou = {iou: 0 for iou in iou_thresh_list}

    print("\nRunning inference & matching...", flush=True)
    for task in tqdm(val_tasks, desc="Validating"):
        file_id = task["fileId"]
        frame_t = task["frame"]
        x1, y1, x2, y2 = task["x1"], task["y1"], task["x2"], task["y2"]
        is_ortho = file_id.startswith("ortho:")

        # Load frame crop (ortho tasks fetch+stitch tiles; video/photo tasks download a frame)
        try:
            if is_ortho:
                pil_img = get_ortho_frame(task, tile_server_url, tile_server_token, cache_dir, ortho_stitch_radius)
            else:
                pil_img = get_frame(api_url, api_key, file_id, frame_t, cache_dir)
        except Exception as e:
            print(f"Skipping {'ortho tile' if is_ortho else 'frame'} {file_id} due to fetch error: {e}", flush=True)
            continue

        W_full, H_full = pil_img.size
        if x2 <= x1 or y2 <= y1:
            x1, y1 = 0, 0
            x2, y2 = W_full, H_full

        crop_pil = pil_img.crop((x1, y1, x2, y2))
        W_crop, H_crop = crop_pil.size

        # Companion context frames — not applicable to ortho tasks (no temporal sequence)
        companion_crops = []
        if num_context_frames > 0 and not is_ortho:
            for n_idx in range(1, num_context_frames + 1):
                frame_t_minus_np = max(0, frame_t - n_idx * context_stride)
                crop_n = None
                try:
                    img_context_full = get_frame(api_url, api_key, file_id, frame_t_minus_np, cache_dir)
                    crop_n = img_context_full.crop((x1, y1, x2, y2))
                except Exception:
                    pass
                companion_crops.append(crop_n)

        # Reconstruct GT boxes centered on the points
        points = [[pt["x"] - x1, pt["y"] - y1] for pt in task.get("points", [])]
        if is_ortho:
            # Exported points are relative to just the task's center tile — shift into the
            # stitched canvas's coordinate space, same as SockeyeApiDataset.__getitem__.
            margin = ortho_stitch_radius * ORTHO_TILE_SIZE
            points = [[px + margin, py + margin] for px, py in points]
        gt_boxes = []
        half_box = box_size / 2.0
        for px, py in points:
            bx1 = max(0.0, px - half_box)
            by1 = max(0.0, py - half_box)
            bx2 = min(float(W_crop), px + half_box)
            by2 = min(float(H_crop), py + half_box)
            if bx2 > bx1 and by2 > by1:
                gt_boxes.append([bx1, by1, bx2, by2])
                
        gt_count = len(gt_boxes)
        
        # Perform tile inference
        tiles = _make_tiles(crop_pil, img_size, companion_crops, num_context_frames)
        all_boxes_xyxy = []
        all_scores = []

        with torch.no_grad():
            for tile_t, tx, ty, orig_W, orig_H in tiles:
                pv = tile_t.unsqueeze(0).to(device_obj)
                pm = torch.ones(1, img_size, img_size, dtype=torch.long, device=device_obj)
                
                out = model(pixel_values=pv, pixel_mask=pm)
                probs = out.logits.softmax(-1)[0, :, 0].cpu()
                boxes = out.pred_boxes[0].cpu()
                
                # Keep raw predictions above a very low threshold (0.01) to compute full PR curves
                keep = probs > 0.01
                if keep.any():
                    b = boxes[keep]
                    s = probs[keep]
                    cx = tx + b[:, 0] * img_size
                    cy = ty + b[:, 1] * img_size
                    w  = b[:, 2] * img_size
                    h  = b[:, 3] * img_size
                    all_boxes_xyxy.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                    all_scores.append(s)

        # Apply Centroid NMS
        pred_boxes = []
        pred_scores = []
        if all_boxes_xyxy:
            boxes_cat = torch.cat(all_boxes_xyxy, 0)
            scores_cat = torch.cat(all_scores, 0)
            
            kept_indices = centroid_nms(boxes_cat, scores_cat, dist_thresh=nms_dist)
            
            pred_boxes = boxes_cat[kept_indices].tolist()
            pred_scores = scores_cat[kept_indices].tolist()

        # Filter predictions at evaluation score_thresh for count & fixed F1 metrics
        eval_pred_boxes = [box for box, score in zip(pred_boxes, pred_scores) if score > score_thresh]
        eval_pred_scores = [score for score in pred_scores if score > score_thresh]
        pred_count = len(eval_pred_boxes)
        
        # Compute count errors for this image
        count_err = pred_count - gt_count
        all_abs_errors.append(abs(count_err))
        all_squared_errors.append(count_err ** 2)
        origin_abs_errors['ortho' if is_ortho else 'video'].append(abs(count_err))

        # Match predictions for each IoU threshold
        for iou in iou_thresh_list:
            # 1. Matching for AP (using all raw/low threshold predictions)
            tp_list_ap, fp_list_ap, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou)
            for score, is_tp in zip(pred_scores, tp_list_ap):
                global_predictions_by_iou[iou].append((score, 1 if is_tp else 0))
            global_total_gt[iou] += gt_count

            # 2. Matching for fixed score threshold F1/P/R
            tp_list_fixed, fp_list_fixed, fn_count_fixed = match_predictions_to_gt(
                eval_pred_boxes, eval_pred_scores, gt_boxes, iou
            )
            fixed_tp_by_iou[iou] += sum(1 for x in tp_list_fixed if x)
            fixed_fp_by_iou[iou] += sum(1 for x in fp_list_fixed if x)
            fixed_fn_by_iou[iou] += fn_count_fixed

    # ---------------------------------------------------------------------------
    # Calculate Final Metrics
    # ---------------------------------------------------------------------------
    n_images = len(all_abs_errors)
    if n_images == 0:
        print("\nNo validation tasks were successfully evaluated.", flush=True)
        return

    mae = np.mean(all_abs_errors)
    rmse = np.sqrt(np.mean(all_squared_errors))
    
    print(f"\n============================================================")
    print(f"COUNT METRICS (Score Threshold = {score_thresh})")
    print(f"============================================================")
    print(f"Total Validation Images Evaluated : {n_images}")
    print(f"Mean Absolute Error (MAE) per img : {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE)    : {rmse:.4f}")
    print(f"Max Absolute Error                : {np.max(all_abs_errors)}")
    for origin, errs in origin_abs_errors.items():
        if errs:
            print(f"  [{origin}] n={len(errs)}  MAE={np.mean(errs):.4f}  RMSE={np.sqrt(np.mean(np.square(errs))):.4f}")

    print(f"\n============================================================")
    print(f"DETECTION METRICS (NMS centroid distance = {nms_dist}px)")
    print(f"============================================================")
    
    # Table header
    print(f"{'IoU Thresh':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AP':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print(f"---------------------------------------------------------------------------------------------")
    
    results = {}
    for iou in iou_thresh_list:
        # AP computation
        preds_info = global_predictions_by_iou[iou]
        if preds_info:
            # Sort all global predictions by score descending
            preds_info = sorted(preds_info, key=lambda x: x[0], reverse=True)
            scores_arr = np.array([x[0] for x in preds_info])
            tp_arr = np.array([x[1] for x in preds_info])
            fp_arr = 1 - tp_arr
            ap = compute_ap(tp_arr, fp_arr, global_total_gt[iou])
        else:
            ap = 0.0

        # Fixed threshold metrics
        tp = fixed_tp_by_iou[iou]
        fp = fixed_fp_by_iou[iou]
        fn = fixed_fn_by_iou[iou]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{iou:<12.2f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {ap:<10.4f} | {tp:<6d} | {fp:<6d} | {fn:<6d}")
        
        results[iou] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
    print(f"============================================================\n")

if __name__ == "__main__":
    fire.Fire(evaluate)
