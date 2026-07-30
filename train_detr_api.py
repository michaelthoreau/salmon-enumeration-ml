"""
DETR salmon box detector training script using Sockeye API.
Loads point-annotated image crops directly from the Sockeye API and trains a DETR model.

Usage:
    python train_detr_api.py
"""

import os
import io
import math
import time
import random
import warnings
from pathlib import Path
from collections import deque

import fire
import mlflow
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

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
        
        # Learnable temporal embeddings: shape [1 + N, 1, 256]
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 1   # fish (background is implicit class inside DETR)


# ---------------------------------------------------------------------------
# API-Driven Dataset
# ---------------------------------------------------------------------------

from dataset_api import SockeyeApiDataset, SameFrameBatchSampler, ORTHO_TILE_SIZE


def collate_fn(batch):
    imgs_and_contexts, targets = zip(*batch)
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    tensors = []
    for img_crop, context_crops in imgs_and_contexts:
        t = normalize(T.functional.to_tensor(img_crop))
        if context_crops:
            tensors_to_cat = [t]
            for crop_n in context_crops:
                if crop_n is not None:
                    t_n = normalize(T.functional.to_tensor(crop_n))
                else:
                    t_n = torch.zeros_like(t)
                tensors_to_cat.append(t_n)
            t = torch.cat(tensors_to_cat, dim=0)
        tensors.append(t)
        
    imgs_tensor = torch.stack(tensors)
    B, C, H, W = imgs_tensor.shape
    pixel_mask = torch.ones((B, H, W), dtype=torch.long)
    
    return imgs_tensor, pixel_mask, list(targets)


# ---------------------------------------------------------------------------
# Network Helper
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
            print(f'  Expanded first conv ({name}): 3 → {in_channels} input channels.', flush=True)
            return
    print('WARNING: could not find Conv2d(3→...) to expand.', flush=True)


# ---------------------------------------------------------------------------
# Validation Debug Logging
# ---------------------------------------------------------------------------

def _log_val_image(writer, epoch, image_t, gt_boxes, pred_boxes, pred_scores, thresh):
    """Draw ground truth (red) and prediction (green) boxes onto the validation crop tile."""
    # Denormalize image
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_t = image_t[:3].cpu() * std + mean
    img_t = torch.clamp(img_t, 0.0, 1.0)
    
    ndarr = img_t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    W, H = im.size
    
    # Draw GT bounding boxes in red
    for box in gt_boxes:
        cx_n, cy_n, w_n, h_n = box
        cx, cy = cx_n * W, cy_n * H
        w, h = w_n * W, h_n * H
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
    # Draw predicted bounding boxes in green (above score threshold)
    for box, score in zip(pred_boxes, pred_scores):
        if score > thresh:
            cx_n, cy_n, w_n, h_n = box
            cx, cy = cx_n * W, cy_n * H
            w, h = w_n * W, h_n * H
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            
    img_arr = np.array(im)
    writer.add_image("val/predictions", img_arr, epoch, dataformats="HWC")


# ---------------------------------------------------------------------------
# Training Main
# ---------------------------------------------------------------------------

def _auto_run_name(num_context_frames, context_stride, temporal_fusion):
    """Build a short run name: 'detr', tagged with context-frame config."""
    parts = ['detr', f'cf{num_context_frames}cs{context_stride}']
    if num_context_frames > 0:
        parts.append(temporal_fusion)
    return '_'.join(parts)


def main(
    api_url:      str   = None,
    api_key:      str   = None,
    save_path:    str   = None,
    epochs:       int   = 20,
    batch_size:   int   = 4,
    lr:           float = 1e-4,
    lr_backbone:  float = 1e-5,
    lr_step:      int   = 240,
    warmup_steps: int   = 1000,
    decay_steps:  int   = 10000,
    val_every:    int   = 1,
    resume:       str   = None,
    workers:      int   = 8,
    prefetch_factor: int = 4,
    cache_dir:    str   = "/tmp/frame_cache",
    img_size:     int   = 512,
    box_size:     int   = 50,
    num_queries:  int   = 100,
    score_thresh: float = 0.25,
    limit_batches: int  = 0,
    min_batches:  int   = 500,
    num_context_frames: int = 1,
    context_stride: int = 5,
    pretrained:   bool  = True,
    temporal_fusion: str = "early",
    verbose:      bool  = False,
    mlflow_tracking_uri: str = None,   # e.g. 'file:./mlruns' or a remote server URI; default local ./mlruns
    mlflow_experiment: str = 'salmon-enumeration',
    mlflow:       str   = None,        # override MLflow run name (skips auto-resume for that name)
    fresh_finetune: bool = False,
    use_video_data: bool = True,
    use_ortho_data: bool = True,
    tile_server_url: str = None,
    tile_server_token: str = None,
    ortho_stitch_radius: int = 1,
):
    """Train DETR on salmon crop tasks pulled from Sockeye API.

    Args:
        api_url:      Base endpoint URL of Sockeye API.
        api_key:      Sockeye Authorization API Token.
        save_path:    Checkpoint output directory (auto-selected if omitted).
        epochs:       Number of training epochs.
        batch_size:   Batch size for loader.
        lr:           Transformer learning rate.
        lr_backbone:  Backbone learning rate.
        lr_step:      Epoch at which to decay LR by 0.1.
        warmup_steps: Warmup steps.
        val_every:    Epoch interval to run validation.
        resume:       Path/directory to resume model weights.
        workers:      DataLoader workers count.
        cache_dir:    Directory where downloaded video frames are cached.
        img_size:     Input dimensions.
        box_size:     Square box size (pixels) centered on points.
        num_queries:  DETR decoder queries count.
        score_thresh: Score threshold to count predictions at validation.
        limit_batches: Stop epoch early (for quick debugging).
        min_batches:  Oversample so each epoch has at least this many batches.
        companion_mode: 'prev10' or similar to enable context frame concatenation.
        context_offset: Number of frames (N) to seek backward for context frame.
        pretrained:   Load facebook/detr-resnet-50 weights.
        mlflow_tracking_uri: MLflow tracking URI. Defaults to local ./mlruns if not given.
        mlflow_experiment: MLflow experiment name.
        fresh_finetune: Set True when --resume points at a DIFFERENT completed run's
            checkpoint (e.g. the currently-deployed production model) rather than
            continuing this exact same interrupted experiment. Loads model weights only —
            epoch/global_step reset to 0 and the LR scheduler starts a fresh warmup+decay
            curve using this run's own --lr/--warmup_steps/--decay_steps, instead of
            inheriting the source checkpoint's scheduler position (which otherwise silently
            resumes wherever THAT run's schedule had already decayed to, potentially near
            zero, defeating the point of a deliberately-configured fine-tune).
        use_video_data: Pull ground truth from /api/v1/labeling-tasks (raw video/photo
            crops). Set to False (with use_ortho_data=True) to train an ortho-only model
            for comparison against a merged run.
        use_ortho_data: Also pull ground truth from /api/v1/ortho-tasks (real orthomosaic
            annotations) and merge into the combined train/val pool. Requires
            tile_server_url/tile_server_token to be set (directly or via .env).
        tile_server_url: Base URL of the ortho tile server (same one /predict_ortho uses).
        tile_server_token: Bearer token for the tile server.
        ortho_stitch_radius: Neighboring-tile radius stitched around each ortho task's
            center tile (1 -> 3x3 grid of 256px tiles = 768x768).
    """
    from transformers import DetrForObjectDetection, DetrConfig

    if api_url is None:
        api_url = os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1")
    if api_key is None:
        api_key = os.environ.get("SOCKEYE_API_KEY")
        
    if not api_key:
        raise ValueError(
            "Sockeye API Key is required. Please set it in .env as SOCKEYE_API_KEY or pass it via --api_key."
        )

    if tile_server_url is None:
        tile_server_url = os.environ.get("MODAL_TILE_SERVER_URL")
    if tile_server_token is None:
        tile_server_token = os.environ.get("MODAL_TILE_TOKEN")
    if use_ortho_data and not (tile_server_url and tile_server_token):
        print(
            "Warning: use_ortho_data=True but tile_server_url/tile_server_token not set "
            "(set MODAL_TILE_SERVER_URL/MODAL_TILE_TOKEN in .env or pass explicitly). "
            "Skipping ortho data for this run.", flush=True
        )
        use_ortho_data = False

    base = os.path.dirname(os.path.abspath(__file__))
    auto_name = mlflow if mlflow else _auto_run_name(num_context_frames, context_stride, temporal_fusion)

    # Initialize MLflow client early to resolve save_path/resume/auto-resume
    mlflow_enabled = False
    client = None
    exp = None
    if mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            mlflow.set_experiment(mlflow_experiment)
            exp = client.get_experiment_by_name(mlflow_experiment)
            mlflow_enabled = True
        except Exception as mlflow_err:
            print(f"Warning: MLflow setup failed: {mlflow_err}. Disabling MLflow tracking.", flush=True)

    def find_mlflow_run_by_name(run_name):
        if not mlflow_enabled or not client or not exp:
            return None
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=100
            )
            for r in runs:
                if getattr(r.info, 'lifecycle_stage', 'active') == 'deleted':
                    continue
                r_name = r.info.run_name or r.data.tags.get("mlflow.runName")
                if r_name == run_name:
                    return r
        except Exception as e:
            print(f"Warning: Failed to search runs by name: {e}", flush=True)
        try:
            r = client.get_run(run_name)
            if getattr(r.info, 'lifecycle_stage', 'active') != 'deleted':
                return r
        except Exception:
            return None

    def find_latest_mlflow_run(resume_name):
        if not mlflow_enabled or not client or not exp:
            return None
        import re
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
        pattern = re.compile(rf"^{re.escape(resume_name)}_\d{{8}}_\d{{6}}$")
        candidates = []
        for r in runs:
            if getattr(r.info, 'lifecycle_stage', 'active') == 'deleted':
                continue
            r_name = r.info.run_name or r.data.tags.get("mlflow.runName")
            if r_name and pattern.match(r_name):
                try:
                    artifacts = client.list_artifacts(r.info.run_id)
                    has_pth = any(a.path.endswith('.pth') for a in artifacts)
                    if has_pth:
                        candidates.append((r, r.info.start_time))
                except Exception:
                    pass
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    resume_run_id = None
    # When the user explicitly provides a run name via --mlflow, skip auto-resume
    # so that a fresh run is always started under that name.
    explicit_run_name = bool(mlflow)

    if save_path:
        run_name_candidate = os.path.basename(save_path.rstrip('/'))
        if not resume and not explicit_run_name:
            existing_run = find_mlflow_run_by_name(run_name_candidate)
            if existing_run:
                resume = run_name_candidate
                resume_run_id = existing_run.info.run_id
                print(f"Auto-resuming from MLflow run: {run_name_candidate} (ID: {resume_run_id})", flush=True)
    else:
        if resume:
            parent_run = find_mlflow_run_by_name(resume)
            if parent_run:
                parent_run_name = parent_run.info.run_name or parent_run.data.tags.get("mlflow.runName")
                active_branched_run = find_latest_mlflow_run(parent_run_name)
                if active_branched_run:
                    branched_name = active_branched_run.info.run_name or active_branched_run.data.tags.get("mlflow.runName")
                    print(f"Detected active branched run '{branched_name}' for parent '{parent_run_name}' in MLflow. Resuming from it.", flush=True)
                    resume = branched_name
                    resume_run_id = active_branched_run.info.run_id
                    auto_name = branched_name
                else:
                    import datetime
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    branched_name = f"{parent_run_name}_{timestamp}"
                    print(f"Branching new run '{branched_name}' from parent run '{parent_run_name}' in MLflow.", flush=True)
                    auto_name = branched_name
            else:
                pass
            save_path = os.path.join(base, 'runs', auto_name)
        else:
            if not explicit_run_name:
                existing_run = find_mlflow_run_by_name(auto_name)
                if existing_run:
                    resume = auto_name
                    resume_run_id = existing_run.info.run_id
                    save_path = os.path.join(base, 'runs', auto_name)
                    print(f"Auto-resuming from default MLflow run: {auto_name} (ID: {resume_run_id})", flush=True)
                else:
                    save_path = os.path.join(base, 'runs', auto_name)
                    print(f"Starting new default MLflow run: {auto_name}", flush=True)
            else:
                save_path = os.path.join(base, 'runs', auto_name)
                print(f"Starting new MLflow run (explicit name): {auto_name}", flush=True)

    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    writer.add_text("config/num_context_frames", str(num_context_frames))
    writer.add_text("config/context_stride", str(context_stride))
    writer.add_text("config/temporal_fusion", str(temporal_fusion))

    if mlflow_enabled:
        try:
            if resume_run_id:
                mlflow.start_run(run_id=resume_run_id)
                print(f"Resumed active MLflow run: {auto_name} (ID: {resume_run_id})", flush=True)
            else:
                mlflow.start_run(run_name=auto_name)
                print(f"Started MLflow run: {auto_name}", flush=True)
                mlflow.log_params({
                    'api_url': api_url, 'save_path': save_path, 'batch_size': batch_size,
                    'lr': lr, 'lr_backbone': lr_backbone, 'lr_step': lr_step, 'warmup_steps': warmup_steps,
                    'decay_steps': decay_steps, 'val_every': val_every, 'resume': resume, 'workers': workers,
                    'img_size': img_size, 'box_size': box_size, 'num_queries': num_queries,
                    'score_thresh': score_thresh, 'limit_batches': limit_batches, 'min_batches': min_batches,
                    'num_context_frames': num_context_frames, 'context_stride': context_stride,
                    'pretrained': pretrained, 'temporal_fusion': temporal_fusion,
                    'use_video_data': use_video_data, 'use_ortho_data': use_ortho_data,
                    'ortho_stitch_radius': ortho_stitch_radius, 'fresh_finetune': fresh_finetune,
                })
        except Exception as e:
            print(f"Warning: Failed to start MLflow run: {e}. Disabling MLflow tracking for this run.", flush=True)
            mlflow_enabled = False

    def log_metric(name, value, step):
        writer.add_scalar(name, value, step)
        if mlflow_enabled:
            try:
                mlflow.log_metric(name, value, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metric {name} to MLflow: {e}", flush=True)

    print(f'Save dir  : {save_path}', flush=True)
    print(f'API Url   : {api_url}', flush=True)
    print(f'Epochs    : {epochs}  |  Batch: {batch_size}  |  LR: {lr}  |  Box: {box_size}px', flush=True)

    if not use_video_data and not use_ortho_data:
        raise ValueError("At least one of use_video_data / use_ortho_data must be True.")

    # 1. Fetch completed tasks from API (raw video/photo crops)
    headers = {"Authorization": f"Bearer {api_key}"}
    tasks = []
    limit = 500
    if use_video_data:
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
        print(f"Loaded {len(tasks)} completed labeling tasks with fish from Sockeye API.", flush=True)
    else:
        print("Skipping video/photo labeling tasks (use_video_data=False) — ortho-only run.", flush=True)

    # 1b. Fetch real orthomosaic ground truth (annotations placed directly on the
    # ortho map) and merge in, so the model sees post-stitch pixels during training
    # instead of only raw video/photo crops.
    if use_ortho_data:
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
        print(f"Loaded {len(ortho_tasks)} ortho tasks with fish from Sockeye API.", flush=True)
        tasks.extend(ortho_tasks)

    if not tasks:
        print("Error: No tasks with fish found in Sockeye API.")
        return

    # 2. Split into train/val BY FLIGHT, not by individual task — a flat shuffle
    # over tasks would leak the same flight's imagery/conditions across both
    # sets, which matters a lot here given the ortho ground truth is heavily
    # skewed toward a handful of flights.
    random.seed(42)
    tasks_by_flight = {}
    for t in tasks:
        fid = t.get("flightId") or f"__no_flight__{id(t)}"
        tasks_by_flight.setdefault(fid, []).append(t)

    flight_ids = list(tasks_by_flight.keys())
    random.shuffle(flight_ids)
    split_idx = max(1, int(len(flight_ids) * 0.9))
    train_flight_ids = flight_ids[:split_idx]
    val_flight_ids = flight_ids[split_idx:]

    train_tasks = [t for fid in train_flight_ids for t in tasks_by_flight[fid]]
    val_tasks   = [t for fid in val_flight_ids for t in tasks_by_flight[fid]]
    random.shuffle(train_tasks)
    random.shuffle(val_tasks)
    print(
        f"Split: {len(train_flight_ids)} train flights ({len(train_tasks)} tasks)  |  "
        f"{len(val_flight_ids)} val flights ({len(val_tasks)} tasks)", flush=True
    )

    # 3. Create datasets & loaders
    train_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=train_tasks,
        cache_dir=cache_dir, img_size=img_size, box_size=box_size,
        augment=True, num_context_frames=num_context_frames, context_stride=context_stride,
        verbose=verbose, mode="detr",
        tile_server_url=tile_server_url, tile_server_token=tile_server_token,
        ortho_stitch_radius=ortho_stitch_radius,
    )
    val_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=val_tasks,
        cache_dir=cache_dir, img_size=img_size, box_size=box_size,
        augment=False, num_context_frames=num_context_frames, context_stride=context_stride,
        verbose=verbose, mode="detr",
        tile_server_url=tile_server_url, tile_server_token=tile_server_token,
        ortho_stitch_radius=ortho_stitch_radius,
    )

    batch_sampler = SameFrameBatchSampler(len(train_ds), batch_size)
    
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=workers, collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=prefetch_factor if workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(workers > 0)
    )

    # 4. Model Setup
    config = DetrConfig(
        num_labels=NUM_CLASSES,
        num_queries=num_queries,
    )
    model = DetrForObjectDetection(config)

    if pretrained:
        print('Loading pretrained DETR from facebook/detr-resnet-50 …', flush=True)
        pretrained_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        pretrained_state = pretrained_model.state_dict()
        model_state      = model.state_dict()
        
        transfer_state = {k: v for k, v in pretrained_state.items()
                          if k in model_state and model_state[k].shape == v.shape}
        model_state.update(transfer_state)
        model.load_state_dict(model_state, strict=False)
        del pretrained_model
        print(f'  Transferred {len(transfer_state)} tensors.', flush=True)

    if num_context_frames > 0:
        if temporal_fusion == "early":
            _expand_first_conv_6ch(model, in_channels=3 * (1 + num_context_frames))
            model = model.cuda()
        elif temporal_fusion == "transformer":
            model.model = DetrModelWithTemporalFusion(model.model, num_context_frames)
            model = model.cuda()
            print(f'  Swapped inner DetrModel with DetrModelWithTemporalFusion.', flush=True)
    else:
        model = model.cuda()
        
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params / 1e6:.1f}M  |  GPU: {torch.cuda.get_device_name()}', flush=True)

    # 5. Optimizer / Scheduler
    backbone_ids = {id(p) for n, p in model.named_parameters() if 'backbone' in n}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad], 'lr': lr},
        {'params': [p for p in model.parameters() if id(p) in backbone_ids and p.requires_grad], 'lr': lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    steps_per_epoch = max(len(train_loader), 1)
    if epochs is None:
        # Derive epochs from the step budget
        total_steps = warmup_steps + decay_steps
        epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        # Epochs explicitly requested — step budget must cover the full run.
        # Keep warmup as-is; stretch decay to fill the remaining steps.
        epoch_steps = epochs * steps_per_epoch
        total_steps = max(warmup_steps + decay_steps, epoch_steps)
        decay_steps  = total_steps - warmup_steps

    if mlflow_enabled:
        try:
            mlflow.log_params({
                'epochs': epochs, 'total_steps': total_steps, 'n_params_M': round(n_params / 1e6, 2),
            })
        except Exception as e:
            print(f"Warning: Failed to log parameters to MLflow: {e}", flush=True)

    warmup_steps_clamped = max(1, warmup_steps)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps_clamped)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, decay_steps), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps_clamped])
    
    global_step = 0
    start_epoch = 0
    best_mae    = 1e9

    # 6. Resume Checkpoint
    if resume:
        checkpoint_loaded = False
        if mlflow_enabled and (mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")):
            try:
                import mlflow
                client = mlflow.tracking.MlflowClient()
                exp = client.get_experiment_by_name(mlflow_experiment)
                if exp is not None:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        filter_string=f"tags.mlflow.runName = '{resume}'",
                        max_results=1
                    )
                    if not runs:
                        try:
                            run = client.get_run(resume)
                            runs = [run]
                        except Exception:
                            pass
                            
                    if runs:
                        run = runs[0]
                        run_id = run.info.run_id
                        print(f"Found MLflow run '{resume}' (ID: {run_id}). Downloading checkpoint...", flush=True)
                        os.makedirs(save_path, exist_ok=True)
                        local_path = None
                        for name in ['last.pth', 'best.pth']:
                            try:
                                local_path = mlflow.artifacts.download_artifacts(
                                    run_id=run_id,
                                    artifact_path=name,
                                    dst_path=save_path
                                )
                                print(f"Successfully downloaded '{name}' from MLflow to {local_path}", flush=True)
                                resume = local_path
                                break
                            except Exception:
                                pass
            except Exception as mlflow_err:
                print(f"Error resolving checkpoint from MLflow: {mlflow_err}", flush=True)

        if os.path.isfile(resume):
            ckpt = torch.load(resume, map_location='cuda')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            if fresh_finetune:
                # Weights only — this checkpoint came from a different run/experiment.
                # Do NOT inherit its epoch/global_step/scheduler position; start a fresh
                # warmup+decay curve using this run's own lr/warmup_steps/decay_steps.
                start_epoch = 0
                best_mae    = 1e9
                global_step = 0
                print(f'Fresh fine-tune from checkpoint: {resume}  (weights only — fresh epoch/LR schedule)', flush=True)
            else:
                start_epoch = ckpt.get('epoch', 0)
                best_mae    = ckpt.get('best_mae', 1e9)
                global_step = ckpt.get('global_step', 0)
                if 'scheduler' in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt['scheduler'])
                    except Exception:
                        pass
                print(f'Resumed from downloaded MLflow checkpoint: {resume}  (epoch {start_epoch}, best MAE {best_mae:.2f}, step {global_step})', flush=True)
            checkpoint_loaded = True
            
        if not checkpoint_loaded:
            print(f'WARNING: checkpoint not found or could not be loaded: {resume}. Starting from scratch.', flush=True)

    # 7. Training/Validation Functions
    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        running = {}
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:4d}/{epochs}', ncols=110, leave=True)
        t_data_sum = 0.0
        t_train_sum = 0.0
        t_start = time.time()
        for i, (pixel_values, pixel_mask, labels) in enumerate(pbar):
            t_data = time.time() - t_start
            t_data_sum += t_data
            
            if global_step >= total_steps:
                break
            if limit_batches > 0 and i >= limit_batches:
                break

            t_train_start = time.time()
            pixel_values = pixel_values.cuda()
            pixel_mask   = pixel_mask.cuda()
            labels       = [{k: v.cuda() for k, v in lab.items()} for lab in labels]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            loss_dict = outputs.loss_dict

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            
            t_train = time.time() - t_train_start
            t_train_sum += t_train
            
            global_step += 1

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()

            log_metric('train/loss',         loss.item(),          global_step)
            log_metric('train/loss_ce',      loss_dict.get('loss_ce',   torch.tensor(0)).item(), global_step)
            log_metric('train/loss_bbox',    loss_dict.get('loss_bbox', torch.tensor(0)).item(), global_step)
            log_metric('train/loss_giou',    loss_dict.get('loss_giou', torch.tensor(0)).item(), global_step)
            log_metric('train/lr',           optimizer.param_groups[0]['lr'], global_step)
            log_metric('train/lr_backbone',  optimizer.param_groups[1]['lr'], global_step)

            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                ce=f'{loss_dict.get("loss_ce", 0):.3f}',
                bb=f'{loss_dict.get("loss_bbox", 0):.3f}',
                gpu=f'{gpu_mem:.1f}GB',
            )
            t_start = time.time()
            
        elapsed = time.time() - t0
        n = max(len(train_loader), 1)
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f'Epoch {epoch:4d}  ce={running.get("loss_ce",0)/n:.4f}  bbox={running.get("loss_bbox",0)/n:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}  gpu_peak={peak_gpu:.1f}GB  time={elapsed:.0f}s (data: {t_data_sum:.1f}s, train: {t_train_sum:.1f}s)', flush=True)
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        mae = mse = 0.0
        origin_stats = {'video': {'mae': 0.0, 'mse': 0.0, 'n': 0}, 'ortho': {'mae': 0.0, 'mse': 0.0, 'n': 0}}
        n = 0
        debug_saved = False
        _normalize_fn = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        def _make_tiles(pil_img, S, companion_pils=None):
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
                    crop      = pil_img.crop((x0, y0, x0 + S, y0 + S))
                    comp_crops = [c.crop((x0, y0, x0 + S, y0 + S)) if c else None for c in (companion_pils or [])]
                    tiles.append((_tile_tensor(crop, comp_crops), x0, y0, W, H))
            return tiles

        for task in tqdm(val_tasks, desc='  Val', ncols=80, leave=False):
            file_id = task["fileId"]
            frame_t = task["frame"]
            x1, y1, x2, y2 = task["x1"], task["y1"], task["x2"], task["y2"]
            
            # Load frame
            pil_img = val_ds._get_frame(file_id, frame_t)
            W_full, H_full = pil_img.size
            if x2 <= x1 or y2 <= y1:
                x1, y1 = 0, 0
                x2, y2 = W_full, H_full
                
            crop_pil = pil_img.crop((x1, y1, x2, y2))
            
            companion_crops = []
            if num_context_frames > 0:
                for n_idx in range(1, num_context_frames + 1):
                    frame_t_minus_np = max(0, frame_t - n_idx * context_stride)
                    crop_n = None
                    try:
                        img_context_full = val_ds._get_frame(file_id, frame_t_minus_np)
                        crop_n = img_context_full.crop((x1, y1, x2, y2))
                    except Exception:
                        pass
                    companion_crops.append(crop_n)
            
            # Get points (these are absolute coordinates!)
            points = [[pt["x"], pt["y"]] for pt in task.get("points", [])]
            if file_id.startswith("ortho:"):
                # Same coordinate-space shift as SockeyeApiDataset.__getitem__:
                # exported points are relative to the task's center tile only.
                margin = val_ds.ortho_stitch_radius * ORTHO_TILE_SIZE
                points = [[px + margin, py + margin] for px, py in points]
            gt_count = len(points)
            
            # Make tiles of size S from the crop
            tiles = _make_tiles(crop_pil, img_size, companion_crops)
            all_boxes_xyxy = []
            all_scores = []

            for tile_t, tx, ty, orig_W, orig_H in tiles:
                pv = tile_t.unsqueeze(0).cuda()
                pm = torch.ones(1, img_size, img_size, dtype=torch.long).cuda()
                
                out = model(pixel_values=pv, pixel_mask=pm)
                probs = out.logits.softmax(-1)[0, :, 0].cpu()
                boxes = out.pred_boxes[0].cpu()
                
                keep = probs > score_thresh
                if keep.any():
                    b = boxes[keep]
                    s = probs[keep]
                    cx = tx + b[:, 0] * img_size
                    cy = ty + b[:, 1] * img_size
                    w  = b[:, 2] * img_size
                    h  = b[:, 3] * img_size
                    all_boxes_xyxy.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                    all_scores.append(s)
                    
            if all_boxes_xyxy:
                from torchvision.ops import nms as tv_nms
                boxes_cat = torch.cat(all_boxes_xyxy, 0)
                scores_cat = torch.cat(all_scores, 0)
                kept = tv_nms(boxes_cat, scores_cat, iou_threshold=0.5)
                pred_count = float(len(kept))
            else:
                pred_count = 0.0
                
            err = pred_count - gt_count
            mae += abs(err)
            mse += err * err
            n += 1

            origin = 'ortho' if file_id.startswith("ortho:") else 'video'
            origin_stats[origin]['mae'] += abs(err)
            origin_stats[origin]['mse'] += err * err
            origin_stats[origin]['n'] += 1

            # Save debug crop to TensorBoard (using first image)
            if not debug_saved:
                debug_saved = True
                
                # Resize the crop to height 1080 (maintaining aspect ratio)
                target_height = 1080
                W_c, H_c = crop_pil.size
                scale = target_height / H_c
                target_width = int(W_c * scale)
                
                draw_crop = crop_pil.resize((target_width, target_height), Image.BILINEAR)
                draw = ImageDraw.Draw(draw_crop)
                
                # Draw GT bounding boxes in red
                for px, py in points:
                    px_rel = (px - x1) * scale
                    py_rel = (py - y1) * scale
                    scaled_box_size = box_size * scale
                    bx1 = px_rel - scaled_box_size / 2
                    by1 = py_rel - scaled_box_size / 2
                    bx2 = px_rel + scaled_box_size / 2
                    by2 = py_rel + scaled_box_size / 2
                    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
                
                # Draw predicted bounding boxes in green (above score threshold)
                if all_boxes_xyxy:
                    boxes_cat = torch.cat(all_boxes_xyxy, 0)
                    scores_cat = torch.cat(all_scores, 0)
                    kept = tv_nms(boxes_cat, scores_cat, iou_threshold=0.5)
                    for idx in kept:
                        box = boxes_cat[idx]
                        score = scores_cat[idx]
                        if score.item() > score_thresh:
                            bx1, by1, bx2, by2 = box.numpy()
                            draw.rectangle([bx1 * scale, by1 * scale, bx2 * scale, by2 * scale], outline="green", width=2)
                            
                img_arr = np.array(draw_crop)
                writer.add_image("val/predictions", img_arr, epoch, dataformats="HWC")
                if mlflow_enabled:
                    try:
                        mlflow.log_image(img_arr, key="val/predictions", step=epoch)
                    except Exception as e:
                        print(f"Warning: Failed to log validation image to MLflow: {e}", flush=True)

        # Compute validation loss on crops
        val_loss_dict = {}
        with torch.no_grad():
            for pixel_values, pixel_mask, labels in tqdm(val_loader, desc='  Val loss', ncols=80, leave=False):
                pv = pixel_values.cuda()
                pm = pixel_mask.cuda()
                labels = [{k: v.cuda() for k, v in lab.items()} for lab in labels]

                out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
                loss_dict = out.loss_dict

                for k, v in loss_dict.items():
                    val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v.item()

        # log validation loss/metrics
        num_batches = len(val_loader)
        if mlflow_enabled:
            try:
                metrics = {f'val/{k}': v / num_batches for k, v in val_loss_dict.items()}
                mlflow.log_metrics(metrics, step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log val metrics to MLflow: {e}", flush=True)

        log_metric('val/loss',       sum(val_loss_dict.values()) / num_batches, epoch)
        log_metric('val/loss_ce',    val_loss_dict.get('loss_ce', 0.0) / num_batches, epoch)
        log_metric('val/loss_bbox',  val_loss_dict.get('loss_bbox', 0.0) / num_batches, epoch)
        log_metric('val/loss_giou',  val_loss_dict.get('loss_giou', 0.0) / num_batches, epoch)

        # Per-origin breakdown (video vs. ortho) — lets us tell "ortho didn't
        # improve" apart from "video regressed" (catastrophic forgetting) when
        # fine-tuning on the merged task pool, rather than only seeing one
        # combined number.
        for origin, stats in origin_stats.items():
            if stats['n'] > 0:
                o_mae = stats['mae'] / stats['n']
                o_rmse = (stats['mse'] / stats['n']) ** 0.5
                log_metric(f'val/count_mae_{origin}', o_mae, epoch)
                log_metric(f'val/count_rmse_{origin}', o_rmse, epoch)
                print(f'  Val[{origin}]  MAE={o_mae:.2f}  RMSE={o_rmse:.2f}  (n={stats["n"]})', flush=True)

        if n == 0:
            return 1e9
        mae  /= n
        rmse  = (mse / n) ** 0.5
        log_metric('val/count_mae',  mae,  epoch)
        log_metric('val/count_rmse', rmse, epoch)
        print(f'  Val  MAE={mae:.2f}  RMSE={rmse:.2f} (thresh={score_thresh})', flush=True)
        return mae

    # 8. Loop
    print("\nStarting Training Loop...", flush=True)
    epoch = start_epoch
    while epoch < epochs:
        if global_step >= total_steps:
            break
        train_one_epoch(epoch)
        
        if epoch == 0 or (epoch + 1) % val_every == 0 or epoch == epochs - 1 or global_step >= total_steps:
            val_mae = validate(epoch + 1)
            
            # Save checkpoint
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
                'global_step': global_step,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(state, os.path.join(save_path, 'last.pth'))
            if mlflow_enabled:
                try:
                    mlflow.log_artifact(os.path.join(save_path, 'last.pth'))
                except Exception as e:
                    print(f"Warning: Failed to log last.pth to MLflow: {e}", flush=True)
            
            if val_mae < best_mae:
                best_mae = val_mae
                state['best_mae'] = best_mae
                torch.save(state, os.path.join(save_path, 'best.pth'))
                print(f"  --> Saved new best checkpoint with MAE {best_mae:.2f}", flush=True)
                if mlflow_enabled:
                    try:
                        mlflow.log_artifact(os.path.join(save_path, 'best.pth'))
                    except Exception as e:
                        print(f"Warning: Failed to log artifact to MLflow: {e}", flush=True)
        epoch += 1

    print("\nTraining completed successfully!", flush=True)
    if mlflow_enabled:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Failed to end MLflow run: {e}", flush=True)


if __name__ == '__main__':
    fire.Fire(main)
