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

class SockeyeApiDataset(Dataset):
    def __init__(self, api_url, api_key, tasks, cache_dir="./frame_cache",
                 img_size=512, box_size=50, augment=False,
                 num_context_frames=1, context_stride=5):
        """
        Args:
            api_url (str): Base URL of API
            api_key (str): Sockeye API Key
            tasks (list): List of task dictionaries
            cache_dir (str): Disk path to cache downloaded frame JPEGs
            img_size (int): Image crop height/width to resize to
            box_size (int): Fixed box size in pixels
            augment (bool): If True, apply horizontal flip and color jitter
            num_context_frames (int): Number of context frames (N) to load (0 to disable context)
            context_stride (int): Step size / separation (P) between context frames
        """
        self.api_url = api_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.tasks = tasks
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.half_box = box_size // 2
        self.augment = augment
        self.num_context_frames = num_context_frames
        self.companion_mode = "prev" if num_context_frames > 0 else None
        self.context_stride = context_stride
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.tasks)

    def _get_frame(self, file_id, frame_num):
        """Downloads and caches a single video frame as a PIL Image."""
        cache_path = os.path.join(self.cache_dir, f"{file_id}_{frame_num:06d}.jpg")
        
        # Check disk cache
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                # Cache corrupted; delete and re-download
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            
        # Download from API
        url = f"{self.api_url}/files/{file_id}/frame/{frame_num}?quality=95"
        res = requests.get(url, headers=self.headers, stream=True)
        if res.status_code != 200:
            raise RuntimeError(f"Error downloading frame {frame_num} for file {file_id}: {res.status_code}")
            
        img_bytes = res.content
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Write to disk cache
        with open(cache_path, "wb") as f:
            f.write(img_bytes)
            
        return img

    def __getitem__(self, idx):
        task = self.tasks[idx]
        file_id = task["fileId"]
        frame_t = task["frame"]
        
        # Crop region coordinates
        x1, y1, x2, y2 = task["x1"], task["y1"], task["x2"], task["y2"]
        
        # 1. Retrieve Frame t (current frame)
        img_t_full = self._get_frame(file_id, frame_t)
        W_full, H_full = img_t_full.size
        
        if x2 <= x1 or y2 <= y1:
            x1, y1 = 0, 0
            x2, y2 = W_full, H_full
            
        W_crop = x2 - x1
        H_crop = y2 - y1
        
        # Extract points (absolute coordinates)
        points = [[pt["x"], pt["y"]] for pt in task.get("points", [])]
        
        S = self.img_size
        
        # If training (augment=True) and crop is larger than S, take a random crop
        if self.augment and W_crop >= S and H_crop >= S:
            x0 = random.randint(x1, x2 - S)
            y0 = random.randint(y1, y2 - S)
            img_crop = img_t_full.crop((x0, y0, x0 + S, y0 + S))
            
            # Filter and shift absolute points to the crop
            points = [[px - x0, py - y0] for px, py in points
                      if x0 <= px < x0 + S and y0 <= py < y0 + S]
            
            context_crops = []
            if self.companion_mode:
                for n_idx in range(1, self.num_context_frames + 1):
                    frame_t_minus_np = max(0, frame_t - n_idx * self.context_stride)
                    crop_n = None
                    try:
                        img_context_full = self._get_frame(file_id, frame_t_minus_np)
                        crop_n = img_context_full.crop((x0, y0, x0 + S, y0 + S))
                    except Exception:
                        pass
                    context_crops.append(crop_n)
            W_crop, H_crop = S, S
        else:
            # Fallback if augment is False or crop area is smaller than S
            img_crop = img_t_full.crop((x1, y1, x2, y2))
            img_crop = img_crop.resize((S, S), Image.BILINEAR)
            
            # Shift absolute points to the crop coordinates
            points = [[px - x1, py - y1] for px, py in points]
            # Scale points to match target S size
            sx, sy = S / W_crop, S / H_crop
            points = [[px * sx, py * sy] for px, py in points]
            
            context_crops = []
            if self.companion_mode:
                for n_idx in range(1, self.num_context_frames + 1):
                    frame_t_minus_np = max(0, frame_t - n_idx * self.context_stride)
                    crop_n = None
                    try:
                        img_context_full = self._get_frame(file_id, frame_t_minus_np)
                        c_crop = img_context_full.crop((x1, y1, x2, y2))
                        crop_n = c_crop.resize((S, S), Image.BILINEAR)
                    except Exception:
                        pass
                    context_crops.append(crop_n)
            W_crop, H_crop = S, S

        # ---- augmentation: horizontal flip ----
        do_flip = self.augment and torch.rand(1).item() > 0.5
        if do_flip:
            img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            points = [[W_crop - 1 - px, py] for px, py in points]
            context_crops = [
                c.transpose(Image.FLIP_LEFT_RIGHT) if c is not None else None
                for c in context_crops
            ]
                
        # ---- augmentation: color jitter ----
        if self.augment and torch.rand(1).item() > 0.5:
            f = 0.7 + torch.rand(1).item() * 0.6
            img_crop = T.functional.adjust_brightness(img_crop, f)
            img_crop = T.functional.adjust_contrast(img_crop,  0.7 + torch.rand(1).item() * 0.6)
            img_crop = T.functional.adjust_saturation(img_crop, 0.7 + torch.rand(1).item() * 0.6)

        # ---- image tensor (normalized, optionally multi-channel) ----
        img_tensor = self.normalize(T.functional.to_tensor(img_crop))   # [3, H, W]
        if self.companion_mode:
            tensors_to_cat = [img_tensor]
            for crop_n in context_crops:
                if crop_n is not None:
                    t_n = self.normalize(T.functional.to_tensor(crop_n))
                else:
                    t_n = torch.zeros_like(img_tensor)
                tensors_to_cat.append(t_n)
            img_tensor = torch.cat(tensors_to_cat, dim=0)   # [3 * (1 + N), H, W]

        # ---- boxes: (cx, cy, w, h) normalized to [0,1] ----
        r = self.half_box
        boxes, labels = [], []
        for x, y in points:
            bx1 = max(0.0, x - r)
            by1 = max(0.0, y - r)
            bx2 = min(float(W_crop), x + r)
            by2 = min(float(H_crop), y + r)
            if bx2 > bx1 and by2 > by1:
                boxes.append([
                    (bx1 + bx2) / 2 / W_crop,
                    (by1 + by2) / 2 / H_crop,
                    (bx2 - bx1) / W_crop,
                    (by2 - by1) / H_crop,
                ])
                labels.append(0)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0,      dtype=torch.int64)

        pixel_mask = torch.ones(img_crop.height, img_crop.width, dtype=torch.long)

        return img_tensor, pixel_mask, {'class_labels': labels_t, 'boxes': boxes_t}


def collate_fn(batch):
    imgs, masks, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), list(labels)


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

def main(
    api_url:      str   = None,
    api_key:      str   = None,
    save_path:    str   = None,
    epochs:       int   = None,
    batch_size:   int   = 2,
    lr:           float = 1e-4,
    lr_backbone:  float = 1e-5,
    lr_step:      int   = 240,
    warmup_steps: int   = 1000,
    decay_steps:  int   = 10000,
    val_every:    int   = 5,
    resume:       str   = None,
    workers:      int   = 8,
    cache_dir:    str   = "frame_cache",
    img_size:     int   = 512,
    box_size:     int   = 50,
    num_queries:  int   = 100,
    score_thresh: float = 0.25,
    limit_batches: int  = 0,
    min_batches:  int   = 100,
    num_context_frames: int = 1,
    context_stride: int = 5,
    pretrained:   bool  = True,
    temporal_fusion: str = "early",
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

    base = os.path.dirname(os.path.abspath(__file__))
    if save_path is None:
        i = 0
        while os.path.exists(os.path.join(base, f'runs/exp{i}')):
            i += 1
        save_path = os.path.join(base, f'runs/exp{i}')
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    writer.add_text("config/num_context_frames", str(num_context_frames))
    writer.add_text("config/context_stride", str(context_stride))
    writer.add_text("config/temporal_fusion", str(temporal_fusion))
    print(f'Save dir  : {save_path}', flush=True)
    print(f'API Url   : {api_url}', flush=True)
    print(f'Epochs    : {epochs}  |  Batch: {batch_size}  |  LR: {lr}  |  Box: {box_size}px', flush=True)

    # 1. Fetch completed tasks from API
    headers = {"Authorization": f"Bearer {api_key}"}
    tasks = []
    limit = 500
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
        
    print(f"Loaded {len(tasks)} completed labeling tasks from Sockeye API.", flush=True)
    if not tasks:
        print("Error: No tasks found in Sockeye API.")
        return

    # 2. Split tasks into train/val
    random.seed(42)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * 0.9)
    train_tasks = tasks[:split_idx]
    val_tasks   = tasks[split_idx:]
    print(f"Split: {len(train_tasks)} train tasks  |  {len(val_tasks)} val tasks", flush=True)

    # 3. Create datasets & loaders
    train_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=train_tasks,
        cache_dir=cache_dir, img_size=img_size, box_size=box_size,
        augment=True, num_context_frames=num_context_frames, context_stride=context_stride
    )
    val_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=val_tasks,
        cache_dir=cache_dir, img_size=img_size, box_size=box_size,
        augment=False, num_context_frames=num_context_frames, context_stride=context_stride
    )

    n_samples = len(train_ds)
    if min_batches > 0:
        n_samples = max(n_samples, min_batches * batch_size)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=n_samples),
        num_workers=workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True
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

    total_steps = warmup_steps + decay_steps
    if epochs is None:
        epochs = math.ceil(total_steps / max(len(train_loader), 1))
        
    print(f'Total steps: {total_steps}  |  Epochs: {epochs}  |  Warmup: {warmup_steps}  |  Decay: {decay_steps}', flush=True)

    warmup_steps_clamped = max(1, warmup_steps)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps_clamped)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, decay_steps), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps_clamped])
    
    global_step = 0
    start_epoch = 0
    best_mae    = 1e9

    # 6. Resume Checkpoint
    if resume:
        if os.path.isdir(resume):
            resume = os.path.join(resume, 'best.pth')
        if os.path.isfile(resume):
            ckpt = torch.load(resume, map_location='cuda')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            start_epoch = ckpt.get('epoch', 0)
            best_mae    = ckpt.get('best_mae', 1e9)
            print(f'Resumed from {resume}  (epoch {start_epoch}, best MAE {best_mae:.2f})', flush=True)
        else:
            print(f'WARNING: checkpoint not found: {resume}', flush=True)

    # 7. Training/Validation Functions
    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        running = {}
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:4d}/{epochs}', ncols=110, leave=True)
        for i, (pixel_values, pixel_mask, labels) in enumerate(pbar):
            if global_step >= total_steps:
                break
            if limit_batches > 0 and i >= limit_batches:
                break

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
            global_step += 1

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()

            writer.add_scalar('train/loss',         loss.item(),          global_step)
            writer.add_scalar('train/loss_ce',      loss_dict.get('loss_ce',   torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/loss_bbox',    loss_dict.get('loss_bbox', torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/loss_giou',    loss_dict.get('loss_giou', torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/lr',           optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/lr_backbone',  optimizer.param_groups[1]['lr'], global_step)

            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                ce=f'{loss_dict.get("loss_ce", 0):.3f}',
                bb=f'{loss_dict.get("loss_bbox", 0):.3f}',
                gpu=f'{gpu_mem:.1f}GB',
            )
        elapsed = time.time() - t0
        n = max(len(train_loader), 1)
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f'Epoch {epoch:4d}  ce={running.get("loss_ce",0)/n:.4f}  bbox={running.get("loss_bbox",0)/n:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}  gpu_peak={peak_gpu:.1f}GB  time={elapsed:.0f}s', flush=True)
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        mae = mse = 0.0
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

        # Compute validation loss on crops
        val_loss_dict = {}
        for pixel_values, pixel_mask, labels in val_loader:
            pv = pixel_values.cuda()
            pm = pixel_mask.cuda()
            labels = [{k: v.cuda() for k, v in lab.items()} for lab in labels]
            
            out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
            loss_dict = out.loss_dict
            
            val_loss_dict['loss'] = val_loss_dict.get('loss', 0.0) + out.loss.item()
            for k, v in loss_dict.items():
                val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v.item()
                
        num_batches = max(len(val_loader), 1)
        writer.add_scalar('val/loss',       val_loss_dict.get('loss', 0.0) / num_batches, epoch)
        writer.add_scalar('val/loss_ce',    val_loss_dict.get('loss_ce', 0.0) / num_batches, epoch)
        writer.add_scalar('val/loss_bbox',  val_loss_dict.get('loss_bbox', 0.0) / num_batches, epoch)
        writer.add_scalar('val/loss_giou',  val_loss_dict.get('loss_giou', 0.0) / num_batches, epoch)

        if n == 0:
            return 1e9
        mae  /= n
        rmse  = (mse / n) ** 0.5
        writer.add_scalar('val/count_mae',  mae,  epoch)
        writer.add_scalar('val/count_rmse', rmse, epoch)
        print(f'  Val  MAE={mae:.2f}  RMSE={rmse:.2f} (thresh={score_thresh})', flush=True)
        return mae

    # 8. Loop
    print("\nStarting Training Loop...", flush=True)
    for epoch in range(start_epoch, epochs):
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
            }
            torch.save(state, os.path.join(save_path, 'last.pth'))
            
            if val_mae < best_mae:
                best_mae = val_mae
                state['best_mae'] = best_mae
                torch.save(state, os.path.join(save_path, 'best.pth'))
                print(f"  --> Saved new best checkpoint with MAE {best_mae:.2f}", flush=True)

    print("\nTraining completed successfully!", flush=True)


if __name__ == '__main__':
    fire.Fire(main)
