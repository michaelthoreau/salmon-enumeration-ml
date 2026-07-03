"""
DETR salmon box detector — single GPU, PyTorch.

Model: facebook/detr-resnet-50 (transformer encoder-decoder, Apache 2.0).
       Pretrained ResNet-50 backbone + random encoder/decoder (allows arbitrary
       num_queries). All weights live in model.model.{backbone,encoder,decoder}
       and model.{class_labels_classifier,bbox_predictor} — modify freely.

Points are converted to fixed-size boxes on-the-fly; no annotation changes needed.

Usage:
    python -u train_detr.py
    python -u train_detr.py --epochs=200 --batch_size=4 --data_root=data_big
    python -u train_detr.py --resume=runs/exp0 --epochs=400
"""

import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

warnings.filterwarnings('ignore')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 1   # fish (background is the implicit +1 class inside DETR)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SalmonBoxDataset(Dataset):
    """
    Loads point-annotated salmon images and yields DETR-compatible targets.

    Each point [x, y] becomes a fixed-size box centered on that point.
    Images are resized to img_size × img_size; boxes are normalized to [0, 1].
    """

    def __init__(self, image_paths: list, img_size: int = 640,
                 box_size: int = 50, augment: bool = False,
                 companion_mode: str = None):
        self.paths          = image_paths
        self.img_size       = img_size
        self.half_box       = box_size // 2
        self.augment        = augment
        self.companion_mode = companion_mode   # None | 'prev10' | 'background'
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        ann_path = (img_path
                    .replace('/images/', '/annotations/')
                    .replace('.jpg', '.json')
                    .replace('.jpeg', '.json')
                    .replace('.png', '.json'))

        img = Image.open(img_path).convert('RGB')
        with open(ann_path) as f:
            ann = json.load(f)

        W, H   = img.size
        points = ann.get('points', [])

        # ---- optional companion image (prev10 / background) ----
        companion = None
        if self.companion_mode:
            stem, ext = os.path.splitext(img_path)
            comp_path = f"{stem}_{self.companion_mode}{ext}"
            if os.path.exists(comp_path):
                companion = Image.open(comp_path).convert('RGB')

        # ---- augmentation (on full image before crop) ----
        do_flip = self.augment and torch.rand(1).item() > 0.5
        if do_flip:
            img    = img.transpose(Image.FLIP_LEFT_RIGHT)
            points = [[W - 1 - x, y] for x, y in points]
            if companion is not None:
                companion = companion.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment and torch.rand(1).item() > 0.5:    # color jitter (main only)
            f = 0.7 + torch.rand(1).item() * 0.6
            img = T.functional.adjust_brightness(img, f)
            img = T.functional.adjust_contrast(img,  0.7 + torch.rand(1).item() * 0.6)
            img = T.functional.adjust_saturation(img, 0.7 + torch.rand(1).item() * 0.6)

        # ---- spatial: random crop for training on large images, resize otherwise ----
        S = self.img_size
        if self.augment and W >= S and H >= S:
            # Random crop — preserves fish scale
            x0 = random.randint(0, W - S)
            y0 = random.randint(0, H - S)
            img    = img.crop((x0, y0, x0 + S, y0 + S))
            points = [[px - x0, py - y0] for px, py in points
                      if x0 <= px < x0 + S and y0 <= py < y0 + S]
            if companion is not None:
                companion = companion.resize((W, H), Image.BILINEAR)  # match main size
                companion = companion.crop((x0, y0, x0 + S, y0 + S))
            W, H = S, S
        else:
            img = img.resize((S, S), Image.BILINEAR)
            if companion is not None:
                companion = companion.resize((S, S), Image.BILINEAR)

        sx, sy = S / W, S / H

        # ---- image tensor (normalized, optionally 6-channel) ----
        img_tensor = self.normalize(T.functional.to_tensor(img))   # [3, H, W]
        if companion is not None:
            comp_t     = self.normalize(T.functional.to_tensor(companion))
            img_tensor = torch.cat([img_tensor, comp_t], dim=0)    # [6, H, W]
        elif self.companion_mode:
            img_tensor = torch.cat([img_tensor, torch.zeros_like(img_tensor)], dim=0)

        # ---- boxes: (cx, cy, w, h) normalized to [0,1] ----
        r   = self.half_box
        boxes, labels = [], []
        for x, y in points:
            cx_px = x * sx
            cy_px = y * sy
            x1 = max(0.0, cx_px - r)
            y1 = max(0.0, cy_px - r)
            x2 = min(float(S), cx_px + r)
            y2 = min(float(S), cy_px + r)
            if x2 > x1 and y2 > y1:
                boxes.append([
                    (x1 + x2) / 2 / S,
                    (y1 + y2) / 2 / S,
                    (x2 - x1) / S,
                    (y2 - y1) / S,
                ])
                labels.append(0)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0,      dtype=torch.int64)

        pixel_mask = torch.ones(S, S, dtype=torch.long)

        return img_tensor, pixel_mask, {'class_labels': labels_t, 'boxes': boxes_t}


def collate_fn(batch):
    imgs, masks, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(masks), list(labels)


def _expand_first_conv_6ch(model, in_channels: int = 6):
    """
    Expand the first Conv2d(3→...) found in the model to `in_channels`.
    Pretrained weights for the original 3 channels are preserved;
    extra channels get a 0.1-scaled copy so their contribution starts near-zero.
    Works for any model by walking named_modules to find the first matching conv.
    """
    import torch.nn as nn

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
            # Navigate to parent module and replace
            parts  = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv.to(old.weight.device))
            print(f'  Expanded first conv ({name}): 3 → {in_channels} input channels.',
                  flush=True)
            return
    print('WARNING: could not find Conv2d(3→...) to expand.', flush=True)


def _list_images(data_root, split):
    """Return image paths that have a matching annotation JSON.
    Annotation-first: only annotated images are included, and suffix variants
    like _prev10.jpg / _background.jpg are naturally excluded.
    """
    ann_dir = os.path.join(data_root, split, 'annotations')
    img_dir = os.path.join(data_root, split, 'images')
    img_exts = ('.jpg', '.jpeg', '.png')
    if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
        return []
    result = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith('.json'):
            continue
        stem = fname[:-5]
        for ext in img_exts:
            img_path = os.path.join(img_dir, stem + ext)
            if os.path.isfile(img_path):
                result.append(img_path)
                break
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    save_path:    str   = None,
    epochs:       int   = 300,
    batch_size:   int   = 16,
    lr:           float = 1e-4,
    lr_backbone:  float = 1e-5,
    lr_step:      int   = 240,      # 80 % of epochs
    warmup_steps: int   = 100,
    val_every:    int   = 5,
    resume:       str   = None,
    workers:      int   = 4,
    data_root:    str   = 'data',
    img_size:     int   = 512,
    box_size:     int   = 50,
    num_queries:  int   = 100,      # toy images have far fewer fish
    score_thresh: float = 0.25,
    limit_batches: int  = 0,
    min_batches:  int   = 100,   # oversample so each epoch has at least this many batches
    companion_mode: str = None,  # None | 'prev10' | 'background' — 6-ch input
    pretrained:   bool  = True,
):
    """Train DETR on salmon point-annotation data using fixed-size boxes.

    Args:
        save_path:    Output directory. Auto-selected (runs/exp0, exp1, …) if omitted.
        epochs:       Total training epochs.
        batch_size:   Images per batch.
        lr:           LR for transformer encoder/decoder / heads.
        lr_backbone:  LR for ResNet-50 backbone (usually 10× smaller).
        lr_step:      Epoch at which LR is multiplied by 0.1.
        warmup_steps: Linear LR warmup over this many optimizer steps (0 = off).
        val_every:    Run validation every N epochs.
        resume:       Checkpoint file or run directory to resume from.
        workers:      DataLoader worker processes.
        data_root:    Root dir with train/val subdirs containing images/ & annotations/.
        img_size:     Images are resized to img_size × img_size before the model.
        box_size:     Fixed box side length in pixels (in the resized image) centered
                      on each annotated point.
        num_queries:  DETR decoder queries. Use ≥ max expected fish per image.
        score_thresh: Fish confidence threshold for counting at val time.
        limit_batches: If >0, stop each epoch after this many batches (debug).
        pretrained:   Load pretrained ResNet-50 backbone from facebook/detr-resnet-50.
    """

    from transformers import DetrForObjectDetection, DetrConfig

    base = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_root):
        data_root = os.path.join(base, data_root)

    if save_path is None:
        i = 0
        while os.path.exists(os.path.join(base, f'runs/exp{i}')):
            i += 1
        save_path = os.path.join(base, f'runs/exp{i}')
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    print(f'Save dir  : {save_path}', flush=True)
    print(f'Epochs    : {epochs}  |  Batch: {batch_size}  |  LR: {lr}  '
          f'|  Queries: {num_queries}  |  Box: {box_size}px', flush=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    train_files = _list_images(data_root, 'train')
    val_files   = _list_images(data_root, 'val')
    print(f'Train: {len(train_files)} images  |  Val: {len(val_files)} images', flush=True)

    train_ds = SalmonBoxDataset(train_files, img_size=img_size,
                                box_size=box_size, augment=True,
                                companion_mode=companion_mode)

    n_samples = len(train_ds)
    if min_batches > 0:
        n_samples = max(n_samples, min_batches * batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=n_samples),
        num_workers=workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )
    # Validation uses a tile-based loop over val_files directly (no DataLoader)
    # to handle full-resolution images without resizing or OOM.

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    config = DetrConfig(
        num_labels=NUM_CLASSES,
        num_queries=num_queries,
        # Keep everything else at DETR defaults (256-dim, 6 enc/dec layers, …)
    )
    model = DetrForObjectDetection(config)

    if pretrained:
        print('Loading pretrained DETR (backbone + encoder + decoder) from '
              'facebook/detr-resnet-50 …', flush=True)
        pretrained_model = DetrForObjectDetection.from_pretrained(
            'facebook/detr-resnet-50')
        pretrained_state = pretrained_model.state_dict()
        model_state      = model.state_dict()
        # Transfer every tensor whose shape matches; class head is skipped
        # automatically because num_labels differs (91 → 1).
        transfer_state = {k: v for k, v in pretrained_state.items()
                          if k in model_state
                          and model_state[k].shape == v.shape}
        model_state.update(transfer_state)
        model.load_state_dict(model_state, strict=False)
        del pretrained_model
        skipped = [k for k in pretrained_state if k not in transfer_state]
        print(f'  Transferred {len(transfer_state)} tensors. '
              f'Skipped (shape mismatch): {skipped}', flush=True)

    model = model.cuda()

    if companion_mode:
        _expand_first_conv_6ch(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params / 1e6:.1f}M  |  GPU: {torch.cuda.get_device_name()}',
          flush=True)

    # -----------------------------------------------------------------------
    # Optimizer / scheduler
    # -----------------------------------------------------------------------

    backbone_ids = {id(p) for n, p in model.named_parameters()
                    if 'backbone' in n}
    param_groups = [
        {'params': [p for p in model.parameters()
                    if id(p) not in backbone_ids and p.requires_grad],
         'lr': lr},
        {'params': [p for p in model.parameters()
                    if id(p) in backbone_ids and p.requires_grad],
         'lr': lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Epoch-based schedule so the cosine curve is clearly visible in TensorBoard:
    #   Phase 1 — LinearLR warmup over warmup_epochs (LR: near-zero → base)
    #   Phase 2 — CosineAnnealingLR decay over the rest of training
    warmup_epochs = max(1, math.ceil(warmup_steps / max(len(train_loader), 1)))
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2],
        milestones=[warmup_epochs])
    global_step = 0

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------

    start_epoch = 0
    best_mae    = 1e9

    if resume:
        if os.path.isdir(resume):
            resume = os.path.join(resume, 'best.pth')
        if os.path.isfile(resume):
            ckpt = torch.load(resume, map_location='cuda')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            start_epoch = ckpt.get('epoch', 0)
            best_mae    = ckpt.get('best_mae', 1e9)
            print(f'Resumed from {resume}  (epoch {start_epoch}, best MAE {best_mae:.2f})',
                  flush=True)
        else:
            print(f'WARNING: checkpoint not found: {resume}', flush=True)

    # -----------------------------------------------------------------------
    # Train loop
    # -----------------------------------------------------------------------

    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        running = {}   # accumulate per-key losses for epoch-level log
        t0      = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:4d}/{epochs}',
                    ncols=110, leave=True)
        for i, (pixel_values, pixel_mask, labels) in enumerate(pbar):
            if limit_batches > 0 and i >= limit_batches:
                break

            pixel_values = pixel_values.cuda()
            pixel_mask   = pixel_mask.cuda()
            labels       = [{k: v.cuda() for k, v in lab.items()}
                            for lab in labels]

            outputs   = model(pixel_values=pixel_values,
                              pixel_mask=pixel_mask,
                              labels=labels)
            loss      = outputs.loss
            loss_dict = outputs.loss_dict   # loss_ce, loss_bbox, loss_giou (+ aux)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            global_step += 1

            # ---- accumulate for epoch summary ----
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()

            # ---- step-level TensorBoard ----
            writer.add_scalar('train/loss',         loss.item(),          global_step)
            writer.add_scalar('train/loss_ce',      loss_dict.get('loss_ce',   torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/loss_bbox',    loss_dict.get('loss_bbox', torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/loss_giou',    loss_dict.get('loss_giou', torch.tensor(0)).item(), global_step)
            writer.add_scalar('train/lr',           optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/lr_backbone',  optimizer.param_groups[1]['lr'], global_step)

            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                ce=f'{loss_dict.get("loss_ce", 0):.3f}',
                bb=f'{loss_dict.get("loss_bbox", 0):.3f}',
            )

        scheduler.step()   # epoch-based: once per epoch
        elapsed = time.time() - t0
        n = max(len(train_loader), 1)
        msg = (f'Epoch {epoch:4d}  '
               f'ce={running.get("loss_ce",0)/n:.4f}  '
               f'bbox={running.get("loss_bbox",0)/n:.4f}  '
               f'giou={running.get("loss_giou",0)/n:.4f}  '
               f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
               f'time={elapsed:.0f}s')
        print(msg, flush=True)

    # -----------------------------------------------------------------------
    # Validation loop  — tiles each image into img_size crops, no resizing
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        mae = mse = 0.0
        n = 0
        debug_saved   = False
        pre_nms = post_nms = 0   # track last image's NMS stats for the summary print
        _normalize_fn = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        def _make_tiles(pil, S, companion_pil=None):
            """Return list of (tile_tensor, x0, y0) non-overlapping tiles.
            If companion_pil is provided, tiles are 6-channel [main | companion].
            """
            def _tile_tensor(main_crop, comp_crop=None):
                t = _normalize_fn(T.functional.to_tensor(main_crop))
                if comp_crop is not None:
                    c = _normalize_fn(T.functional.to_tensor(comp_crop))
                    t = torch.cat([t, c], dim=0)
                elif companion_pil is not None:
                    t = torch.cat([t, torch.zeros_like(t)], dim=0)
                return t

            W, H = pil.size
            if W < S or H < S:
                pil_r  = pil.resize((S, S), Image.BILINEAR)
                comp_r = companion_pil.resize((S, S), Image.BILINEAR) if companion_pil else None
                return [(_tile_tensor(pil_r, comp_r), 0, 0, W, H)]
            def origins(dim):
                os_ = list(range(0, dim - S + 1, S))
                if not os_ or os_[-1] + S < dim:
                    os_.append(dim - S)
                return sorted(set(os_))
            tiles = []
            for y0 in origins(H):
                for x0 in origins(W):
                    crop      = pil.crop((x0, y0, x0 + S, y0 + S))
                    comp_crop = companion_pil.crop((x0, y0, x0 + S, y0 + S)) if companion_pil else None
                    tiles.append((_tile_tensor(crop, comp_crop), x0, y0, W, H))
            return tiles

        for img_path in tqdm(val_files[:100], desc='  Val', ncols=80, leave=False):
            ann_path = (img_path
                        .replace('/images/', '/annotations/')
                        .replace('.jpg',  '.json')
                        .replace('.jpeg', '.json')
                        .replace('.png',  '.json'))
            pil = Image.open(img_path).convert('RGB')
            # Load companion for 6-channel validation
            companion_pil = None
            if companion_mode:
                stem, ext = os.path.splitext(img_path)
                comp_path = f"{stem}_{companion_mode}{ext}"
                if os.path.exists(comp_path):
                    companion_pil = Image.open(comp_path).convert('RGB')
            with open(ann_path) as f:
                ann_data = json.load(f)
            points   = ann_data.get('points', [])
            gt_count = len(points)

            tiles = _make_tiles(pil, img_size, companion_pil)
            tile_preds  = [] if not debug_saved else None  # (probs, boxes, tx, ty) for first image
            first_probs = None
            # Accumulate all per-tile boxes in full-image coords for cross-tile NMS
            all_boxes_xyxy: list = []
            all_scores:     list = []

            for tile_t, tx, ty, orig_W, orig_H in tiles:
                pv = tile_t.unsqueeze(0).cuda()
                pm = torch.ones(1, img_size, img_size, dtype=torch.long).cuda()
                out   = model(pixel_values=pv, pixel_mask=pm)
                probs = out.logits.softmax(-1)[0, :, 0].cpu()   # fish scores
                boxes = out.pred_boxes[0].cpu()                   # (cx,cy,w,h) in [0,1]

                if first_probs is None:
                    first_probs = probs
                if tile_preds is not None:
                    tile_preds.append((probs, boxes, tx, ty))

                # Convert above-threshold boxes to full-image (x1,y1,x2,y2) pixels
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

            # NMS across all tiles (removes duplicates near tile borders)
            if all_boxes_xyxy:
                from torchvision.ops import nms as tv_nms
                boxes_cat  = torch.cat(all_boxes_xyxy, 0)
                scores_cat = torch.cat(all_scores, 0)
                kept       = tv_nms(boxes_cat, scores_cat, iou_threshold=0.5)
                pred_count = float(len(kept))
                pre_nms    = len(scores_cat)
                post_nms   = int(pred_count)
            else:
                pred_count = 0.0
                pre_nms = post_nms = 0

            err  = pred_count - gt_count
            mae += abs(err)
            mse += err * err
            n   += 1
            writer.add_scalar('val/max_fish_score', first_probs.max().item(), epoch)

            # ---- TensorBoard debug image: full image with all tile predictions ----
            if not debug_saved:
                debug_saved = True
                _log_val_image_full(writer, epoch, pil, points,
                                    tile_preds, score_thresh, img_size)

        if n == 0:
            return 1e9
        mae  /= n
        rmse  = (mse / n) ** 0.5
        writer.add_scalar('val/count_mae',  mae,  epoch)
        writer.add_scalar('val/count_rmse', rmse, epoch)
        print(f'  Val  MAE={mae:.2f}  RMSE={rmse:.2f}  '
              f'(thresh={score_thresh}  tiles/img≈{len(tiles)}  '
              f'last pre-NMS={pre_nms}→post-NMS={post_nms})',
              flush=True)
        return mae

    # -----------------------------------------------------------------------
    # Debug image helper
    # -----------------------------------------------------------------------

    def _log_val_image_full(writer, epoch, pil, gt_points, tile_preds, thresh, S):
        """
        Draw GT dots (green) and all tile predictions (red boxes) on the
        full original image and log to TensorBoard as val/detections.

        tile_preds: list of (probs [Q], boxes [Q,4], tile_x0, tile_y0)
                    boxes are (cx, cy, w, h) normalised to [0,1] within the tile.
        """
        try:
            import PIL.Image as PILImage
            import PIL.ImageDraw as PILDraw
            import torchvision.transforms.functional as TF

            canvas  = pil.convert('RGBA')
            overlay = PILImage.new('RGBA', canvas.size, (0, 0, 0, 0))
            draw    = PILDraw.Draw(overlay)

            # GT — green filled dots at each annotated point
            r_dot = max(4, S // 64)
            for px, py in gt_points:
                draw.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot],
                             fill=(0, 220, 0, 210))

            # Predictions — red hollow boxes, converted to full-image coordinates
            for probs, boxes, tx, ty in tile_preds:
                keep = probs > thresh
                for box in boxes[keep]:
                    cx_n, cy_n, w_n, h_n = box.tolist()
                    cx = tx + cx_n * S
                    cy = ty + cy_n * S
                    w  = w_n * S
                    h  = h_n * S
                    draw.rectangle([cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                                   outline=(220, 40, 40, 210), width=2)

            result = PILImage.alpha_composite(canvas, overlay).convert('RGB')
            writer.add_image('val_img/gt_and_pred', TF.to_tensor(result), epoch)
        except Exception:
            pass   # never crash training on a debug image failure

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------

    for epoch in range(start_epoch, epochs):
        train_one_epoch(epoch)

        if epoch % val_every == 0 or epoch == epochs - 1:
            mae = validate(epoch)

            if mae < best_mae:
                best_mae = mae
                torch.save({
                    'epoch':      epoch,
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'best_mae':   best_mae,
                    'config':     config.to_dict(),
                }, os.path.join(save_path, 'best.pth'))
                print(f'  ↳ New best MAE={best_mae:.2f}, saved checkpoint.', flush=True)

        torch.save({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'best_mae':   best_mae,
            'config':     config.to_dict(),
        }, os.path.join(save_path, 'last.pth'))

    writer.close()
    print(f'\nDone. Best val MAE: {best_mae:.2f}', flush=True)


if __name__ == '__main__':
    fire.Fire(main)
