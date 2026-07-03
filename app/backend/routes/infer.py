"""
Inference routes — model listing and single-image inference.
"""
from __future__ import annotations

import base64
import glob
import io
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import PROJECT_ROOT

router = APIRouter()

RUNS_ROOT = PROJECT_ROOT / "runs"

# Lazy-loaded model cache (path → model)
_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Model config — matches train.py / annotate.py
# ---------------------------------------------------------------------------

def _build_cfg():
    return SimpleNamespace(
        backbone="resnet50", dilation=True, lr_backbone=1e-4,
        position_embedding="sine", masks=False, frozen_weights=None,
        enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=256,
        dropout=0.1, nheads=8, num_queries=700, pre_norm=False,
        transformer_flag="merge3", dm_count=True, branch_merge=True,
        branch_merge_way=2, channel_point=3, only_dm=False, with_weights=False,
        aux_loss=True, cls_loss_coef=2, point_loss_coef=5, giou_loss_coef=2,
        mask_loss_coef=1, dice_loss_coef=1, set_cost_class=2, set_cost_point=5,
        set_cost_giou=2, focal_alpha=0.25, dataset_file="crowd_data", device="cuda",
    )


def _load_model(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]

    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from Networks.CDETR import build_model

    model, _, _ = build_model(_build_cfg())
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _model_cache[model_path] = model
    return model


def _run_infer(model, img_bytes: bytes, threshold: float = 0.35, crop_size: int = 640):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    device = next(model.parameters()).device
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W_orig, H_orig = pil.size
    img_t = tf(pil)  # [3, H, W]

    # Pad so dims are multiples of crop_size
    _, H, W = img_t.shape
    pad_h = (crop_size - H % crop_size) % crop_size
    pad_w = (crop_size - W % crop_size) % crop_size
    img_pad = F.pad(img_t, (0, pad_w, 0, pad_h), "constant", 0)
    _, H_pad, W_pad = img_pad.shape
    num_h = H_pad // crop_size
    num_w = W_pad // crop_size
    n_patches = num_h * num_w

    all_patches = (
        img_pad
        .view(3, num_h, crop_size, W_pad)
        .view(3, num_h, crop_size, num_w, crop_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(3, n_patches, crop_size, crop_size)
        .permute(1, 0, 2, 3)
    )

    batch_size = 4
    all_logits, all_points_t = [], []
    dm_patches = []  # [P, 1, h, w] predicted density maps
    dm_count = 0.0

    for start in range(0, n_patches, batch_size):
        batch = all_patches[start:start + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
        if isinstance(out, list):
            d6, out_dm = out
            dm_patches.append(out_dm[1].cpu())  # [B, 1, h, w]
            dm_count += float(out_dm[1].sum().item())
        else:
            d6 = out
        all_logits.append(d6["pred_logits"].cpu())
        all_points_t.append(d6["pred_points"].cpu())

    all_logits   = torch.cat(all_logits, dim=0)   # [P, Q, 2]
    all_points_t = torch.cat(all_points_t, dim=0) # [P, Q, 3]
    scores = all_logits.sigmoid()[:, :, 1]         # [P, Q]

    patch_rows = torch.arange(num_h).repeat_interleave(num_w)
    patch_cols = torch.arange(num_w).repeat(num_h)
    origins_y  = patch_rows * crop_size
    origins_x  = patch_cols * crop_size

    # Collect predicted points in full-image pixel coords
    pred_points = []  # list of (px, py)
    for pidx in range(n_patches):
        keep = scores[pidx] > threshold
        pts  = all_points_t[pidx][keep].numpy()[:, :2]  # (y_norm, x_norm)
        oy, ox = int(origins_y[pidx]), int(origins_x[pidx])
        for (yn, xn) in pts:
            py = oy + yn * crop_size
            px = ox + xn * crop_size
            if py >= H_orig or px >= W_orig:
                continue
            pred_points.append((int(px), int(py)))

    # Draw overlay on original image
    import cv2
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for (px, py) in pred_points:
        cv2.circle(arr, (px, py), 7, (0, 60, 255), -1)
        cv2.circle(arr, (px, py), 7, (0, 220, 255), 2)

    _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    overlay_b64 = base64.b64encode(buf.tobytes()).decode()

    # Build density map image (if available)
    dm_b64: Optional[str] = None
    if dm_patches:
        dm_all = torch.cat(dm_patches, dim=0)  # [P, 1, h, w]
        dm_h, dm_w = dm_all.shape[-2], dm_all.shape[-1]
        # Stitch patches into full padded map
        dm_full = (
            dm_all
            .view(num_h, num_w, dm_h, dm_w)
            .permute(0, 2, 1, 3)
            .reshape(num_h * dm_h, num_w * dm_w)
        )
        # Crop to original image size scaled by downsample ratio
        ds = crop_size // dm_h  # downsample ratio (typically 8)
        dm_H = H_orig // ds
        dm_W = W_orig // ds
        dm_crop = dm_full[:dm_H, :dm_W].numpy()
        # Normalise and colorize
        mn, mx = dm_crop.min(), dm_crop.max()
        dm_norm = (dm_crop - mn) / (mx - mn + 1e-8)
        dm_uint8 = (dm_norm * 255).astype(np.uint8)
        dm_color = cv2.applyColorMap(dm_uint8, cv2.COLORMAP_INFERNO)
        # Upscale to original image size
        dm_up = cv2.resize(dm_color, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        # Blend density map over original image (60% original, 40% heatmap)
        orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        dm_blended = cv2.addWeighted(orig_bgr, 0.55, dm_up, 0.45, 0)
        # Draw points on top of density blend too
        for (px, py) in pred_points:
            cv2.circle(dm_blended, (px, py), 5, (255, 255, 255), -1)
            cv2.circle(dm_blended, (px, py), 5, (0, 0, 0), 1)
        _, dm_buf = cv2.imencode(".jpg", dm_blended, [cv2.IMWRITE_JPEG_QUALITY, 88])
        dm_b64 = base64.b64encode(dm_buf.tobytes()).decode()

    return {
        "ok": True,
        "overlay": overlay_b64,
        "density_map": dm_b64,
        "count": len(pred_points),
        "dm_count": round(dm_count, 1),
        "threshold": threshold,
        "points": pred_points,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/infer/models")
def list_models():
    models = []
    for path in sorted(glob.glob(str(RUNS_ROOT / "**" / "*.pth"), recursive=True)):
        label = os.path.relpath(path, str(RUNS_ROOT))
        models.append({"path": path, "label": label})
    return models


@router.post("/infer/run")
async def run_infer(
    image: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(0.35),
):
    if not os.path.isfile(model):
        raise HTTPException(status_code=400, detail=f"Model not found: {model}")
    try:
        img_bytes = await image.read()
        m = _load_model(model)
        result = _run_infer(m, img_bytes, threshold)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
