"""
Run the trained DETR salmon detector on a video and produce an annotated output.

Features:
  - Tiled inference (handles full-resolution frames without resizing)
  - Cross-tile NMS at IoU 0.5
  - Predicted centroids drawn as dots on the frame
  - Count graph (raw + moving average) overlaid in the bottom-right corner
  - Optional companion-frame support (prev10 / background) for 6-ch models

Usage:
    python infer_video_detr.py \\
        --checkpoint runs/exp0/best.pth \\
        --video data/input/videos/DJI_20260630093119_0001_V.mp4 \\
        --output out.mp4

Full options:
    --img_size       512      Tile crop size (must match training)
    --score_thresh   0.25     Confidence threshold
    --iou_thresh     0.5      NMS IoU threshold
    --num_queries    100      DETR query count (must match training)
    --companion_mode None     'prev10' | 'background' | None
    --ma_window      30       Moving-average window in frames
    --graph_height   110      Pixel height of the count-graph overlay
    --graph_width    380      Pixel width of the count-graph overlay
    --dot_radius     6        Radius of predicted-centroid dots
    --max_frames     0        Maximum frames to process (0 = all)
"""

from __future__ import annotations

import math
import os
import json
import glob
from collections import deque
from pathlib import Path

import cv2
import fire
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import nms as tv_nms
from tqdm import tqdm
from tracker import BYTETracker, get_motion_vector, reject_outliers_iqr

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 1


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _expand_first_conv_6ch(model, in_channels: int = 6):
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
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv.to(old.weight.device))
            print(f'  Expanded first conv ({name}): 3 → {in_channels} channels.')
            return


def load_model(checkpoint: str, num_queries: int, companion_mode: str | None):
    from transformers import DetrConfig, DetrForObjectDetection

    ckpt        = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config_dict = ckpt.get('config', {})
    config = DetrConfig(
        num_labels=NUM_CLASSES,
        num_queries=config_dict.get('num_queries', num_queries),
    )
    model = DetrForObjectDetection(config)
    if companion_mode:
        _expand_first_conv_6ch(model)
        
    # Map state dict keys dynamically to handle transformers version mismatches (e.g. model.backbone.model vs model.backbone.conv_encoder.model, self_attn.o_proj vs self_attn.out_proj, etc.)
    model_keys = set(model.state_dict().keys())
    state_dict = ckpt['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in model_keys:
            # 1. Backbone mapping
            if "model.backbone.model." in k:
                k_alt = k.replace("model.backbone.model.", "model.backbone.conv_encoder.model.", 1)
                if k_alt in model_keys:
                    k = k_alt
            elif "model.backbone.conv_encoder.model." in k:
                k_alt = k.replace("model.backbone.conv_encoder.model.", "model.backbone.model.", 1)
                if k_alt in model_keys:
                    k = k_alt
            
            # 2. Attention projection mapping
            if "self_attn.o_proj." in k:
                k_alt = k.replace("self_attn.o_proj.", "self_attn.out_proj.", 1)
                if k_alt in model_keys:
                    k = k_alt
            elif "self_attn.out_proj." in k:
                k_alt = k.replace("self_attn.out_proj.", "self_attn.o_proj.", 1)
                if k_alt in model_keys:
                    k = k_alt
                    
            if "encoder_attn.o_proj." in k:
                k_alt = k.replace("encoder_attn.o_proj.", "encoder_attn.out_proj.", 1)
                if k_alt in model_keys:
                    k = k_alt
            elif "encoder_attn.out_proj." in k:
                k_alt = k.replace("encoder_attn.out_proj.", "encoder_attn.o_proj.", 1)
                if k_alt in model_keys:
                    k = k_alt
            
            # 3. MLP fc mapping
            if ".mlp.fc1." in k:
                k_alt = k.replace(".mlp.fc1.", ".fc1.", 1)
                if k_alt in model_keys:
                    k = k_alt
            elif ".fc1." in k:
                k_alt = k.replace(".fc1.", ".mlp.fc1.", 1)
                if k_alt in model_keys:
                    k = k_alt
            
            if ".mlp.fc2." in k:
                k_alt = k.replace(".mlp.fc2.", ".fc2.", 1)
                if k_alt in model_keys:
                    k = k_alt
            elif ".fc2." in k:
                k_alt = k.replace(".fc2.", ".mlp.fc2.", 1)
                if k_alt in model_keys:
                    k = k_alt
                    
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval().cuda()
    return model


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _tile_tensor(crop: Image.Image, companion_crop: Image.Image | None) -> torch.Tensor:
    t = _normalize(T.functional.to_tensor(crop))
    if companion_crop is not None:
        c = _normalize(T.functional.to_tensor(companion_crop))
        t = torch.cat([t, c], dim=0)
    return t


def _make_tiles(pil: Image.Image, S: int, companion_pil: Image.Image | None):
    """Non-overlapping tiles; last tile slides back at edges."""
    W, H = pil.size

    def origins(dim):
        if dim <= S:
            return [0]
        os_ = list(range(0, dim - S + 1, S))
        if os_[-1] + S < dim:
            os_.append(dim - S)
        return sorted(set(os_))

    if W < S or H < S:
        pil_r  = pil.resize((S, S), Image.BILINEAR)
        comp_r = companion_pil.resize((S, S), Image.BILINEAR) if companion_pil else None
        return [(_tile_tensor(pil_r, comp_r), 0, 0)]

    tiles = []
    for y0 in origins(H):
        for x0 in origins(W):
            crop = pil.crop((x0, y0, x0 + S, y0 + S))
            comp = companion_pil.crop((x0, y0, x0 + S, y0 + S)) if companion_pil else None
            tiles.append((_tile_tensor(crop, comp), x0, y0))
    return tiles


@torch.no_grad()
def infer_frame(
    model,
    frame_bgr: np.ndarray,
    img_size: int,
    score_thresh: float,
    iou_thresh: float,
    companion_bgr: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Single-frame inference (convenience wrapper around the batch version)."""
    batch_res = _infer_batch(model, [(frame_bgr, companion_bgr)], img_size, score_thresh, iou_thresh)[0]
    return [(r[0], r[1]) for r in batch_res]


@torch.no_grad()
def _infer_batch(
    model,
    frames: list[tuple[np.ndarray, np.ndarray | None]],
    img_size: int,
    score_thresh: float,
    iou_thresh: float,
    tile_batch_size: int = 16,
) -> list[list[tuple[int, int, float, float, float, float, float]]]:
    """
    Infer on multiple frames.  Tiles are collected across all frames and sent
    to the GPU in sub-batches of `tile_batch_size`, balancing throughput vs VRAM.
    Returns: list of centroid lists, one per frame.
    """
    n_frames = len(frames)
    S        = img_size

    # 1. Build the full tile list (CPU side — cheap)
    all_tile_info: list[tuple[torch.Tensor, int, int, int]] = []
    for fi, (frame_bgr, companion_bgr) in enumerate(frames):
        pil      = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        comp_pil = (Image.fromarray(cv2.cvtColor(companion_bgr, cv2.COLOR_BGR2RGB))
                    if companion_bgr is not None else None)
        for tile_t, tx, ty in _make_tiles(pil, S, comp_pil):
            all_tile_info.append((tile_t, tx, ty, fi))

    # 2. Forward pass in GPU sub-batches
    frame_boxes  = [[] for _ in range(n_frames)]
    frame_scores = [[] for _ in range(n_frames)]

    for start in range(0, len(all_tile_info), tile_batch_size):
        sub = all_tile_info[start: start + tile_batch_size]
        pv  = torch.stack([t for t, _, _, _ in sub]).cuda()
        pm  = torch.ones(len(sub), S, S, dtype=torch.long).cuda()
        out = model(pixel_values=pv, pixel_mask=pm)
        logits = out.logits.cpu()      # [B, Q, C+1]
        boxes  = out.pred_boxes.cpu()  # [B, Q, 4]

        for i, (_, tx, ty, fi) in enumerate(sub):
            probs = logits[i].softmax(-1)[:, 0]
            bxs   = boxes[i]
            keep  = probs > score_thresh
            if keep.any():
                b  = bxs[keep]; s = probs[keep]
                cx = tx + b[:, 0] * S; cy = ty + b[:, 1] * S
                w  = b[:, 2] * S;      h  = b[:, 3] * S
                frame_boxes[fi].append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1))
                frame_scores[fi].append(s)

    # 3. Per-frame NMS
    results = []
    for fi in range(n_frames):
        if frame_boxes[fi]:
            bc   = torch.cat(frame_boxes[fi], 0)
            sc   = torch.cat(frame_scores[fi], 0)
            kept = tv_nms(bc, sc, iou_threshold=iou_thresh)
            frame_res = []
            for i in kept.tolist():
                x1, y1, x2, y2 = bc[i].tolist()
                score = sc[i].item()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                frame_res.append((cx, cy, x1, y1, x2, y2, score))
            results.append(frame_res)
        else:
            results.append([])
    return results


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _draw_centroids(frame: np.ndarray, centroids: list, radius: int = 6) -> np.ndarray:
    frame = frame.copy()
    for item in centroids:
        if isinstance(item, dict):
            x = item["cx"]
            y = item["cy"]
            track_id = item["track_id"]
            prev_pos = item.get("prev_pos", None)
        else:
            x, y, track_id, prev_pos = item
        
        if prev_pos is not None:
            px, py = prev_pos
            # Draw arrowed line representing the motion vector
            cv2.arrowedLine(frame, (px, py), (x, y), (255, 220, 0), 2, cv2.LINE_AA, 0, 0.15)
            # Also draw a small green circle at the head
            cv2.circle(frame, (x, y), 4, (0, 220, 80), -1)
        else:
            # Too new, draw a simple dot
            cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), radius,     (0, 220, 80), -1)
            
        if track_id is not None:
            cv2.putText(frame, f"#{track_id}", (x + radius + 4, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"#{track_id}", (x + radius + 4, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_motion_panel(frame: np.ndarray, mean_speed: float, num_tracks: int, mean_vector: np.ndarray | None, margin: int = 20) -> np.ndarray:
    """Overlay a premium-looking motion stats panel in the top-left corner."""
    H, W = frame.shape[:2]
    px, py = margin, margin
    pw, ph = 420, 115
    
    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
    
    # Border
    accent_color = (0, 215, 90)  # Bright green
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), accent_color, 2)
    
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    
    # Title
    cv2.putText(frame, "MOTION ANALYSIS (1s window)", (px + 16, py + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Speed value
    speed_text = f"{mean_speed:.1f} px/s" if mean_speed > 0 else "Calculating..."
    cv2.putText(frame, speed_text, (px + 16, py + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
    # Active tracks count
    tracks_text = f"Tracked Fish: {num_tracks}"
    cv2.putText(frame, tracks_text, (px + 16, py + 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 180), 1, cv2.LINE_AA)
                
    # Compass / Arrow for predominant direction
    if mean_vector is not None and np.linalg.norm(mean_vector) > 1e-3:
        cx_c, cy_c = px + 355, py + 60
        r_c = 28
        cv2.circle(frame, (cx_c, cy_c), r_c, (60, 60, 60), 1, cv2.LINE_AA)
        
        vx, vy = mean_vector
        norm_v = np.sqrt(vx*vx + vy*vy)
        ux, uy = vx / norm_v, vy / norm_v
        
        arrow_end_x = int(cx_c + r_c * ux)
        arrow_end_y = int(cy_c + r_c * uy)
        
        cv2.arrowedLine(frame, (cx_c, cy_c), (arrow_end_x, arrow_end_y), (255, 220, 0), 3, cv2.LINE_AA, 0, 0.35)
        cv2.circle(frame, (cx_c, cy_c), 3, (255, 255, 255), -1)
        
    return frame


def _draw_count_graph(
    frame: np.ndarray,
    count_history: deque,
    ma_window: int,
    hourly_rate: float | None = None,
    traverse_seconds: float   = 25.0,
    graph_h: int = 110,
    graph_w: int = 380,
    margin: int  = 14,
) -> np.ndarray:
    """Overlay a count graph (raw + moving average) in the bottom-right corner.
    Right y-axis shows equivalent hourly escapement values.
    """
    """Overlay a count graph (raw + moving average) in the bottom-right corner."""
    H, W = frame.shape[:2]
    counts = list(count_history)
    n      = len(counts)
    if n < 2:
        return frame

    # Panel position
    px = W - graph_w - margin
    py = H - graph_h - margin

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 6, py - 22), (W - margin + 2, H - margin + 2), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    max_c = max(max(counts), 1)

    def to_px(i: int, c: float) -> tuple[int, int]:
        x = px + int(i / max(n - 1, 1) * graph_w)
        y = py + graph_h - int(c / max_c * graph_h)
        return x, y

    # Raw count (dim grey)
    for i in range(1, n):
        cv2.line(frame, to_px(i - 1, counts[i - 1]), to_px(i, counts[i]), (120, 120, 120), 2,
                 lineType=cv2.LINE_AA)

    # Moving average (bright green)
    ma = []
    for i in range(n):
        win = counts[max(0, i - ma_window + 1): i + 1]
        ma.append(sum(win) / len(win))

    for i in range(1, n):
        cv2.line(frame, to_px(i - 1, ma[i - 1]), to_px(i, ma[i]), (0, 215, 90), 4,
                 lineType=cv2.LINE_AA)

    # Axis reference line at max
    cv2.line(frame, (px, py), (px + graph_w, py), (60, 60, 60), 1)
    # Left y-axis: count label at top
    cv2.putText(frame, str(int(max_c)), (px - 4, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, '0', (px - 4, py + graph_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1, cv2.LINE_AA)

    # Right y-axis: hourly escapement values
    esc_per_count = 3600.0 / max(traverse_seconds, 1e-6)
    esc_max = int(max_c * esc_per_count)
    rx = px + graph_w + 3
    cv2.putText(frame, f'{esc_max:,}/hr', (rx, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, '0/hr', (rx, py + graph_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 200, 255), 1, cv2.LINE_AA)

    # Current count + MA text
    label = f'Count: {counts[-1]}   MA-{ma_window}: {ma[-1]:.1f}'
    cv2.putText(frame, label, (px, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 90), 2, cv2.LINE_AA)

    # Hourly escapement
    if hourly_rate is not None:
        esc_label = f'Escapement: {hourly_rate:,.0f} fish/hr'
        cv2.putText(frame, esc_label, (px, py - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 200, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Pipeline Wrapper
# ---------------------------------------------------------------------------

class Pipeline:
    def detect(
        self,
        checkpoint: str,
        video: str,
        output_json: str | None = None,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        img_size: int = 512,
        score_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        num_queries: int = 100,
        companion_mode: str | None = None,  # 'prev10' | 'background' | None
        batch_frames: int = 1,
        tile_batch_size: int = 16,
        infer_every: int = 1,
        detect_height: int | None = None
    ) -> dict:
        """
        Stage 1: GPU-bound inference. Processes video from start_sec to start_sec+duration_sec,
        computes bounding boxes, and returns them. Also saves them to output_json if provided.
        """
        video_path = Path(video)
        
        # ---- load model ----
        print(f'Loading checkpoint: {checkpoint}')
        model = load_model(checkpoint, num_queries, companion_mode)
        print(f'Model loaded  |  companion_mode={companion_mode}  |  img_size={img_size}')

        # ---- open video ----
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_sec * fps)
        if duration_sec is not None:
            n_to_process = int(duration_sec * fps)
            end_frame = min(start_frame + n_to_process, total_frames)
        else:
            end_frame = total_frames
            n_to_process = end_frame - start_frame

        # Resizing configuration
        scale = None
        if detect_height is not None and detect_height < h_vid:
            scale = detect_height / h_vid
            detect_w = int(w_vid * scale)
            detect_h = detect_height
            print(f'Resizing video frames from {w_vid}x{h_vid} to {detect_w}x{detect_h} for inference (scale={scale:.4f})')
        else:
            scale = None

        print(f'Video: {video_path.name}  {w_vid}×{h_vid}  {fps:.2f} fps  '
              f'Frames {start_frame} → {end_frame} ({n_to_process} frames)')

        # ---- frame buffer for companion ----
        PREV_OFFSET    = 10
        frame_buffer: deque[np.ndarray] = deque(maxlen=PREV_OFFSET + 1)
        
        # Pre-populate companion buffer if starting in the middle
        if companion_mode == 'prev10' and start_frame > 0:
            buffer_start = max(0, start_frame - PREV_OFFSET)
            cap.set(cv2.CAP_PROP_POS_FRAMES, buffer_start)
            for _ in range(start_frame - buffer_start):
                ok, f = cap.read()
                if not ok:
                    break
                frame_buffer.append(f)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # ---- process in batched blocks ----
        block_size      = batch_frames * max(infer_every, 1)
        frame_block:    list[tuple[np.ndarray, np.ndarray | None]] = []
        infer_in_block: list[tuple[np.ndarray, np.ndarray | None, int]] = []
        
        detections_out = {}

        pbar = tqdm(total=n_to_process, desc='Detecting', unit='frame', ncols=100)

        def _write_block():
            # If scaling is requested, resize frames in memory before inference
            frames_to_infer = []
            for f, c, _ in infer_in_block:
                if scale is not None:
                    f_resized = cv2.resize(f, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
                    c_resized = cv2.resize(c, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR) if c is not None else None
                    frames_to_infer.append((f_resized, c_resized))
                else:
                    frames_to_infer.append((f, c))

            batch_results = (_infer_batch(model, frames_to_infer,
                                          img_size, score_thresh, iou_thresh,
                                          tile_batch_size=tile_batch_size)
                              if infer_in_block else [])
            infer_by_pos = {pos: res for (_, _, pos), res in zip(infer_in_block, batch_results)}

            for pos, _ in enumerate(frame_block):
                abs_frame_idx = frame_idx - len(frame_block) + pos
                if pos in infer_by_pos:
                    detections = infer_by_pos[pos]
                    if scale is not None:
                        # Scale bounding box coordinates back to original video pixels
                        detections_out[abs_frame_idx] = [
                            [
                                round(d[2] / scale, 2),
                                round(d[3] / scale, 2),
                                round(d[4] / scale, 2),
                                round(d[5] / scale, 2),
                                round(d[6], 4)
                            ]
                            for d in detections
                        ]
                    else:
                        detections_out[abs_frame_idx] = [
                            [round(d[2], 2), round(d[3], 2), round(d[4], 2), round(d[5], 2), round(d[6], 4)]
                            for d in detections
                        ]
                pbar.update(1)

        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_buffer.append(frame_bgr)
            companion_bgr = frame_buffer[0] if companion_mode == 'prev10' else None

            block_pos = len(frame_block)
            frame_block.append((frame_bgr, companion_bgr))
            if (frame_idx - start_frame) % infer_every == 0:
                infer_in_block.append((frame_bgr, companion_bgr, block_pos))

            frame_idx += 1

            if len(frame_block) >= block_size:
                _write_block()
                frame_block.clear()
                infer_in_block.clear()

        if frame_block:   # flush remainder
            _write_block()

        pbar.close()
        cap.release()

        out_data = {
            "video_path": video,
            "fps": fps,
            "width": w_vid,
            "height": h_vid,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "infer_every": infer_every,
            "detections": detections_out
        }

        if output_json is not None:
            output_json_path = Path(output_json)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(out_data, f)
            print(f"Saved detections to {output_json}")

        return out_data

    def track(
        self,
        detection_jsons: list[str] | str | dict,
        output_track_json: str | None = None,
        track_thresh: float = 0.35,
        min_box_score: float = 0.1,
        ma_window: int = 30
    ) -> dict:
        """
        Stage 2: CPU-bound tracking. Loads and merges bounding boxes from one or more
        detection files/dicts, runs the BYTETracker, filters outliers, and computes paths + metrics.
        """
        video_path = None
        fps = 30.0
        width = 1920
        height = 1080
        merged_detections = {}

        if isinstance(detection_jsons, dict):
            video_path = detection_jsons.get("video_path")
            fps = detection_jsons.get("fps", 30.0)
            width = detection_jsons.get("width", 1920)
            height = detection_jsons.get("height", 1080)
            for k, v in detection_jsons.get("detections", {}).items():
                merged_detections[int(k)] = v
        else:
            files_to_load = []
            if isinstance(detection_jsons, str):
                if "*" in detection_jsons or "?" in detection_jsons:
                    files_to_load = sorted(glob.glob(detection_jsons))
                else:
                    files_to_load = [p.strip() for p in detection_jsons.split(",") if p.strip()]
            elif isinstance(detection_jsons, list):
                files_to_load = detection_jsons

            if not files_to_load:
                raise ValueError("No detection inputs provided.")

            print(f"Loading and merging {len(files_to_load)} detection file(s)...")
            for idx, file_path in enumerate(files_to_load):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if idx == 0:
                    video_path = data.get("video_path")
                    fps = data.get("fps", 30.0)
                    width = data.get("width", 1920)
                    height = data.get("height", 1080)
                for k, v in data.get("detections", {}).items():
                    merged_detections[int(k)] = v

        sorted_frame_indices = sorted(merged_detections.keys())
        if not sorted_frame_indices:
            print("Warning: No detections to track.")
            
        tracker = BYTETracker(track_thresh=track_thresh, track_buffer=ma_window, min_box_score=min_box_score)

        tracks_data = {}
        metrics_counts = {}
        metrics_speeds = {}
        metrics_directions = {}
        
        count_history = deque(maxlen=max(ma_window * 3, 90))
        last_mean_speed = 0.0
        last_mean_vector = None

        print(f"Tracking over {len(sorted_frame_indices)} frames...")
        
        for frame_idx in tqdm(sorted_frame_indices, desc="Tracking", unit="frame", ncols=100):
            detections = merged_detections[frame_idx]
            tracker.frame_id = frame_idx
            active_tracks = tracker.update(detections)

            frame_centroids = []
            motion_vectors = []
            for track in active_tracks:
                cx = int((track.tlbr[0] + track.tlbr[2]) / 2)
                cy = int((track.tlbr[1] + track.tlbr[3]) / 2)

                mv_res = get_motion_vector(track, frame_idx, fps)
                if mv_res is not None:
                    mv, cx_prev, cy_prev = mv_res
                    motion_vectors.append(mv)
                    prev_pos = [int(cx_prev), int(cy_prev)]
                else:
                    prev_pos = None

                frame_centroids.append({
                    "track_id": track.track_id,
                    "cx": cx,
                    "cy": cy,
                    "prev_pos": prev_pos
                })

            if len(motion_vectors) > 0:
                filtered_vectors = reject_outliers_iqr(motion_vectors)
                if len(filtered_vectors) > 0:
                    speeds = np.linalg.norm(filtered_vectors, axis=1)
                    last_mean_speed = float(np.mean(speeds))
                    last_mean_vector = np.mean(filtered_vectors, axis=0)
                else:
                    last_mean_speed = 0.0
                    last_mean_vector = None
            else:
                last_mean_speed = 0.0
                last_mean_vector = None

            tracks_data[frame_idx] = frame_centroids
            metrics_counts[frame_idx] = len(frame_centroids)
            metrics_speeds[frame_idx] = round(last_mean_speed, 2)
            metrics_directions[frame_idx] = [round(float(v), 3) for v in last_mean_vector] if last_mean_vector is not None else None
            count_history.append(len(frame_centroids))

        out_data = {
            "video_path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "tracks": tracks_data,
            "metrics": {
                "frame_counts": metrics_counts,
                "frame_speeds": metrics_speeds,
                "frame_directions": metrics_directions
            }
        }

        if output_track_json is not None:
            output_track_path = Path(output_track_json)
            output_track_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_track_path, 'w') as f:
                json.dump(out_data, f)
            print(f"Saved tracks to {output_track_json}")

        return out_data

    def render(
        self,
        track_json: str | dict,
        video: str,
        output_video: str | None = None,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        graph_height: int = 280,
        graph_width: int = 700,
        dot_radius: int = 6,
        traverse_seconds: float = 25.0,
        ma_window: int = 30
    ) -> str:
        """
        Stage 3: CPU/UI-bound rendering. Takes tracking JSON/dict and original video, 
        applies all visual overlays, and saves to an annotated video.
        """
        if isinstance(track_json, dict):
            track_data = track_json
        else:
            with open(track_json, 'r') as f:
                track_data = json.load(f)

        video_path = Path(video)
        if output_video is None:
            output_video = str(video_path.with_name(video_path.stem + '_annotated.mp4'))

        # ---- open video ----
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_sec * fps)
        if duration_sec is not None:
            n_to_process = int(duration_sec * fps)
            end_frame = min(start_frame + n_to_process, total_frames)
        else:
            end_frame = total_frames
            n_to_process = end_frame - start_frame

        print(f'Rendering video: {output_video}  ({n_to_process} frames)...')

        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        writer     = cv2.VideoWriter(output_video, fourcc, fps, (w_vid, h_vid))

        count_history = deque(maxlen=max(ma_window * 3, 90))
        
        tracks_dict = track_data.get("tracks", {})
        metrics = track_data.get("metrics", {})
        frame_counts = metrics.get("frame_counts", {})
        frame_speeds = metrics.get("frame_speeds", {})
        frame_directions = metrics.get("frame_directions", {})

        last_centroids = []
        last_mean_speed = 0.0
        last_mean_vector = None
        last_count = 0

        pbar = tqdm(total=n_to_process, desc='Rendering', unit='frame', ncols=100)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            f_key = str(frame_idx)
            if f_key in tracks_dict:
                last_centroids = tracks_dict[f_key]
                last_mean_speed = frame_speeds.get(f_key, 0.0)
                raw_dir = frame_directions.get(f_key, None)
                last_mean_vector = np.array(raw_dir) if raw_dir is not None else None
                
                last_count = frame_counts.get(f_key, 0)
                count_history.append(last_count)

            ma_val    = sum(count_history) / max(len(count_history), 1)
            out_frame = _draw_centroids(frame_bgr, last_centroids, radius=dot_radius)
            out_frame = _draw_count_graph(
                out_frame, count_history, ma_window,
                hourly_rate=ma_val * (3600 / traverse_seconds),
                traverse_seconds=traverse_seconds,
                graph_h=graph_height, graph_w=graph_width,
            )
            out_frame = _draw_motion_panel(out_frame, last_mean_speed, len(last_centroids), last_mean_vector)
            writer.write(out_frame)
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()
        
        print(f'\nRender Done. Output: {output_video}')
        return output_video

    def run_all(
        self,
        checkpoint: str,
        video: str,
        output: str | None = None,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        img_size: int = 512,
        score_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        num_queries: int = 100,
        companion_mode: str | None = None,
        ma_window: int = 30,
        graph_height: int = 280,
        graph_width: int = 700,
        dot_radius: int = 6,
        infer_every: int = 1,
        batch_frames: int = 1,
        tile_batch_size: int = 16,
        traverse_seconds: float = 25.0,
        track_thresh: float = 0.35,
        min_box_score: float = 0.1,
        detect_height: int | None = None
    ):
        """
        Run the entire pipeline (Detect -> Track -> Render) sequentially in memory.
        """
        print("=== STAGE 1: DETECTION ===")
        detection_data = self.detect(
            checkpoint=checkpoint, video=video, output_json=None,
            start_sec=start_sec, duration_sec=duration_sec, img_size=img_size,
            score_thresh=score_thresh, iou_thresh=iou_thresh, num_queries=num_queries,
            companion_mode=companion_mode, batch_frames=batch_frames,
            tile_batch_size=tile_batch_size, infer_every=infer_every,
            detect_height=detect_height
        )

        print("\n=== STAGE 2: TRACKING ===")
        t_thresh = track_thresh if track_thresh is not None else score_thresh
        track_data = self.track(
            detection_jsons=detection_data, output_track_json=None,
            track_thresh=t_thresh, min_box_score=min_box_score, ma_window=ma_window
        )

        print("\n=== STAGE 3: RENDERING ===")
        self.render(
            track_json=track_data, video=video, output_video=output,
            start_sec=start_sec, duration_sec=duration_sec,
            graph_height=graph_height, graph_width=graph_width,
            dot_radius=dot_radius, traverse_seconds=traverse_seconds,
            ma_window=ma_window
        )

    def detect_parallel(
        self,
        checkpoint: str,
        video: str,
        output_dir: str,
        num_workers: int = 2,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
        img_size: int = 512,
        score_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        num_queries: int = 100,
        companion_mode: str | None = None,
        batch_frames: int = 1,
        tile_batch_size: int = 16,
        infer_every: int = 1,
        detect_height: int | None = None
    ):
        """
        Splits the video duration (or specified slice) into equal chunks and processes them
        in parallel using multiprocessing workers. Detections are saved to output_dir.
        """
        import multiprocessing
        
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        video_duration = total_frames / fps
        if duration_sec is not None:
            slice_duration = min(duration_sec, video_duration - start_sec)
        else:
            slice_duration = video_duration - start_sec
            
        chunk_duration = slice_duration / num_workers
        
        print(f"Parallel Detection: splitting {slice_duration:.1f}s segment (starting at {start_sec:.1f}s) into {num_workers} chunks of {chunk_duration:.1f}s...")
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        tasks = []
        for i in range(num_workers):
            chunk_start_sec = start_sec + i * chunk_duration
            dur = chunk_duration if i < num_workers - 1 else slice_duration - (i * chunk_duration)
            out_json = str(out_path / f"detections_chunk_{i}.json")
            
            tasks.append((
                checkpoint, video, out_json, chunk_start_sec, dur, img_size,
                score_thresh, iou_thresh, num_queries, companion_mode,
                batch_frames, tile_batch_size, infer_every, detect_height
            ))
            
        ctx = multiprocessing.get_context("spawn")
        processes = []
        for task in tasks:
            p = ctx.Process(target=_detect_worker, args=(task,))
            p.start()
            processes.append(p)
            
        print(f"Started {num_workers} detection workers.")
        for p in processes:
            p.join()
            
        print(f"Parallel detection complete. Outputs written to {output_dir}")


def _detect_worker(args):
    """
    Pickleable worker function for detect_parallel.
    """
    (
        checkpoint, video, output_json, start_sec, duration_sec, img_size,
        score_thresh, iou_thresh, num_queries, companion_mode,
        batch_frames, tile_batch_size, infer_every, detect_height
    ) = args

    pipeline = Pipeline()
    pipeline.detect(
        checkpoint=checkpoint, video=video, output_json=output_json,
        start_sec=start_sec, duration_sec=duration_sec, img_size=img_size,
        score_thresh=score_thresh, iou_thresh=iou_thresh, num_queries=num_queries,
        companion_mode=companion_mode, batch_frames=batch_frames,
        tile_batch_size=tile_batch_size, infer_every=infer_every,
        detect_height=detect_height
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        sys.argv.insert(1, 'run_all')
    fire.Fire(Pipeline)
