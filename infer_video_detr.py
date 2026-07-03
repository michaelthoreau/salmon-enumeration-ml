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
    model.load_state_dict(ckpt['state_dict'], strict=False)
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
    return _infer_batch(model, [(frame_bgr, companion_bgr)], img_size, score_thresh, iou_thresh)[0]


@torch.no_grad()
def _infer_batch(
    model,
    frames: list[tuple[np.ndarray, np.ndarray | None]],
    img_size: int,
    score_thresh: float,
    iou_thresh: float,
    tile_batch_size: int = 2,
) -> list[list[tuple[int, int]]]:
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
            results.append([(int((bc[i][0]+bc[i][2])/2), int((bc[i][1]+bc[i][3])/2))
                             for i in kept.tolist()])
        else:
            results.append([])
    return results


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _draw_centroids(frame: np.ndarray, centroids: list, radius: int = 6) -> np.ndarray:
    frame = frame.copy()
    for x, y in centroids:
        cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)          # black border
        cv2.circle(frame, (x, y), radius,     (0, 220, 80), -1)        # green fill
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
# Main
# ---------------------------------------------------------------------------

def main(
    checkpoint: str,
    video: str,
    output: str         = None,
    img_size: int       = 512,
    score_thresh: float = 0.25,
    iou_thresh: float   = 0.5,
    num_queries: int    = 100,
    companion_mode: str = None,   # 'prev10' | 'background' | None
    ma_window: int      = 30,     # moving-average window (frames)
    graph_height: int   = 220,
    graph_width: int    = 560,
    dot_radius: int     = 6,
    max_frames: int     = 0,
    infer_every: int    = 1,
    batch_frames: int   = 1,
    tile_batch_size: int = 2,      # tiles per GPU forward pass; raise if VRAM allows
    traverse_seconds: float = 25.0,
):
    """
    Run the DETR model on a video and produce an annotated output video.

    Args:
        checkpoint:     Path to best.pth checkpoint.
        video:          Input video path.
        output:         Output video path. Defaults to <input>_annotated.mp4.
        img_size:       Tile size — must match training.
        score_thresh:   Confidence threshold for detections.
        iou_thresh:     NMS IoU threshold.
        num_queries:    DETR query count — must match training.
        companion_mode: 'prev10' | 'background' | None.
                        For 'prev10', uses the frame 10 positions earlier as the companion.
        ma_window:      Moving-average window for the count graph (frames).
        graph_height:   Height of the count-graph overlay in pixels.
        graph_width:    Width of the count-graph overlay in pixels.
        dot_radius:     Radius of predicted-centroid dots.
        infer_every:        Run inference on one frame in every N. Skips writing intermediate frames.
        traverse_seconds:   Time (s) for one fish to traverse the frame top-to-bottom.
                            Used to convert instantaneous count to hourly escapement.
                            hourly_escapement = moving_avg_count × (3600 / traverse_seconds)
    """
    video_path = Path(video)
    if output is None:
        output = str(video_path.with_name(video_path.stem + '_annotated.mp4'))

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
    n_to_process = total_frames if max_frames <= 0 else min(max_frames, total_frames)

    print(f'Video: {video_path.name}  {w_vid}×{h_vid}  {fps:.2f} fps  '
          f'{total_frames} frames  → processing {n_to_process}')

    # ---- video writer (always at input FPS) ----
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = fps
    writer     = cv2.VideoWriter(output, fourcc, output_fps, (w_vid, h_vid))

    # ---- frame buffer for companion ----
    # For 'prev10': companion is frame i-10 (or frame 0 if not yet accumulated)
    PREV_OFFSET    = 10
    frame_buffer: deque[np.ndarray] = deque(maxlen=PREV_OFFSET + 1)

    # ---- count history for graph ----
    count_history: deque[int] = deque(maxlen=max(ma_window * 3, 90))

    # ---- process ----
    pbar = tqdm(total=n_to_process, desc='Inferring', unit='frame', ncols=100)

    # ---- process in batched blocks ----
    # Each block = batch_frames × infer_every frames.
    # Inference frames within a block are batched into one GPU forward pass,
    # then the entire block is written in temporal order.
    block_size      = batch_frames * max(infer_every, 1)
    frame_block:    list[tuple[np.ndarray, np.ndarray | None]] = []
    infer_in_block: list[tuple[np.ndarray, np.ndarray | None, int]] = []
    last_centroids: list = []
    last_count: int      = 0

    pbar = tqdm(total=n_to_process, desc='Inferring', unit='frame', ncols=100)

    def _write_block():
        nonlocal last_centroids, last_count
        batch_results = (_infer_batch(model, [(f, c) for f, c, _ in infer_in_block],
                                      img_size, score_thresh, iou_thresh,
                                      tile_batch_size=tile_batch_size)
                         if infer_in_block else [])
        infer_by_pos = {pos: res for (_, _, pos), res in zip(infer_in_block, batch_results)}

        for pos, (frame_bgr, _) in enumerate(frame_block):
            if pos in infer_by_pos:
                last_centroids = infer_by_pos[pos]
                last_count     = len(last_centroids)
                count_history.append(last_count)
            ma_val    = sum(count_history) / max(len(count_history), 1)
            out_frame = _draw_centroids(frame_bgr, last_centroids, radius=dot_radius)
            out_frame = _draw_count_graph(
                out_frame, count_history, ma_window,
                hourly_rate=ma_val * (3600 / traverse_seconds),
                traverse_seconds=traverse_seconds,
                graph_h=graph_height, graph_w=graph_width,
            )
            writer.write(out_frame)
            pbar.update(1)
        pbar.set_postfix(count=last_count,
                         ma=f'{sum(count_history)/max(len(count_history),1):.1f}')

    frame_idx = 0
    while frame_idx < n_to_process:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_buffer.append(frame_bgr)
        companion_bgr = frame_buffer[0] if companion_mode == 'prev10' else None

        block_pos = len(frame_block)
        frame_block.append((frame_bgr, companion_bgr))
        if frame_idx % infer_every == 0:
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
    writer.release()

    mean_count  = sum(count_history) / max(len(count_history), 1)
    mean_escape = mean_count * (3600 / traverse_seconds)
    print(f'\nDone. Frames: {n_to_process}  batch_frames: {batch_frames}  '
          f'Mean count: {mean_count:.1f}  '
          f'Mean hourly escapement: {mean_escape:,.0f} fish/hr  |  {output}')


if __name__ == '__main__':
    fire.Fire(main)
