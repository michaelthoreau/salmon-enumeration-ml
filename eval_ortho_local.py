"""Local A/B harness for orthomosaic fish detection.

Scores inference strategies against point labels on a local GeoTIFF ortho:

  endpoint_zN   — a faithful replica of /predict_ortho in predict_modal.py: 256-px
                  web-mercator tiles at zoom N, each upscaled to 512 and run on its
                  own, per-tile box NMS, _pixel_to_latlon back to geo. Scored as-is.
  stitched_Gmm  — the ortho resampled to G mm/px ground sample distance, run through
                  overlapping 512-px windows, optionally under test-time augmentation
                  (the 8 dihedral rotations/flips). Raw detections are merged in ground
                  metres: a cluster within --merge_m becomes one point whose score is
                  the mean over augmentations of each augmentation's best score there
                  (0 for augmentations that missed it), so with no TTA it's plain
                  distance NMS and with TTA it's a vote.

Raw stitched detections (pre-merge, tagged with their augmentation) are saved, so
TTA subsets and multi-scale ensembles are scored without rerunning the model:
  stitched_Gmm_none / _flip / _rot4 / _d4   — TTA subsets of one run
  ens_<G1>+<G2>                              — union of runs at several GSDs, merged

The tile server is emulated with gdalwarp (EPSG:3857, tile-aligned), processed in
strips so neither RAM nor disk has to hold the whole resampled ortho.

Example:
  python eval_ortho_local.py \
      --tif ortho/inputs/44a810ef-4ac8-4f73-af46-8041572bf8b2.tif \
      --labels ortho/inputs/44a810ef-4ac8-4f73-af46-8041572bf8b2.json \
      --gsd_mm 6,7.4,9.6 --tta d4 --ensembles 7.4+9.6
  # re-score saved detections with different settings:
  python eval_ortho_local.py --tif ... --labels ... --score_only --merge_m 0.6
"""
import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from torchvision import transforms as T
from torchvision.ops import nms

from point_metrics import match_points, point_metrics
from predict_modal import (IMAGENET_MEAN, IMAGENET_STD, _latlon_to_tile,
                           _pixel_to_latlon, load_model_from_checkpoint)

EARTH_R = 6378137.0
HALF_WORLD = math.pi * EARTH_R
TILE = 256
S = 512
NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# Dihedral group as (rot90 count, hflip). Index order makes the TTA subsets prefixes
# or simple picks: none={0}, flip={0,4}, rot4={0,1,2,3}, d4=all.
AUGS = [(k, f) for f in (False, True) for k in range(4)]
TTA_SUBSETS = {"none": [0], "flip": [0, 4], "rot4": [0, 1, 2, 3], "d4": list(range(8))}


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def lonlat_to_merc(lon, lat):
    return EARTH_R * math.radians(lon), EARTH_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def merc_to_lonlat(x, y):
    return math.degrees(x / EARTH_R), math.degrees(2 * math.atan(math.exp(y / EARTH_R)) - math.pi / 2)


def merc_res(z):
    """Web-mercator metres per pixel of a 256-px tile at zoom z."""
    return 2 * HALF_WORLD / (TILE * 2 ** z)


def tile_origin_merc(z, tx, ty):
    """Upper-left corner of tile (tx, ty) in EPSG:3857 metres."""
    size = 2 * HALF_WORLD / 2 ** z
    return -HALF_WORLD + tx * size, HALF_WORLD - ty * size


def local_metres(lonlat, lat0):
    """Equirectangular projection to ground metres — fine over a few hundred metres."""
    k = math.cos(math.radians(lat0))
    return [[EARTH_R * math.radians(lon) * k, EARTH_R * math.radians(lat)] for lon, lat in lonlat]


def warp(tif, minx, maxy, width, height, res, workdir):
    """Resamples [minx, maxy] + (width, height) px at res metres/px (EPSG:3857) → HxWx4 uint8."""
    out = os.path.join(workdir, "strip.raw")
    maxx, miny = minx + width * res, maxy - height * res
    subprocess.run(
        ["gdalwarp", "-q", "-overwrite", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
         "-t_srs", "EPSG:3857", "-te", repr(minx), repr(miny), repr(maxx), repr(maxy),
         "-ts", str(width), str(height), "-r", "bilinear",
         "-of", "ENVI", "-co", "INTERLEAVE=BIP", tif, out],
        check=True,
    )
    raw = np.fromfile(out, dtype=np.uint8)
    bands = raw.size // (width * height)
    arr = raw.reshape(height, width, bands)
    if bands == 3:
        alpha = np.where(arr.any(axis=2), 255, 0).astype(np.uint8)[..., None]
        arr = np.concatenate([arr, alpha], axis=2)
    return arr


# ---------------------------------------------------------------------------
# Test-time augmentation
# ---------------------------------------------------------------------------

def augment(t, aug):
    """Applies AUGS[aug] to a CHW tensor: rotate k×90° (torch.rot90 over H,W), then hflip."""
    k, flip = AUGS[aug]
    t = torch.rot90(t, k, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if flip else t


def unaugment_points(uv, aug):
    """Maps normalised (u=x, v=y) points predicted on augment(img, aug) back onto img."""
    k, flip = AUGS[aug]
    u, v = uv[:, 0].clone(), uv[:, 1].clone()
    if flip:
        u = 1 - u
    for _ in range(k):
        # torch.rot90 sends (u, v) → (v, 1 - u); undo one step.
        u, v = 1 - v, u
    return torch.stack([u, v], 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Detector:
    def __init__(self, checkpoint, batch_size):
        self.model, model_type, self.context, _ = load_model_from_checkpoint(checkpoint, device="cuda")
        if model_type != "detr":
            raise ValueError(f"Unsupported model type {model_type}")
        self.batch_size = batch_size

    @torch.no_grad()
    def _forward(self, tensors):
        pv = torch.stack(tensors)
        if self.context:
            # Same policy the trainer uses for photos: zero context after normalisation.
            pv = torch.cat([pv, torch.zeros(pv.shape[0], 3 * self.context, S, S)], dim=1)
        pv = pv.cuda()
        pm = torch.ones(pv.shape[0], S, S, dtype=torch.long, device="cuda")
        out = self.model(pixel_values=pv, pixel_mask=pm)
        return out.logits.softmax(-1)[:, :, 0].cpu(), out.pred_boxes[:, :, :2].cpu()

    def __call__(self, images, augs=(0,)):
        """images: list of 512x512 RGB uint8 arrays.
        Returns, per image, a list of (aug, probs[Q], centers[Q, 2] in 0..1 of the original image)."""
        jobs = [(i, a) for i in range(len(images)) for a in augs]
        base = {}
        results = [[] for _ in images]
        for start in range(0, len(jobs), self.batch_size):
            chunk = jobs[start:start + self.batch_size]
            tensors = []
            for i, a in chunk:
                if i not in base:
                    base = {i: NORMALIZE(T.functional.to_tensor(images[i]))}
                tensors.append(augment(base[i], a))
            probs, centers = self._forward(tensors)
            for (i, a), p, c in zip(chunk, probs, centers):
                results[i].append((a, p, unaugment_points(c, a)))
        return results


# ---------------------------------------------------------------------------
# Strategy A: replica of /predict_ortho
# ---------------------------------------------------------------------------

def run_endpoint(tif, bounds, zoom, detector, keep_thresh, iou_thresh, workdir, strip_rows=8):
    minlng, minlat, maxlng, maxlat = bounds
    x_min, y_min = _latlon_to_tile(zoom, maxlat, minlng)
    x_max, y_max = _latlon_to_tile(zoom, minlat, maxlng)
    n_tx = x_max - x_min + 1
    res = merc_res(zoom)
    points, tiles_run = [], 0

    for row0 in range(y_min, y_max + 1, strip_rows):
        rows = min(strip_rows, y_max + 1 - row0)
        ox, oy = tile_origin_merc(zoom, x_min, row0)
        strip = warp(tif, ox, oy, n_tx * TILE, rows * TILE, res, workdir)
        tiles, keys = [], []
        for r in range(rows):
            for c in range(n_tx):
                tile = strip[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
                if not tile[..., 3].any():  # outside the ortho — tile server has nothing here
                    continue
                img = Image.fromarray(np.ascontiguousarray(tile[..., :3])).resize((S, S), Image.BILINEAR)
                tiles.append(np.array(img))
                keys.append((x_min + c, row0 + r))
        tiles_run += len(tiles)
        for (tx, ty), [(_, probs, centers)] in zip(keys, detector(tiles)):
            keep = probs > keep_thresh
            if not keep.any():
                continue
            c_k, s_k = centers[keep] * TILE, probs[keep]
            bsize = 10.0  # identical to predict_ortho
            boxes = torch.cat([c_k - bsize, c_k + bsize], 1)
            for idx in nms(boxes, s_k, iou_thresh).tolist():
                lng, lat = _pixel_to_latlon(c_k[idx, 0].item(), c_k[idx, 1].item(), zoom, tx, ty, TILE)
                points.append((lng, lat, s_k[idx].item()))
        print(f"  [endpoint z{zoom}] tile rows {row0 - y_min + rows}/{y_max - y_min + 1}, "
              f"{tiles_run} tiles, {len(points)} detections", flush=True)

    return points, {"tile_count": n_tx * (y_max - y_min + 1), "tiles_run": tiles_run,
                    "ground_mm_per_px": res * math.cos(math.radians((minlat + maxlat) / 2)) * 1000 * TILE / S}


# ---------------------------------------------------------------------------
# Strategy B: stitched, overlapping windows, raw detections for later merging
# ---------------------------------------------------------------------------

def window_origins(dim, stride):
    if dim <= S:
        return [0]
    origins = list(range(0, dim - S + 1, stride))
    if origins[-1] + S < dim:
        origins.append(dim - S)
    return origins


def run_stitched(tif, bounds, gsd_mm, detector, keep_thresh, overlap, augs, workdir, strip_windows=4):
    """Returns raw (lng, lat, score, aug) detections — unmerged, across all windows and augs."""
    minlng, minlat, maxlng, maxlat = bounds
    lat0 = (minlat + maxlat) / 2
    res = gsd_mm / 1000 / math.cos(math.radians(lat0))  # mercator metres per pixel
    ox, oy = lonlat_to_merc(minlng, maxlat)
    ex, ey = lonlat_to_merc(maxlng, minlat)
    W, H = max(S, math.ceil((ex - ox) / res)), max(S, math.ceil((oy - ey) / res))
    xs, ys = window_origins(W, S - overlap), window_origins(H, S - overlap)

    raw, windows_run = [], 0
    for i in range(0, len(ys), strip_windows):
        row_origins = ys[i:i + strip_windows]
        top, bottom = row_origins[0], row_origins[-1] + S
        strip = warp(tif, ox, oy - top * res, W, bottom - top, res, workdir)
        crops, offsets = [], []
        for y0 in row_origins:
            for x0 in xs:
                crop = strip[y0 - top:y0 - top + S, x0:x0 + S]
                if not crop[..., 3].any():
                    continue
                crops.append(np.ascontiguousarray(crop[..., :3]))
                offsets.append((x0, y0))
        windows_run += len(crops)
        for (x0, y0), per_aug in zip(offsets, detector(crops, augs)):
            for aug, probs, centers in per_aug:
                keep = probs > keep_thresh
                for (u, v), s in zip((centers[keep] * S).tolist(), probs[keep].tolist()):
                    lng, lat = merc_to_lonlat(ox + (x0 + u) * res, oy - (y0 + v) * res)
                    raw.append((lng, lat, s, aug))
        print(f"  [stitched {gsd_mm:g}mm] window rows {min(i + strip_windows, len(ys))}/{len(ys)}, "
              f"{windows_run} windows, {len(raw)} raw detections", flush=True)

    return raw, {"raster_px": [W, H], "windows_run": windows_run, "ground_mm_per_px": gsd_mm}


# ---------------------------------------------------------------------------
# Merging, scoring and overlays
# ---------------------------------------------------------------------------

def merge_points(raw, n_views, merge_m, lat0):
    """Greedy distance clustering of raw (lng, lat, score, view) detections.

    Highest-scoring unassigned detection seeds a cluster of everything unassigned
    within merge_m. Cluster score = sum over views of that view's best score in the
    cluster / n_views; position = score-weighted mean. With n_views=1 this is plain
    distance NMS (score = max)."""
    if not raw:
        return []
    pts = np.array(local_metres([(r[0], r[1]) for r in raw], lat0))
    lnglat = np.array([(r[0], r[1]) for r in raw])
    scores = np.array([r[2] for r in raw])
    views = np.array([r[3] for r in raw])
    tree = cKDTree(pts)
    taken = np.zeros(len(raw), bool)
    merged = []
    for i in np.argsort(-scores):
        if taken[i]:
            continue
        members = [j for j in tree.query_ball_point(pts[i], merge_m) if not taken[j]]
        taken[members] = True
        members = np.array(members)
        best = {}
        for j in members:
            best[views[j]] = max(best.get(views[j], 0.0), scores[j])
        w = scores[members]
        lng, lat = (lnglat[members] * w[:, None]).sum(0) / w.sum()
        merged.append((float(lng), float(lat), sum(best.values()) / n_views))
    return merged


def in_bounds(lng, lat, bounds):
    return bounds[0] <= lng <= bounds[2] and bounds[1] <= lat <= bounds[3]


def score(name, preds, labels_ll, bounds, radius_m, thresholds, extra):
    lat0 = (bounds[1] + bounds[3]) / 2
    preds = [p for p in preds if in_bounds(p[0], p[1], bounds)]
    gt = local_metres(labels_ll, lat0)
    all_m = local_metres([(p[0], p[1]) for p in preds], lat0)
    ap = point_metrics(gt, all_m, [p[2] for p in preds], radius_m)["ap"]
    rows = []
    for th in thresholds:
        kept = [m for m, p in zip(all_m, preds) if p[2] >= th]
        tp = len(match_points(gt, kept, radius_m)[0])
        precision = tp / len(kept) if kept else 0.0
        recall = tp / len(gt)
        rows.append({
            "method": name, "score_thresh": th, "gt": len(gt), "pred": len(kept), "tp": tp,
            "fp": len(kept) - tp, "fn": len(gt) - tp, "count_error": len(kept) - len(gt),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if tp else 0.0,
            "ap": round(ap, 4) if ap is not None else None,
            **{k: v for k, v in extra.items() if not isinstance(v, (list, dict))},
        })
    return rows


def densest_centres(labels_ll, n, min_sep_deg):
    pts = np.array(labels_ll)
    density = [(np.linalg.norm(pts - p, axis=1) < min_sep_deg).sum() for p in pts]
    chosen = []
    for i in np.argsort(density)[::-1]:
        if all(np.linalg.norm(pts[i] - pts[j]) > min_sep_deg for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    return [tuple(pts[i]) for i in chosen]


def render_overlays(tif, labels_ll, method_preds, bounds, thresh, radius_m, out_dir, workdir,
                    crop_m=25.0, gsd_mm=8.0, n_crops=3):
    lat0 = (bounds[1] + bounds[3]) / 2
    res = gsd_mm / 1000 / math.cos(math.radians(lat0))
    side = int(crop_m * 1000 / gsd_mm)
    gt_m = local_metres(labels_ll, lat0)
    centres = densest_centres(labels_ll, n_crops, crop_m / 111000)
    for k, (clng, clat) in enumerate(centres):
        cx, cy = lonlat_to_merc(clng, clat)
        ox, oy = cx - side / 2 * res, cy + side / 2 * res
        base = Image.fromarray(np.ascontiguousarray(warp(tif, ox, oy, side, side, res, workdir)[..., :3]))

        def to_px(lng, lat):
            x, y = lonlat_to_merc(lng, lat)
            return (x - ox) / res, (oy - y) / res

        for name, preds in method_preds.items():
            img = base.copy()
            draw = ImageDraw.Draw(img)
            kept = [p for p in preds if p[2] >= thresh]
            pr_m = local_metres([(p[0], p[1]) for p in kept], lat0)
            matches, missed, false_pos = match_points(gt_m, pr_m, radius_m)
            r = 0.25 * 1000 / gsd_mm
            for gi, pi in matches:
                x, y = to_px(*kept[pi][:2])
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=3)
            for gi in missed:
                x, y = to_px(*labels_ll[gi])
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 215, 0), width=3)
            for pi in false_pos:
                x, y = to_px(*kept[pi][:2])
                draw.rectangle([x - r, y - r, x + r, y + r], outline=(255, 40, 40), width=3)
            draw.text((10, 10), f"{name} @ {thresh}  green=TP yellow=missed red=FP", fill=(255, 255, 255))
            img.save(os.path.join(out_dir, f"overlay_{k}_{name}.jpg"), quality=90)


# ---------------------------------------------------------------------------

def build_variants(detections, tta_subsets, ensembles, merge_m, lat0):
    """Turns saved detections into scored-ready variants: endpoint as-is, stitched TTA
    subsets merged, and multi-GSD ensembles merged across (gsd, aug) views."""
    variants = {}
    for name, d in detections.items():
        if d["kind"] == "endpoint":
            variants[name] = d["points"]
            continue
        have = set(d["augs"])
        for subset, augs in TTA_SUBSETS.items():
            if subset in tta_subsets and set(augs) <= have:
                raw = [r for r in d["points"] if r[3] in augs]
                variants[f"{name}_{subset}"] = merge_points(raw, len(augs), merge_m, lat0)
    for ens in ensembles:
        members = [f"stitched_{float(g):g}mm" for g in ens.split("+")]
        if not all(m in detections for m in members):
            print(f"Skipping ensemble {ens}: missing one of {members}")
            continue
        raw, n_views = [], 0
        for m in members:
            augs = detections[m]["augs"]
            index = {a: n_views + i for i, a in enumerate(augs)}
            raw += [(r[0], r[1], r[2], index[r[3]]) for r in detections[m]["points"]]
            n_views += len(augs)
        variants[f"ens_{ens}"] = merge_points(raw, n_views, merge_m, lat0)
    return variants


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", required=True)
    ap.add_argument("--labels", required=True, help="Annotation export JSON (point geometries, label=fish)")
    ap.add_argument("--checkpoint", default="models/best_without_context/best.pth")
    ap.add_argument("--zooms", default="21,22", help="Zooms for the /predict_ortho replica ('' to skip)")
    ap.add_argument("--gsd_mm", default="7.4,9.6,12", help="GSDs for the stitched strategy ('' to skip)")
    ap.add_argument("--tta", default="none", choices=list(TTA_SUBSETS),
                    help="Augmentations to run for the stitched strategy; smaller subsets are scored too")
    ap.add_argument("--ensembles", default="", help="Comma-separated GSD ensembles, e.g. '7.4+9.6,6+9.6'")
    ap.add_argument("--overlap", type=int, default=128, help="Window overlap in px for the stitched strategy")
    ap.add_argument("--merge_m", type=float, default=0.5, help="Distance (ground m) within which detections merge")
    ap.add_argument("--iou", type=float, default=0.5, help="Box NMS IoU for the endpoint replica")
    ap.add_argument("--radius_m", type=float, default=0.5, help="Match radius in ground metres")
    ap.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6")
    ap.add_argument("--margin_m", type=float, default=5.0, help="Margin around the label bbox to evaluate")
    ap.add_argument("--batch_size", type=int, default=8, help="Images per forward pass (augmentations count)")
    ap.add_argument("--overlay_thresh", type=float, default=0.25)
    ap.add_argument("--overlay_methods", default="",
                    help="Comma-separated variants to draw overlays for (default: endpoints + best AP)")
    ap.add_argument("--score_only", action="store_true", help="Re-score saved detections.json, no inference")
    ap.add_argument("--out", default="debug/ortho_eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    keep_thresh = min(min(thresholds), 0.05)
    det_path = os.path.join(args.out, "detections.json")

    with open(args.labels) as f:
        annotations = json.load(f)
    labels_ll = [json.loads(a["geometry"])["coordinates"][:2] for a in annotations
                 if a.get("shape") == "point" and a.get("label") == "fish" and a.get("deleted_at") is None]
    lngs, lats = zip(*labels_ll)
    lat0 = sum(lats) / len(lats)
    dlat = args.margin_m / 111320
    dlng = dlat / math.cos(math.radians(lat0))
    bounds = [min(lngs) - dlng, min(lats) - dlat, max(lngs) + dlng, max(lats) + dlat]
    print(f"{len(labels_ll)} fish labels; eval bounds {bounds}", flush=True)

    if args.score_only:
        with open(det_path) as f:
            detections = json.load(f)["detections"]
    else:
        detector = Detector(args.checkpoint, args.batch_size)
        detections = {}

        def save():
            # After every method, so an interrupted run keeps what finished (--score_only).
            with open(det_path, "w") as f:
                json.dump({"bounds": bounds, "args": vars(args), "detections": detections}, f)

        with tempfile.TemporaryDirectory() as workdir:
            for z in [int(z) for z in args.zooms.split(",") if z]:
                t0 = time.time()
                points, extra = run_endpoint(args.tif, bounds, z, detector, keep_thresh, args.iou, workdir)
                extra["seconds"] = round(time.time() - t0, 1)
                detections[f"endpoint_z{z}"] = {"kind": "endpoint", "points": points, "extra": extra}
                save()
            for g in [float(g) for g in args.gsd_mm.split(",") if g]:
                augs = TTA_SUBSETS[args.tta]
                t0 = time.time()
                points, extra = run_stitched(args.tif, bounds, g, detector, keep_thresh, args.overlap, augs, workdir)
                extra["seconds"] = round(time.time() - t0, 1)
                detections[f"stitched_{g:g}mm"] = {"kind": "stitched", "augs": augs, "points": points, "extra": extra}
                save()

    subsets = list(TTA_SUBSETS)
    ensembles = [e for e in args.ensembles.split(",") if e]
    variants = build_variants(detections, subsets, ensembles, args.merge_m, lat0)
    all_rows = []
    for name, preds in variants.items():
        base = name if name in detections else name.rsplit("_", 1)[0]
        extra = detections.get(base, {}).get("extra", {})
        all_rows += score(name, preds, labels_ll, bounds, args.radius_m, thresholds, extra)

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in all_rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    best = {}
    for r in all_rows:
        if r["method"] not in best or r["f1"] > best[r["method"]]["f1"]:
            best[r["method"]] = r
    print(f"\nmatch radius {args.radius_m} m, merge {args.merge_m} m — best F1 per method")
    print(f"{'method':<26}{'AP':>7}{'F1':>7}{'thr':>6}{'pred':>6}{'P':>7}{'R':>7}{'sec':>8}")
    for name, r in sorted(best.items(), key=lambda kv: -(kv[1]["ap"] or 0)):
        print(f"{name:<26}{(r['ap'] or 0):>7.3f}{r['f1']:>7.3f}{r['score_thresh']:>6}{r['pred']:>6}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r.get('seconds', ''):>8}")

    top = max((n for n in variants if not n.startswith("endpoint")), key=lambda n: best[n]["ap"] or 0, default=None)
    if args.overlay_methods:
        overlay = {n: variants[n] for n in args.overlay_methods.split(",") if n in variants}
    else:
        overlay = {n: variants[n] for n in variants if n.startswith("endpoint")}
        if top:
            overlay[top] = variants[top]
    with tempfile.TemporaryDirectory() as workdir:
        render_overlays(args.tif, labels_ll, overlay, bounds, args.overlay_thresh, args.radius_m, args.out, workdir)
    print(f"\nWrote {args.out}/summary.csv, detections.json, overlay_*.jpg (best: {top})")


if __name__ == "__main__":
    main()
