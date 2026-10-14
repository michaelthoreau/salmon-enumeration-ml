import os
import io
import time
import math
import random
import hashlib
import urllib.parse
import requests
import numpy as np
import scipy.spatial
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from functools import lru_cache

ORTHO_TILE_SIZE = 256

def is_photo_task(task):
    filename = urllib.parse.urlparse(task.get('fileKey', '')).path
    return os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}

class SameFrameBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(self.dataset_size))
        random.shuffle(indices)
        for idx in indices:
            yield [idx] * self.batch_size

    def __len__(self):
        return self.dataset_size

# Helpers

def calculate_knn_distance(gt_points, num_knn=4):
    num_point = gt_points.shape[0]
    if num_point >= 4:
        tree = scipy.spatial.cKDTree(gt_points.numpy(), leafsize=2048)
        distances, locations = tree.query(gt_points.numpy(), k=min(num_knn, num_point))
        distances = np.delete(distances, 0, axis=1)
        distances = np.mean(distances, axis=1)
        distances = torch.from_numpy(distances).unsqueeze(1)
    elif num_point == 0:
        distances = torch.zeros((0, 1), dtype=torch.float32)
    elif num_point == 1:
        distances = torch.zeros((1, 1), dtype=torch.float32)
    elif num_point == 2:
        tree = scipy.spatial.cKDTree(gt_points.numpy(), leafsize=2048)
        distances, locations = tree.query(gt_points.numpy(), k=num_point)
        distances = np.delete(distances, 0, axis=1)
        distances = torch.from_numpy(distances[:, 0] / 1.0).unsqueeze(1)
    elif num_point == 3:
        tree = scipy.spatial.cKDTree(gt_points.numpy(), leafsize=2048)
        distances, locations = tree.query(gt_points.numpy(), k=num_point)
        distances = np.delete(distances, 0, axis=1)
        distances = torch.from_numpy((distances[:, 0] + distances[:, 1]) / 2.0).unsqueeze(1)
    return distances


def gen_discrete_map(im_height, im_width, points):
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = points.detach().cpu().numpy().round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(
        0, index=p_index, src=torch.ones(im_width*im_height)
    ).view(im_height, im_width).numpy()
    assert np.sum(discrete_map) == num_gt
    return discrete_map


def generate_heatmap(points, shape=(512, 512), sigma=4.0):
    """Draws 2D Gaussian blobs centered at each target keypoint on a grid."""
    heatmap = np.zeros(shape, dtype=np.float32)
    h, w = shape
    r = int(math.ceil(3 * sigma))
    
    for x, y in points:
        cx, cy = int(round(x)), int(round(y))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
            
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        
        for i in range(y0, y1):
            for j in range(x0, x1):
                d2 = (j - x)**2 + (i - y)**2
                val = math.exp(-d2 / (2.0 * sigma**2))
                heatmap[i, j] = max(heatmap[i, j], val)
                
    return torch.tensor(heatmap)


class SockeyeApiDataset(Dataset):
    def __init__(self, api_url, api_key, tasks, cache_dir="/tmp/frame_cache",
                 img_size=512, augment=False, num_context_frames=1, context_stride=5,
                 box_size=50, sigma=4.0, verbose=False, mode="iocformer",
                 tile_server_url=None, tile_server_token=None, ortho_stitch_radius=1,
                 deterministic_crop=False):
        """
        Args:
            api_url (str): Base URL of API
            api_key (str): Sockeye API Key
            tasks (list): List of task dictionaries
            cache_dir (str): Disk path to cache downloaded frame JPEGs
            img_size (int): Image crop height/width to resize to
            augment (bool): If True, apply horizontal flip and color jitter
            num_context_frames (int): Number of context frames (N) to load (0 to disable context)
            context_stride (int): Step size / separation (P) between context frames
            box_size (int): Fixed box size in pixels (used for DETR mode)
            sigma (float): Sigma for Gaussian heatmap generation (used for CNN mode)
            verbose (bool): If True, print detailed timing logs
            mode (str): One of "iocformer", "detr", "cnn"
            tile_server_url (str): Base URL of the ortho tile server (required if any
                task has an orthoKey, i.e. comes from /api/v1/ortho-tasks)
            tile_server_token (str): Bearer token for the tile server
            ortho_stitch_radius (int): Neighboring-tile radius to stitch around an ortho
                task's center tile (1 -> 3x3 grid). A single 256px tile is smaller than
                img_size, which would always take the plain-resize path in __getitem__
                and skip random-crop augmentation; stitching gives that branch room.
        """
        self.api_url = api_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.tasks = tasks
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.augment = augment
        self.deterministic_crop = deterministic_crop
        self.num_context_frames = num_context_frames
        self.companion_mode = "prev" if num_context_frames > 0 else None
        self.context_stride = context_stride
        self.box_size = box_size
        self.half_box = box_size // 2
        self.sigma = sigma
        self.verbose = verbose
        self.mode = mode

        self.tile_server_url = tile_server_url.rstrip('/') if tile_server_url else None
        self.tile_server_token = tile_server_token
        self.ortho_stitch_radius = ortho_stitch_radius
        self._ortho_lookup = {}
        for t in tasks:
            fid = t.get("fileId", "")
            if fid.startswith("ortho:") and t.get("orthoKey") is not None:
                self._ortho_lookup[fid] = {
                    "z": t["tileZ"], "x": t["tileX"], "y": t["tileY"],
                    "orthoKey": t["orthoKey"], "bucket": t.get("bucket", ""),
                }

        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.tasks)

    def _get_ortho_tile_raw(self, ortho_key, bucket, z, tx, ty):
        """Downloads and caches a single ORTHO_TILE_SIZE ortho tile as a PIL Image,
        fetched from the same tile server /predict_ortho uses at inference time."""
        key_hash = hashlib.md5(ortho_key.encode()).hexdigest()[:12]
        cache_path = os.path.join(self.cache_dir, f"ortho_{key_hash}_{z}_{tx}_{ty}.jpg")

        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

        url = (
            f"{self.tile_server_url}/tiles/{z}/{tx}/{ty}"
            f"?key={urllib.parse.quote(ortho_key, safe='')}"
            f"&bucket={urllib.parse.quote(bucket, safe='')}"
            f"&tilesize={ORTHO_TILE_SIZE}"
        )
        headers = {"Authorization": f"Bearer {self.tile_server_token}"}

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
            # Neighboring tile may legitimately fall outside the ortho's coverage
            # (edge of flight) — return blank rather than failing the whole stitch.
            return Image.new("RGB", (ORTHO_TILE_SIZE, ORTHO_TILE_SIZE), (0, 0, 0))

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        tmp_path = cache_path + f".tmp.{os.getpid()}.{random.randint(0, 100000)}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return img

    def _get_ortho_tiles_stitched(self, file_id):
        """Fetches a (2*radius+1)^2 neighborhood of ortho tiles around the task's
        center tile and stitches them into one square image. The task's own
        points stay relative to just the center tile — callers (__getitem__,
        validate()) are responsible for shifting them by +radius*ORTHO_TILE_SIZE
        into this canvas's coordinate space."""
        info = self._ortho_lookup[file_id]
        z, cx, cy = info["z"], info["x"], info["y"]
        ortho_key, bucket = info["orthoKey"], info["bucket"]

        r = self.ortho_stitch_radius
        n = 2 * r + 1
        canvas = Image.new("RGB", (n * ORTHO_TILE_SIZE, n * ORTHO_TILE_SIZE))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                tile_img = self._get_ortho_tile_raw(ortho_key, bucket, z, cx + dx, cy + dy)
                canvas.paste(tile_img, ((dx + r) * ORTHO_TILE_SIZE, (dy + r) * ORTHO_TILE_SIZE))
        return canvas

    @lru_cache(maxsize=64)
    def _get_full_frame(self, file_id, frame_num):
        if file_id.startswith("ortho:"):
            return self._get_ortho_tiles_stitched(file_id)

        t0 = time.time()
        # Always use the full frame cache path
        cache_path = os.path.join(self.cache_dir, f"{file_id}_{frame_num:06d}.jpg")
        
        # Check disk cache for full frame
        if os.path.exists(cache_path):
            try:
                img = Image.open(cache_path).convert("RGB")
                if self.verbose:
                    print(f"[WORKER] Cache hit: {file_id}_{frame_num:06d} | Read took {time.time() - t0:.4f}s", flush=True)
                return img
            except Exception:
                # Cache corrupted; delete and re-download
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            
        # Download from API (always download full frame)
        url = f"{self.api_url}/files/{file_id}/frame/{frame_num}?quality=95"
        img_bytes = None
        t_download_start = time.time()
        for attempt in range(3):
            try:
                res = requests.get(url, headers=self.headers, stream=True, timeout=15)
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
            if self.verbose:
                print(f"[WORKER] Cache miss & DOWNLOAD FAILED: {file_id}_{frame_num:06d} | Total took {time.time() - t0:.4f}s", flush=True)
            raise RuntimeError(f"Error downloading frame {frame_num} for file {file_id} after 3 attempts")
            
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Save the full frame (original) to cache
        tmp_path = cache_path + f".tmp.{os.getpid()}.{random.randint(0, 100000)}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            
        if self.verbose:
            print(f"[WORKER] Cache miss: Downloaded + Saved {file_id}_{frame_num:06d} | Fetch + Save took {time.time() - t0:.4f}s (download part took {time.time() - t_download_start:.4f}s)", flush=True)
            
        return img

    def _get_frame(self, file_id, frame_num, bbox=None):
        """Downloads and caches a single video frame as a PIL Image.
        Always downloads and saves the full frame. Crops in-memory if bbox is provided.
        """
        img = self._get_full_frame(file_id, frame_num)
        
        # Crop after loading/downloading
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
                
        return img

    def __getitem__(self, idx):
        t_start = time.time()
        task = self.tasks[idx]
        photo = is_photo_task(task)
        file_id = task["fileId"]
        frame_t = task["frame"]
        
        if self.verbose:
            print(f"[DataLoader Progress] [Worker {os.getpid()}] Loading item {idx + 1}/{len(self.tasks)} (File: {file_id}, Frame: {frame_t})...", flush=True)
        
        x1, y1, x2, y2 = task["x1"], task["y1"], task["x2"], task["y2"]
        bbox = (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
        
        t0 = time.time()
        img_t_crop = self._get_frame(file_id, frame_t, bbox=bbox)
        W_crop, H_crop = img_t_crop.size
        t_target = time.time() - t0
        
        if bbox is None:
            x1, y1 = 0, 0
            x2, y2 = W_crop, H_crop
            W_crop = x2 - x1
            H_crop = y2 - y1
            
        points = [[pt["x"], pt["y"]] for pt in task.get("points", [])]
        if file_id.startswith("ortho:"):
            # Exported points are relative to just the task's center tile, but
            # _get_full_frame returns the whole stitched (2*radius+1)-tile canvas
            # for ortho tasks — shift into that canvas's coordinate space.
            margin = self.ortho_stitch_radius * ORTHO_TILE_SIZE
            points = [[px + margin, py + margin] for px, py in points]
        S = self.img_size
        
        t0 = time.time()
        # Extract/augment crops
        if (self.augment or self.deterministic_crop) and W_crop >= S and H_crop >= S:
            points_crop = [[px - x1, py - y1] for px, py in points]
            
            # Try up to 15 times to crop a region with at least one point, if points exist
            x0, y0 = 0, 0
            if self.deterministic_crop and not self.augment:
                x0, y0 = (W_crop - S) // 2, (H_crop - S) // 2
            elif len(points_crop) > 0:
                for _ in range(15):
                    x0 = random.randint(0, W_crop - S)
                    y0 = random.randint(0, H_crop - S)
                    inside_pts = [
                        (px, py) for px, py in points_crop
                        if x0 <= px < x0 + S and y0 <= py < y0 + S
                    ]
                    if len(inside_pts) > 0:
                        break
            else:
                x0 = random.randint(0, W_crop - S)
                y0 = random.randint(0, H_crop - S)
                
            img_crop = img_t_crop.crop((x0, y0, x0 + S, y0 + S))
            
            points = [[px - x0, py - y0] for px, py in points_crop
                      if x0 <= px < x0 + S and y0 <= py < y0 + S]
            
            context_crops = []
            if self.companion_mode:
                for n_idx in range(1, self.num_context_frames + 1):
                    if photo:
                        context_crops.append(None)
                        continue
                    frame_t_minus_np = max(0, frame_t - n_idx * self.context_stride)
                    crop_n = None
                    try:
                        img_context_crop = self._get_frame(file_id, frame_t_minus_np, bbox=bbox)
                        crop_n = img_context_crop.crop((x0, y0, x0 + S, y0 + S))
                    except Exception as error:
                        print(f'Warning: context unavailable for {file_id} frame {frame_t_minus_np}: {error}', flush=True)
                    context_crops.append(crop_n)
            W_crop, H_crop = S, S
        else:
            img_crop = img_t_crop.resize((S, S), Image.BILINEAR)
            points_crop = [[px - x1, py - y1] for px, py in points]
            sx, sy = S / W_crop, S / H_crop
            points = [[px * sx, py * sy] for px, py in points_crop]
            
            context_crops = []
            if self.companion_mode:
                for n_idx in range(1, self.num_context_frames + 1):
                    if photo:
                        context_crops.append(None)
                        continue
                    frame_t_minus_np = max(0, frame_t - n_idx * self.context_stride)
                    crop_n = None
                    try:
                        img_context_crop = self._get_frame(file_id, frame_t_minus_np, bbox=bbox)
                        crop_n = img_context_crop.resize((S, S), Image.BILINEAR)
                    except Exception as error:
                        print(f'Warning: context unavailable for {file_id} frame {frame_t_minus_np}: {error}', flush=True)
                    context_crops.append(crop_n)
            W_crop, H_crop = S, S
            
        t_context = time.time() - t0
        
        t0 = time.time()
        # Flip augmentation
        do_flip = self.augment and torch.rand(1).item() > 0.5
        if do_flip:
            img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
            points = [[W_crop - 1 - px, py] for px, py in points]
            context_crops = [
                c.transpose(Image.FLIP_LEFT_RIGHT) if c is not None else None
                for c in context_crops
            ]
            
        # Color jitter augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            f = 0.7 + torch.rand(1).item() * 0.6
            img_crop = T.functional.adjust_brightness(img_crop, f)
            img_crop = T.functional.adjust_contrast(img_crop,  0.7 + torch.rand(1).item() * 0.6)
            img_crop = T.functional.adjust_saturation(img_crop, 0.7 + torch.rand(1).item() * 0.6)
            
        # Return format based on mode
        if self.mode == "iocformer":
            points_t = torch.tensor(points, dtype=torch.float32)
            num_points = points_t.shape[0]
            if num_points > 0:
                gt_points_yx = torch.stack([points_t[:, 1], points_t[:, 0]], dim=1)
                distances = calculate_knn_distance(gt_points_yx, num_knn=4)
                points_yx_dist = torch.cat([gt_points_yx, distances], dim=1).float()
            else:
                points_yx_dist = torch.zeros((0, 3), dtype=torch.float32)
                
            target = {}
            target['labels'] = torch.ones(num_points, dtype=torch.long)
            target['points_macher'] = (points_yx_dist / float(S)).float()
            if num_points > 0:
                pts_yx_norm = points_yx_dist[:, 0:2] / float(S)
                dist_norm = points_yx_dist[:, 2:3] / float(S)
                target['points'] = torch.cat([pts_yx_norm[:, 1:2], pts_yx_norm[:, 0:1], dist_norm], dim=1).float()
            else:
                target['points'] = torch.zeros((0, 3), dtype=torch.float32)
                
            discrete_map = gen_discrete_map(S, S, points_t)
            target['gt_map'] = torch.from_numpy(discrete_map).unsqueeze(0).float()
            target['points_true'] = points_t.float()
            
            t_total = time.time() - t_start
            if self.verbose:
                print(f"[DataLoader Progress] [Worker {os.getpid()}] Loaded item {idx + 1}/{len(self.tasks)} (File: {file_id}, Frame: {frame_t}) | Took {t_total:.4f}s", flush=True)
            return (img_crop, context_crops), target
            
        elif self.mode == "detr":
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
                
            t_processing = time.time() - t0
            t_total = time.time() - t_start
            if self.verbose:
                print(f"[DataLoader Progress] [Worker {os.getpid()}] Loaded item {idx + 1}/{len(self.tasks)} (File: {file_id}, Frame: {frame_t}) | Took {t_total:.4f}s", flush=True)
                print(f"[DATASET] Getitem (idx={idx:04d}): target_load={t_target:.4f}s | context_load={t_context:.4f}s | process={t_processing:.4f}s | total={t_total:.4f}s", flush=True)
                
            target = {'class_labels': labels_t, 'boxes': boxes_t}
            if photo:
                target['_zero_context'] = True
            return (img_crop, context_crops), target
            
        elif self.mode == "cnn":
            # Target heatmap
            target_heatmap = generate_heatmap(points, shape=(self.img_size, self.img_size), sigma=self.sigma)
            
            # Convert to Tensors
            target_t = torch.from_numpy(np.array(img_crop)).permute(2, 0, 1).float() / 255.0
            context_ts = []
            for img in context_crops:
                if img is not None:
                    ctx_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                else:
                    ctx_t = torch.zeros_like(target_t)
                context_ts.append(ctx_t)
                
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            target_t = (target_t - mean) / std
            context_ts = [(t - mean) / std for t in context_ts]
            
            # Concat target and context tensors into a single input tensor: [Channel, H, W]
            all_tensors = [target_t] + context_ts
            input_tensor = torch.cat(all_tensors, dim=0) # shape [3 * (1 + N), H, W]
            t_processing = time.time() - t0
            
            t_total = time.time() - t_start
            if self.verbose:
                print(f"[DataLoader Progress] [Worker {os.getpid()}] Loaded item {idx + 1}/{len(self.tasks)} (File: {file_id}, Frame: {frame_t}) | Took {t_total:.4f}s", flush=True)
                print(f"[DATASET] Getitem (idx={idx:04d}): target_load={t_target:.4f}s | context_load={t_context:.4f}s | process={t_processing:.4f}s | total={t_total:.4f}s", flush=True)
                
            return input_tensor, target_heatmap
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
