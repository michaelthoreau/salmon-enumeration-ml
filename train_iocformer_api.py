import os
import io
import time
import math
import random
import requests
import fire
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import torchvision.transforms as T
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path
import mlflow

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

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _auto_run_name(num_context_frames, context_stride, temporal_fusion):
    parts = ['iocformer', f'cf{num_context_frames}cs{context_stride}']
    if num_context_frames > 0:
        parts.append(temporal_fusion)
    return '_'.join(parts)

from dataset_api import SockeyeApiDataset, SameFrameBatchSampler


def collate_fn(batch):
    imgs_and_contexts, targets = zip(*batch)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
        
    return torch.stack(tensors), list(targets)


def main(
    api_url:      str   = None,
    api_key:      str   = None,
    save_path:    str   = None,
    epochs:       int   = 20,
    batch_size:   int   = 8,
    lr:           float = 1e-4,
    lr_backbone:  float = 1e-4,
    lr_step:      int   = 240,
    warmup_steps: int   = 100,
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
    temporal_fusion: str = "early",
    verbose:      bool  = False,
    mlflow_tracking_uri: str = None,
    mlflow_experiment: str = 'salmon-enumeration',
):
    """Train IOCFormer on salmon crop tasks pulled from Sockeye API.

    Args:
        api_url:            Base endpoint URL of Sockeye API.
        api_key:            Sockeye Authorization API Token.
        save_path:          Checkpoint output directory (auto-selected if omitted).
        epochs:             Number of training epochs.
        batch_size:         Batch size for loader.
        lr:                 Decoder learning rate.
        lr_backbone:        Backbone learning rate.
        lr_step:            Epoch at which to decay LR by 0.1.
        warmup_steps:       Warmup steps.
        decay_steps:        Cosine decay steps.
        val_every:          Epoch interval to run validation.
        resume:             Path/directory to resume model weights.
        workers:            DataLoader workers count.
        cache_dir:          Directory where downloaded video frames are cached.
        img_size:           Input dimensions.
        box_size:           Square box size (pixels) centered on points.
        num_queries:        DETR decoder queries count.
        score_thresh:       Score threshold to count predictions at validation.
        limit_batches:      Stop epoch early (for quick debugging).
        min_batches:        Oversample so each epoch has at least this many batches.
        num_context_frames: Number of context frames (N) to seek backward.
        context_stride:     Stride between context frames.
        temporal_fusion:    Temporal fusion mode (IOCFormer only supports early).
    """
    from Networks.CDETR import build_model

    if api_url is None:
        api_url = os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1")
    if api_key is None:
        api_key = os.environ.get("SOCKEYE_API_KEY")
        
    if not api_key:
        raise ValueError(
            "Sockeye API Key is required. Please set it in .env as SOCKEYE_API_KEY or pass it via --api_key."
        )

    base = os.path.dirname(os.path.abspath(__file__))
    auto_name = _auto_run_name(num_context_frames, context_stride, temporal_fusion)

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

    if save_path:
        run_name_candidate = os.path.basename(save_path.rstrip('/'))
        if not resume:
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
            existing_run = find_mlflow_run_by_name(auto_name)
            if existing_run:
                resume = auto_name
                resume_run_id = existing_run.info.run_id
                save_path = os.path.join(base, 'runs', auto_name)
                print(f"Auto-resuming from default MLflow run: {auto_name} (ID: {resume_run_id})", flush=True)
            else:
                save_path = os.path.join(base, 'runs', auto_name)
                print(f"Starting new default MLflow run: {auto_name}", flush=True)

    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    writer.add_text("config/num_context_frames", str(num_context_frames))
    writer.add_text("config/context_stride", str(context_stride))

    if mlflow_enabled:
        try:
            if resume_run_id:
                mlflow.start_run(run_id=resume_run_id)
                print(f"Resumed active MLflow run: {auto_name} (ID: {resume_run_id})", flush=True)
            else:
                mlflow.start_run(run_name=auto_name)
                print(f"Started MLflow run: {auto_name}", flush=True)
                mlflow.log_params({
                    'model': 'iocformer', 'api_url': api_url, 'save_path': save_path,
                    'batch_size': batch_size, 'lr': lr, 'warmup_steps': warmup_steps,
                    'decay_steps': decay_steps, 'val_every': val_every, 'workers': workers,
                    'img_size': img_size, 'score_thresh': score_thresh,
                    'limit_batches': limit_batches, 'min_batches': min_batches,
                    'num_context_frames': num_context_frames, 'context_stride': context_stride,
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
        
    tasks = [t for t in tasks if len(t.get("points", [])) > 0]
    print(f"Loaded {len(tasks)} completed labeling tasks with fish from Sockeye API.", flush=True)
    if not tasks:
        print("Error: No tasks with fish found in Sockeye API.")
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
        cache_dir=cache_dir, img_size=img_size,
        augment=True, num_context_frames=num_context_frames, context_stride=context_stride,
        mode="iocformer"
    )
    val_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=val_tasks,
        cache_dir=cache_dir, img_size=img_size,
        augment=False, num_context_frames=num_context_frames, context_stride=context_stride,
        mode="iocformer"
    )

    n_samples = len(train_ds)
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
    MODEL_CFG = SimpleNamespace(
        backbone='resnet50',
        dilation=True,
        lr_backbone=lr_backbone,
        position_embedding='sine',
        masks=False,
        frozen_weights=None,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=num_queries,
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

    model, criterion, postprocessors = build_model(MODEL_CFG)

    if num_context_frames > 0:
        if temporal_fusion != "early":
            print(f"Warning: IOCFormer only supports 'early' temporal fusion. Forcing 'early' fusion.", flush=True)
        from Networks.CDETR.backbone import expand_first_conv
        expand_first_conv(model.backbone, in_channels=3 * (1 + num_context_frames))

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
            print(f"Resuming weights from downloaded MLflow checkpoint: {resume}...", flush=True)
            ckpt = torch.load(resume, map_location="cuda")
            model_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
            model.load_state_dict(ckpt[model_key], strict=False)
            
            opt_key = "optimizer_state_dict" if "optimizer_state_dict" in ckpt else "optimizer"
            if opt_key in ckpt:
                try:
                    optimizer.load_state_dict(ckpt[opt_key])
                except Exception:
                    print("Could not load optimizer state dict. Skipping.", flush=True)
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if "best_mae" in ckpt:
                best_mae = ckpt["best_mae"]
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass
            print(f"Loaded checkpoint at epoch {start_epoch} (best MAE: {best_mae:.2f}, step {global_step})", flush=True)
            checkpoint_loaded = True
        else:
            print(f"WARNING: checkpoint not found: {resume}", flush=True)

    # 7. Helper functions for training loop
    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        criterion.train()
        
        running = {}
        t0 = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:4d}/{epochs}', ncols=110, leave=True)
        t_data_sum = 0.0
        t_train_sum = 0.0
        t_start = time.time()
        for i, (imgs, targets) in enumerate(pbar):
            t_data = time.time() - t_start
            t_data_sum += t_data
            
            if limit_batches > 0 and i >= limit_batches:
                break
                
            t_train_start = time.time()
            imgs = imgs.cuda()
            targets = [{k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            
            out_all = model(imgs)
            
            if isinstance(out_all, list):
                d6, out_dm = out_all
                gd_count = torch.tensor([t['points'].shape[0] for t in targets], dtype=torch.float32, device=imgs.device)
                pred_dm_sum = out_dm[1].sum(dim=(1, 2, 3))
                count_l = F.l1_loss(pred_dm_sum, gd_count)
                loss_dm = count_l + 0.0 * out_dm[0].sum()
            else:
                d6 = out_all
                loss_dm = 0.0
                
            loss_dict = criterion(d6, targets)
            loss_detr = sum(loss_dict[k] * criterion.weight_dict[k]
                            for k in loss_dict if k in criterion.weight_dict)
            
            loss = loss_detr
            if isinstance(out_all, list):
                loss = loss + loss_dm * 0.5
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            t_train = time.time() - t_train_start
            t_train_sum += t_train
            
            global_step += 1
            
            running['loss'] = running.get('loss', 0.0) + loss.item()
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()
            if isinstance(out_all, list):
                running['count_l'] = running.get('count_l', 0.0) + count_l.item()
                
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            steps_done = i + 1
            pbar.set_postfix(
                loss=f'{running["loss"] / steps_done:.4f}',
                ce=f'{running.get("loss_ce", 0.0) / steps_done:.3f}',
                pt=f'{running.get("loss_point", 0.0) / steps_done:.3f}',
                gpu=f'{gpu_mem:.1f}GB'
            )
            
            log_metric('train/loss',         loss.item(),         global_step)
            log_metric('train/loss_ce',      loss_dict.get('loss_ce',   torch.tensor(0)).item(), global_step)
            log_metric('train/loss_point',   loss_dict.get('loss_point', torch.tensor(0)).item(), global_step)
            if isinstance(out_all, list):
                log_metric('train/count_l',  count_l.item(),      global_step)
            log_metric('train/lr',           optimizer.param_groups[0]['lr'], global_step)
            log_metric('train/lr_backbone',  optimizer.param_groups[1]['lr'], global_step)
            
            t_start = time.time()
            
        elapsed = time.time() - t0
        n = max(len(train_loader), 1)
        peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f'Epoch {epoch:4d}  loss={running.get("loss",0)/n:.4f}  ce={running.get("loss_ce",0)/n:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}  gpu_peak={peak_gpu:.1f}GB  time={elapsed:.0f}s (data: {t_data_sum:.1f}s, train: {t_train_sum:.1f}s)', flush=True)
        torch.cuda.reset_peak_memory_stats()

    # 8. Tile builder helper for validation
    def _make_tiles(pil_img, S, companion_crops=None):
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

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        criterion.eval()
        
        mae = mse = 0.0
        n = 0
        debug_saved = False

        for task in tqdm(val_tasks, desc='  Val', ncols=80, leave=False):
            file_id = task["fileId"]
            frame_t = task["frame"]
            x1, y1, x2, y2 = task["x1"], task["y1"], task["x2"], task["y2"]
            
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
            
            points = [[pt["x"], pt["y"]] for pt in task.get("points", [])]
            gt_count = len(points)
            
            tiles = _make_tiles(crop_pil, img_size, companion_crops)
            all_boxes_xyxy = []
            all_scores = []

            for tile_t, tx, ty, orig_W, orig_H in tiles:
                pv = tile_t.unsqueeze(0).cuda()
                
                out_all = model(pv)
                if isinstance(out_all, list):
                    d6 = out_all[0]
                else:
                    d6 = out_all
                    
                probs = d6['pred_logits'].sigmoid()[0, :, 1].cpu()
                pts = d6['pred_points'][0].cpu()
                
                keep = probs > score_thresh
                if keep.any():
                    p_kept = pts[keep]
                    s_kept = probs[keep]
                    
                    cx = tx + p_kept[:, 0] * img_size
                    cy = ty + p_kept[:, 1] * img_size
                    w = torch.full_like(cx, box_size)
                    h = torch.full_like(cy, box_size)
                    
                    all_boxes_xyxy.append(torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1))
                    all_scores.append(s_kept)
                    
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
            
            if not debug_saved:
                debug_saved = True
                
                target_height = 1080
                W_c, H_c = crop_pil.size
                scale = target_height / H_c
                target_width = int(W_c * scale)
                
                draw_crop = crop_pil.resize((target_width, target_height), Image.BILINEAR)
                draw = ImageDraw.Draw(draw_crop)
                
                for px, py in points:
                    px_rel = (px - x1) * scale
                    py_rel = (py - y1) * scale
                    scaled_box_size = box_size * scale
                    bx1 = px_rel - scaled_box_size / 2
                    by1 = py_rel - scaled_box_size / 2
                    bx2 = px_rel + scaled_box_size / 2
                    by2 = py_rel + scaled_box_size / 2
                    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
                
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
        for pixel_values, targets in val_loader:
            pv = pixel_values.cuda()
            targets = [{k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            
            out_all = model(pv)
            if isinstance(out_all, list):
                d6 = out_all[0]
            else:
                d6 = out_all
                
            loss_dict = criterion(d6, targets)
            loss_detr = sum(loss_dict[k] * criterion.weight_dict[k]
                            for k in loss_dict if k in criterion.weight_dict)
            
            val_loss_dict['loss'] = val_loss_dict.get('loss', 0.0) + loss_detr.item()
            for k, v in loss_dict.items():
                val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v.item()
                
        num_batches = max(len(val_loader), 1)
        log_metric('val/loss',       val_loss_dict.get('loss', 0.0) / num_batches, epoch)
        log_metric('val/loss_ce',    val_loss_dict.get('loss_ce', 0.0) / num_batches, epoch)
        log_metric('val/loss_point', val_loss_dict.get('loss_point', 0.0) / num_batches, epoch)
        
        mae /= n
        rmse = (mse / n) ** 0.5
        log_metric('val/count_mae',  mae,  epoch)
        log_metric('val/count_rmse', rmse, epoch)
        
        print(f'  Val  MAE={mae:.2f}  RMSE={rmse:.2f}', flush=True)
        return mae, rmse

    # 9. Training Main Loop
    print("Starting Training Loop...", flush=True)
    epoch = start_epoch
    while epoch < epochs:
        train_one_epoch(epoch)
        
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            mae, rmse = validate(epoch + 1)
            
            # Save last checkpoint
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'global_step': global_step,
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state, os.path.join(save_path, 'last.pth'))
            if mlflow_enabled:
                try:
                    mlflow.log_artifact(os.path.join(save_path, 'last.pth'))
                except Exception as e:
                    print(f"Warning: Failed to log last.pth to MLflow: {e}", flush=True)

            if mae < best_mae:
                best_mae = mae
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mae': best_mae,
                    'global_step': global_step,
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))
                print(f"  --> Saved new best checkpoint with MAE {best_mae:.2f}", flush=True)
                if mlflow_enabled:
                    try:
                        mlflow.log_artifact(os.path.join(save_path, 'best.pth'))
                    except Exception as e:
                        print(f"Warning: Failed to log best.pth to MLflow: {e}", flush=True)
        epoch += 1
                
    writer.close()
    if mlflow_enabled:
        try:
            mlflow.end_run()
        except Exception:
            pass


if __name__ == '__main__':
    fire.Fire(main)
