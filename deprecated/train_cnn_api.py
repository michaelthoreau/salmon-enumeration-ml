import os
import math
import time
import datetime
import warnings
from pathlib import Path
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import fire
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import mlflow

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# MLflow Wrapper
# ---------------------------------------------------------------------------
mlflow_enabled = True

def _load_dotenv():
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

# ---------------------------------------------------------------------------
# Dataset with Gaussian Heatmap Generation
# ---------------------------------------------------------------------------
def _auto_run_name(num_context_frames, context_stride):
    return f"exp_cnn_cf{num_context_frames}_cs{context_stride}"

from dataset_api import SockeyeApiDataset, SameFrameBatchSampler

def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    heatmaps = torch.stack([item[1] for item in batch], dim=0)
    return inputs, heatmaps

# ---------------------------------------------------------------------------
# PyTorch Lightweight CNN Model with Late Temporal Fusion
# ---------------------------------------------------------------------------
import torchvision.models as models

class LightweightCNNWithLateFusion(nn.Module):
    def __init__(self, num_context_frames=1):
        super().__init__()
        self.num_context_frames = num_context_frames
        
        # Shared ResNet-18 Backbone
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,  # [B, 64, 128, 128]
            resnet.layer2,  # [B, 128, 64, 64]
            resnet.layer3,  # [B, 256, 32, 32]
            resnet.layer4   # [B, 512, 16, 16]
        )
        
        # Late Fusion layer at bottleneck
        in_channels = 512 * (1 + num_context_frames)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsamples from 16x16 back to 512x512)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        groups = 1 + self.num_context_frames
        
        # Split channels back to individual frame tensors [B, 3, H, W]
        chunks = torch.chunk(x, groups, dim=1)
        
        # Run each frame through the shared backbone
        features = []
        for chunk in chunks:
            features.append(self.backbone(chunk)) # shape: [B, 512, 16, 16]
            
        # Late Fusion: Concatenate features along channel dimension
        fused = torch.cat(features, dim=1) # shape: [B, 512 * groups, 16, 16]
        fused = self.fusion(fused)         # shape: [B, 512, 16, 16]
        
        # Upsample back to 512x512
        h = self.up1(fused) # 32x32
        h = self.up2(h)     # 64x64
        h = self.up3(h)     # 128x128
        h = self.up4(h)     # 256x256
        out = self.up5(h)   # 512x512, channel=1
        
        return out.squeeze(1) # shape [B, 512, 512]

# ---------------------------------------------------------------------------
# Peak Extraction Post-processing
# ---------------------------------------------------------------------------
def extract_peaks(heatmap, threshold=0.25, kernel_size=7):
    """Identifies local maxima above threshold to return point coordinates."""
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    
    h_tensor = heatmap.unsqueeze(0).unsqueeze(0) if heatmap.ndim == 2 else heatmap.unsqueeze(1)
    
    # 2D local max pooling
    pad = kernel_size // 2
    local_max = nn.functional.max_pool2d(h_tensor, kernel_size=kernel_size, stride=1, padding=pad)
    
    # Find peaks that match the local maximum and exceed the threshold
    peaks = (h_tensor == local_max) & (h_tensor > threshold)
    peaks = peaks.squeeze(1) # [B, H, W]
    
    batch_coords = []
    for b in range(peaks.shape[0]):
        coords = torch.nonzero(peaks[b]) # y, x indices
        points = []
        for cy, cx in coords:
            score = h_tensor[b, 0, cy, cx].item()
            points.append((cx.item(), cy.item(), score))
        batch_coords.append(points)
        
    return batch_coords

def log_validation_images(writer, epoch, input_tensor, target_heatmap, pred_heatmap, score_thresh=0.25):
    """Logs target frame overlaid with predicted heatmap (red channel), GT dots, and predicted peak circles."""
    try:
        from PIL import ImageDraw
        
        # 1. Unnormalize target frame (first 3 channels)
        img_t = input_tensor[:3].cpu().detach().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_t = np.clip((img_t * std + mean) * 255.0, 0, 255).astype(np.uint8)

        # 2. Build red-channel heatmap overlay (pure numpy, no cv2)
        pred_np   = pred_heatmap.cpu().detach().numpy()          # [H, W], 0-1
        heat_alpha = np.clip(pred_np * 0.6, 0.0, 0.6)[:, :, np.newaxis]  # opacity
        heat_rgb   = np.zeros_like(img_t)
        heat_rgb[:, :, 0] = (pred_np * 255).clip(0, 255)        # red channel only
        blended = np.clip(img_t * (1 - heat_alpha) + heat_rgb * heat_alpha, 0, 255).astype(np.uint8)

        # 3. Draw GT peaks (red filled dots) and predicted peaks (green hollow circles)
        im   = Image.fromarray(blended)
        draw = ImageDraw.Draw(im)

        gt_peaks = extract_peaks(target_heatmap.unsqueeze(0), threshold=0.9)[0]
        for cx, cy, _ in gt_peaks:
            r = 3
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red", outline="red")

        pred_peaks = extract_peaks(pred_heatmap.unsqueeze(0), threshold=score_thresh)[0]
        for cx, cy, _ in pred_peaks:
            r = 5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="lime", width=2)

        combined = np.array(im)

        # 4. Log to TensorBoard
        writer.add_image("val/predictions", combined, epoch, dataformats="HWC")

        # 5. Log to MLflow with key+step so images are slideable per epoch
        if mlflow_enabled:
            try:
                mlflow.log_image(combined, key="val/predictions", step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log validation image to MLflow: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"Warning: Error during validation image logging: {e}", flush=True)
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Main Training Logic
# ---------------------------------------------------------------------------
def main(
    api_url:      str   = None,
    api_key:      str   = None,
    save_path:    str   = None,
    epochs:       int   = 20,
    batch_size:   int   = 8,
    lr:           float = 1e-4,
    warmup_steps: int   = 100,
    decay_steps:  int   = 5000,
    val_every:    int   = 1,
    resume:       str   = None,
    workers:      int   = 8,
    prefetch_factor: int = 4,
    cache_dir:    str   = "/tmp/frame_cache",
    img_size:     int   = 512,
    sigma:        float = 4.0,
    score_thresh: float = 0.25,
    limit_batches: int  = 0,
    min_batches:  int   = 200,
    num_context_frames: int = 1,
    context_stride: int = 5,
    verbose:      bool  = False,
    mlflow_tracking_uri: str = None,
    mlflow_experiment: str = 'salmon-enumeration',
):
    global mlflow_enabled
    
    # Resolve API Key
    if not api_key:
        api_key = os.environ.get("SOCKEYE_API_KEY")
    if not api_url:
        api_url = os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1")
        
    if not api_key:
        raise ValueError("Missing Sockeye API Key. Add it to .env or pass --api_key.")

    # Resolve paths
    base = os.path.dirname(os.path.abspath(__file__))
    auto_name = _auto_run_name(num_context_frames, context_stride)

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
    writer = SummaryWriter(log_dir=save_path)

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
                    'lr': lr, 'warmup_steps': warmup_steps, 'decay_steps': decay_steps,
                    'val_every': val_every, 'resume': resume, 'workers': workers,
                    'img_size': img_size, 'sigma': sigma, 'score_thresh': score_thresh,
                    'limit_batches': limit_batches, 'min_batches': min_batches,
                    'num_context_frames': num_context_frames, 'context_stride': context_stride
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

    # Fetch labeling tasks from API
    headers = {"Authorization": f"Bearer {api_key}"}
    res = requests.get(f"{api_url.rstrip('/')}/labeling-tasks?completed=true&limit=2000", headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch labeling tasks: {res.status_code} - {res.text}")
    all_tasks = res.json()
    all_tasks = [t for t in all_tasks if len(t.get("points", [])) > 0]
    print(f"Loaded {len(all_tasks)} completed labeling tasks with fish from Sockeye API.", flush=True)
    
    # Split into train/val
    train_tasks = [t for i, t in enumerate(all_tasks) if i % 10 != 0]
    val_tasks = [t for i, t in enumerate(all_tasks) if i % 10 == 0]

    # Create datasets
    train_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=train_tasks,
        cache_dir=cache_dir, img_size=img_size, augment=True,
        num_context_frames=num_context_frames, context_stride=context_stride, sigma=sigma,
        verbose=verbose, mode="cnn"
    )
    val_ds = SockeyeApiDataset(
        api_url=api_url, api_key=api_key, tasks=val_tasks,
        cache_dir=cache_dir, img_size=img_size, augment=False,
        num_context_frames=num_context_frames, context_stride=context_stride, sigma=sigma,
        verbose=verbose, mode="cnn"
    )

    batch_sampler = SameFrameBatchSampler(len(train_ds), batch_size)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=prefetch_factor if workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=(workers > 0)
    )

    # Setup Model, Optimizer, Loss, Scheduler
    model = LightweightCNNWithLateFusion(num_context_frames=num_context_frames).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    total_steps = warmup_steps + decay_steps
    if epochs is None:
        epochs = math.ceil(total_steps / max(len(train_loader), 1))

    # Cosine Annealing learning rate scheduler
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, warmup_steps))
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, decay_steps), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    global_step = 0
    best_mae = 1e9
    start_epoch = 0

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
            model_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
            model.load_state_dict(ckpt[model_key], strict=False)
            
            opt_key = "optimizer_state_dict" if "optimizer_state_dict" in ckpt else "optimizer"
            if opt_key in ckpt:
                try:
                    optimizer.load_state_dict(ckpt[opt_key])
                except Exception:
                    pass
            start_epoch = ckpt.get('epoch', 0)
            best_mae = ckpt.get('best_mae', 1e9)
            global_step = ckpt.get('global_step', 0)
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
            print(f"Resumed from downloaded MLflow checkpoint: {resume} (epoch {start_epoch}, best MAE {best_mae:.2f}, step {global_step})", flush=True)
            checkpoint_loaded = True
        else:
            print(f"WARNING: checkpoint not found: {resume}", flush=True)

    # Helper Training Functions
    def train_one_epoch(epoch):
        nonlocal global_step
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:4d}/{epochs}', ncols=110, leave=True)
        
        for i, (inputs, targets) in enumerate(pbar):
            if limit_batches > 0 and i >= limit_batches:
                break
            if global_step >= total_steps:
                break
                
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            log_metric('train/loss', loss.item(), global_step)
            log_metric('train/lr', optimizer.param_groups[0]['lr'], global_step)

    @torch.no_grad()
    def validate(epoch):
        model.eval()
        running_val_loss = 0.0
        mae, mse = 0.0, 0.0
        n = 0
        
        for inputs, targets in val_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()
            
            # Peak coordinates extraction to compare counts
            batch_predictions = extract_peaks(outputs, threshold=score_thresh)
            
            # Log validation images (just for the very first batch of validation)
            if n == 0:
                log_validation_images(writer, epoch, inputs[0], targets[0], outputs[0], score_thresh=score_thresh)
            
            # Compare against task target points count
            for b in range(inputs.shape[0]):
                task_idx = n + b
                if task_idx >= len(val_ds.tasks):
                    break
                gt_count = len(val_ds.tasks[task_idx].get("points", []))
                pred_count = len(batch_predictions[b])
                
                diff = abs(pred_count - gt_count)
                mae += diff
                mse += diff**2
                
            n += inputs.shape[0]
            
        val_loss = running_val_loss / max(len(val_loader), 1)
        log_metric('val/loss', val_loss, epoch)
        
        if n == 0:
            return 1e9
            
        mae /= n
        rmse = (mse / n) ** 0.5
        log_metric('val/count_mae', mae, epoch)
        log_metric('val/count_rmse', rmse, epoch)
        print(f"  Val  Loss={val_loss:.6f}  MAE={mae:.2f}  RMSE={rmse:.2f} (thresh={score_thresh})", flush=True)
        return mae

    # Main training execution loop
    print("\nStarting Training Loop...", flush=True)
    epoch = start_epoch
    while epoch < epochs:
        if global_step >= total_steps:
            break
        train_one_epoch(epoch)
        
        if epoch == 0 or (epoch + 1) % val_every == 0 or epoch == epochs - 1 or global_step >= total_steps:
            val_mae = validate(epoch + 1)
            
            # Checkpoint saving
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
                        print(f"Warning: Failed to log best.pth to MLflow: {e}", flush=True)
        epoch += 1

    print("\nTraining completed successfully!", flush=True)
    if mlflow_enabled:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Failed to end MLflow run: {e}")

if __name__ == '__main__':
    fire.Fire(main)
