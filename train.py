"""
IOCFormer training — single GPU, no distributed, no config.py.
Usage:
    python -u train.py
    python -u train.py --epochs=1500 --batch_size=4 --lr=1e-5 --save_path=runs/exp1
"""

import math
import os
import time
import warnings
from types import SimpleNamespace

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(
    save_path: str  = None,
    epochs: int     = 1500,
    batch_size: int = 2,
    lr: float       = 1e-4,
    lr_step: int    = 1200,
    warmup_steps: int = 100,
    val_every: int  = 5,
    resume: str     = None,
    workers: int    = 4,
    data_root: str  = 'data',
    limit_batches: int = 0,
    min_batches: int = 50,   # oversample small datasets; 20 batches/epoch min
    crop_size: int  = 640,   # train_distributed.py default; 640 for full-res
    num_queries: int = 100,  # max expected fish per crop; 700 for data_big
    companion_mode: str = None,  # None | 'prev10' | 'background' — 6-ch input
):
    """Train IOCFormer on the salmon dataset.

    Args:
        save_path:  Directory to save checkpoints and logs. Auto-selected (runs/exp0, exp1, …) if not given.
        epochs:     Total number of training epochs.
        batch_size: Training batch size.
        lr:         Initial learning rate.
        lr_step:    Epoch at which LR is multiplied by 0.1.
        warmup_steps: Linear LR warmup over this many optimizer steps (0 to disable).
        val_every:  Run validation every N epochs (starting at epoch 800).
        resume:     Path to a checkpoint to resume from.
        workers:    DataLoader worker processes.
        data_root:  Root directory containing train/val/test split folders.
        limit_batches: If >0, stop each epoch after this many batches (debug mode).
        min_batches: Repeat-sample so each epoch has at least this many batches. Default 20
                     gives 40 crops/epoch from 10 toy images. Use 0 to disable.
        crop_size:  Crop size fed to the model. train_distributed.py default is 256;
                    use 640 for data_big to match original full-res training.
        num_queries: Max detectable fish per crop. 100 for toy/256px, 700 for data_big/640px.
    """

    # ---------------------------------------------------------------------------
    # Hardcoded model config — matches CVPR23 paper settings
    # ---------------------------------------------------------------------------

    MODEL_CFG = SimpleNamespace(
        # backbone
        backbone='resnet50',
        dilation=True,          # DC5 — gives downsample_ratio=8
        lr_backbone=1e-4,
        position_embedding='sine',
        masks=False,
        frozen_weights=None,
        # transformer
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=700,
        pre_norm=False,
        transformer_flag='merge3',
        # IOCFormer-specific
        dm_count=True,
        branch_merge=True,
        branch_merge_way=2,
        channel_point=3,
        only_dm=False,
        with_weights=False,
        # loss coefficients
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
        # misc
        dataset_file='crowd_data',
        device='cuda',
    )

    DOWNSAMPLE_RATIO = 8    # dilation=True
    WDM  = 0.5              # density-map loss weight
    WOT  = 0.0              # optimal-transport loss (off)
    WTV  = 0.0              # total-variation loss (off)

    CROP_SIZE   = crop_size
    NUM_QUERIES = num_queries
    MODEL_CFG.num_queries = num_queries
    THRESHOLD   = 0.35

    # ---------------------------------------------------------------------------
    # Imports (no config.py)
    # ---------------------------------------------------------------------------

    import dataset
    from Networks.CDETR import build_model
    from ot_loss import OT_Loss
    from utils import get_root_logger

    # ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------

    torch.manual_seed(1); torch.cuda.manual_seed(1); np.random.seed(1)

    base = os.path.dirname(os.path.abspath(__file__))
    # Resolve data_root relative to the script location if not absolute
    if not os.path.isabs(data_root):
        data_root = os.path.join(base, data_root)
    if save_path is None:
        i = 0
        while os.path.exists(os.path.join(base, f'runs/exp{i}')):
            i += 1
        save_path = f'runs/exp{i}'
    SAVE_DIR = os.path.join(base, save_path)
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger = get_root_logger(os.path.join(SAVE_DIR, 'train.log'))
    writer = SummaryWriter(SAVE_DIR)

    print(f'Save dir : {SAVE_DIR}', flush=True)
    print(f'Epochs   : {epochs}  |  Batch: {batch_size}  |  LR: {lr}  '
          f'|  Crop: {crop_size}  |  Queries: {num_queries}  |  MinBatches: {min_batches}', flush=True)

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------

    def _list_images(split_dir):
        """Return image paths that have a matching annotation JSON.
        Looks in annotations/ first so only annotated images are included,
        and suffix variants like _prev10.jpg / _background.jpg are naturally excluded.
        """
        ann_dir = os.path.join(data_root, split_dir, 'annotations')
        img_dir = os.path.join(data_root, split_dir, 'images')
        img_exts = ('.jpg', '.jpeg', '.png')
        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            return []
        result = []
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith('.json'):
                continue
            stem = fname[:-5]   # strip .json
            for ext in img_exts:
                img_path = os.path.join(img_dir, stem + ext)
                if os.path.isfile(img_path):
                    result.append(img_path)
                    break
        return result

    train_files = _list_images('train')
    val_files   = _list_images('val')

    print(f'Train: {len(train_files)} images  |  Val: {len(val_files)} images', flush=True)

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------

    model, criterion, postprocessors = build_model(MODEL_CFG)

    if companion_mode:
        from Networks.CDETR.backbone import expand_first_conv
        expand_first_conv(model.backbone, in_channels=6)

    model = model.cuda()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params / 1e6:.2f}M  |  GPU: {torch.cuda.get_device_name()}', flush=True)

    # ---------------------------------------------------------------------------
    # Losses / optimiser / scheduler
    # ---------------------------------------------------------------------------

    ot_loss   = OT_Loss(CROP_SIZE, DOWNSAMPLE_RATIO, 0, None, 100, 10.0)
    tv_loss   = nn.L1Loss(reduction='none').cuda()
    mae_loss  = nn.L1Loss().cuda()
    dm_losses = [ot_loss, tv_loss, mae_loss]

    wd = 5e-4
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': lr}],
        lr=lr, weight_decay=wd)

    # LR schedule: MultiStepLR stepping once per epoch (matching train_distributed.py)
    # Linear warmup applied manually for the first warmup_steps batches.
    scheduler   = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_step], gamma=0.1)
    global_step = 0
    warmup_done = warmup_steps == 0
    # store base LRs for warmup scaling
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    if warmup_steps > 0:
        for pg in optimizer.param_groups:
            pg['lr'] = 0.0

    # ---------------------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------------------

    start_epoch = 0
    best_mae    = 1e5

    if resume:
        # Allow passing a run directory — auto-resolve to best.pth inside it
        if os.path.isdir(resume):
            resume = os.path.join(resume, 'best.pth')
        if os.path.isfile(resume):
            ckpt = torch.load(resume, map_location='cuda')
            model.load_state_dict(ckpt['state_dict'], strict=False)
            start_epoch = ckpt.get('epoch', 0)
            best_mae    = ckpt.get('best_prec1', 1e5)
            print(f'Resumed from {resume}  (epoch {start_epoch}, best MAE {best_mae:.2f})', flush=True)
        else:
            print(f'WARNING: resume path not found: {resume}', flush=True)

    # ---------------------------------------------------------------------------
    # Dataset args dict (what dataset.listDataset expects)
    # ---------------------------------------------------------------------------

    DS_ARGS = {
        'crop_size': CROP_SIZE,
        'num_queries': NUM_QUERIES,
        'threshold': THRESHOLD,
        'num_patch': 1,
        'test_patch': True,
        'gray_aug': True,
        'gray_p': 0.1,
        'scale_aug': True,
        'scale_type': 1,
        'scale_p': 0.3,
        'min_num': -1,
        'num_knn': 4,
        'channel_point': 3,
        'dm_count': True,
        'only_dm': False,
        'save_path': SAVE_DIR,
        'local_rank': 0,
        'distributed': False,
        'companion_mode': companion_mode,  # None | 'prev10' | 'background'
    }

    # ---------------------------------------------------------------------------
    # Collate
    # ---------------------------------------------------------------------------

    def collate_fn(batch):
        fnames, imgs, targets = [], [], []
        for item in batch:
            fnames.append(item[0])
            for img in item[1]:
                imgs.append(img)
            for t in item[2]:
                targets.append(t)
        return fnames, torch.stack(imgs, 0), targets

    # ---------------------------------------------------------------------------
    # Train loop
    # ---------------------------------------------------------------------------

    def train_one_epoch(epoch):
        ds = dataset.listDataset(
            train_files, SAVE_DIR,
            shuffle=True,
            transform=transforms.Compose([
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]),
            train=True,
            args=DS_ARGS,
        )
        # If min_batches set, oversample so epoch has at least that many batches
        n_samples = len(ds)
        if min_batches > 0:
            n_samples = max(n_samples, min_batches * batch_size)
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=n_samples)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, drop_last=False,
            collate_fn=collate_fn, num_workers=workers, pin_memory=True,
            sampler=sampler,
        )

        model.train()
        losses = []
        t0 = time.time()

        pbar = tqdm(loader, desc=f'Epoch {epoch:4d}/{epochs}', ncols=110, leave=True)
        for i, (fname, img, targets) in enumerate(pbar):
            if limit_batches > 0 and i >= limit_batches:
                break
            img = img.cuda()
            N   = img.shape[0]

            # LR warmup: scale LR for first warmup_steps batches
            nonlocal global_step, warmup_done
            global_step += 1
            if not warmup_done:
                frac = min(global_step / warmup_steps, 1.0)
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * frac
                if global_step >= warmup_steps:
                    warmup_done = True

            out_all = model(img)

            if isinstance(out_all, list):
                d6, out_dm = out_all
                gd_count = np.array([t['points'].shape[0] for t in targets], dtype=np.float32)
                points   = [t['points_true'][:, :2].cuda() for t in targets]

                # Only compute OT if WOT>0 AND all items have at least 1 point
                if WOT > 0 and all(len(p) > 0 for p in points):
                    ot, _, _ = dm_losses[0](out_dm[2], out_dm[1], points)
                else:
                    ot = torch.tensor(0.0, device=img.device)
                count_l  = dm_losses[2](
                    out_dm[1].sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().cuda())

                gd_tensor = torch.from_numpy(gd_count).float().cuda().view(N, 1, 1, 1)
                gt_disc   = torch.stack([t['gt_map'].cuda() for t in targets])
                dh = gt_disc.shape[-2] // DOWNSAMPLE_RATIO
                dw = gt_disc.shape[-1] // DOWNSAMPLE_RATIO
                gt_disc   = gt_disc.reshape(N, 1, dh, DOWNSAMPLE_RATIO, dw, DOWNSAMPLE_RATIO).sum(dim=(3, 5))
                gt_disc_n = gt_disc / (gd_tensor + 1e-6)
                tv_l = (dm_losses[1](out_dm[2], gt_disc_n).sum(1).sum(1).sum(1)
                        * torch.from_numpy(gd_count).float().cuda()).mean(0) * WTV

                loss_dm = (ot * WOT + count_l + tv_l) + 0.0 * out_dm[0].sum()
            else:
                d6      = out_all
                loss_dm = 0.0

            loss_dict = criterion(d6, targets)
            loss_detr = sum(loss_dict[k] * criterion.weight_dict[k]
                       for k in loss_dict if k in criterion.weight_dict)
            loss = loss_detr
            if isinstance(out_all, list):
                loss = loss + loss_dm * WDM

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # DETR predicted count: queries whose fg score > THRESHOLD
            with torch.no_grad():
                pred_scores = d6['pred_logits'].sigmoid()[:, :, 1]  # [N, Q]
                pred_count  = (pred_scores > THRESHOLD).sum(dim=1).float()  # [N]
                gt_count_detr = torch.tensor(
                    [t['points'].shape[0] for t in targets], dtype=torch.float32)
                detr_count_err = (pred_count.cpu() - gt_count_detr).abs().mean().item()

            pbar.set_postfix(loss=f'{loss.item():.4f}', avg=f'{np.mean(losses):.4f}',
                             cnt_err=f'{detr_count_err:.1f}')
            writer.add_scalar('train/loss',            loss.item(),                         global_step)
            writer.add_scalar('train/loss_detr',       loss_detr.item(),                    global_step)
            writer.add_scalar('train/loss_ce',         loss_dict.get('loss_ce', 0),         global_step)
            writer.add_scalar('train/loss_point',      loss_dict.get('loss_point', 0),      global_step)
            writer.add_scalar('train/detr_count_err',  detr_count_err,                      global_step)
            writer.add_scalar('train/detr_pred_count', pred_count.mean().item(),             global_step)
            writer.add_scalar('train/detr_gt_count',   gt_count_detr.mean().item(),          global_step)
            if isinstance(out_all, list):
                writer.add_scalar('train/count_l',  count_l.item(),                  global_step)
                writer.add_scalar('train/dm_pred',  out_dm[1].sum().item() / N,       global_step)
                writer.add_scalar('train/dm_gt',    float(gd_count.mean()),            global_step)
            writer.add_scalar('train/lr',         optimizer.param_groups[0]['lr'],    global_step)
            # writer.add_scalar('gpu/mem_used_mb',  torch.cuda.memory_allocated() / 1e6, global_step)
            # writer.add_scalar('gpu/mem_reserved_mb', torch.cuda.memory_reserved() / 1e6, global_step)
            # try:
            #     writer.add_scalar('gpu/utilization_pct', torch.cuda.utilization(), global_step)
            # except Exception:
            #     pass

        scheduler.step()   # once per epoch
        elapsed = time.time() - t0
        msg = (f'Epoch {epoch:4d}  loss={np.mean(losses):.4f}  '
               f'lr={optimizer.param_groups[-1]["lr"]:.2e}  time={elapsed:.0f}s')
        logger.info(msg)
        print(msg, flush=True)

    # ---------------------------------------------------------------------------
    # Validation loop
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def validate(epoch):
        ds = dataset.listDataset(
            val_files, SAVE_DIR,
            shuffle=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]),
            train=False,
            args=DS_ARGS,
        )
        n_val = min(100, len(ds))
        if n_val == 0:
            print('  Val: no labeled val images, skipping.', flush=True)
            return best_mae, 0.0
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=2,
            sampler=torch.utils.data.RandomSampler(ds, num_samples=n_val))

        model.eval()
        mae = mse = 0.0
        total_tp = total_pred = total_gt = 0
        MATCH_RADIUS = 15  # pixels
        debug_saved = False  # save overlay debug images for the first val sample only

        for fname, img, kpoint, targets, patch_info in tqdm(loader, desc='  Val', ncols=80, leave=False):
            if img.dim() == 5:    img = img.squeeze(0)
            if img.dim() == 3:    img = img.unsqueeze(0)
            if kpoint.dim() == 5: kpoint = kpoint.squeeze(0)

            # Process patches one at a time to avoid OOM on large images.
            # A 1920×1080 image tiles into ~6 patches; batching all at once OOMs.
            all_logits, all_pts_out, dm_count = [], [], 0.0
            for pi in range(img.shape[0]):
                patch_out = model(img[pi:pi+1].cuda())
                if isinstance(patch_out, list):
                    d6p, dmp = patch_out
                    dm_count += dmp[1].sum().item()
                else:
                    d6p = patch_out
                all_logits.append(d6p['pred_logits'].cpu())
                all_pts_out.append(d6p['pred_points'].cpu())
            outputs = {'pred_logits': torch.cat(all_logits, 0),
                       'pred_points': torch.cat(all_pts_out, 0)}

            count = dm_count
            gt    = kpoint.sum().item()
            mae  += abs(count - gt)
            mse  += (count - gt) ** 2

            # ----- precision / recall via point matching -----
            # GT points from kpoint patches -> reassemble full padded map -> (y, x) coords
            num_h  = int(patch_info[0].item())
            num_w  = int(patch_info[1].item())
            H_pad  = int(patch_info[2].item())
            W_pad  = int(patch_info[3].item())
            crop   = int(patch_info[4].item())
            pad_w  = int(patch_info[5].item())
            pad_h  = int(patch_info[6].item())
            # kpoint: [num_h*num_w, 1, crop, crop] -> [H_pad, W_pad]
            kp2d   = (kpoint.view(num_h, num_w, crop, crop)
                            .permute(0, 2, 1, 3)
                            .reshape(H_pad, W_pad))
            gt_yx  = torch.nonzero(kp2d).float()   # [N, 2] (y, x) in padded space
            # undo padding offset
            gt_pts = gt_yx - torch.tensor([[pad_h, pad_w]], dtype=torch.float)

            # Predicted points from DETR head across ALL patches.
            # outputs['pred_logits']: [P, Q, 2],  outputs['pred_points']: [P, Q, >=2]
            # Patch layout: patches are ordered row-major (patch_row * num_w + patch_col).
            # Padding was added to LEFT and TOP (dataset.py: F.pad(img,(pad_w,0,pad_h,0))),
            # so patch origin in padded image is (row*crop, col*crop), and to convert to
            # original-image coords we subtract the padding.
            orig_H = H_pad - pad_h
            orig_W = W_pad - pad_w
            all_pred_pts = []
            for p_idx in range(num_h * num_w):
                p_scores = outputs['pred_logits'][p_idx].sigmoid()[:, 1].cpu()
                keep     = p_scores > THRESHOLD
                pts_norm = outputs['pred_points'][p_idx][keep].cpu()[:, :2]  # (x_norm, y_norm)
                patch_row = p_idx // num_w
                patch_col = p_idx  % num_w
                # origin in padded image (pixels)
                oy = patch_row * crop   # y offset (includes pad_h at top)
                ox = patch_col * crop   # x offset (includes pad_w at left)
                # convert to original-image (y, x) by subtracting padding
                py = oy + pts_norm[:, 1] * crop - pad_h  # y_norm * crop + row_origin - pad_h
                px = ox + pts_norm[:, 0] * crop - pad_w  # x_norm * crop + col_origin - pad_w
                # keep only points inside original image bounds
                inside = (py >= 0) & (py < orig_H) & (px >= 0) & (px < orig_W)
                if inside.any():
                    all_pred_pts.append(torch.stack([py[inside], px[inside]], dim=1))
            pred_pts = torch.cat(all_pred_pts, dim=0) if all_pred_pts else torch.zeros((0, 2))

            # ----- per-epoch image logging to TensorBoard (first sample only) -----
            if not debug_saved:
                debug_saved = True
                import PIL.Image as PILImage, PIL.ImageDraw as PILDraw

                # Unnormalize and stitch patches -> full original image
                _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                img_un = (img.cpu() * _std + _mean).clamp(0, 1)  # [P, 3, crop, crop]
                img_grid = (img_un
                            .view(num_h, num_w, 3, crop, crop)
                            .permute(2, 0, 3, 1, 4)
                            .reshape(3, num_h * crop, num_w * crop))
                img_orig = img_grid[:, pad_h:, pad_w:]     # [3, H, W] — strip top/left padding
                img_np = (img_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                def _draw_pts(base_np, pts_yx, color, radius=8, outline_only=False):
                    base    = PILImage.fromarray(base_np).convert('RGBA')
                    overlay = PILImage.new('RGBA', base.size, (0, 0, 0, 0))
                    draw    = PILDraw.Draw(overlay)
                    for y, x in pts_yx:
                        y, x = int(y), int(x)
                        bbox = [x - radius, y - radius, x + radius, y + radius]
                        if outline_only:
                            draw.ellipse(bbox, outline=color, width=2)
                        else:
                            draw.ellipse(bbox, fill=color)
                    return PILImage.alpha_composite(base, overlay).convert('RGB')

                gt_np   = gt_pts.numpy()   if gt_pts.numel()   > 0 else np.zeros((0, 2))
                pred_np = pred_pts.numpy() if pred_pts.numel() > 0 else np.zeros((0, 2))

                # Single image: GT (green filled) + preds (red hollow) overlaid
                combined = _draw_pts(img_np, gt_np,   (0, 220, 0,   200), radius=8, outline_only=False)
                combined = _draw_pts(np.array(combined), pred_np, (220, 50, 50, 255), radius=8, outline_only=True)
                img_tensor = torch.from_numpy(np.array(combined)).float().div(255).permute(2, 0, 1)
                writer.add_image('val_img/gt_and_pred', img_tensor, epoch)
                logger.info(f'  Logged val image: {len(gt_np)} GT pts, {len(pred_np)} pred pts')

            n_gt   = gt_pts.shape[0]
            n_pred = pred_pts.shape[0]
            tp = 0
            if n_gt > 0 and n_pred > 0:
                # pairwise distances [n_gt, n_pred]
                dists  = torch.cdist(gt_pts.float(), pred_pts.float())
                # greedy matching
                matched_gt   = set()
                matched_pred = set()
                vals, idxs   = dists.flatten().sort()
                for v, flat_i in zip(vals.tolist(), idxs.tolist()):
                    if v > MATCH_RADIUS:
                        break
                    gi, pi = divmod(flat_i, n_pred)
                    if gi not in matched_gt and pi not in matched_pred:
                        matched_gt.add(gi); matched_pred.add(pi); tp += 1

            total_tp   += tp
            total_pred += n_pred
            total_gt   += n_gt

        n = len(loader)
        mae /= n
        mse  = math.sqrt(mse / n)
        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall    = total_tp / total_gt   if total_gt   > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        writer.add_scalar('val/dm_count_mae',  mae,       epoch)
        writer.add_scalar('val/dm_count_rmse', mse,       epoch)
        writer.add_scalar('val/precision', precision, epoch)
        writer.add_scalar('val/recall',    recall,    epoch)
        writer.add_scalar('val/f1',        f1,        epoch)
        msg = (f'  --> Val  dm_MAE={mae:.3f}  dm_RMSE={mse:.3f}  '
               f'P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  best={best_mae:.3f}')
        logger.info(msg)
        print(msg, flush=True)
        return mae, mse

    # ---------------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------------

    print('Starting training...', flush=True)

    nonlocal_best_mae = best_mae

    # Run a validation pass before any training so epoch-0 images are visible
    if start_epoch == 0 and val_files:
        validate(0)

    for epoch in range(start_epoch, epochs + 1):
        train_one_epoch(epoch)

        if epoch % val_every == 0 and val_files:
            mae, mse = validate(epoch)

            is_best           = mae < nonlocal_best_mae
            nonlocal_best_mae = min(mae, nonlocal_best_mae)

            if is_best:
                torch.save({
                    'epoch':       epoch + 1,
                    'state_dict':  model.state_dict(),
                    'best_prec1':  nonlocal_best_mae,
                }, os.path.join(SAVE_DIR, 'best.pth'))
                print(f'  --> New best! MAE={nonlocal_best_mae:.3f}', flush=True)

    writer.close()
    print('Done.', flush=True)


if __name__ == '__main__':
    fire.Fire(main)

