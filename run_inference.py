"""
Inference script to run salmon count prediction on a single image and visualize outputs.
Usage:
    python run_inference.py --img_path=data/val/images/some-uuid.jpg --model_path=runs/exp0/best.pth
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from types import SimpleNamespace

# Add workspace directory to system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Networks.CDETR import build_model


def load_trained_model(model_path: str, device: str = 'cuda'):
    """Initializes the model architecture and loads weight checkpoint."""
    # Base configuration matching CVPR23 paper / train.py hardcoded settings
    cfg = SimpleNamespace(
        backbone='resnet50',
        dilation=True,          # dc5 layout
        lr_backbone=1e-4,
        position_embedding='sine',
        masks=False,
        frozen_weights=None,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=700,
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
        device=device,
    )
    
    print(f"Building IOCFormer model structure...", flush=True)
    model, _, _ = build_model(cfg)
    
    print(f"Loading weights from: {model_path}", flush=True)
    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    
    # Strip any potential Dataparallel prefix ('module.') from keys
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    
    model = model.to(device).eval()
    return model


def run_single_image_inference(model, img_path: str, threshold: float = 0.35, crop_size: int = 640):
    """
    Runs model inference on a single image.
    Applies image tiling/padding similar to training datasets in dataset.py.
    """
    device = next(model.parameters()).device
    
    # Image loading and training transformations
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    pil_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    img_t = transform(pil_img)  # [3, H, W]
    
    # Calculate padding to ensure dimensions are multiples of crop_size (same as test dataset)
    padding_h = img_t.shape[1] % crop_size
    padding_w = img_t.shape[2] % crop_size

    if padding_w != 0:
        padding_w = crop_size - padding_w
    if padding_h != 0:
        padding_h = crop_size - padding_h
        
    pd = (padding_w, 0, padding_h, 0) # left/right padding is (w, 0) which means pad left by padding_w! Wait! F.pad with (0, pad_w, 0, pad_h) pads right and bottom! But dataset.py has pd = (padding_w, 0, padding_h, 0) which means pad left for width and top for height!
    img_pad = F.pad(img_t, pd, 'constant', 0)
    _, H_pad, W_pad = img_pad.shape
    num_h = H_pad // crop_size
    num_w = W_pad // crop_size
    n_patches = num_h * num_w
    
    # Tile padded image into individual patches: [num_patches, 3, crop_size, crop_size]
    patches = (img_pad
               .view(3, num_h, crop_size, W_pad)
               .view(3, num_h, crop_size, num_w, crop_size)
               .permute(0, 1, 3, 2, 4)
               .contiguous()
               .view(3, n_patches, crop_size, crop_size)
               .permute(1, 0, 2, 3))
    
    print(f"Segmented image into {n_patches} patch(es) ({num_h}x{num_w}) of size {crop_size}x{crop_size}.", flush=True)
    
    all_logits = []
    all_points = []
    all_dms = []
    dm_predicted_count = 0.0
    
    # Batch through patches to prevent OOM
    batch_size = 4
    for start in range(0, n_patches, batch_size):
        batch = patches[start:start + batch_size].to(device)
        with torch.no_grad():
            out = model(batch)
        if isinstance(out, list):
            d6, out_dm = out
            all_dms.append(out_dm[1].cpu()) # out_dm[1] is density map predicted [Batch_size, 1, h_dm, w_dm]
            dm_predicted_count += float(out_dm[1].sum().item())
        else:
            d6 = out
        all_logits.append(d6['pred_logits'].cpu())
        all_points.append(d6['pred_points'].cpu())
        
    all_logits = torch.cat(all_logits, dim=0)   # [n_patches, Q, 2]
    all_points = torch.cat(all_points, dim=0)   # [n_patches, Q, 3]
    scores = all_logits.sigmoid()[:, :, 1]      # [n_patches, Q]
    
    # Assemble density map if available
    density_map_img = None
    if len(all_dms) > 0:
        all_dms = torch.cat(all_dms, dim=0)     # [n_patches, 1, h_dm, w_dm]
        # h_dm and w_dm are resolution of density map patch
        h_dm, w_dm = all_dms.shape[-2], all_dms.shape[-1]
        
        # Stitch patches back together to match padded image dimensions
        # Originally tiled from: permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, crop_size, crop_size)
        # To undo tiling: [num_patches, 1, h_dm, w_dm] -> [num_h, num_w, h_dm, w_dm] -> [num_h, h_dm, num_w, w_dm] -> [num_h * h_dm, num_w * w_dm]
        stitched_dm = (all_dms
                       .view(num_h, num_w, h_dm, w_dm)
                       .permute(0, 2, 1, 3)
                       .reshape(num_h * h_dm, num_w * w_dm))
        
        # Crop to matching aspect ratio of original image (same downsample ratio layout)
        downsample_ratio = crop_size / h_dm # usually 8
        orig_dm_h = int(orig_h / downsample_ratio)
        orig_dm_w = int(orig_w / downsample_ratio)
        
        # Since F.pad (padding_w, 0, padding_h, 0) padded the image on the LEFT and TOP,
        # the stitched density map has the original image contents shifted to the RIGHT and BOTTOM by padding_w/downsample_ratio and padding_h/downsample_ratio indices respectively.
        # We must slice the density map starting from the padded offset to get the correct original area!
        pad_offset_y = int(padding_h / downsample_ratio)
        pad_offset_x = int(padding_w / downsample_ratio)
        cropped_dm = stitched_dm[pad_offset_y:pad_offset_y + orig_dm_h, pad_offset_x:pad_offset_x + orig_dm_w].numpy()
        
        # Normalize and apply color colormap for visualization
        if cropped_dm.max() > 0:
            norm_dm = (cropped_dm - cropped_dm.min()) / (cropped_dm.max() - cropped_dm.min() + 1e-8)
        else:
            norm_dm = cropped_dm
        norm_dm = (norm_dm * 255).astype(np.uint8)
        # Resize density map to original representation image width/height using OpenCV
        density_map_img = cv2.resize(norm_dm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        density_map_img = cv2.applyColorMap(density_map_img, cv2.COLORMAP_JET)
        
    # Map patch-relative points back to origin coordinates
    patch_rows = torch.arange(num_h).repeat_interleave(num_w)
    patch_cols = torch.arange(num_w).repeat(num_h)
    origins_y = patch_rows * crop_size
    origins_x = patch_cols * crop_size
    
    detected_points = []
    for p_idx in range(n_patches):
        keep = scores[p_idx] > threshold
        pts_norm = all_points[p_idx][keep].numpy()[:, :2]  # model outputs (x_norm, y_norm)
        oy, ox = int(origins_y[p_idx]), int(origins_x[p_idx])

        for (x_norm, y_norm) in pts_norm:
            py = oy + y_norm * crop_size
            px = ox + x_norm * crop_size
            # Since F.pad with (padding_w, 0, padding_h, 0) pads LEFT and TOP, 
            # we must subtract the padding offset to get true original image coordinate position!
            py = py - padding_h
            px = px - padding_w
            # Only keep points inside original unpadded bounds
            if 0 <= py < orig_h and 0 <= px < orig_w:
                detected_points.append((float(px), float(py)))
                
    print(f"Inference completed. Detected {len(detected_points)} point(s) using threshold {threshold}. DM predicted count: {dm_predicted_count:.2f}", flush=True)
    return detected_points, pil_img, density_map_img


def visualize_predictions(img_path: str, detected_points, pil_img, density_map_img=None, out_path: str = None):
    """Overlays predicted points with circle annotations and density maps onto the image."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Load ground truth points if available
    gt_points = []
    ann_path = img_path.replace('/images/', '/annotations/').replace('.jpg', '.json')
    if os.path.exists(ann_path):
        try:
            with open(ann_path) as f:
                ann = json.load(f)
                gt_points = ann.get('points', [])
        except Exception:
            pass
            
    # Draw Ground Truth circles in GREEN if available
    for (gx, gy) in gt_points:
        cv2.circle(cv_img, (int(gx), int(gy)), 8, (0, 255, 0), 2)
        
    # Draw Predicted circles in RED with a dot in the center
    for (px, py) in detected_points:
        cv2.circle(cv_img, (int(px), int(py)), 6, (0, 0, 255), -1)
        cv2.circle(cv_img, (int(px), int(py)), 8, (0, 255, 255), 1)
        
    # Add count summary on the top-left portion of the image
    txt = f"Pred Count: {len(detected_points)}"
    if len(gt_points) > 0:
        txt += f" | GT Count: {len(gt_points)}"
        
    cv2.putText(cv_img, txt, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # If density map exists, blend or stack them side-by-side
    if density_map_img is not None:
        # Create a blended density overlay image (alpha blending)
        blended = cv2.addWeighted(cv_img, 0.6, density_map_img, 0.4, 0)
        # Create a clean side-by-side horizontal stack: [Annotation Overlay] | [Density Map] | [Alpha Blended]
        stacked_img = np.hstack((cv_img, density_map_img, blended))
    else:
        stacked_img = cv_img

    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(base_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        out_path = os.path.join(debug_dir, f"viz_{img_name}")
        
    cv2.imwrite(out_path, stacked_img)
    print(f"Visualization saved to: {out_path}", flush=True)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single Image Salmon Enumeration Inference and Visualization")
    parser.add_argument('--img_path', type=str, required=True, help="Path to input image file")
    parser.add_argument('--model_path', type=str, default="runs/exp2/best.pth", help="Path to checkpoint weights")
    parser.add_argument('--threshold', type=float, default=0.35, help="Salmon query scoring threshold")
    parser.add_argument('--crop_size', type=int, default=640, help="Tile crop size used during training")
    parser.add_argument('--out_path', type=str, default=None, help="Output path for final image")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run inference on ('cuda' or 'cpu')")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    model = load_trained_model(args.model_path, device=device)
    
    pts, pil_img, density_map_img = run_single_image_inference(
        model=model,
        img_path=args.img_path,
        threshold=args.threshold,
        crop_size=args.crop_size
    )
    
    visualize_predictions(
        img_path=args.img_path,
        detected_points=pts,
        pil_img=pil_img,
        density_map_img=density_map_img,
        out_path=args.out_path
    )


if __name__ == '__main__':
    main()
