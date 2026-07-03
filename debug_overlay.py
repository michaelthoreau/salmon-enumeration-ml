"""
Overlay annotation points onto an image and save to a debug file.

Usage:
    python debug_overlay.py data/train/images/some-uuid.jpg
    python debug_overlay.py data/train/images/some-uuid.jpg --out debug/overlay.jpg
    python debug_overlay.py data/train/images/  # overlay all images in a directory
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


POINT_RADIUS = 8
POINT_COLOR  = (255, 50, 50, 200)   # red, semi-transparent
TEXT_COLOR   = (255, 255, 255, 255)


def overlay_image(img_path: Path, out_path: Path):
    ann_path = Path(str(img_path).replace('/images/', '/annotations/')).with_suffix('.json')
    if not ann_path.exists():
        print(f'  MISSING annotation: {ann_path}')
        return

    with open(ann_path) as f:
        ann = json.load(f)

    points = ann.get('points', [])
    img = Image.open(img_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, (x, y) in enumerate(points):
        r = POINT_RADIUS
        draw.ellipse([x - r, y - r, x + r, y + r], fill=POINT_COLOR, outline=(255, 50, 50, 255), width=2)
        draw.text((x + r + 2, y - 6), str(i + 1), fill=TEXT_COLOR)

    result = Image.alpha_composite(img, overlay).convert('RGB')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f'  {img_path.name}: {len(points)} fish -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Image file or directory of images')
    parser.add_argument('--out', default=None, help='Output file (single image) or directory (batch)')
    args = parser.parse_args()

    src = Path(args.input)

    if src.is_dir():
        images = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
        out_dir = Path(args.out) if args.out else Path('debug/overlays') / src.parent.name
        print(f'Overlaying {len(images)} images -> {out_dir}')
        for img_path in images:
            overlay_image(img_path, out_dir / img_path.name)
    else:
        out_path = Path(args.out) if args.out else Path('debug') / ('overlay_' + src.name)
        overlay_image(src, out_path)


if __name__ == '__main__':
    main()
