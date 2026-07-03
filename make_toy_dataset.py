"""
Generate a small synthetic dataset of white rectangles on a dark background.
Images and JSON annotations are written in the same format as the real salmon dataset.

Usage:
    python make_toy_dataset.py
    python make_toy_dataset.py --data_root=data_toy --n_train=20 --n_val=8 --seed=42
"""

import json
import os
import random
import uuid
from pathlib import Path

import fire
import numpy as np
from PIL import Image, ImageDraw


def make_image(
    width: int = 1280,
    height: int = 960,
    n_rects: int = None,
    min_rects: int = 5,
    max_rects: int = 30,
    rect_w: int = 14,
    rect_h: int = 10,
    bg_color: int = 30,
    rect_color: int = 230,
    noise_std: float = 8.0,
    rng: random.Random = None,
):
    """Return (PIL.Image, list of (cx, cy) centre-points)."""
    if rng is None:
        rng = random.Random()
    if n_rects is None:
        n_rects = rng.randint(min_rects, max_rects)

    # Dark grey background with a little noise
    bg = np.full((height, width, 3), bg_color, dtype=np.float32)
    bg += rng.gauss(0, noise_std) * np.random.randn(height, width, 1).astype(np.float32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg, 'RGB')
    draw = ImageDraw.Draw(img)

    points = []
    half_w = rect_w // 2
    half_h = rect_h // 2

    for _ in range(n_rects):
        cx = rng.randint(half_w + 1, width  - half_w - 1)
        cy = rng.randint(half_h + 1, height - half_h - 1)
        x0, y0 = cx - half_w, cy - half_h
        x1, y1 = cx + half_w, cy + half_h
        # Slightly randomise brightness so they aren't all identical
        brightness = rng.randint(rect_color - 20, min(255, rect_color + 20))
        fill = (brightness, brightness, brightness)
        draw.rectangle([x0, y0, x1, y1], fill=fill)
        points.append([cx, cy])

    return img, points


def write_split(
    split_dir: Path,
    n_images: int,
    width: int,
    height: int,
    min_rects: int,
    max_rects: int,
    rect_w: int,
    rect_h: int,
    rng: random.Random,
    base_image=None,
    base_points=None,
):
    images_dir = split_dir / 'images'
    ann_dir    = split_dir / 'annotations'
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        name = str(uuid.uuid4())
        img_path = images_dir / f'{name}.jpg'
        ann_path = ann_dir    / f'{name}.json'

        if base_image is not None:
            # Reuse the shared base image (overfit test)
            img, points = base_image, base_points
        else:
            img, points = make_image(
                width=width, height=height,
                min_rects=min_rects, max_rects=max_rects,
                rect_w=rect_w, rect_h=rect_h,
                rng=rng,
            )

        img.save(img_path, quality=95)

        with open(ann_path, 'w') as f:
            json.dump({'width': img.width, 'height': img.height, 'points': points}, f)

    print(f'  {split_dir.name}: {n_images} images -> {split_dir}')


def main(
    data_root: str = 'data_toy',
    n_train:   int = 20,
    n_val:     int = 8,
    width:     int = 1280,
    height:    int = 960,
    min_rects: int = 5,
    max_rects: int = 30,
    rect_w:    int = 14,
    rect_h:    int = 10,
    seed:      int = 0,
    one_image: bool = False,  # if True, all splits use the exact same image (overfit smoke-test)
):
    """
    Args:
        data_root:  Root directory to write the dataset into.
        n_train:    Number of training images to generate.
        n_val:      Number of validation images to generate.
        width:      Image width in pixels.
        height:     Image height in pixels.
        min_rects:  Minimum number of rectangles per image.
        max_rects:  Maximum number of rectangles per image.
        rect_w:     Rectangle width in pixels.
        rect_h:     Rectangle height in pixels.
        seed:       Random seed.
        one_image:  Reuse a single image for every sample in every split (overfit test).
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    root = Path(data_root)

    base_image, base_points = None, None
    if one_image:
        base_image, base_points = make_image(
            width=width, height=height,
            min_rects=min_rects, max_rects=max_rects,
            rect_w=rect_w, rect_h=rect_h,
            rng=rng,
        )
        print(f'one_image mode: single image with {len(base_points)} rectangles')

    print(f'Writing toy dataset to {root.resolve()}')
    for split, n in [('train', n_train), ('val', n_val)]:
        write_split(
            split_dir=root / split,
            n_images=n,
            width=width, height=height,
            min_rects=min_rects, max_rects=max_rects,
            rect_w=rect_w, rect_h=rect_h,
            rng=rng,
            base_image=base_image,
            base_points=base_points,
        )

    # Quick sanity check: open the first train image and print count
    first_img = sorted((root / 'train' / 'images').glob('*.jpg'))[0]
    ann_path  = root / 'train' / 'annotations' / (first_img.stem + '.json')
    with open(ann_path) as f:
        ann = json.load(f)
    img_check = Image.open(first_img)
    print(f'\nSample: {first_img.name}  size={img_check.size}  points={len(ann["points"])}')
    print('Done.')


if __name__ == '__main__':
    fire.Fire(main)
