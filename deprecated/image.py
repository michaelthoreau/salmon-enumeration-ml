import json
import numpy as np
from PIL import Image


def load_data(img_path, args, train=True):
    """Load image and kpoint map from JPEG + JSON annotation."""
    # annotations live alongside images at ../annotations/<uuid>.json
    ann_path = img_path.replace('/images/', '/annotations/').replace('.jpg', '.json')

    img = Image.open(img_path).convert('RGB')
    with open(ann_path) as f:
        ann = json.load(f)

    width, height = ann['width'], ann['height']
    k = np.zeros((height, width), dtype=np.float32)
    for x, y in ann['points']:
        if 0 <= x < width and 0 <= y < height:
            k[y, x] += 1

    return img, k


def load_data_test(img_path, args, train=True):
    img = Image.open(img_path).convert('RGB')
    return img

