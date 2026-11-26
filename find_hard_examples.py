import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import ImageDraw
from scipy.spatial import cKDTree
from point_metrics import match_points, point_metrics
from tqdm import tqdm

from dataset_api import SockeyeApiDataset
from export_annotation_debug import task_bounds
from eval_detr_api import centroid_nms
from generate_review_gallery import write_gallery
from infer_video_detr import load_model, _make_tiles
from train_detr_api import _validation_tile_predictions
from torchvision.ops import nms


def infer_context_crop(model, images, batch_size, threshold, amp):
    image_tiles = [_make_tiles(image, 512, None, target_channels=3, tile_overlap=128) for image in images]
    tiles = []
    for aligned in zip(*image_tiles):
        tensor = torch.cat([tile[0] for tile in aligned], dim=0)
        horizontal, vertical = aligned[0][1:]
        tiles.append((tensor, horizontal, vertical, images[0].width, images[0].height))
    boxes, scores = [], []
    for (horizontal, vertical, _, _), probabilities, predictions in _validation_tile_predictions(model, tiles, batch_size, amp):
        keep = probabilities > threshold
        selected = predictions[keep]
        if len(selected):
            centers = selected[:, :2] * 512 + torch.tensor([horizontal, vertical])
            boxes.append(torch.cat((centers - 25, centers + 25), dim=1))
            scores.append(probabilities[keep])
    if not boxes:
        return []
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    detections = []
    for index in nms(boxes, scores, 0.5).tolist():
        left, top, right, bottom = boxes[index].tolist()
        detections.append([(left + right) / 2, (top + bottom) / 2, left, top, right, bottom, scores[index].item()])
    return detections


def suppress_detections(detections, scale_horizontal, scale_vertical, radius):
    if not detections or radius == 0:
        return detections
    boxes = torch.tensor([[item[2] / scale_horizontal, item[3] / scale_vertical,
                           item[4] / scale_horizontal, item[5] / scale_vertical] for item in detections])
    scores = torch.tensor([item[6] for item in detections])
    return [detections[index] for index in centroid_nms(boxes, scores, radius).tolist()]


def ground_truth_crowding(labels, nms_radius, match_radius):
    count = len(labels)
    if not count:
        return {'gt_close_pairs': 0, 'gt_crowded': 0, 'gt_crowded_fraction': None,
                'gt_nms_min_lost': 0, 'gt_center_recall_cap': None, 'gt_recall_upper_bound': None}
    coordinates = np.asarray(labels, dtype=float)
    neighbors = [set() for _ in labels]
    forced_neighbors = [set() for _ in labels]
    pairs = 0
    if nms_radius > 0:
        for first, second in cKDTree(coordinates).query_pairs(nms_radius):
            distance = np.linalg.norm(coordinates[first] - coordinates[second])
            if distance >= nms_radius:
                continue
            pairs += 1
            neighbors[first].add(second)
            neighbors[second].add(first)
            if distance + 2 * match_radius < nms_radius:
                forced_neighbors[first].add(second)
                forced_neighbors[second].add(first)

    def clique_cover_size(graph):
        remaining = set(range(count))
        groups = 0
        while remaining:
            seed = min(remaining, key=lambda index: (-len(graph[index]), index))
            remaining.remove(seed)
            candidates = graph[seed] & remaining
            while candidates:
                member = min(candidates, key=lambda index: (-len(graph[index]), index))
                remaining.remove(member)
                candidates &= graph[member] & remaining
            groups += 1
        return groups

    center_cap = clique_cover_size(neighbors)
    crowded = sum(bool(adjacent) for adjacent in neighbors)
    return {'gt_close_pairs': pairs, 'gt_crowded': crowded, 'gt_crowded_fraction': crowded / count,
            'gt_nms_min_lost': count - center_cap, 'gt_center_recall_cap': center_cap / count,
            'gt_recall_upper_bound': clique_cover_size(forced_neighbors) / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='debug/annotations/tasks.json')
    parser.add_argument('--checkpoint', default='models/best_without_context/best.pth')
    parser.add_argument('--out', default='debug/hard_examples')
    parser.add_argument('--cache-dir', default='/tmp/frame_cache')
    parser.add_argument('--score-thresh', type=float, default=0.25)
    parser.add_argument('--match-radius', type=float, default=25)
    parser.add_argument('--nms-distance', type=float, default=50, help='Centroid suppression radius in model-input pixels; zero disables it.')
    parser.add_argument('--tile-batch-size', type=int, default=4)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--num-context-frames', type=int, default=0)
    parser.add_argument('--context-stride', type=int, default=20)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Score only the first N tasks; zero selects all tasks.')
    parser.add_argument('--cached-only', action='store_true', help='Render existing predictions without running inference.')
    args = parser.parse_args()
    if args.match_radius <= 0 or args.tile_batch_size < 1 or args.top < 1:
        raise ValueError('Radius, tile batch size, and top must be positive.')
    if args.limit < 0:
        raise ValueError('Task limit must be nonnegative.')
    if args.nms_distance < 0:
        raise ValueError('NMS distance must be nonnegative.')
    load_dotenv(Path(__file__).with_name('.env'))
    tasks_path = Path(args.tasks)
    tasks = json.loads(tasks_path.read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    settings = {'checkpoint': str(checkpoint), 'checkpoint_size': checkpoint.stat().st_size,
                'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
                'tasks_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
                'score_thresh': args.score_thresh, 'match_radius': args.match_radius,
                'img_size': 512, 'tile_overlap': 128, 'nms_iou': 0.5, 'nms_distance': args.nms_distance,
                'num_context_frames': args.num_context_frames, 'context_stride': args.context_stride, 'amp': args.amp}
    settings_path = output / 'settings.json'
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if {key: value for key, value in previous.items() if key != 'nms_distance'} != {key: value for key, value in settings.items() if key != 'nms_distance'}:
            raise ValueError('Existing output uses different settings. Choose a new --out directory.')
    settings_path.write_text(json.dumps(settings, indent=2))
    api_key = os.environ.get('SOCKEYE_API_KEY')
    if not api_key:
        raise ValueError('SOCKEYE_API_KEY is required for uncached frames.')
    dataset = SockeyeApiDataset(os.environ.get('SOCKEYE_API_URL', 'https://salmon.projectkiwi.io/api/v1'),
                               api_key, tasks, cache_dir=args.cache_dir, num_context_frames=0, mode='detr')
    prediction_file = output / 'predictions.jsonl'
    cached = {}
    if prediction_file.exists():
        for line in prediction_file.read_text().splitlines():
            record = json.loads(line)
            cached[record['index']] = record['detections']
    model = None
    rows, details, failures = [], {}, []
    selected_tasks = list(enumerate(tasks, 1))
    if args.limit:
        selected_tasks = selected_tasks[:args.limit]
    if args.cached_only:
        selected_tasks = [(index, task) for index, task in selected_tasks if index in cached]
    if not selected_tasks:
        raise ValueError('No tasks selected; cached-only mode requires saved predictions.')
    with prediction_file.open('a') as predictions_log:
        for index, task in tqdm(selected_tasks, desc='Hard-example audit'):
            try:
                image = dataset._get_frame(task['fileId'], task['frame'])
                left, top, right, bottom = task_bounds(task, image.size)
                crop = image.crop((left, top, right, bottom))
                labels = [[point['x'] - left, point['y'] - top] for point in task.get('points', [])]
                resized = crop.width < 512 or crop.height < 512
                scale_horizontal = crop.width / 512 if resized else 1
                scale_vertical = crop.height / 512 if resized else 1
                if index in cached:
                    detections = cached[index]
                else:
                    if model is None:
                        model = load_model(str(checkpoint), 100, None)
                        channels = next(module.in_channels for module in model.modules() if isinstance(module, torch.nn.Conv2d))
                        if channels != 3 * (1 + args.num_context_frames):
                            raise ValueError('Checkpoint channels do not match --num-context-frames.')
                    images = [crop]
                    for context_index in range(1, args.num_context_frames + 1):
                        context_frame = max(0, task['frame'] - context_index * args.context_stride)
                        context_image = dataset._get_frame(task['fileId'], context_frame)
                        if context_image.size != image.size:
                            raise ValueError('Context frame dimensions differ from target frame.')
                        images.append(context_image.crop((left, top, right, bottom)))
                    if resized:
                        images = [item.resize((512, 512)) for item in images]
                    with torch.inference_mode():
                        detections = infer_context_crop(model, images, args.tile_batch_size, args.score_thresh, args.amp)
                    detections = [[horizontal * scale_horizontal, vertical * scale_vertical,
                                   bx1 * scale_horizontal, by1 * scale_vertical, bx2 * scale_horizontal,
                                   by2 * scale_vertical, score]
                                  for horizontal, vertical, bx1, by1, bx2, by2, score in detections]
                    predictions_log.write(json.dumps({'index': index, 'detections': detections}) + '\n')
                    predictions_log.flush()
                matching_labels = [[horizontal / scale_horizontal, vertical / scale_vertical] for horizontal, vertical in labels]
                detections = suppress_detections(detections, scale_horizontal, scale_vertical, args.nms_distance)
                matching_predictions = [[detection[0] / scale_horizontal, detection[1] / scale_vertical] for detection in detections]
                metrics = point_metrics(matching_labels, matching_predictions, [item[6] for item in detections], args.match_radius)
                crowding = ground_truth_crowding(matching_labels, args.nms_distance, args.match_radius)
                matches, missed, extra = match_points(matching_labels, matching_predictions, args.match_radius)
                mismatch = len(missed) + len(extra)
                rows.append({'index': index, 'task_id': task.get('id', index), 'file_id': task['fileId'],
                             'frame': task['frame'], 'labels': len(labels), 'predictions': len(detections),
                             'matched': len(matches), 'missed_labels': len(missed), 'extra_predictions': len(extra), **metrics, **crowding,
                             'high_confidence_extras': sum(detections[number][6] >= 0.75 for number in extra),
                             'count_error': abs(len(labels) - len(detections)), 'disagreement': mismatch,
                             'disagreement_fraction': mismatch / max(len(labels) + len(detections), 1),
                             'review_score': mismatch / max(len(labels) + len(detections), 1) * np.log1p(mismatch)})
                details[index] = (labels, detections, missed, extra)
            except Exception as error:
                failures.append({'index': index, 'error': str(error)})
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    break
    rankings = {'ranked': sorted(rows, key=lambda row: row['review_score'], reverse=True),
                'absolute_disagreement': sorted(rows, key=lambda row: row['disagreement'], reverse=True),
                'empty_with_detections': sorted([row for row in rows if row['labels'] == 0 and row['predictions']],
                                               key=lambda row: (row['high_confidence_extras'], row['predictions']), reverse=True)}
    for name, ranking in rankings.items():
        if ranking:
            with (output / f'{name}.csv').open('w', newline='') as table:
                writer = csv.DictWriter(table, fieldnames=list(ranking[0]))
                writer.writeheader()
                writer.writerows(ranking)
    review = output / 'review'
    review.mkdir(exist_ok=True)
    for rank, row in enumerate(rankings['ranked'][:args.top], 1):
        task = tasks[row['index'] - 1]
        image = dataset._get_frame(task['fileId'], task['frame'])
        image = image.crop(task_bounds(task, image.size)).copy()
        labels, detections, missed, extra = details[row['index']]
        draw = ImageDraw.Draw(image)
        for number, (horizontal, vertical) in enumerate(labels):
            color = 'red' if number in missed else 'lime'
            draw.ellipse((horizontal-7, vertical-7, horizontal+7, vertical+7), outline=color, width=3)
        for number, detection in enumerate(detections):
            color = 'cyan' if number in extra else 'lime'
            horizontal, vertical = detection[:2]
            draw.ellipse((horizontal-3, vertical-3, horizontal+3, vertical+3), fill=color)
        draw.text((10, 10), f"Rank {rank} | labels {row['labels']} | predictions {row['predictions']} | missed {row['missed_labels']} | extra {row['extra_predictions']}",
                  fill='white', stroke_width=2, stroke_fill='black')
        image.save(review / f"rank_{rank:03d}_task_{row['index']:05d}.jpg", quality=95)
    write_gallery(output, rankings['ranked'], args.top)
    summary = {'settings': settings, 'total_tasks': len(tasks), 'selected_tasks': len(selected_tasks),
               'limit': args.limit, 'cached_only': args.cached_only, 'scored_tasks': len(rows), 'failures': failures,
               'empty_with_detections': len(rankings['empty_with_detections']), 'top': rankings['ranked'][:10]}
    labeled_rows = [row for row in rows if row['labels']]
    summary['mean_ap'] = float(np.mean([row['ap'] for row in labeled_rows])) if labeled_rows else None
    summary['mean_ar'] = float(np.mean([row['ar'] for row in labeled_rows])) if labeled_rows else None
    summary['mean_count_error_fraction'] = float(np.mean([row['count_error_fraction'] for row in labeled_rows])) if labeled_rows else None
    summary['count_error_definition'] = 'Absolute prediction count minus GT count, divided by GT count. Multiply by 100 for percent; zero GT yields null. Mean excludes zero-GT tasks.'
    summary['crowding_definition'] = 'GT pairs strictly closer than NMS in model pixels. Disjoint greedy clique cover bounds ideal-center recall/AP from above; minimum lost is a lower bound, not an exact optimum. Tolerance-aware recall bound uses conflicts with distance + 2 * match_radius < NMS; it can be 100% even for crowded GT.'
    summary['metric_definition'] = f'Per-task point AP and maximum recall at {args.match_radius}px, scores > {args.score_thresh}, distance NMS {args.nms_distance}px; means exclude empty ground truth. Not COCO AP/AR.'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()