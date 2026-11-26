import argparse
import csv
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import ImageDraw
from tqdm import tqdm
import torch

from dataset_api import SockeyeApiDataset

TRAINING_SAMPLE_LOCK = threading.Lock()


def annotation_boxes(points, bounds, img_size=512, box_size=50):
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    resized = width < img_size or height < img_size
    scale_horizontal = width / img_size if resized else 1.0
    scale_vertical = height / img_size if resized else 1.0
    half_box = box_size // 2
    boxes = []
    for point in points:
        horizontal, vertical = point["x"], point["y"]
        box = (max(left, horizontal - half_box * scale_horizontal),
               max(top, vertical - half_box * scale_vertical),
               min(right, horizontal + half_box * scale_horizontal),
               min(bottom, vertical + half_box * scale_vertical))
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
    return boxes


def draw_boxes(image, boxes, offset=(0, 0)):
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom in boxes:
        draw.rectangle((left - offset[0], top - offset[1], right - offset[0], bottom - offset[1]),
                       outline="lime", width=2)


def training_preview(dataset, task_index):
    with TRAINING_SAMPLE_LOCK:
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(task_index)
            torch.manual_seed(task_index)
            (image, _), target = dataset[task_index]
        finally:
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)
    image = image.copy()
    boxes = []
    for horizontal, vertical, width, height in target["boxes"].tolist():
        boxes.append(((horizontal - width / 2) * image.width,
                      (vertical - height / 2) * image.height,
                      (horizontal + width / 2) * image.width,
                      (vertical + height / 2) * image.height))
    draw_boxes(image, boxes)
    return image, len(boxes)

def draw_points(image, points, offset=(0, 0)):
    draw = ImageDraw.Draw(image)
    for number, point in enumerate(points, 1):
        horizontal = point["x"] - offset[0]
        vertical = point["y"] - offset[1]
        draw.ellipse((horizontal - 7, vertical - 7, horizontal + 7, vertical + 7),
                     outline="yellow", width=3)
        draw.line((horizontal - 10, vertical, horizontal + 10, vertical), fill="red", width=2)
        draw.line((horizontal, vertical - 10, horizontal, vertical + 10), fill="red", width=2)
        draw.text((horizontal + 9, vertical - 9), str(number), fill="white", stroke_width=2, stroke_fill="black")


def task_bounds(task, image_size):
    bounds = tuple(task[name] for name in ("x1", "y1", "x2", "y2"))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return (0, 0, *image_size)
    return bounds


def export_frame(dataset, training_dataset, output, file_id, frame, numbered_tasks):
    image = dataset._get_frame(file_id, frame)
    full_overlay = image.copy()
    full_draw = ImageDraw.Draw(full_overlay)
    records = []
    for index, task in numbered_tasks:
        bounds = task_bounds(task, image.size)
        left, top, right, bottom = bounds
        points = task.get("points", [])
        crop = image.crop(bounds)
        boxes = annotation_boxes(points, bounds, dataset.img_size, dataset.box_size)
        draw_boxes(crop, boxes, (left, top))
        draw_points(crop, points, (left, top))
        identifier = str(task.get("id", task.get("_id", index)))
        filename = f"task_{index:05d}_frame_{frame:06d}_count_{len(points):04d}.jpg"
        crop.save(output / "crops" / filename, quality=95)
        preview, training_box_count = training_preview(training_dataset, index - 1)
        preview.save(output / "training_inputs" / filename, quality=95)
        full_draw.rectangle(bounds, outline="cyan", width=3)
        full_draw.text((left + 4, top + 4), f"task {index}: {len(points)} points",
                       fill="cyan", stroke_width=2, stroke_fill="black")
        draw_points(full_overlay, points)
        draw_boxes(full_overlay, boxes)
        outside_crop = sum(not (left <= point["x"] < right and top <= point["y"] < bottom) for point in points)
        outside_image = sum(not (0 <= point["x"] < image.width and 0 <= point["y"] < image.height) for point in points)
        records.append({
            "index": index, "task_id": identifier, "file_id": file_id, "frame": frame,
            "flight_id": task.get("flightId", ""), "count": len(points),
            "bounds": list(bounds), "image_size": list(image.size),
            "outside_crop": outside_crop, "outside_image": outside_image,
            "crop_outside_image": left < 0 or top < 0 or right > image.width or bottom > image.height,
            "crop": f"crops/{filename}",
            "training_input": f"training_inputs/{filename}",
            "training_box_count": training_box_count,
            "box_size": dataset.box_size, "img_size": dataset.img_size,
            "source_box_mode": "resized" if crop.width < dataset.img_size or crop.height < dataset.img_size else "native_random_crop",
        })
    frame_filename = f"frame_{numbered_tasks[0][0]:05d}_{frame:06d}.jpg"
    full_overlay.save(output / "frames" / frame_filename, quality=95)
    for record in records:
        record["full_frame"] = f"frames/{frame_filename}"
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="debug/annotations")
    parser.add_argument("--cache-dir", default="/tmp/frame_cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--box-size", type=int, default=50)
    parser.add_argument("--tasks-json", help="Reuse an exported task snapshot instead of fetching tasks")
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    api_url = os.environ.get("SOCKEYE_API_URL", "https://salmon.projectkiwi.io/api/v1").rstrip("/")
    api_key = os.environ.get("SOCKEYE_API_KEY")
    if not api_key:
        raise ValueError("SOCKEYE_API_KEY is required.")
    output = Path(args.out)
    for directory in (output, output / "crops", output / "frames", output / "training_inputs"):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    if args.tasks_json:
        with Path(args.tasks_json).open() as task_file:
            tasks = json.load(task_file)
    while not args.tasks_json:
        response = requests.get(f"{api_url}/labeling-tasks", params={"completed": "true", "limit": 500, "offset": len(tasks)},
                                headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        tasks.extend(page)
    with (output / "tasks.json").open("w") as task_file:
        json.dump(tasks, task_file, indent=2)
    grouped = defaultdict(list)
    for index, task in enumerate(tasks, 1):
        grouped[(task["fileId"], task["frame"])].append((index, task))
    dataset = SockeyeApiDataset(api_url, api_key, tasks, cache_dir=args.cache_dir,
                               img_size=args.img_size, box_size=args.box_size,
                               augment=False, num_context_frames=0, mode="detr")
    training_dataset = SockeyeApiDataset(api_url, api_key, tasks, cache_dir=args.cache_dir,
                                        img_size=args.img_size, box_size=args.box_size,
                                        augment=True, num_context_frames=0, mode="detr")
    records, failures = [], []
    print(f"Exporting {len(tasks)} completed tasks from {len(grouped)} frames to {output}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_frame, dataset, training_dataset, output, file_id, frame, entries): (file_id, frame)
                   for (file_id, frame), entries in grouped.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Annotation frames"):
            try:
                records.extend(future.result())
            except Exception as error:
                file_id, frame = futures[future]
                failures.append({"file_id": file_id, "frame": frame, "error": type(error).__name__})
    records.sort(key=lambda record: record["index"])
    with (output / "manifest.json").open("w") as manifest:
        json.dump({"total_tasks": len(tasks), "exported_tasks": len(records), "frames": len(grouped),
                   "failures": failures, "tasks": records}, manifest, indent=2)
    if records:
        with (output / "manifest.csv").open("w", newline="") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"Exported {len(records)}/{len(tasks)} crops; failed frames: {len(failures)}; "
          f"tasks with out-of-crop points: {sum(record['outside_crop'] > 0 for record in records)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()