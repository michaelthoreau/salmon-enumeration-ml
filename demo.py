import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from infer_video_detr import Pipeline, _infer_batch, load_model


def annotate(frame, detections):
    output = frame.copy()
    for detection in detections:
        cv2.circle(output, (round(detection['x']), round(detection['y'])), 5, (0, 255, 255), 2)
    cv2.putText(output, f'Fish: {len(detections)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 5)
    cv2.putText(output, f'Fish: {len(detections)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the released salmon detector on an image or video.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, help='Default: <input>_annotated beside the input.')
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent / 'release/salmon-detr/model.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.35, help='Detection and tracking confidence (default: 0.35).')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--detect-height', type=int, default=1440,
                        help='Downscale taller inputs before inference (default: 1440; 0 disables).')
    parser.add_argument('--frame-stride', type=int, default=3, help='Infer every Nth video frame (default: 3).')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit decoded source frames; 0 means the whole video.')
    parser.add_argument('--start-sec', type=float, default=0.0, help='Start at this video time in seconds.')
    parser.add_argument('--duration-sec', type=float, help='Process this many seconds of source video.')
    args = parser.parse_args(argv)
    if args.detect_height < 0:
        parser.error('detect-height must be nonnegative; 0 disables resizing.')
    if args.batch_size < 1 or args.frame_stride < 1 or args.max_frames < 0 or not 0 <= args.threshold <= 1:
        parser.error('batch-size and frame-stride must be positive; max-frames nonnegative; threshold between 0 and 1.')
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        parser.error('start-sec must be finite and nonnegative.')
    if args.duration_sec is not None and (not math.isfinite(args.duration_sec) or args.duration_sec <= 0):
        parser.error('duration-sec must be finite and positive.')
    if not args.input.is_file():
        parser.error(f'Input does not exist: {args.input}')
    is_image = args.input.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    if is_image and (args.start_sec != 0 or args.duration_sec is not None):
        parser.error('start-sec and duration-sec apply only to video inputs.')
    if args.output is None:
        args.output = args.input.with_name(args.input.stem + ('_annotated.png' if is_image else '_annotated.mp4'))
    if args.input.resolve() in {args.output.resolve(), args.output.with_suffix('.json').resolve()}:
        parser.error('Output must not overwrite the input.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if is_image:
        if args.output.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            parser.error('Image output must be JPG or PNG.')
        image = Image.open(args.input).convert('RGB')
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        inference_frame = frame
        if args.detect_height and frame.shape[0] > args.detect_height:
            width = max(1, round(frame.shape[1] * args.detect_height / frame.shape[0]))
            inference_frame = cv2.resize(frame, (width, args.detect_height), interpolation=cv2.INTER_AREA)
        model = load_model(str(args.checkpoint), 100, None, device=args.device, strict=True)
        results = _infer_batch(model, [(inference_frame, None)], 512, args.threshold, 0.5,
                       tile_batch_size=args.batch_size, zero_context=True)[0]
        scale_horizontal = frame.shape[1] / inference_frame.shape[1]
        scale_vertical = frame.shape[0] / inference_frame.shape[0]
        detections = [{'x': result[0] * scale_horizontal, 'y': result[1] * scale_vertical,
                       'score': result[6]} for result in results]
        if not cv2.imwrite(str(args.output), annotate(frame, detections)):
            raise RuntimeError('Could not write image output.')
        records.append({'frame': 0, 'count': len(detections), 'detections': detections})
    else:
        if args.output.suffix.lower() != '.mp4':
            parser.error('Video output must be MP4.')
        Pipeline().run_all(checkpoint=str(args.checkpoint), video=str(args.input), output=str(args.output),
                           start_sec=args.start_sec, duration_sec=args.duration_sec,
                           score_thresh=args.threshold, track_thresh=args.threshold,
                           infer_every=args.frame_stride, tile_batch_size=args.batch_size,
                           companion_mode='temporal', context_stride=20,
                           detect_height=args.detect_height or None,
                           device=args.device, max_frames=args.max_frames, strict=True)
        print(f'Saved {args.output}')
        return
    args.output.with_suffix('.json').write_text(json.dumps({'input': str(args.input), 'threshold': args.threshold,
                                                          'frame_stride': args.frame_stride,
                                                          'frames': records}, indent=2))
    print(f'Saved {args.output} and {args.output.with_suffix(".json")}')


if __name__ == '__main__':
    main()