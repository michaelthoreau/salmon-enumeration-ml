import argparse
import csv
import json
from pathlib import Path


def write_gallery(output, ranking, top):
    output = Path(output)
    examples = []
    for rank, row in enumerate(ranking[:top], 1):
        image = f"review/rank_{rank:03d}_task_{int(row['index']):05d}.jpg"
        if (output / image).is_file():
            example = dict(row)
            for field in ('index', 'frame', 'labels', 'predictions', 'missed_labels', 'extra_predictions', 'count_error'):
                example[field] = int(example[field])
            example['review_score'] = float(example['review_score'])
            example.pop('rmse', None)
            example['count_error_fraction'] = abs(example['predictions'] - example['labels']) / example['labels'] if example['labels'] else None
            for field in ('ap', 'ar', 'precision', 'count_error_fraction', 'gt_close_pairs', 'gt_crowded', 'gt_crowded_fraction',
                          'gt_nms_min_lost', 'gt_center_recall_cap', 'gt_recall_upper_bound'):
                example[field] = float(example[field]) if example.get(field) not in (None, '') else None
            example['image'] = image
            examples.append(example)
    template = Path(__file__).with_name('review_gallery.html').read_text()
    settings_path = output / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    labeled = [row for row in ranking if row.get('ap') not in (None, '')]
    count_errors = [abs(int(row['predictions']) - int(row['labels'])) / int(row['labels'])
                    for row in ranking if int(row['labels']) > 0]
    metadata = {'mean_ap': sum(float(row['ap']) for row in labeled) / len(labeled) if labeled else None,
                'mean_ar': sum(float(row['ar']) for row in labeled) / len(labeled) if labeled else None,
                'mean_count_error_fraction': sum(count_errors) / len(count_errors) if count_errors else None,
                'scored': len(ranking), 'radius': settings.get('match_radius', 25),
                'threshold': settings.get('score_thresh', .25), 'nms': settings.get('nms_distance', 0)}
    data = json.dumps(examples).replace('<', '\\u003c')
    destination = output / 'index.html'
    destination.write_text(template.replace('__REVIEW_DATA__', data).replace('__REVIEW_METADATA__', json.dumps(metadata)))
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with (args.output / 'ranked.csv').open(newline='') as table:
        ranking = list(csv.DictReader(table))
    print(write_gallery(args.output, ranking, len(ranking)))


if __name__ == '__main__':
    main()