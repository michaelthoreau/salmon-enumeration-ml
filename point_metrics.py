import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_points(labels, predictions, radius):
    if not labels or not predictions:
        return [], list(range(len(labels))), list(range(len(predictions)))
    distances = cdist(labels, predictions)
    penalty = (min(len(labels), len(predictions)) + 1) * (radius + 1)
    costs = np.where(distances <= radius, distances, penalty)
    label_indices, prediction_indices = linear_sum_assignment(costs)
    matches = [(int(label), int(prediction)) for label, prediction in zip(label_indices, prediction_indices)
               if distances[label, prediction] <= radius]
    matched_labels = {label for label, _ in matches}
    matched_predictions = {prediction for _, prediction in matches}
    return matches, [index for index in range(len(labels)) if index not in matched_labels], [index for index in range(len(predictions)) if index not in matched_predictions]


def point_metrics(labels, predictions, scores, radius):
    matches, _, _ = match_points(labels, predictions, radius)
    count_error_fraction = abs(len(predictions) - len(labels)) / len(labels) if labels else None
    precision = len(matches) / len(predictions) if predictions else 0.0
    if not labels:
        return {'ap': None, 'ar': None, 'precision': precision, 'count_error_fraction': count_error_fraction}
    remaining = set(range(len(labels)))
    true_positives = []
    for index in sorted(range(len(predictions)), key=lambda number: scores[number], reverse=True):
        nearest = min(remaining, key=lambda number: np.linalg.norm(np.array(labels[number]) - predictions[index]), default=None)
        matched = nearest is not None and np.linalg.norm(np.array(labels[nearest]) - predictions[index]) <= radius
        true_positives.append(int(matched))
        if matched:
            remaining.remove(nearest)
    cumulative = np.cumsum(true_positives)
    recalls = np.concatenate(([0.0], cumulative / len(labels), [1.0]))
    precisions = np.concatenate(([0.0], cumulative / np.arange(1, len(predictions) + 1), [0.0]))
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = float(np.sum(np.diff(recalls) * precisions[1:]))
    return {'ap': ap, 'ar': len(matches) / len(labels), 'precision': precision, 'count_error_fraction': count_error_fraction}