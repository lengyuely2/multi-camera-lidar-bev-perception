from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .nuscenes_source import Object3D
from .predictions import DETECTION_CLASSES, Prediction3D, detection_class


@dataclass(frozen=True)
class DetectionMetrics:
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_center_error_m: float | None
    mean_yaw_error_deg: float | None


def evaluate_detections(
    predictions: list[Prediction3D],
    ground_truth: list[Object3D],
    center_threshold_m: float = 2.0,
) -> tuple[DetectionMetrics, dict[str, DetectionMetrics]]:
    per_class: dict[str, DetectionMetrics] = {}
    all_center_errors: list[float] = []
    all_yaw_errors: list[float] = []

    for class_name in DETECTION_CLASSES:
        predicted = [item for item in predictions if item.class_name == class_name]
        expected = [item for item in ground_truth if detection_class(item.category) == class_name]
        metrics, center_errors, yaw_errors = _evaluate_class(predicted, expected, center_threshold_m)
        per_class[class_name] = metrics
        all_center_errors.extend(center_errors)
        all_yaw_errors.extend(yaw_errors)

    tp = sum(item.true_positives for item in per_class.values())
    fp = sum(item.false_positives for item in per_class.values())
    fn = sum(item.false_negatives for item in per_class.values())
    overall = _metrics(
        ground_truth=sum(item.ground_truth for item in per_class.values()),
        predictions=sum(item.predictions for item in per_class.values()),
        tp=tp,
        fp=fp,
        fn=fn,
        center_errors=all_center_errors,
        yaw_errors=all_yaw_errors,
    )
    return overall, per_class


def _evaluate_class(
    predictions: list[Prediction3D],
    ground_truth: list[Object3D],
    threshold: float,
) -> tuple[DetectionMetrics, list[float], list[float]]:
    matches: list[tuple[int, int, float]] = []
    if predictions and ground_truth:
        predicted_xy = np.asarray([item.object.center_ego[:2] for item in predictions])
        expected_xy = np.asarray([item.center_ego[:2] for item in ground_truth])
        distances = np.linalg.norm(predicted_xy[:, None, :] - expected_xy[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        matches = [(int(row), int(column), float(distances[row, column]))
                   for row, column in zip(rows, columns) if distances[row, column] <= threshold]

    center_errors = [distance for _, _, distance in matches]
    yaw_errors = [
        _yaw_error(predictions[row].object.yaw_ego, ground_truth[column].yaw_ego)
        for row, column, _ in matches
    ]
    tp = len(matches)
    metrics = _metrics(len(ground_truth), len(predictions), tp, len(predictions) - tp,
                       len(ground_truth) - tp, center_errors, yaw_errors)
    return metrics, center_errors, yaw_errors


def _metrics(
    ground_truth: int,
    predictions: int,
    tp: int,
    fp: int,
    fn: int,
    center_errors: list[float],
    yaw_errors: list[float],
) -> DetectionMetrics:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        ground_truth, predictions, tp, fp, fn, precision, recall, f1,
        float(np.mean(center_errors)) if center_errors else None,
        float(np.degrees(np.mean(yaw_errors))) if yaw_errors else None,
    )


def _yaw_error(first: float, second: float) -> float:
    return float(abs((first - second + np.pi) % (2 * np.pi) - np.pi))
