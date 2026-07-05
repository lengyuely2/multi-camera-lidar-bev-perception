from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.evaluation import evaluate_detections
from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import detection_class, load_bevfusion_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one BEVFusion frame against nuScenes ground truth")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--score", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=Path("output/bevfusion_evaluation.json"))
    parser.add_argument("--image", type=Path, default=Path("output/bevfusion_evaluation.jpg"))
    args = parser.parse_args()

    source = NuScenesSource(args.dataroot, radar_enabled=False)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("No nuScenes sample found")

    lidar_to_ego = frame.calibrations["LIDAR_TOP"].sensor_to_ego
    predictions = load_bevfusion_predictions(args.predictions, lidar_to_ego, args.score)
    ground_truth = [
        obj for obj in frame.objects
        if detection_class(obj.category) is not None
        and -54 <= obj.center_ego[0] < 54
        and -54 <= obj.center_ego[1] < 54
        and -5 <= obj.center_ego[2] < 3
    ]
    overall, per_class = evaluate_detections(predictions, ground_truth, args.distance)
    threshold_curve = []
    all_predictions = load_bevfusion_predictions(args.predictions, lidar_to_ego, 0.0)
    for threshold in np.arange(0.1, 0.91, 0.1):
        selected = [item for item in all_predictions if item.score >= threshold]
        threshold_metrics, _ = evaluate_detections(selected, ground_truth, args.distance)
        threshold_curve.append({"score_threshold": round(float(threshold), 2), **asdict(threshold_metrics)})
    best_threshold = max(threshold_curve, key=lambda item: item["f1"])

    report = {
        "scope": "single-frame diagnostic; not official nuScenes mAP",
        "center_distance_threshold_m": args.distance,
        "score_threshold": args.score,
        "overall": asdict(overall),
        "per_class": {name: asdict(metrics) for name, metrics in per_class.items()},
        "threshold_curve": threshold_curve,
        "best_single_frame_threshold": best_threshold,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    renderer = MetricBEVRenderer()
    ground_truth_image = renderer.render(replace(frame, radar_ego={}, objects=tuple(ground_truth)))
    prediction_image = renderer.render(replace(
        frame, radar_ego={}, objects=tuple(item.object for item in predictions)))
    cv2.putText(ground_truth_image, "GROUND TRUTH", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(prediction_image, "BEVFUSION PREDICTIONS", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (60, 240, 255), 2, cv2.LINE_AA)
    comparison = np.hstack((ground_truth_image, prediction_image))
    if not cv2.imwrite(str(args.image), comparison):
        raise RuntimeError(f"Could not write {args.image}")

    summary = asdict(overall)
    summary["score_threshold"] = args.score
    summary["best_single_frame_score_threshold"] = best_threshold["score_threshold"]
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.report.resolve()}")
    print(f"Image: {args.image.resolve()}")


if __name__ == "__main__":
    main()
