from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.evaluation import DetectionMetrics, evaluate_detections
from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import (
    DETECTION_CLASSES,
    bevfusion_predictions_from_records,
    detection_class,
    within_detection_range,
)


def _empty_accumulator() -> dict:
    return {
        "ground_truth": 0, "predictions": 0, "true_positives": 0,
        "false_positives": 0, "false_negatives": 0,
        "center_error_sum": 0.0, "center_error_count": 0,
        "yaw_error_sum": 0.0, "yaw_error_count": 0,
    }


def _accumulate(target: dict, metrics: DetectionMetrics) -> None:
    for key in ("ground_truth", "predictions", "true_positives", "false_positives", "false_negatives"):
        target[key] += getattr(metrics, key)
    if metrics.mean_center_error_m is not None:
        target["center_error_sum"] += metrics.mean_center_error_m * metrics.true_positives
        target["center_error_count"] += metrics.true_positives
    if metrics.mean_yaw_error_deg is not None:
        target["yaw_error_sum"] += metrics.mean_yaw_error_deg * metrics.true_positives
        target["yaw_error_count"] += metrics.true_positives


def _finalize(accumulator: dict) -> dict:
    tp, fp, fn = (accumulator[key] for key in ("true_positives", "false_positives", "false_negatives"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        key: accumulator[key]
        for key in ("ground_truth", "predictions", "true_positives", "false_positives", "false_negatives")
    } | {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_center_error_m": accumulator["center_error_sum"] / accumulator["center_error_count"]
        if accumulator["center_error_count"] else None,
        "mean_yaw_error_deg": accumulator["yaw_error_sum"] / accumulator["yaw_error_count"]
        if accumulator["yaw_error_count"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and render one complete BEVFusion scene")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_scene_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--score", type=float, default=0.2)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--report", type=Path, default=Path("output/bevfusion_scene_evaluation.json"))
    parser.add_argument("--video", type=Path, default=Path("output/bevfusion_scene_evaluation.mp4"))
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(args.dataroot, cameras_enabled=False, radar_enabled=False)
    renderer = MetricBEVRenderer()
    overall_accumulator = _empty_accumulator()
    class_accumulators = {name: _empty_accumulator() for name in DETECTION_CLASSES}
    per_frame = []
    evaluation_frames = []

    args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1600, 800))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.video}")

    try:
        for frame_record in payload["frames"]:
            frame = source.read_token(frame_record["sample_token"])
            all_predictions = [item for item in bevfusion_predictions_from_records(
                frame_record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                0.0,
            ) if within_detection_range(item.class_name, item.object.center_ego)]
            predictions = [item for item in all_predictions if item.score >= args.score]
            ground_truth = [
                obj for obj in frame.objects
                if (class_name := detection_class(obj.category)) is not None
                and within_detection_range(class_name, obj.center_ego)
            ]
            overall, per_class = evaluate_detections(predictions, ground_truth, args.distance)
            evaluation_frames.append((all_predictions, ground_truth))
            _accumulate(overall_accumulator, overall)
            for name, metrics in per_class.items():
                _accumulate(class_accumulators[name], metrics)
            per_frame.append({
                "frame_index": frame_record["frame_index"],
                "sample_token": frame.token,
                "inference_seconds": frame_record["inference_seconds"],
                **asdict(overall),
            })

            gt_image = renderer.render(replace(frame, radar_ego={}, objects=tuple(ground_truth)))
            pred_image = renderer.render(replace(
                frame, radar_ego={}, objects=tuple(item.object for item in predictions)))
            cv2.putText(gt_image, "GROUND TRUTH", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (80, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(pred_image, "BEVFUSION PREDICTIONS", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 240, 255), 2, cv2.LINE_AA)
            cv2.putText(pred_image, f"Frame {frame_record['frame_index']:02d}  F1 {overall.f1:.2f}",
                        (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(np.hstack((gt_image, pred_image)))
    finally:
        writer.release()

    threshold_curve = []
    for threshold in np.arange(0.1, 0.91, 0.1):
        accumulator = _empty_accumulator()
        for all_predictions, ground_truth in evaluation_frames:
            selected = [item for item in all_predictions if item.score >= threshold]
            metrics, _ = evaluate_detections(selected, ground_truth, args.distance)
            _accumulate(accumulator, metrics)
        threshold_curve.append({"score_threshold": round(float(threshold), 2), **_finalize(accumulator)})
    best_threshold = max(threshold_curve, key=lambda item: item["f1"])

    report = {
        "scope": "one complete nuScenes mini scene; not official nuScenes mAP/NDS",
        "scene_name": payload["scene_name"],
        "frames": len(per_frame),
        "score_threshold": args.score,
        "center_distance_threshold_m": args.distance,
        "performance": payload.get("summary", {}),
        "overall": _finalize(overall_accumulator),
        "per_class": {name: _finalize(value) for name, value in class_accumulators.items()},
        "threshold_curve": threshold_curve,
        "best_scene_threshold": best_threshold,
        "per_frame": per_frame,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"scene_name": report["scene_name"], "frames": report["frames"],
                      "performance": report["performance"], "overall": report["overall"]}, indent=2))
    print(f"Report: {args.report.resolve()}")
    print(f"Video: {args.video.resolve()}")


if __name__ == "__main__":
    main()
