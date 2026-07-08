from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from parking_bev.evaluation import DetectionMetrics, evaluate_detections
from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import (
    DETECTION_CLASSES,
    bevfusion_predictions_from_records,
    detection_class,
    within_detection_range,
)
from parking_bev.tracking import TimestampAwareTracker, prediction_to_global_measurement
from parking_bev.tracking_evaluation import TrackingIdentityEvaluator


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


def _prediction_files(prediction_dir: Path, pattern: str) -> list[Path]:
    files = sorted(prediction_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matched {prediction_dir / pattern}")
    return files


def _evaluate_detection_scene(
    payload: dict,
    source: NuScenesSource,
    score: float,
    distance: float,
) -> tuple[dict, dict, list[dict]]:
    overall_accumulator = _empty_accumulator()
    class_accumulators = {name: _empty_accumulator() for name in DETECTION_CLASSES}
    per_frame = []

    for frame_record in payload["frames"]:
        frame = source.read_token(frame_record["sample_token"])
        predictions = [
            item for item in bevfusion_predictions_from_records(
                frame_record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                score,
            ) if within_detection_range(item.class_name, item.object.center_ego)
        ]
        ground_truth = [
            obj for obj in frame.objects
            if (class_name := detection_class(obj.category)) is not None
            and within_detection_range(class_name, obj.center_ego)
        ]
        overall, per_class = evaluate_detections(predictions, ground_truth, distance)
        _accumulate(overall_accumulator, overall)
        for name, metrics in per_class.items():
            _accumulate(class_accumulators[name], metrics)
        per_frame.append({
            "frame_index": frame_record["frame_index"],
            "sample_token": frame.token,
            "inference_seconds": frame_record.get("inference_seconds"),
            **asdict(overall),
        })

    return (
        _finalize(overall_accumulator),
        {name: _finalize(value) for name, value in class_accumulators.items()},
        per_frame,
    )


def _evaluate_tracking_scene(
    payload: dict,
    source: NuScenesSource,
    score: float,
    distance: float,
    association_distance: float,
    max_missed_seconds: float,
) -> dict:
    tracker = TimestampAwareTracker(
        history_size=10,
        association_distance_m=association_distance,
        max_missed_seconds=max_missed_seconds,
    )
    evaluator = TrackingIdentityEvaluator(distance)

    for frame_record in payload["frames"]:
        frame = source.read_token(frame_record["sample_token"])
        predictions = [
            item for item in bevfusion_predictions_from_records(
                frame_record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                score,
            ) if within_detection_range(item.class_name, item.object.center_ego)
        ]
        measurements = [
            prediction_to_global_measurement(item, frame.ego_to_global)
            for item in predictions
        ]
        snapshots = tracker.update(frame.timestamp_us / 1_000_000.0, measurements)
        confirmed = [snapshot for snapshot in snapshots if snapshot.hits >= 2]
        ground_truth = [
            obj for obj in frame.objects
            if (class_name := detection_class(obj.category)) is not None
            and within_detection_range(class_name, obj.center_ego)
        ]
        evaluator.update(confirmed, ground_truth, frame.ego_to_global)

    return evaluator.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate detection and motion-only tracking diagnostics over BEVFusion scene files")
    parser.add_argument("--prediction-dir", type=Path, default=Path("output/bevfusion_mini/scenes"))
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--score", type=float, default=0.2)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--association-distance", type=float, default=2.0)
    parser.add_argument("--max-missed-seconds", type=float, default=1.2)
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_mini/evaluation_summary.json"))
    args = parser.parse_args()

    files = _prediction_files(args.prediction_dir, args.pattern)
    source = NuScenesSource(args.dataroot, cameras_enabled=False, radar_enabled=False)
    overall_accumulator = _empty_accumulator()
    class_accumulators = {name: _empty_accumulator() for name in DETECTION_CLASSES}
    scene_reports = []
    tracking_reports = []
    inference_times = []

    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        detection_overall, detection_per_class, per_frame = _evaluate_detection_scene(
            payload, source, args.score, args.distance)
        _accumulate(overall_accumulator, DetectionMetrics(**{
            key: detection_overall[key]
            for key in DetectionMetrics.__dataclass_fields__
        }))
        for name, class_metrics in detection_per_class.items():
            _accumulate(class_accumulators[name], DetectionMetrics(**{
                key: class_metrics[key]
                for key in DetectionMetrics.__dataclass_fields__
            }))
        tracking = _evaluate_tracking_scene(
            payload, source, args.score, args.distance,
            args.association_distance, args.max_missed_seconds)
        tracking_reports.append(tracking)
        inference_times.extend([
            item["inference_seconds"]
            for item in per_frame
            if item["inference_seconds"] is not None
        ])
        scene_reports.append({
            "scene_name": payload["scene_name"],
            "scene_index": payload.get("scene_index"),
            "frames": len(payload["frames"]),
            "prediction_file": str(file),
            "detection": detection_overall,
            "tracking": tracking,
        })
        print(f"{payload['scene_name']}: F1={detection_overall['f1']:.3f}, "
              f"IDF1={tracking['id_f1']:.3f}", flush=True)

    tracking_summary = {
        "frames": sum(item["frames"] for item in tracking_reports),
        "unique_ground_truth_instances": sum(item["unique_ground_truth_instances"] for item in tracking_reports),
        "unique_track_ids": sum(item["unique_track_ids"] for item in tracking_reports),
        "ground_truth_detections": sum(item["ground_truth_detections"] for item in tracking_reports),
        "track_outputs": sum(item["track_outputs"] for item in tracking_reports),
        "spatial_matches": sum(item["spatial_matches"] for item in tracking_reports),
        "identity_true_positives": sum(item["identity_true_positives"] for item in tracking_reports),
        "identity_false_positives": sum(item["identity_false_positives"] for item in tracking_reports),
        "identity_false_negatives": sum(item["identity_false_negatives"] for item in tracking_reports),
        "id_switches": sum(item["id_switches"] for item in tracking_reports),
        "fragments": sum(item["fragments"] for item in tracking_reports),
    }
    tracking_summary["detection_coverage"] = (
        tracking_summary["spatial_matches"] / tracking_summary["ground_truth_detections"]
        if tracking_summary["ground_truth_detections"] else 0.0)
    tracking_summary["track_output_precision"] = (
        tracking_summary["spatial_matches"] / tracking_summary["track_outputs"]
        if tracking_summary["track_outputs"] else 0.0)
    tracking_summary["id_precision"] = (
        tracking_summary["identity_true_positives"] / tracking_summary["track_outputs"]
        if tracking_summary["track_outputs"] else 0.0)
    tracking_summary["id_recall"] = (
        tracking_summary["identity_true_positives"] / tracking_summary["ground_truth_detections"]
        if tracking_summary["ground_truth_detections"] else 0.0)
    tracking_summary["id_f1"] = (
        2 * tracking_summary["identity_true_positives"] /
        (2 * tracking_summary["identity_true_positives"] +
         tracking_summary["identity_false_positives"] +
         tracking_summary["identity_false_negatives"])
        if tracking_summary["track_outputs"] + tracking_summary["ground_truth_detections"] else 0.0)
    velocity_errors = [
        item["mean_velocity_error_mps"]
        for item in tracking_reports
        if item["mean_velocity_error_mps"] is not None
    ]
    tracking_summary["mean_scene_velocity_error_mps"] = (
        float(np.mean(velocity_errors)) if velocity_errors else None)
    tracking_summary["scope"] = "sum of per-scene motion-only diagnostics; not official nuScenes AMOTA/AMOTP"

    report = {
        "scope": "nuScenes mini batch diagnostics; not official nuScenes mAP/NDS",
        "prediction_dir": str(args.prediction_dir),
        "files": len(files),
        "score_threshold": args.score,
        "center_distance_threshold_m": args.distance,
        "detection": {
            "overall": _finalize(overall_accumulator),
            "per_class": {name: _finalize(value) for name, value in class_accumulators.items()},
        },
        "tracking": tracking_summary,
        "performance": {
            "frames": len(inference_times),
            "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else None,
            "p95_inference_seconds": float(np.percentile(inference_times, 95)) if inference_times else None,
            "approx_fps": float(1.0 / np.mean(inference_times)) if inference_times else None,
        },
        "scenes": scene_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "files": report["files"],
        "performance": report["performance"],
        "detection": report["detection"]["overall"],
        "tracking": report["tracking"],
    }, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
