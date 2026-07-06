from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import (
    bevfusion_predictions_from_records,
    detection_class,
    within_detection_range,
)
from parking_bev.tracking import TimestampAwareTracker, prediction_to_global_measurement
from parking_bev.tracking_evaluation import TrackingIdentityEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate timestamp-aware tracking IDs")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_scene_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--score", type=float, default=0.2)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--association-distance", type=float, default=2.0)
    parser.add_argument("--max-missed-seconds", type=float, default=1.2)
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_tracking_evaluation.json"))
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(args.dataroot, cameras_enabled=False, radar_enabled=False)
    tracker = TimestampAwareTracker(
        history_size=10,
        association_distance_m=args.association_distance,
        max_missed_seconds=args.max_missed_seconds,
    )
    evaluator = TrackingIdentityEvaluator(args.distance)

    for frame_record in payload["frames"]:
        frame = source.read_token(frame_record["sample_token"])
        predictions = [
            item for item in bevfusion_predictions_from_records(
                frame_record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                args.score,
            ) if within_detection_range(item.class_name, item.object.center_ego)
        ]
        measurements = [
            prediction_to_global_measurement(item, frame.ego_to_global) for item in predictions
        ]
        snapshots = tracker.update(frame.timestamp_us / 1_000_000.0, measurements)
        confirmed = [snapshot for snapshot in snapshots if snapshot.hits >= 2]
        ground_truth = [
            obj for obj in frame.objects
            if (class_name := detection_class(obj.category)) is not None
            and within_detection_range(class_name, obj.center_ego)
        ]
        evaluator.update(confirmed, ground_truth, frame.ego_to_global)

    report = evaluator.finalize() | {
        "scene_name": payload["scene_name"],
        "score_threshold": args.score,
        "center_distance_threshold_m": args.distance,
        "tracker": {
            "model": "constant-velocity Kalman",
            "association": "class-aware Hungarian",
            "association_distance_m": args.association_distance,
            "max_missed_seconds": args.max_missed_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
