from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_bev.appearance import extract_object_appearance
from parking_bev.learned_appearance import ResNetAppearanceEncoder, extract_learned_appearances
from parking_bev.radar_fusion import blend_object_and_radar_velocity, estimate_radar_velocity
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
    parser.add_argument("--appearance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--appearance-weight", type=float, default=0.5)
    parser.add_argument("--learned-appearance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radar-weight", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_tracking_evaluation.json"))
    args = parser.parse_args()
    if args.appearance and args.learned_appearance:
        parser.error("Choose either --appearance or --learned-appearance")

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(
        args.dataroot,
        cameras_enabled=args.appearance or args.learned_appearance,
        radar_enabled=args.radar)
    learned_encoder = ResNetAppearanceEncoder() if args.learned_appearance else None
    tracker = TimestampAwareTracker(
        history_size=10,
        association_distance_m=args.association_distance,
        max_missed_seconds=args.max_missed_seconds,
        appearance_weight=args.appearance_weight
        if args.appearance or args.learned_appearance else 0.0,
    )
    evaluator = TrackingIdentityEvaluator(args.distance)
    total_measurements = 0
    radar_measurements = 0
    radar_points = 0

    for frame_record in payload["frames"]:
        frame = source.read_token(frame_record["sample_token"])
        predictions = [
            item for item in bevfusion_predictions_from_records(
                frame_record["predictions"],
                frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                args.score,
            ) if within_detection_range(item.class_name, item.object.center_ego)
        ]
        learned_features = (extract_learned_appearances(
            [item.object for item in predictions], frame.cameras, frame.calibrations, learned_encoder)
            if learned_encoder is not None else [None] * len(predictions))
        measurements = []
        for item, learned_feature in zip(predictions, learned_features):
            total_measurements += 1
            appearance = (learned_feature if args.learned_appearance else
                          extract_object_appearance(item.object, frame.cameras, frame.calibrations)
                          if args.appearance else None)
            radar_estimate = estimate_radar_velocity(item.object, frame.radar_ego) if args.radar else None
            if radar_estimate is not None:
                radar_measurements += 1
                radar_points += radar_estimate.point_count
            velocity = (blend_object_and_radar_velocity(
                        item.object.velocity_ego, radar_estimate, args.radar_weight)
                        if radar_estimate is not None else None)
            measurements.append(prediction_to_global_measurement(
                item,
                frame.ego_to_global,
                appearance=appearance,
                velocity_ego=velocity,
                velocity_confidence=(radar_estimate.confidence * args.radar_weight
                                     if radar_estimate is not None else 0.0),
            ))
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
            "camera_appearance": args.appearance,
            "learned_camera_appearance": args.learned_appearance,
            "appearance_encoder": "torchvision_resnet18_imagenet" if args.learned_appearance else None,
            "appearance_weight": args.appearance_weight
            if args.appearance or args.learned_appearance else 0.0,
            "radar_velocity": args.radar,
            "radar_weight": args.radar_weight if args.radar else 0.0,
            "radar_associated_measurements": radar_measurements,
            "radar_association_rate": radar_measurements / total_measurements if total_measurements else 0.0,
            "radar_points_used": radar_points,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
