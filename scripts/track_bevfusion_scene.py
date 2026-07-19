from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.appearance import extract_object_appearance
from parking_bev.learned_appearance import ResNetAppearanceEncoder, extract_learned_appearances
from parking_bev.radar_fusion import blend_object_and_radar_velocity, estimate_radar_velocity
from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import NuScenesSource, Object3D
from parking_bev.predictions import (
    bevfusion_predictions_from_records,
    category_for_detection_class,
    within_detection_range,
)
from parking_bev.tracking import TimestampAwareTracker, prediction_to_global_measurement


def _snapshot_to_ego(snapshot, ego_to_global: np.ndarray) -> Object3D:
    global_to_ego = np.linalg.inv(ego_to_global)
    global_z = float(ego_to_global[2, 3])
    center_ego = global_to_ego @ np.asarray([
        snapshot.position_global[0], snapshot.position_global[1], global_z, 1.0
    ])
    velocity_ego = (global_to_ego[:3, :3]
                    @ np.asarray([snapshot.velocity_global[0], snapshot.velocity_global[1], 0.0]))[:2]
    heading_global = np.asarray([np.cos(snapshot.yaw_global), np.sin(snapshot.yaw_global), 0.0])
    heading_ego = global_to_ego[:3, :3] @ heading_global
    return Object3D(
        token=f"track-{snapshot.track_id}",
        category=category_for_detection_class(snapshot.class_name),
        center_ego=center_ego[:3].astype(np.float32),
        size_wlh=snapshot.size_wlh.astype(np.float32),
        yaw_ego=float(np.arctan2(heading_ego[1], heading_ego[0])),
        velocity_ego=velocity_ego.astype(np.float32),
    )


def _track_color(track_id: int) -> tuple[int, int, int]:
    return ((53 * track_id) % 180 + 60, (97 * track_id) % 180 + 60, (151 * track_id) % 180 + 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timestamp-aware tracking over BEVFusion scene predictions")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_scene_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--score", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--appearance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--appearance-weight", type=float, default=0.5)
    parser.add_argument("--learned-appearance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--radar-weight", type=float, default=0.25)
    parser.add_argument("--min-visible-hits", type=int, default=3)
    parser.add_argument("--max-missed-seconds", type=float, default=2.5)
    parser.add_argument("--video", type=Path, default=Path("output/bevfusion_scene_tracking.mp4"))
    parser.add_argument("--report", type=Path, default=Path("output/bevfusion_scene_tracking.json"))
    args = parser.parse_args()
    if args.appearance and args.learned_appearance:
        parser.error("Choose either --appearance or --learned-appearance")

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(
        args.dataroot, cameras_enabled=args.appearance or args.learned_appearance,
        radar_enabled=args.radar,
        annotations_enabled=False)
    learned_encoder = ResNetAppearanceEncoder() if args.learned_appearance else None
    tracker = TimestampAwareTracker(
        history_size=10,
        association_distance_m=4.0,
        max_missed_seconds=args.max_missed_seconds,
        appearance_weight=args.appearance_weight
        if args.appearance or args.learned_appearance else 0.0,
    )
    renderer = MetricBEVRenderer()
    timestamps = []
    active_counts = []
    track_stats: dict[int, dict] = {}
    total_measurements = 0
    radar_measurements = 0
    radar_points = 0

    args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (800, 800))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.video}")

    try:
        for frame_record in payload["frames"]:
            frame = source.read_token(frame_record["sample_token"])
            timestamp_s = frame.timestamp_us / 1_000_000.0
            timestamps.append(timestamp_s)
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
            snapshots = tracker.update(timestamp_s, measurements)
            visible = [snapshot for snapshot in snapshots if snapshot.hits >= args.min_visible_hits]
            active_counts.append(len(visible))
            objects = [_snapshot_to_ego(snapshot, frame.ego_to_global) for snapshot in visible]
            tracked_frame = replace(frame, objects=tuple(objects), radar_ego={})
            image = renderer.render(tracked_frame, draw_labels=False)
            global_to_ego = np.linalg.inv(frame.ego_to_global)

            for snapshot, obj in zip(visible, objects):
                color = _track_color(snapshot.track_id)
                global_z = float(frame.ego_to_global[2, 3])
                history_h = np.column_stack((
                    snapshot.history_global,
                    np.full(len(snapshot.history_global), global_z),
                    np.ones(len(snapshot.history_global)),
                ))
                history_ego = (global_to_ego @ history_h.T).T[:, :2]
                in_range = ((history_ego[:, 0] >= renderer.x_min) & (history_ego[:, 0] < renderer.x_max)
                            & (history_ego[:, 1] >= renderer.y_min) & (history_ego[:, 1] < renderer.y_max))
                pixels = renderer.ego_to_pixel(history_ego[in_range]).astype(np.int32)
                if len(pixels) >= 2:
                    cv2.polylines(image, [pixels], False, color, 2, cv2.LINE_AA)
                center = renderer.ego_to_pixel(obj.center_ego[:2].reshape(1, 2))[0].astype(int)
                stale = " P" if snapshot.missed else ""
                cv2.putText(image, f"ID{snapshot.track_id}{stale}", tuple(center + [4, 12]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
                stats = track_stats.setdefault(snapshot.track_id, {
                    "class": snapshot.class_name,
                    "first_timestamp_s": timestamp_s,
                    "last_timestamp_s": timestamp_s,
                    "max_hits": snapshot.hits,
                })
                stats["last_timestamp_s"] = timestamp_s
                stats["max_hits"] = max(stats["max_hits"], snapshot.hits)

            cv2.putText(image, "TIME-AWARE TRACKING", (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (80, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"t={timestamp_s - timestamps[0]:.1f}s  active={len(visible)}",
                        (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()

    time_deltas = np.diff(timestamps)
    durations = [value["last_timestamp_s"] - value["first_timestamp_s"] for value in track_stats.values()]
    report = {
        "scene_name": payload["scene_name"],
        "frames": len(timestamps),
        "duration_s": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "mean_frame_dt_s": float(time_deltas.mean()) if len(time_deltas) else 0.0,
        "min_frame_dt_s": float(time_deltas.min()) if len(time_deltas) else 0.0,
        "max_frame_dt_s": float(time_deltas.max()) if len(time_deltas) else 0.0,
        "unique_confirmed_tracks": len(track_stats),
        "mean_active_tracks": float(np.mean(active_counts)) if active_counts else 0.0,
        "max_active_tracks": max(active_counts, default=0),
        "mean_track_duration_s": float(np.mean(durations)) if durations else 0.0,
        "tracks_by_class": dict(Counter(value["class"] for value in track_stats.values())),
        "camera_appearance": args.appearance,
        "learned_camera_appearance": args.learned_appearance,
        "appearance_encoder": "torchvision_resnet18_imagenet" if args.learned_appearance else None,
        "appearance_weight": args.appearance_weight
        if args.appearance or args.learned_appearance else 0.0,
        "display_stabilization": {
            "min_visible_hits": args.min_visible_hits,
            "max_missed_seconds": args.max_missed_seconds,
            "note": "New tracks must be observed repeatedly before the BEV overlay shows them.",
        },
        "radar_velocity": args.radar,
        "radar_weight": args.radar_weight if args.radar else 0.0,
        "radar_associated_measurements": radar_measurements,
        "radar_association_rate": radar_measurements / total_measurements if total_measurements else 0.0,
        "radar_points_used": radar_points,
        "note": "P suffix means a temporarily predicted track during a missed detection.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Video: {args.video.resolve()}")


if __name__ == "__main__":
    main()
