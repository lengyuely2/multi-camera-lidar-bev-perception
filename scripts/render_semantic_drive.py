from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import bevfusion_predictions_from_records, within_detection_range
from parking_bev.radar_fusion import blend_object_and_radar_velocity, estimate_radar_velocity
from parking_bev.semantic_3d import Semantic3DRenderer, snapshot_to_ego
from parking_bev.tracking import TimestampAwareTracker, prediction_to_global_measurement


def _radar_array(radars: dict[str, np.ndarray]) -> np.ndarray:
    available = [points for points in radars.values() if len(points)]
    return np.vstack(available) if available else np.empty((0, 5), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Tesla-style semantic surround visualization from BEVFusion predictions")
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("output/bevfusion_mini/scenes/07_scene-1077.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--score", type=float, default=0.3)
    parser.add_argument("--radar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radar-weight", type=float, default=0.25)
    parser.add_argument("--engineering-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--screenshot-frame", type=int, default=20)
    parser.add_argument("--video", type=Path, default=Path("output/semantic_surround_drive.mp4"))
    parser.add_argument("--screenshot", type=Path, default=Path("output/semantic_surround_drive.jpg"))
    parser.add_argument("--report", type=Path, default=Path("output/semantic_surround_drive.json"))
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(
        args.dataroot,
        cameras_enabled=False,
        lidar_enabled=args.engineering_mode,
        radar_enabled=args.radar,
        annotations_enabled=False,
    )
    tracker = TimestampAwareTracker(
        history_size=14,
        association_distance_m=4.0,
        max_missed_seconds=2.0,
        appearance_weight=0.0,
    )
    renderer = Semantic3DRenderer(args.width, args.height)

    args.video.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.video}")

    first_timestamp_s = None
    radar_associations = 0
    total_measurements = 0
    rendered_objects = 0
    rendered_frames = 0
    screenshot_written = False
    try:
        for index, record in enumerate(payload["frames"]):
            frame = source.read_token(record["sample_token"])
            timestamp_s = frame.timestamp_us / 1_000_000.0
            if first_timestamp_s is None:
                first_timestamp_s = timestamp_s
            predictions = [
                item for item in bevfusion_predictions_from_records(
                    record["predictions"],
                    frame.calibrations["LIDAR_TOP"].sensor_to_ego,
                    args.score,
                )
                if within_detection_range(item.class_name, item.object.center_ego)
            ]
            measurements = []
            for item in predictions:
                total_measurements += 1
                radar_estimate = (
                    estimate_radar_velocity(item.object, frame.radar_ego)
                    if args.radar else None
                )
                if radar_estimate is not None:
                    radar_associations += 1
                velocity = (
                    blend_object_and_radar_velocity(
                        item.object.velocity_ego, radar_estimate, args.radar_weight)
                    if radar_estimate is not None else None
                )
                measurements.append(prediction_to_global_measurement(
                    item,
                    frame.ego_to_global,
                    velocity_ego=velocity,
                    velocity_confidence=(
                        radar_estimate.confidence * args.radar_weight
                        if radar_estimate is not None else 0.0
                    ),
                ))

            snapshots = tracker.update(timestamp_s, measurements)
            tracks = [
                snapshot_to_ego(snapshot, frame.ego_to_global)
                for snapshot in snapshots if snapshot.hits >= 2
            ]
            rendered_objects += len(tracks)
            rendered_frames += 1
            radar_points = _radar_array(frame.radar_ego) if args.radar else None
            image = renderer.render(
                tracks,
                elapsed_s=timestamp_s - first_timestamp_s,
                frame_index=index,
                radar_enabled=args.radar,
                radar_points_ego=radar_points,
                lidar_points_ego=frame.lidar_ego if args.engineering_mode else None,
                engineering_mode=args.engineering_mode,
            )
            writer.write(image)
            if index == args.screenshot_frame:
                if not cv2.imwrite(str(args.screenshot), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"Could not write {args.screenshot}")
                screenshot_written = True
    finally:
        writer.release()

    if rendered_frames and not screenshot_written:
        raise ValueError(
            f"Screenshot frame {args.screenshot_frame} is outside the rendered sequence")
    report = {
        "scene_name": payload["scene_name"],
        "frames": rendered_frames,
        "video_fps": args.fps,
        "resolution": [args.width, args.height],
        "score_threshold": args.score,
        "camera_lidar_bevfusion": True,
        "radar_velocity_fusion": args.radar,
        "radar_weight": args.radar_weight if args.radar else 0.0,
        "radar_associated_measurements": radar_associations,
        "radar_association_rate": (
            radar_associations / total_measurements if total_measurements else 0.0
        ),
        "engineering_mode": args.engineering_mode,
        "mean_visible_tracks": rendered_objects / rendered_frames if rendered_frames else 0.0,
        "renderer": "stylized metric 3D semantic surround visualization",
        "note": "Road grid is a reference plane; detected map and lane geometry are future work.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Video: {args.video.resolve()}")
    print(f"Screenshot: {args.screenshot.resolve()}")


if __name__ == "__main__":
    main()
