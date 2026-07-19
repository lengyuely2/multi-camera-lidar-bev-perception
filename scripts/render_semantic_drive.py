from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.nuscenes_source import NuScenesSource
from parking_bev.predictions import bevfusion_predictions_from_records, within_detection_range
from parking_bev.radar_fusion import blend_object_and_radar_velocity, estimate_radar_velocity
from parking_bev.semantic_3d import (
    Semantic3DRenderer,
    SemanticTrack,
    interpolate_semantic_tracks,
    stabilized_snapshot_to_ego,
)
from parking_bev.tracking import TimestampAwareTracker, prediction_to_global_measurement
from parking_bev.world_model import WorldModelLite, world_model_output_to_dict


def _radar_array(radars: dict[str, np.ndarray]) -> np.ndarray:
    available = [points for points in radars.values() if len(points)]
    return np.vstack(available) if available else np.empty((0, 5), dtype=np.float32)


def _camera_tile(image: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (round(source_width * scale), round(source_height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    x = max(0, (resized.shape[1] - width) // 2)
    y = max(0, (resized.shape[0] - height) // 2)
    tile = resized[y:y + height, x:x + width].copy()
    cv2.rectangle(tile, (0, 0), (width, 27), (13, 15, 18), -1)
    cv2.putText(tile, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def _camera_montage(cameras: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """Place all six nuScenes camera keyframes in a compact real-sensor panel."""
    canonical = np.zeros((720, 640, 3), dtype=np.uint8)
    canonical[0:360, 0:640] = _camera_tile(cameras["CAM_FRONT"], (640, 360), "CAM_FRONT - RAW")
    canonical[360:540, 0:320] = _camera_tile(
        cameras["CAM_FRONT_LEFT"], (320, 180), "FRONT_LEFT")
    canonical[360:540, 320:640] = _camera_tile(
        cameras["CAM_FRONT_RIGHT"], (320, 180), "FRONT_RIGHT")
    thirds = (213, 214, 213)
    x = 0
    for channel, tile_width, label in zip(
        ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
        thirds,
        ("BACK_LEFT", "CAM_BACK", "BACK_RIGHT"),
    ):
        canonical[540:720, x:x + tile_width] = _camera_tile(
            cameras[channel], (tile_width, 180), label)
        x += tile_width
    if (width, height) == (640, 720):
        return canonical
    return cv2.resize(canonical, (width, height), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class RenderKeyframe:
    source_index: int
    timestamp_s: float
    tracks: list[SemanticTrack]
    ego_velocity_ego: np.ndarray
    radar_points: np.ndarray | None
    lidar_points: np.ndarray | None
    camera_montage: np.ndarray | None


def _ego_velocity_ego(
    previous_pose: np.ndarray | None,
    previous_timestamp_s: float | None,
    ego_to_global: np.ndarray,
    timestamp_s: float,
) -> np.ndarray:
    if previous_pose is None or previous_timestamp_s is None:
        return np.zeros(2, dtype=np.float32)
    dt = timestamp_s - previous_timestamp_s
    if dt <= 1e-6:
        return np.zeros(2, dtype=np.float32)
    velocity_global = (ego_to_global[:3, 3] - previous_pose[:3, 3]) / dt
    velocity_ego = np.linalg.inv(ego_to_global)[:3, :3] @ velocity_global
    return velocity_ego[:2].astype(np.float32)


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
    parser.add_argument("--camera-comparison", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--world-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", default="FUSION DRIVE")
    parser.add_argument("--sensor-label", default=None)
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-visible-hits", type=int, default=2)
    parser.add_argument("--fade-in-hits", type=int, default=4)
    parser.add_argument("--fade-out-misses", type=int, default=4)
    parser.add_argument("--fps", type=float, default=12.0)
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
        cameras_enabled=args.camera_comparison,
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
    world_model = WorldModelLite(horizon_s=3.0, step_s=0.5)
    renderer = Semantic3DRenderer(args.width, args.height)
    camera_panel_width = round(args.height * 8 / 9) if args.camera_comparison else 0
    output_width = args.width + camera_panel_width + (4 if args.camera_comparison else 0)
    keyframes: list[RenderKeyframe] = []

    args.video.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    radar_associations = 0
    total_measurements = 0
    previous_pose: np.ndarray | None = None
    previous_timestamp_s: float | None = None
    for index, record in enumerate(payload["frames"]):
        frame = source.read_token(record["sample_token"])
        timestamp_s = frame.timestamp_us / 1_000_000.0
        ego_velocity = _ego_velocity_ego(
            previous_pose, previous_timestamp_s, frame.ego_to_global, timestamp_s)
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
        tracks = []
        for snapshot in snapshots:
            track = stabilized_snapshot_to_ego(
                snapshot,
                frame.ego_to_global,
                min_visible_hits=args.min_visible_hits,
                fade_in_hits=args.fade_in_hits,
                fade_out_misses=args.fade_out_misses,
            )
            if track is not None:
                tracks.append(track)
        keyframes.append(RenderKeyframe(
            source_index=index,
            timestamp_s=timestamp_s,
            tracks=tracks,
            ego_velocity_ego=ego_velocity,
            radar_points=_radar_array(frame.radar_ego) if args.radar else None,
            lidar_points=frame.lidar_ego if args.engineering_mode else None,
            camera_montage=(
                _camera_montage(frame.cameras, camera_panel_width, args.height)
                if args.camera_comparison else None
            ),
        ))
        previous_pose = frame.ego_to_global.copy()
        previous_timestamp_s = timestamp_s

    if not keyframes:
        raise ValueError("Prediction file contains no frames")

    source_duration_s = keyframes[-1].timestamp_s - keyframes[0].timestamp_s
    output_fps = (
        args.fps if args.smooth else
        (len(keyframes) - 1) / source_duration_s if source_duration_s > 0 else 1.0
    )
    writer = cv2.VideoWriter(
        str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
        (output_width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.video}")

    first_timestamp_s = keyframes[0].timestamp_s
    rendered_objects = 0
    rendered_frames = 0
    screenshot_written = False
    risk_counts = {level: 0 for level in ("clear", "watch", "caution", "critical")}
    max_risk_score = 0.0
    highest_risk_report: dict | None = None

    def write_frame(
        tracks: list[SemanticTrack],
        timestamp_s: float,
        source_index: int,
        sensor_frame: RenderKeyframe,
    ) -> None:
        nonlocal rendered_objects, rendered_frames, screenshot_written
        nonlocal max_risk_score, highest_risk_report
        world_output = (
            world_model.assess(tracks, sensor_frame.ego_velocity_ego)
            if args.world_model else None
        )
        if world_output is not None:
            risk_counts[world_output.risk_level] += 1
            if world_output.risk_score >= max_risk_score:
                max_risk_score = world_output.risk_score
                highest_risk_report = world_model_output_to_dict(world_output)
                highest_risk_report["source_index"] = source_index
                highest_risk_report["elapsed_s"] = timestamp_s - first_timestamp_s
        image = renderer.render(
            tracks,
            elapsed_s=timestamp_s - first_timestamp_s,
            frame_index=source_index,
            radar_enabled=args.radar,
            radar_points_ego=sensor_frame.radar_points,
            lidar_points_ego=sensor_frame.lidar_points,
            engineering_mode=args.engineering_mode,
            world_output=world_output,
            title=args.title,
            sensor_label=args.sensor_label,
        )
        if args.camera_comparison:
            assert sensor_frame.camera_montage is not None
            separator = np.full((args.height, 4, 3), (32, 34, 38), dtype=np.uint8)
            image = np.hstack((sensor_frame.camera_montage, separator, image))
        writer.write(image)
        rendered_objects += len(tracks)
        rendered_frames += 1
        if source_index == args.screenshot_frame and not screenshot_written:
            if not cv2.imwrite(str(args.screenshot), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Could not write {args.screenshot}")
            screenshot_written = True

    try:
        for before, after in zip(keyframes, keyframes[1:]):
            duration_s = max(after.timestamp_s - before.timestamp_s, 1.0 / output_fps)
            steps = max(1, round(duration_s * args.fps)) if args.smooth else 1
            for step in range(steps):
                alpha = step / steps
                tracks = (
                    interpolate_semantic_tracks(before.tracks, after.tracks, alpha)
                    if args.smooth else before.tracks
                )
                sensor_frame = before if alpha < 0.5 else after
                write_frame(
                    tracks,
                    (1.0 - alpha) * before.timestamp_s + alpha * after.timestamp_s,
                    before.source_index,
                    sensor_frame,
                )
        final = keyframes[-1]
        write_frame(final.tracks, final.timestamp_s, final.source_index, final)
    finally:
        writer.release()

    if rendered_frames and not screenshot_written:
        raise ValueError(
            f"Screenshot frame {args.screenshot_frame} is outside the rendered sequence")
    report = {
        "scene_name": payload["scene_name"],
        "source_keyframes": len(keyframes),
        "video_frames": rendered_frames,
        "source_duration_s": source_duration_s,
        "video_fps": output_fps,
        "smooth_interpolation": args.smooth,
        "display_stabilization": {
            "min_visible_hits": args.min_visible_hits,
            "fade_in_hits": args.fade_in_hits,
            "fade_out_misses": args.fade_out_misses,
            "risk_activation_alpha": world_model.risk_activation_alpha,
        },
        "resolution": [output_width, args.height],
        "camera_comparison": args.camera_comparison,
        "score_threshold": args.score,
        "camera_lidar_bevfusion": True,
        "radar_velocity_fusion": args.radar,
        "radar_weight": args.radar_weight if args.radar else 0.0,
        "radar_associated_measurements": radar_associations,
        "radar_association_rate": (
            radar_associations / total_measurements if total_measurements else 0.0
        ),
        "world_model": (
            {
                "name": "WorldModelLite",
                "horizon_s": world_model.horizon_s,
                "step_s": world_model.step_s,
                "risk_frame_counts": risk_counts,
                "max_risk_score": max_risk_score,
                "highest_risk": highest_risk_report,
            }
            if args.world_model else {"enabled": False}
        ),
        "engineering_mode": args.engineering_mode,
        "title": args.title,
        "sensor_label": args.sensor_label,
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
