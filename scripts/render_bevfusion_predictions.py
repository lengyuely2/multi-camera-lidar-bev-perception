from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import NuScenesSource, Object3D


def main() -> None:
    parser = argparse.ArgumentParser(description="Render BEVFusion predictions on the metric BEV")
    parser.add_argument("--predictions", type=Path, default=Path("output/bevfusion_predictions.json"))
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("output/bevfusion_predictions.jpg"))
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    source = NuScenesSource(args.dataroot, radar_enabled=False, annotations_enabled=False)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("No nuScenes sample found")
    lidar_to_ego = frame.calibrations["LIDAR_TOP"].sensor_to_ego
    rotation = lidar_to_ego[:3, :3]
    translation = lidar_to_ego[:3, 3]

    objects = []
    scores = []
    for index, prediction in enumerate(payload["predictions"]):
        box = np.asarray(prediction["box_lidar"], dtype=np.float32)
        center_ego = rotation @ box[:3] + translation
        heading_lidar = np.asarray([np.cos(box[6]), np.sin(box[6]), 0.0], np.float32)
        heading_ego = rotation @ heading_lidar
        yaw_ego = float(np.arctan2(heading_ego[1], heading_ego[0]))
        velocity_lidar = (np.asarray([box[7], box[8], 0.0], np.float32)
                          if len(box) >= 9 else np.zeros(3, np.float32))
        velocity_ego = (rotation @ velocity_lidar)[:2]
        predicted_class = prediction["class"]
        if predicted_class == "pedestrian":
            category = "human.pedestrian"
        elif predicted_class in {"barrier", "traffic_cone"}:
            category = f"movable_object.{predicted_class}"
        else:
            category = f"vehicle.{predicted_class}"
        objects.append(Object3D(
            token=f"prediction-{index}",
            category=category,
            center_ego=center_ego,
            # MMDetection3D LiDAR boxes store length, width, height.
            size_wlh=np.asarray([box[4], box[3], box[5]], dtype=np.float32),
            yaw_ego=yaw_ego,
            velocity_ego=velocity_ego,
        ))
        scores.append(float(prediction["score"]))

    predicted_frame = type(frame)(
        token=frame.token,
        timestamp_us=frame.timestamp_us,
        cameras=frame.cameras,
        lidar_ego=frame.lidar_ego,
        radar_ego={},
        objects=tuple(objects),
        calibrations=frame.calibrations,
    )
    image = MetricBEVRenderer().render(predicted_frame)
    cv2.putText(image, "AI PREDICTIONS (not ground truth)", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 240, 255), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"Wrote {len(objects)} predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
