from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..sensors.nuscenes_source import Object3D


DETECTION_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
DETECTION_RANGES_M = {
    "car": 50.0,
    "truck": 50.0,
    "construction_vehicle": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "barrier": 30.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
    "pedestrian": 40.0,
    "traffic_cone": 30.0,
}


@dataclass(frozen=True)
class Prediction3D:
    object: Object3D
    class_name: str
    score: float


def detection_class(category: str) -> str | None:
    if category.startswith("human.pedestrian"):
        return "pedestrian"
    if category.startswith("vehicle.bus"):
        return "bus"
    mapping = {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.trailer": "trailer",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
    }
    return mapping.get(category)


def category_for_detection_class(class_name: str) -> str:
    if class_name == "pedestrian":
        return "human.pedestrian"
    if class_name in {"barrier", "traffic_cone"}:
        return f"movable_object.{class_name}"
    if class_name == "construction_vehicle":
        return "vehicle.construction"
    return f"vehicle.{class_name}"


def within_detection_range(class_name: str, center_ego: np.ndarray) -> bool:
    return float(np.linalg.norm(np.asarray(center_ego)[:2])) <= DETECTION_RANGES_M[class_name]


def load_bevfusion_predictions(
    path: str | Path,
    lidar_to_ego: np.ndarray,
    score_threshold: float | None = None,
) -> list[Prediction3D]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    threshold = payload.get("score_threshold", 0.0) if score_threshold is None else score_threshold
    return bevfusion_predictions_from_records(payload["predictions"], lidar_to_ego, threshold)


def bevfusion_predictions_from_records(
    records: list[dict],
    lidar_to_ego: np.ndarray,
    score_threshold: float = 0.2,
) -> list[Prediction3D]:
    rotation = np.asarray(lidar_to_ego[:3, :3], dtype=np.float32)
    translation = np.asarray(lidar_to_ego[:3, 3], dtype=np.float32)

    output: list[Prediction3D] = []
    for index, prediction in enumerate(records):
        score = float(prediction["score"])
        if score < score_threshold:
            continue
        box = np.asarray(prediction["box_lidar"], dtype=np.float32)
        center_ego = rotation @ box[:3] + translation
        heading_ego = rotation @ np.asarray([np.cos(box[6]), np.sin(box[6]), 0.0], np.float32)
        yaw_ego = float(np.arctan2(heading_ego[1], heading_ego[0]))
        velocity_lidar = (np.asarray([box[7], box[8], 0.0], np.float32)
                          if len(box) >= 9 else np.zeros(3, np.float32))
        class_name = prediction["class"]
        obj = Object3D(
            token=f"prediction-{index}",
            category=category_for_detection_class(class_name),
            center_ego=center_ego,
            # MMDetection3D LiDAR boxes store length, width, height.
            size_wlh=np.asarray([box[4], box[3], box[5]], dtype=np.float32),
            yaw_ego=yaw_ego,
            velocity_ego=(rotation @ velocity_lidar)[:2],
        )
        output.append(Prediction3D(obj, class_name, score))
    return output
