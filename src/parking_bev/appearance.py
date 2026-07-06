from __future__ import annotations

import cv2
import numpy as np

from .nuscenes_source import Object3D, SensorCalibration


def extract_object_appearance(
    obj: Object3D,
    cameras: dict[str, np.ndarray],
    calibrations: dict[str, SensorCalibration],
) -> np.ndarray | None:
    """Project a 3D box into the best camera and return a normalized HSV histogram."""
    best_crop = extract_object_crop(obj, cameras, calibrations)
    if best_crop is None or not best_crop.size:
        return None
    hsv = cv2.cvtColor(best_crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    return (histogram / norm).astype(np.float32) if norm > 0 else None


def extract_object_crop(
    obj: Object3D,
    cameras: dict[str, np.ndarray],
    calibrations: dict[str, SensorCalibration],
) -> np.ndarray | None:
    """Return the largest valid projection of an ego-frame 3D box."""
    corners = _box_corners_ego(obj)
    best_crop: np.ndarray | None = None
    best_area = 0
    for channel, image in cameras.items():
        calibration = calibrations[channel]
        if calibration.intrinsic is None:
            continue
        ego_to_camera = np.linalg.inv(calibration.sensor_to_ego)
        corners_h = np.column_stack((corners, np.ones(len(corners), np.float32)))
        camera_points = (ego_to_camera @ corners_h.T).T[:, :3]
        visible = camera_points[:, 2] > 0.5
        if visible.sum() < 4:
            continue
        projected = (calibration.intrinsic @ camera_points[visible].T).T
        pixels = projected[:, :2] / projected[:, 2:3]
        x0, y0 = np.floor(pixels.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pixels.max(axis=0)).astype(int)
        height, width = image.shape[:2]
        x0, x1 = np.clip([x0, x1], 0, width)
        y0, y1 = np.clip([y0, y1], 0, height)
        area = int(max(0, x1 - x0) * max(0, y1 - y0))
        if area > best_area and x1 - x0 >= 3 and y1 - y0 >= 5:
            best_crop = image[y0:y1, x0:x1]
            best_area = area
    return best_crop


def _box_corners_ego(obj: Object3D) -> np.ndarray:
    width, length, height = obj.size_wlh.astype(float)
    xy = np.asarray([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ], dtype=np.float32)
    cosine, sine = np.cos(obj.yaw_ego), np.sin(obj.yaw_ego)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    xy = xy @ rotation.T + obj.center_ego[:2]
    bottom = np.column_stack((xy, np.full(4, obj.center_ego[2] - height / 2)))
    top = np.column_stack((xy, np.full(4, obj.center_ego[2] + height / 2)))
    return np.vstack((bottom, top)).astype(np.float32)
