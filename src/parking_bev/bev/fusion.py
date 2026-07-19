from __future__ import annotations

import cv2
import numpy as np

from ..sensors.lidar import LidarBEV


def render_camera(camera_bev: np.ndarray, draw_overlay: bool = False) -> np.ndarray:
    output = camera_bev.copy()
    if not draw_overlay:
        return output

    height, width = output.shape[:2]
    center = (width // 2, height // 2)
    cv2.rectangle(output, (center[0] - 22, center[1] - 45),
                  (center[0] + 22, center[1] + 45), (245, 245, 245), 2)
    cv2.putText(output, "CAMERA BEV ONLY", (14, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 245, 255), 2, cv2.LINE_AA)
    return output


def render_fusion(camera_bev: np.ndarray, lidar_bev: LidarBEV, lidar_enabled: bool) -> np.ndarray:
    output = camera_bev.copy()
    if lidar_enabled:
        dilated = cv2.dilate(lidar_bev.occupancy, np.ones((3, 3), np.uint8))
        overlay = np.zeros_like(output)
        overlay[dilated > 0] = (0, 80, 255)
        output = cv2.addWeighted(output, 1.0, overlay, 0.8, 0.0)

    height, width = output.shape[:2]
    center = (width // 2, height // 2)
    cv2.rectangle(output, (center[0] - 22, center[1] - 45),
                  (center[0] + 22, center[1] + 45), (245, 245, 245), 2)
    label = "LiDAR ON" if lidar_enabled else "LiDAR OFF"
    color = (50, 230, 80) if lidar_enabled else (90, 90, 240)
    cv2.putText(output, label, (14, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return output


def render_lidar(lidar_bev: LidarBEV) -> np.ndarray:
    image = np.zeros((*lidar_bev.occupancy.shape, 3), np.uint8)
    image[lidar_bev.occupancy > 0] = (0, 180, 255)
    return image
