"""Label-free sensor packet shared by the ROS process and Python 3.8 model worker."""
from __future__ import annotations

import numpy as np

# Preserve CMT's camera order (CAM_BACK_LEFT has index 4).
CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def pointcloud_matrix(rows, fields=("x", "y", "z", "intensity")):
    """Support both ROS structured NumPy arrays and older tuple iterators."""
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        return np.column_stack([rows[name].ravel() for name in fields]).astype(np.float32)
    return np.asarray(list(rows), dtype=np.float32).reshape(-1, len(fields))


def validate_transform(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(name + " must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(name + " must be a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError(name + " must contain a proper rotation")
    return matrix


def validate_packet(packet, max_delay_s=0.6, max_camera_skew_s=0.08):
    points = np.asarray(packet["points"])
    if points.ndim != 2 or points.shape[1] != 5 or not len(points) or not np.isfinite(points).all():
        raise ValueError("points must be nonempty finite Nx5 [x,y,z,intensity,sweep_age_seconds]")
    if np.any(points[:, 4] < 0):
        raise ValueError("Sweep ages must be nonnegative seconds relative to the source LiDAR timestamp")
    timestamp = float(packet["timestamp_s"])
    lidar_timestamp = float(packet["lidar_timestamp_s"])
    camera_timestamps = np.asarray(packet["camera_timestamps_s"])
    if camera_timestamps.shape != (6,) or not np.isfinite(camera_timestamps).all() or not np.isfinite([timestamp, lidar_timestamp]).all():
        raise ValueError("Sensor timestamps must be finite seconds")
    if abs(timestamp - lidar_timestamp) > max_delay_s:
        raise ValueError("LiDAR/reference timestamp difference exceeds maximum delay")
    if np.ptp(camera_timestamps) > max_camera_skew_s or np.max(np.abs(camera_timestamps - timestamp)) > max_camera_skew_s:
        raise ValueError("Cameras are not synchronized to the reference time")
    if not str(packet["frame_id"]):
        raise ValueError("frame_id is required")
    for name in ("lidar_to_ego", "ego_to_global", "source_ego_to_global"):
        validate_transform(packet[name], name)
    projections = np.asarray(packet["lidar2cam"])
    intrinsics = np.asarray(packet["intrinsics"])
    if projections.shape != (6, 4, 4) or intrinsics.shape != (6, 3, 3):
        raise ValueError("Require six lidar2cam transforms and six camera intrinsics")
    for index in range(6):
        validate_transform(projections[index], "lidar2cam")
        image = np.asarray(packet["image_" + str(index)])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) == 0:
            raise ValueError("Images must be nonempty uint8 BGR HxWx3")
    if not np.isfinite(intrinsics).all() or np.any(intrinsics[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Invalid camera intrinsics")
    if any(key.startswith("gt_") or key.startswith("pts_in_boxes") for key in packet):
        raise ValueError("Deployment packets must not contain ground truth")


def compensate_sweep(points, source_lidar_to_world, reference_lidar_to_world, age_s):
    """Move historical scan into source-scan LiDAR coordinates; fifth channel is age."""
    transform = np.linalg.inv(reference_lidar_to_world) @ source_lidar_to_world
    output = np.empty((len(points), 5), dtype=np.float32)
    output[:, :3] = np.asarray(points)[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    output[:, 3] = np.asarray(points)[:, 3]
    output[:, 4] = age_s
    return output
