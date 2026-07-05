from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
RADAR_CHANNELS = (
    "RADAR_FRONT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_FRONT_LEFT",
)


@dataclass(frozen=True)
class NuScenesFrame:
    token: str
    timestamp_us: int
    cameras: dict[str, np.ndarray]
    lidar_ego: np.ndarray
    radar_ego: dict[str, np.ndarray]


class NuScenesSource:
    """Synchronized nuScenes keyframes transformed into the ego frame."""

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        cameras_enabled: bool = True,
        lidar_enabled: bool = True,
        radar_enabled: bool = True,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError(
                "nuScenes support requires: python -m pip install -e .[datasets]"
            ) from exc

        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        self.cameras_enabled = cameras_enabled
        self.lidar_enabled = lidar_enabled
        self.radar_enabled = radar_enabled
        self._index = 0

    def __len__(self) -> int:
        return len(self.nusc.sample)

    def read(self) -> tuple[bool, NuScenesFrame | None]:
        if self._index >= len(self):
            return False, None
        sample = self.nusc.sample[self._index]
        self._index += 1

        cameras = self._load_cameras(sample) if self.cameras_enabled else {}
        lidar = self._load_lidar(sample) if self.lidar_enabled else np.empty((0, 4), np.float32)
        radars = self._load_radars(sample) if self.radar_enabled else {}
        return True, NuScenesFrame(
            token=sample["token"],
            timestamp_us=int(sample["timestamp"]),
            cameras=cameras,
            lidar_ego=lidar,
            radar_ego=radars,
        )

    def _load_cameras(self, sample: dict) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for channel in CAMERA_CHANNELS:
            sample_data = self.nusc.get("sample_data", sample["data"][channel])
            image = cv2.imread(str(Path(self.nusc.dataroot) / sample_data["filename"]))
            if image is None:
                raise RuntimeError(f"Could not read {sample_data['filename']}")
            frames[channel] = image
        return frames

    def _load_lidar(self, sample: dict) -> np.ndarray:
        from nuscenes.utils.data_classes import LidarPointCloud

        sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cloud = LidarPointCloud.from_file(str(Path(self.nusc.dataroot) / sample_data["filename"]))
        xyz = self._sensor_xyz_to_ego(cloud.points[:3].T, sample_data)
        return np.column_stack((xyz, cloud.points[3])).astype(np.float32)

    def _load_radars(self, sample: dict) -> dict[str, np.ndarray]:
        from nuscenes.utils.data_classes import RadarPointCloud

        output: dict[str, np.ndarray] = {}
        for channel in RADAR_CHANNELS:
            sample_data = self.nusc.get("sample_data", sample["data"][channel])
            cloud = RadarPointCloud.from_file(str(Path(self.nusc.dataroot) / sample_data["filename"]))
            xyz = self._sensor_xyz_to_ego(cloud.points[:3].T, sample_data)
            velocity = self._sensor_velocity_to_ego(cloud.points[[8, 9]].T, sample_data)
            output[channel] = np.column_stack((xyz, velocity)).astype(np.float32)
        return output

    def _rotation_translation(self, sample_data: dict) -> tuple[np.ndarray, np.ndarray]:
        from pyquaternion import Quaternion

        calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        rotation = Quaternion(calibrated["rotation"]).rotation_matrix
        translation = np.asarray(calibrated["translation"], dtype=np.float64)
        return rotation, translation

    def _sensor_xyz_to_ego(self, xyz: np.ndarray, sample_data: dict) -> np.ndarray:
        rotation, translation = self._rotation_translation(sample_data)
        return xyz @ rotation.T + translation

    def _sensor_velocity_to_ego(self, velocity_xy: np.ndarray, sample_data: dict) -> np.ndarray:
        rotation, _ = self._rotation_translation(sample_data)
        velocity_xyz = np.column_stack((velocity_xy, np.zeros(len(velocity_xy))))
        return (velocity_xyz @ rotation.T)[:, :2]
