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
class Object3D:
    token: str
    category: str
    center_ego: np.ndarray
    size_wlh: np.ndarray
    yaw_ego: float
    velocity_ego: np.ndarray


@dataclass(frozen=True)
class SensorCalibration:
    sensor_to_ego: np.ndarray
    intrinsic: np.ndarray | None


@dataclass(frozen=True)
class NuScenesFrame:
    token: str
    timestamp_us: int
    cameras: dict[str, np.ndarray]
    lidar_ego: np.ndarray
    radar_ego: dict[str, np.ndarray]
    objects: tuple[Object3D, ...]
    calibrations: dict[str, SensorCalibration]
    ego_to_global: np.ndarray


class NuScenesSource:
    """Synchronized nuScenes keyframes transformed into the ego frame."""

    def __init__(
        self,
        dataroot: str | Path,
        version: str = "v1.0-mini",
        cameras_enabled: bool = True,
        lidar_enabled: bool = True,
        radar_enabled: bool = True,
        annotations_enabled: bool = True,
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
        self.annotations_enabled = annotations_enabled
        self._index = 0

    def __len__(self) -> int:
        return len(self.nusc.sample)

    def read(self) -> tuple[bool, NuScenesFrame | None]:
        if self._index >= len(self):
            return False, None
        sample = self.nusc.sample[self._index]
        self._index += 1

        return True, self._build_frame(sample)

    def read_token(self, token: str) -> NuScenesFrame:
        """Read one keyframe by nuScenes sample token without changing iteration state."""
        return self._build_frame(self.nusc.get("sample", token))

    def _build_frame(self, sample: dict) -> NuScenesFrame:
        cameras = self._load_cameras(sample) if self.cameras_enabled else {}
        lidar = self._load_lidar(sample) if self.lidar_enabled else np.empty((0, 5), np.float32)
        radars = self._load_radars(sample) if self.radar_enabled else {}
        objects = self._load_annotations(sample) if self.annotations_enabled else ()
        calibrations = self._load_calibrations(sample)
        ego_to_global = self._load_ego_to_global(sample)
        return NuScenesFrame(
            token=sample["token"],
            timestamp_us=int(sample["timestamp"]),
            cameras=cameras,
            lidar_ego=lidar,
            radar_ego=radars,
            objects=objects,
            calibrations=calibrations,
            ego_to_global=ego_to_global,
        )

    def _load_ego_to_global(self, sample: dict) -> np.ndarray:
        from pyquaternion import Quaternion

        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        transform[:3, 3] = np.asarray(ego_pose["translation"], dtype=np.float32)
        return transform

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
        sample_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        raw = np.fromfile(Path(self.nusc.dataroot) / sample_data["filename"], dtype=np.float32).reshape(-1, 5)
        xyz = self._sensor_xyz_to_ego(raw[:, :3], sample_data)
        time_lag = np.zeros((len(raw), 1), dtype=np.float32)
        return np.column_stack((xyz, raw[:, 3], time_lag)).astype(np.float32)

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

    def _load_annotations(self, sample: dict) -> tuple[Object3D, ...]:
        from pyquaternion import Quaternion

        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_rotation = Quaternion(ego_pose["rotation"])
        ego_translation = np.asarray(ego_pose["translation"], dtype=np.float64)
        global_to_ego = ego_rotation.rotation_matrix.T

        objects: list[Object3D] = []
        for token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", token)
            center_global = np.asarray(annotation["translation"], dtype=np.float64)
            center_ego = global_to_ego @ (center_global - ego_translation)
            orientation_ego = ego_rotation.inverse * Quaternion(annotation["rotation"])

            velocity_global = np.asarray(self.nusc.box_velocity(token), dtype=np.float64)
            if not np.isfinite(velocity_global).all():
                velocity_global = np.zeros(3, dtype=np.float64)
            velocity_ego = (global_to_ego @ velocity_global)[:2]

            objects.append(Object3D(
                token=token,
                category=annotation["category_name"],
                center_ego=center_ego.astype(np.float32),
                size_wlh=np.asarray(annotation["size"], dtype=np.float32),
                yaw_ego=float(orientation_ego.yaw_pitch_roll[0]),
                velocity_ego=velocity_ego.astype(np.float32),
            ))
        return tuple(objects)

    def _load_calibrations(self, sample: dict) -> dict[str, SensorCalibration]:
        output: dict[str, SensorCalibration] = {}
        for channel in (*CAMERA_CHANNELS, "LIDAR_TOP", *RADAR_CHANNELS):
            sample_data = self.nusc.get("sample_data", sample["data"][channel])
            calibrated = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            rotation, translation = self._rotation_translation(sample_data)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            camera_intrinsic = calibrated.get("camera_intrinsic", [])
            intrinsic = (np.asarray(camera_intrinsic, dtype=np.float32)
                         if len(camera_intrinsic) else None)
            output[channel] = SensorCalibration(transform, intrinsic)
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
