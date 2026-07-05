from __future__ import annotations

import cv2
import numpy as np

from .nuscenes_source import NuScenesFrame, Object3D


class MetricBEVRenderer:
    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        x_limits_m: tuple[float, float] = (-60.0, 60.0),
        y_limits_m: tuple[float, float] = (-60.0, 60.0),
    ) -> None:
        self.width = width
        self.height = height
        self.x_min, self.x_max = x_limits_m
        self.y_min, self.y_max = y_limits_m

    def ego_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float32)
        u = (self.y_max - points[..., 1]) / (self.y_max - self.y_min) * self.width
        v = (self.x_max - points[..., 0]) / (self.x_max - self.x_min) * self.height
        return np.stack((u, v), axis=-1)

    @staticmethod
    def object_corners_xy(obj: Object3D) -> np.ndarray:
        width, length = float(obj.size_wlh[0]), float(obj.size_wlh[1])
        local = np.asarray([
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ], dtype=np.float32)
        cosine, sine = np.cos(obj.yaw_ego), np.sin(obj.yaw_ego)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        return local @ rotation.T + obj.center_ego[:2]

    def render(self, frame: NuScenesFrame) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 16, np.uint8)
        self._draw_grid(image)
        self._draw_lidar(image, frame.lidar_ego)
        self._draw_radar(image, frame.radar_ego)
        for obj in frame.objects:
            self._draw_object(image, obj)
        self._draw_ego(image)
        cv2.putText(image, f"Objects: {len(frame.objects)}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        return image

    def _inside(self, xy: np.ndarray) -> np.ndarray:
        return ((xy[:, 0] >= self.x_min) & (xy[:, 0] < self.x_max)
                & (xy[:, 1] >= self.y_min) & (xy[:, 1] < self.y_max))

    def _draw_grid(self, image: np.ndarray) -> None:
        maximum = int(max(abs(self.x_min), abs(self.x_max), abs(self.y_min), abs(self.y_max)))
        center = tuple(self.ego_to_pixel(np.asarray([[0.0, 0.0]]))[0].astype(int))
        pixels_per_metre = self.width / (self.y_max - self.y_min)
        for distance in range(10, maximum + 1, 10):
            cv2.circle(image, center, int(distance * pixels_per_metre), (42, 42, 42), 1)
            point = self.ego_to_pixel(np.asarray([[float(distance), 0.0]]))[0].astype(int)
            cv2.putText(image, f"{distance}m", tuple(point + [5, -3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (105, 105, 105), 1, cv2.LINE_AA)

    def _draw_lidar(self, image: np.ndarray, points: np.ndarray) -> None:
        if not len(points):
            return
        valid = self._inside(points[:, :2])
        pixels = self.ego_to_pixel(points[valid, :2]).astype(np.int32)
        image[pixels[:, 1], pixels[:, 0]] = (255, 185, 35)

    def _draw_radar(self, image: np.ndarray, radars: dict[str, np.ndarray]) -> None:
        for points in radars.values():
            valid = self._inside(points[:, :2])
            for x, y, _, vx, vy in points[valid]:
                velocity = self._limited_velocity(np.asarray([vx, vy], dtype=np.float32))
                start = tuple(self.ego_to_pixel(np.asarray([[x, y]]))[0].astype(int))
                end = tuple(self.ego_to_pixel(np.asarray([[x, y]]) + velocity)[0].astype(int))
                cv2.arrowedLine(image, start, end, (30, 70, 255), 1, tipLength=0.3)

    def _draw_object(self, image: np.ndarray, obj: Object3D) -> None:
        center = obj.center_ego[:2].reshape(1, 2)
        if not self._inside(center)[0]:
            return
        color = self._category_color(obj.category)
        corners = self.ego_to_pixel(self.object_corners_xy(obj)).astype(np.int32)
        cv2.polylines(image, [corners], True, color, 2, cv2.LINE_AA)

        center_pixel = self.ego_to_pixel(center)[0].astype(int)
        velocity = self._limited_velocity(obj.velocity_ego)
        velocity_tip = self.ego_to_pixel(center + velocity)[0].astype(int)
        if np.linalg.norm(obj.velocity_ego) > 0.2:
            cv2.arrowedLine(image, tuple(center_pixel), tuple(velocity_tip), color, 2, tipLength=0.25)
        if not obj.category.startswith("movable_object"):
            label = self._short_category(obj.category)
            cv2.putText(image, label, tuple(center_pixel + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, color, 1, cv2.LINE_AA)

    def _draw_ego(self, image: np.ndarray) -> None:
        ego = Object3D("ego", "ego", np.zeros(3, np.float32),
                       np.asarray([1.9, 4.6, 1.6], np.float32), 0.0, np.zeros(2, np.float32))
        corners = self.ego_to_pixel(self.object_corners_xy(ego)).astype(np.int32)
        cv2.fillPoly(image, [corners], (70, 230, 80))

    @staticmethod
    def _short_category(category: str) -> str:
        if category.startswith("human.pedestrian"):
            return "pedestrian"
        parts = category.split(".")
        return parts[-1].replace("construction", "const")

    @staticmethod
    def _category_color(category: str) -> tuple[int, int, int]:
        if category.startswith("vehicle"):
            return 60, 220, 255
        if category.startswith("human"):
            return 255, 100, 210
        if category.startswith("movable_object"):
            return 150, 150, 255
        return 180, 255, 120

    @staticmethod
    def _limited_velocity(velocity: np.ndarray, horizon_s: float = 0.5, max_length_m: float = 6.0) -> np.ndarray:
        displacement = np.asarray(velocity, dtype=np.float32).reshape(1, 2) * horizon_s
        length = float(np.linalg.norm(displacement))
        if length > max_length_m:
            displacement *= max_length_m / length
        return displacement
