from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from parking_bev.nuscenes_source import CAMERA_CHANNELS, NuScenesSource


def _sensor_bev(frame, size: int = 600, extent_m: float = 60.0) -> np.ndarray:
    image = np.full((size, size, 3), 18, np.uint8)
    scale = size / (2 * extent_m)

    for distance in (10, 20, 30, 40, 50, 60):
        radius = int(distance * scale)
        cv2.circle(image, (size // 2, size // 2), radius, (45, 45, 45), 1)

    lidar = frame.lidar_ego
    valid = (np.abs(lidar[:, 0]) < extent_m) & (np.abs(lidar[:, 1]) < extent_m)
    x, y = lidar[valid, 0], lidar[valid, 1]
    u = (size / 2 - y * scale).astype(np.int32)
    v = (size / 2 - x * scale).astype(np.int32)
    image[v, u] = (255, 190, 30)

    for points in frame.radar_ego.values():
        valid = (np.abs(points[:, 0]) < extent_m) & (np.abs(points[:, 1]) < extent_m)
        for x, y, _, vx, vy in points[valid]:
            start = (int(size / 2 - y * scale), int(size / 2 - x * scale))
            end = (int(start[0] - vy * scale), int(start[1] - vx * scale))
            cv2.arrowedLine(image, start, end, (40, 60, 255), 1, tipLength=0.3)

    center = size // 2
    cv2.rectangle(image, (center - 8, center - 18), (center + 8, center + 18), (70, 230, 80), -1)
    cv2.putText(image, "LiDAR (cyan) + Radar velocity (red)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    return image


def render_preview(dataroot: Path, output: Path) -> None:
    source = NuScenesSource(dataroot)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("nuScenes mini contains no readable samples")

    thumb_size = (320, 180)
    camera_panel = np.zeros((360, 960, 3), np.uint8)
    for index, channel in enumerate(CAMERA_CHANNELS):
        image = cv2.resize(frame.cameras[channel], thumb_size)
        cv2.putText(image, channel, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        row, column = divmod(index, 3)
        camera_panel[row * 180:(row + 1) * 180, column * 320:(column + 1) * 320] = image

    bev = _sensor_bev(frame)
    canvas = np.zeros((960, 960, 3), np.uint8)
    canvas[:360] = camera_panel
    canvas[360:, 180:780] = bev
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview synchronized nuScenes sensors")
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("output/nuscenes_first_sample.jpg"))
    args = parser.parse_args()
    render_preview(args.dataroot, args.output)
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
