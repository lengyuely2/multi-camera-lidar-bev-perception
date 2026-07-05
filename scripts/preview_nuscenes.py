from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from parking_bev.metric_bev import MetricBEVRenderer
from parking_bev.nuscenes_source import CAMERA_CHANNELS, NuScenesSource


def compose_preview(frame) -> np.ndarray:
    thumb_size = (320, 180)
    camera_panel = np.zeros((360, 960, 3), np.uint8)
    for index, channel in enumerate(CAMERA_CHANNELS):
        image = cv2.resize(frame.cameras[channel], thumb_size)
        cv2.putText(image, channel, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        row, column = divmod(index, 3)
        camera_panel[row * 180:(row + 1) * 180, column * 320:(column + 1) * 320] = image

    bev = MetricBEVRenderer(width=600, height=600).render(frame)
    canvas = np.zeros((960, 960, 3), np.uint8)
    canvas[:360] = camera_panel
    canvas[360:, 180:780] = bev
    return canvas


def render_preview(dataroot: Path, output: Path) -> None:
    source = NuScenesSource(dataroot)
    ok, frame = source.read()
    if not ok or frame is None:
        raise RuntimeError("nuScenes mini contains no readable samples")

    canvas = compose_preview(frame)
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
