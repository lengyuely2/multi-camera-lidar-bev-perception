from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from parking_bev.nuscenes_source import NuScenesSource
from preview_nuscenes import compose_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synchronized nuScenes BEV sequence")
    parser.add_argument("--dataroot", type=Path, default=Path("data/external/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("output/nuscenes_metric_bev.mp4"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    source = NuScenesSource(args.dataroot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960, 960))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")

    count = 0
    try:
        while count < args.frames:
            ok, frame = source.read()
            if not ok or frame is None:
                break
            writer.write(compose_preview(frame))
            count += 1
    finally:
        writer.release()
    print(f"Wrote {count} frames: {args.output.resolve()}")


if __name__ == "__main__":
    main()
