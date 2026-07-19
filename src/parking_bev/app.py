from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .bev import CameraBEVProjector
from .camera import build_camera_rig
from .config import load_config
from .fusion import render_camera, render_fusion, render_lidar
from .lidar import LidarBEVProjector, SyntheticLidarSource


def run(config_path: str, max_frames: int | None = None, display: bool = True) -> Path:
    config = load_config(config_path)
    runtime = config["runtime"]
    bev_config = config["bev"]
    camera_config = config["cameras"]
    lidar_config = config["lidar"]

    width = int(bev_config["width_px"])
    height = int(bev_config["height_px"])
    camera_rig = build_camera_rig(camera_config)
    camera_projector = CameraBEVProjector(width, height, camera_config["destination_quads"])
    lidar_projector = LidarBEVProjector(bev_config, lidar_config)
    lidar_source = SyntheticLidarSource() if lidar_config.get("source") == "synthetic" else None

    output_path = Path(runtime["output_video"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(runtime["fps"]), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")

    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frames = camera_rig.read()
            if not ok:
                break
            camera_bev = camera_projector.project(frames)
            lidar_enabled = bool(lidar_config.get("enabled", False))
            if lidar_enabled:
                if lidar_source is None:
                    raise NotImplementedError("The MVP currently supports synthetic LiDAR input only")
                lidar_bev = lidar_projector.project(lidar_source.read())
            else:
                lidar_bev = lidar_projector.empty()

            mode = runtime.get("view_mode", "fused")
            if mode == "camera":
                output = render_camera(camera_bev, bool(runtime.get("draw_overlay", False)))
            elif mode == "lidar":
                output = render_lidar(lidar_bev)
            else:
                output = render_fusion(camera_bev, lidar_bev, lidar_enabled)

            writer.write(output)
            if display:
                cv2.imshow("Parking BEV MVP", output)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_index += 1
    finally:
        camera_rig.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the parking BEV MVP")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    output = run(args.config, args.max_frames, not args.no_display)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
