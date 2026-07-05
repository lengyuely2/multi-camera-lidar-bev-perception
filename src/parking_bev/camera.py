from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

CAMERA_NAMES = ("front", "rear", "left", "right")


class SyntheticCameraRig:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.index = 0

    def read(self) -> tuple[bool, dict[str, np.ndarray]]:
        frames: dict[str, np.ndarray] = {}
        colors = {
            "front": (70, 110, 170),
            "rear": (130, 85, 65),
            "left": (80, 145, 90),
            "right": (145, 90, 130),
        }
        for camera_index, name in enumerate(CAMERA_NAMES):
            frame = np.full((self.height, self.width, 3), colors[name], np.uint8)
            for x in range(0, self.width, 40):
                cv2.line(frame, (x, 0), (x, self.height), (65, 65, 65), 1)
            for y in range(0, self.height, 40):
                cv2.line(frame, (0, y), (self.width, y), (65, 65, 65), 1)
            offset = int((self.index * 4 + camera_index * 45) % (self.width + 80)) - 40
            cv2.rectangle(frame, (offset, 100), (offset + 48, 145), (25, 25, 25), -1)
            cv2.putText(frame, name.upper(), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            frames[name] = frame
        self.index += 1
        return True, frames

    def release(self) -> None:
        return None


class VideoCameraRig:
    def __init__(self, sources: Mapping[str, str | int]) -> None:
        missing = set(CAMERA_NAMES) - set(sources)
        if missing:
            raise ValueError(f"Missing camera sources: {sorted(missing)}")
        self.captures = {name: cv2.VideoCapture(_coerce_source(sources[name])) for name in CAMERA_NAMES}
        failed = [name for name, capture in self.captures.items() if not capture.isOpened()]
        if failed:
            self.release()
            raise RuntimeError(f"Could not open camera sources: {failed}")

    def read(self) -> tuple[bool, dict[str, np.ndarray]]:
        frames: dict[str, np.ndarray] = {}
        for name, capture in self.captures.items():
            ok, frame = capture.read()
            if not ok:
                return False, {}
            frames[name] = frame
        return True, frames

    def release(self) -> None:
        for capture in self.captures.values():
            capture.release()


def build_camera_rig(config: Mapping) -> SyntheticCameraRig | VideoCameraRig:
    sources = config["sources"]
    if all(str(source).lower() == "synthetic" for source in sources.values()):
        return SyntheticCameraRig(int(config["frame_width"]), int(config["frame_height"]))
    return VideoCameraRig(sources)


def _coerce_source(source: str | int) -> str | int:
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source

