from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from .camera import CAMERA_NAMES


class CameraBEVProjector:
    def __init__(self, width: int, height: int, destination_quads: Mapping[str, list]) -> None:
        self.width = width
        self.height = height
        self.destination_quads = destination_quads

    def project(self, frames: Mapping[str, np.ndarray]) -> np.ndarray:
        accumulator = np.zeros((self.height, self.width, 3), np.float32)
        weights = np.zeros((self.height, self.width, 1), np.float32)

        for name in CAMERA_NAMES:
            frame = frames[name]
            src_h, src_w = frame.shape[:2]
            source_quad = np.float32([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]])
            destination_quad = np.float32(self.destination_quads[name])
            homography = cv2.getPerspectiveTransform(source_quad, destination_quad)
            warped = cv2.warpPerspective(frame, homography, (self.width, self.height))
            source_mask = np.full((src_h, src_w), 255, np.uint8)
            mask = cv2.warpPerspective(source_mask, homography, (self.width, self.height))
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            accumulator += warped.astype(np.float32) * alpha
            weights += alpha

        result = accumulator / np.maximum(weights, 1.0)
        return np.clip(result, 0, 255).astype(np.uint8)

