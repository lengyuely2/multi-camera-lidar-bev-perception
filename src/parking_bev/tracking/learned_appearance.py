from __future__ import annotations

import cv2
import numpy as np

from .appearance import extract_object_crop
from ..sensors.nuscenes_source import Object3D, SensorCalibration


class ResNetAppearanceEncoder:
    """Lazy TorchVision ResNet-18 encoder for generic object appearance."""

    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("Learned appearance requires PyTorch and TorchVision") from exc
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def encode_crops(self, crops: list[np.ndarray | None]) -> list[np.ndarray | None]:
        valid_indices = [index for index, crop in enumerate(crops) if crop is not None and crop.size]
        output: list[np.ndarray | None] = [None] * len(crops)
        if not valid_indices:
            return output
        arrays = []
        for index in valid_indices:
            crop = crops[index]
            assert crop is not None
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            arrays.append(rgb.transpose(2, 0, 1))
        batch = self.torch.from_numpy(np.stack(arrays)).to(self.device, dtype=self.torch.float32) / 255.0
        batch = (batch - self.mean) / self.std
        with self.torch.inference_mode():
            features = self.model(batch)
            features = self.torch.nn.functional.normalize(features, dim=1)
        for index, feature in zip(valid_indices, features.detach().cpu().numpy()):
            output[index] = feature.astype(np.float32)
        return output


def extract_learned_appearances(
    objects: list[Object3D],
    cameras: dict[str, np.ndarray],
    calibrations: dict[str, SensorCalibration],
    encoder: ResNetAppearanceEncoder,
) -> list[np.ndarray | None]:
    crops = [extract_object_crop(obj, cameras, calibrations) for obj in objects]
    return encoder.encode_crops(crops)
