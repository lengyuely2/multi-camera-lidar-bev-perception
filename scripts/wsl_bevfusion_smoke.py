from __future__ import annotations

import json

import mmcv
import mmdet
import mmengine
import torch

import mmdet3d
import projects.BEVFusion.bevfusion  # noqa: F401
from projects.BEVFusion.bevfusion.ops.bev_pool import bev_pool_ext  # noqa: F401
from projects.BEVFusion.bevfusion.ops.voxel import voxel_layer  # noqa: F401


print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    if torch.cuda.is_available() else 0,
    "mmcv": mmcv.__version__,
    "mmdet": mmdet.__version__,
    "mmengine": mmengine.__version__,
    "mmdet3d": mmdet3d.__version__,
    "bevfusion_cuda_ops": True,
}, indent=2))
