#!/usr/bin/env bash
set -euo pipefail
# Run in a NEW Python 3.8 environment. Never run inside the BEVFusion/ROS env.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
python -c 'import sys; assert sys.version_info[:2] == (3, 8), "Requires an isolated Python 3.8 environment"'
CONSTRAINTS="$ROOT/data/external/AsyncBEV/requirements/model-constraints.txt"
python -m pip install 'pip<25' 'setuptools<60' wheel
python -m pip install -c "$CONSTRAINTS" torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -c "$CONSTRAINTS" mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
python -m pip install -c "$CONSTRAINTS" mmdet==2.24.0 mmsegmentation==0.29.1 spconv-cu113==2.1.21 cumm-cu113==0.2.9 nuscenes-devkit==1.1.11 lyft-dataset-sdk==0.0.8 plyfile einops ninja packaging prettytable tensorboard 'numba<0.59' 'numpy<1.24' 'opencv-python<4.12' 'scikit-image<0.20' 'trimesh<4' 'networkx<3' 'yapf==0.40.1'
python -m pip install --no-deps "$ROOT/data/external/AsyncBEV/wheels/flash_attn-0.2.2+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl"
bash "$ROOT/integrations/asyncbev/build_legacy_ops.sh"
