#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
python -c 'import sys; assert sys.version_info[:2] == (3,8)'
if ! command -v nvcc >/dev/null; then
  echo 'Install CUDA 11.3 development tools in the model env (e.g. micromamba install -n asyncbev -c conda-forge cudatoolkit-dev=11.3).' >&2
  exit 1
fi
export CUDA_HOME="${CONDA_PREFIX:?Activate the asyncbev environment first}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.6+PTX'
export MAX_JOBS=2
SOURCE="$ROOT/data/external/mmdetection3d_asyncbev"
if [ ! -d "$SOURCE" ]; then
  git clone --branch v1.0.0rc5 --depth 1 https://github.com/open-mmlab/mmdetection3d.git "$SOURCE"
fi
python -m pip install --no-build-isolation --no-deps -v -e "$SOURCE"
python "$ROOT/integrations/asyncbev/worker.py" --doctor
