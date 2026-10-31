#!/usr/bin/env bash
set -euo pipefail
PREFIX=${1:?First argument must be the isolated asyncbev environment path}
shift
export CONDA_PREFIX="$PREFIX"
export CUDA_HOME="$PREFIX"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST='8.6+PTX'
exec "$PREFIX/bin/python" "$@"
