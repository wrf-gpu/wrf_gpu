#!/usr/bin/env bash
set -euo pipefail

WRF_GPU_REFERENCE_ROOT="${WRF_GPU_REFERENCE_ROOT:-reference_data}"
WRF_GPU_ENV="${WRF_GPU_ENV:-$WRF_GPU_REFERENCE_ROOT/artifacts/wrf_gpu_src/env_wrf_gpu.sh}"

if [[ -f "$WRF_GPU_ENV" ]]; then
  # shellcheck disable=SC1091
  source "$WRF_GPU_ENV"
fi

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT/src/gpuwrf/backends/cuda_tile"
if make all; then
  exit 0
fi

echo "nvcc build failed; retrying with nvc++ -cuda because local CUDA 13.1 + glibc exposes rsqrt prototype conflicts" >&2
make clean
make all \
  NVCC=nvc++ \
  NVCCFLAGS="-cuda -gpu=cc120 -O3 -std=c++17 --diag_suppress=cuda_compile" \
  LDFLAGS="-lz"
