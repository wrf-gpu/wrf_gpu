#!/usr/bin/env bash
set -euo pipefail

if [[ -f /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
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
