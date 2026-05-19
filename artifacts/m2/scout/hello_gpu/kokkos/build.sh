#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$ROOT/data/scratch/m2-scout-build/hello_gpu/kokkos"
source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
source "$ROOT/data/scratch/m2-scout-venv/bin/activate"

cmake -S "$PWD" -B "$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$ROOT/data/scratch/m2-scout-install/kokkos" \
  -DCMAKE_CXX_COMPILER="$ROOT/data/scratch/m2-scout-install/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --parallel 2 >/dev/null
