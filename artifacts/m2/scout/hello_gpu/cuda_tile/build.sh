#!/usr/bin/env bash
set -euo pipefail

source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$ROOT/data/scratch/m2-scout-build/hello_gpu/cuda_tile"
mkdir -p "$BUILD_DIR"
nvcc -std=c++17 -arch=sm_120 -O2 hello.cu -o "$BUILD_DIR/hello_cuda_tile"
