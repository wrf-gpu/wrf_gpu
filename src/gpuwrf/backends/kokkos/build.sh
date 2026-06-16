#!/usr/bin/env bash
set -euo pipefail

if [[ -f /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh ]]; then
  # shellcheck disable=SC1091
  source /home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
fi

ROOT=$(git rev-parse --show-toplevel)
SRC="$ROOT/data/scratch/kokkos-src"
BUILD="$ROOT/data/scratch/kokkos-build"
INSTALL="$ROOT/data/scratch/kokkos-install"
BENCH_BUILD="$ROOT/data/scratch/kokkos-build-bench"
BENCH_DIR="$ROOT/data/scratch/kokkos"
BENCH="$BENCH_DIR/bench"

mkdir -p "$ROOT/data/scratch" "$BENCH_DIR"

if [[ ! -d "$SRC/.git" ]]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch 4.7.01 https://github.com/kokkos/kokkos.git "$SRC"
fi

if [[ ! -x "$INSTALL/bin/nvcc_wrapper" || ! -f "$INSTALL/lib/cmake/Kokkos/KokkosConfig.cmake" ]]; then
  cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DCMAKE_CXX_COMPILER="$SRC/bin/nvcc_wrapper" \
    -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_BLACKWELL120=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_SERIAL=ON
  cmake --build "$BUILD" --parallel 8
  cmake --install "$BUILD"
fi

cmake -S "$ROOT/src/gpuwrf/backends/kokkos" -B "$BENCH_BUILD" \
  -DCMAKE_PREFIX_PATH="$INSTALL" \
  -DCMAKE_CXX_COMPILER="$INSTALL/bin/nvcc_wrapper" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BENCH_BUILD" --parallel 4
cp "$BENCH_BUILD/bench" "$BENCH"
"$BENCH" >/dev/null
