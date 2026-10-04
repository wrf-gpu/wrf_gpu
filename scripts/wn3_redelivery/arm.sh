#!/usr/bin/env bash
# One WN3 twin forecast on W8/source with RELEASE DEFAULTS (no flag file, full WRF history is the default from LW12).
# Env: ROOT (=W8). Args: case_dir out hours tag
set -euo pipefail
: "${ROOT:?}"
src=$ROOT/source; case_dir=$1; out=$2; hours=$3; tag=$4
export GPUWRF_CENSUS=${GPUWRF_CENSUS:-0}
frac=$(cat "$ROOT/mem_fraction" 2>/dev/null || echo 0.1675)
export XLA_PYTHON_CLIENT_ALLOCATOR=${W8_ALLOCATOR:-cuda_async} XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=$frac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} JAX_PLATFORMS=cuda PYTHONPATH="$src/src" TMPDIR="$out/tmp" LC_ALL=C
export GPUWRF_JAX_CACHE_DIR=$ROOT/cache/jax JAX_COMPILATION_CACHE_DIR=$ROOT/cache/jax
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$ROOT/cache/autotune CUDA_CACHE_PATH=$ROOT/cache/cuda TRITON_CACHE_DIR=$ROOT/cache/triton
export GPUWRF_WRF_ROOT=<DATA_ROOT>/wrf_gpu2/v025/tenerife_b4/wrf_root TF_CPP_MIN_LOG_LEVEL=1
mkdir -p "$out/tmp"
env | grep -E '^(GPUWRF_|JAX_|XLA_|OMP_|WN3_)' | grep -v TOKEN | sort > "$out/arm_env.txt"
exec <USER_HOME>/miniconda3/bin/python3 "$src/scripts/wn3_forecast.py" --input-dir "$case_dir" --out "$out" --hours "$hours" --tag "$tag"
