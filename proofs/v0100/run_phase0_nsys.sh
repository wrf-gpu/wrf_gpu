#!/usr/bin/env bash
# Serialize these runs on the single GPU. Do not launch this script in parallel
# with any other GPU workload.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODE="${1:?usage: $0 <coupled|dycore> <guards_on|guards_off> [profile_steps]}"
GUARDS="${2:?usage: $0 <coupled|dycore> <guards_on|guards_off> [profile_steps]}"
PROFILE_STEPS="${3:-240}"

case "$GUARDS" in
  guards_on) GUARD_FLAG= ;;
  guards_off) GUARD_FLAG=--disable-guards ;;
  *) echo "guards must be guards_on or guards_off" >&2; exit 2 ;;
esac

OUT_BASE="proofs/v0100/nsys_${MODE}_${GUARDS}_${PROFILE_STEPS}steps"
JSON_OUT="${OUT_BASE}.driver.json"

export PYTHONPATH=src
export GPUWRF_CANAIRY_ROOT="${GPUWRF_CANAIRY_ROOT:-<DATA_ROOT>/canairy_meteo}"
export OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

taskset -c 0-3 nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --output="$OUT_BASE" \
  python proofs/v0100/phase0_nsys_driver.py \
    --mode "$MODE" \
    $GUARD_FLAG \
    --warm-steps 200 \
    --profile-steps "$PROFILE_STEPS" \
    --out-json "$JSON_OUT"

nsys stats --force-overwrite=true --force-export=true \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_sum \
  --report cuda_api_sum \
  --report nvtx_sum \
  --format csv \
  --output "${OUT_BASE}_stats" \
  "${OUT_BASE}.nsys-rep"

python proofs/v0100/parse_nsys_stats.py \
  --driver-json "$JSON_OUT" \
  --stats-prefix "${OUT_BASE}_stats" \
  --out-json "${OUT_BASE}.summary.json"
