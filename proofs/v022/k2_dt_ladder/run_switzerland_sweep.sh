#!/usr/bin/env bash
# K2 dt/n_sound CFL-ladder SWEEP on the FAST Switzerland 128x128 single-domain case.
#
# Cost guard (per task + prep doc): the SWEEP runs on Switzerland 128 only.
# The best safe rung is validated on the 3-dom Canary by a SEPARATE script.
# We do NOT run the ladder on the all-7 9-nest.
#
# Ladder (scaled to this case's EFFECTIVE baseline dt=10s; config.dt_s default):
#   R1: dt=10s n_sound=10   (baseline)
#   R2: dt=12s n_sound=9
#   R3: dt=15s n_sound=8
#   R4: dt=18s n_sound=7
#
# All four run under ONE GPU-lock acquisition (serial), so we never collide with
# the corpus / sibling GPU openers and never hold the lock between rungs idle.
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root (worktree)
ROOT="$(pwd)"
LADDER_DIR="$ROOT/proofs/v022/k2_dt_ladder"
LOG_DIR="$LADDER_DIR/logs"
mkdir -p "$LOG_DIR"

INPUT_DIR="${K2_INPUT_DIR:-<DATA_ROOT>/wrf_gpu_switzerland_128/run_gpu_input}"
HOURS="${K2_HOURS:-2}"

scripts/with_gpu_lock.sh --timeout 21600 --label k2-dt-ladder-switz -- bash -lc '
set -euo pipefail
cd "'"$ROOT"'"
INPUT_DIR="'"$INPUT_DIR"'"
HOURS="'"$HOURS"'"
LADDER_DIR="'"$LADDER_DIR"'"
LOG_DIR="'"$LOG_DIR"'"

COMMON=(
  OMP_NUM_THREADS=4
  PYTHONPATH=src
  JAX_ENABLE_X64=true
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_ALLOCATOR=platform
  GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo
  K2_INPUT_DIR="$INPUT_DIR"
  K2_HOURS="$HOURS"
  K2_DOMAIN=d01
)

run_rung() {
  local tag="$1" dt="$2" ns="$3"
  echo "=== RUNG $tag : dt=${dt}s n_sound=${ns} (hours=$HOURS) ==="
  taskset -c 0-3 env "${COMMON[@]}" \
    K2_TAG="$tag" K2_DT_S="$dt" K2_ACOUSTIC_SUBSTEPS="$ns" \
    python proofs/v022/k2_dt_ladder/run_rung.py 2>&1 | tee "$LOG_DIR/${tag}.log"
}

run_rung R1 10 10
run_rung R2 12 9
run_rung R3 15 8
run_rung R4 18 7

# Compare each rung final wrfout to R1 (operational tolerance).
R1_OUT=$(find "$LADDER_DIR"/runs/R1/out -name "wrfout_d01_*" 2>/dev/null | sort | tail -1 || true)
for tag in R2 R3 R4; do
  RUNG_OUT=$(find "$LADDER_DIR"/runs/$tag/out -name "wrfout_d01_*" 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$R1_OUT" && -n "$RUNG_OUT" ]]; then
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true \
      python proofs/v022/k2_dt_ladder/compare_to_r1.py \
      "$RUNG_OUT" "$R1_OUT" "$LADDER_DIR/runs/$tag/compare_to_r1.json" \
      > "$LOG_DIR/${tag}_compare.log" 2>&1 || echo "compare $tag FAILED"
  fi
done

echo "=== SWEEP DONE ==="
'
