#!/usr/bin/env bash
# v0.17 compile-pathology flag test (SHORT: 2 forecast hours, Switzerland d01 replay).
# Runs the SAME cpu_wrf_replay CLI path that hit the 72h slow-compile, but only 2h,
# under three configs, capturing wall + the XLA "slow compile" alarm durations:
#   A. default  (run_forecast_operational while-loop) + autotune ON  (baseline pathology)
#   B. default  while-loop + autotune OFF (--xla_gpu_autotune_level=0)  (is autotune the cause?)
#   C. GPUWRF_REPLAY_SEGMENTED=1 (run_forecast_operational_segmented advance_chunk) + autotune ON (the fix)
# Each config is a SEPARATE python process => cold compile each time (no warm cache leakage).
set -euo pipefail

ROOT=/home/enric/src/wrf_gpu2/.wt-rc
OUT=/mnt/data/wrf_gpu_validation/v017_compile_flag_test_$(date -u +%Y%m%dT%H%M%SZ)
INPUT=/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu
mkdir -p "$OUT"
echo "OUT=$OUT"

run_one () {
  local tag="$1"; shift
  local odir="$OUT/$tag"
  mkdir -p "$odir"
  echo "=== [$tag] start $(date -u +%H:%M:%S) ==="
  ( cd "$ROOT" && \
    env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
        JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=24 GPUWRF_MYNN_BOULAC_ONZ=1 \
        "$@" \
        nice -n 10 taskset -c 0-23 python -m gpuwrf.cli run \
          --input-dir "$INPUT" \
          --output-dir "$odir/gpu_output" \
          --scratch-dir "$odir/scratch" \
          --domain d01 --hours 2 \
          --proof-dir "$odir/proofs" ) > "$odir/run.log" 2>&1 &
  local pid=$!
  local t0=$(date +%s)
  wait $pid; local rc=$?
  local t1=$(date +%s)
  echo "rc=$rc wall_s=$((t1-t0))" | tee "$odir/wall.txt"
  echo "  slow-compile alarms:"
  grep -E "Compiling module|The operation took" "$odir/run.log" 2>/dev/null | sed 's/^/    /' || true
  echo "  wrfout written: $(ls "$odir/gpu_output" 2>/dev/null | grep -c wrfout || echo 0)"
  echo "=== [$tag] done $(date -u +%H:%M:%S) ==="
}

run_one A_default_autotune_on
run_one B_default_autotune_off XLA_FLAGS=--xla_gpu_autotune_level=0
run_one C_segmented_autotune_on GPUWRF_REPLAY_SEGMENTED=1

echo "ALL FLAG TESTS DONE -> $OUT"
