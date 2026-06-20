#!/usr/bin/env bash
# v0.17 all-7 nested GPU-idle A/B arm driver (opus hostgap fix).
# Runs ONE arm (sync mode) of the canary all-7 live-nested forecast with the
# WARM shared JAX compile cache + canonical env, sampling GPU util + VRAM at
# 0.5 s for the whole run.  Intended to be wrapped in scripts/with_gpu_lock.sh.
#
# Usage:  run_all7_hostgap_arm.sh <MODE> <HOURS> <RR>
#   MODE  = advance | root | root:K | segment   (GPUWRF_NESTED_SYNC_MODE)
#   HOURS = forecast hours (e.g. 2 or 3)
#   RR    = run root dir (created)
set -uo pipefail

MODE="${1:?need MODE}"
HOURS="${2:?need HOURS}"
RR="${3:?need RR}"

ROOT=<USER_HOME>/src/wrf_gpu2/.wt-opus-hostgap
INPUT=<DATA_ROOT>/wrf_downscale/canary_all7/run
mkdir -p "$RR/gpu_output" "$RR/proofs" "$RR/scratch"
LOG="$RR/run.log"
SAMP="$RR/gpu_samples.csv"

cat > "$RR/runinfo.txt" <<EOF
kind=v017_all7_hostgap_arm
sync_mode=$MODE
hours=$HOURS
root=$ROOT
input_dir=$INPUT
cache=<DATA_ROOT>/gpuwrf_jax_cache (warm, shared)
cpu_cores=0-3
EOF

cd "$ROOT"

# ---- 0.5 s GPU util + VRAM sampler (single nvidia-smi process) -------------
echo "timestamp,util_gpu_pct,mem_used_mib" > "$SAMP"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -lms 500 >> "$SAMP" 2>/dev/null &
SAMP_PID=$!

S=$(date -u +%s)
echo "[arm] START mode=$MODE hours=$HOURS $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

env PYTHONPATH="$ROOT/src" \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=true \
    JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache \
    OMP_NUM_THREADS=4 \
    GPUWRF_MYNN_BOULAC_ONZ=1 \
    GPUWRF_SCRATCH="$RR/scratch" \
    GPUWRF_NESTED_SYNC_MODE="$MODE" \
    nice -n 5 taskset -c 0-3 \
    python -m gpuwrf run \
      --input-dir "$INPUT" \
      --namelist "$INPUT/namelist.input" \
      --output-dir "$RR/gpu_output" \
      --proof-dir "$RR/proofs" \
      --scratch-dir "$RR/scratch" \
      --max-dom 9 \
      --hours "$HOURS" \
  >> "$LOG" 2>&1
RC=$?
E=$(date -u +%s)

kill "$SAMP_PID" 2>/dev/null || true
wait "$SAMP_PID" 2>/dev/null || true

WALL=$((E - S))
COMPILES=$(grep -c "Compiling module jit__advance_chunk" "$LOG" 2>/dev/null || echo 0)
SLOWOPS=$(grep -c "The operation took" "$LOG" 2>/dev/null || echo 0)
PEAK_VRAM=$(awk -F',' 'NR>1{if($3+0>m)m=$3+0} END{print m+0}' "$SAMP" 2>/dev/null)

{
  echo "rc=$RC"
  echo "sync_mode=$MODE"
  echo "hours=$HOURS"
  echo "wall_s=$WALL"
  echo "advance_chunk_compile_alarms=$COMPILES"
  echo "slow_operation_completed=$SLOWOPS"
  echo "peak_vram_mib=$PEAK_VRAM"
  echo "start_epoch=$S"
  echo "end_epoch=$E"
} | tee "$RR/result.txt"

# Per-domain wrfout counts + d01 per-hour mtimes (epoch) for warm-rate timing.
echo "--- wrfout counts ---" | tee -a "$RR/result.txt"
for d in 01 02 03 04 05 06 07 08 09; do
  n=$(ls "$RR"/gpu_output/wrfout_d${d}_* 2>/dev/null | wc -l)
  echo "d$d=$n" | tee -a "$RR/result.txt"
done
echo "--- d01 hourly mtimes (epoch path) ---" | tee -a "$RR/result.txt"
for f in $(ls "$RR"/gpu_output/wrfout_d01_* 2>/dev/null | sort); do
  echo "$(stat -c %Y "$f") $(basename "$f")" | tee -a "$RR/result.txt"
done

echo "[arm] END rc=$RC wall_s=$WALL compiles=$COMPILES peak_vram_mib=$PEAK_VRAM $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
exit "$RC"
