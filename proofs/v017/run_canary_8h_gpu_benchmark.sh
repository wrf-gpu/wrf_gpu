#!/usr/bin/env bash
# v0.17 canary 8h NESTED GPU benchmark (AC1_FIT 9/3/1 live-coupled one-way nest).
# Uses the gpuwrf CLI nested path (execute_nested_pipeline / advance_chunk = the
# FAST-compile path), NOT the single-domain per-hour replay path. fp64 +
# GPUWRF_MYNN_BOULAC_ONZ=1. Honest capability: STANDALONE LIVE-NESTED 9/3/1
# (parent feeds each child LBC live; IC from wrfinput_d0N; only wrfbdy_d01 forces
# the root; NO CPU-WRF wrfout dependency). One-way nesting (default).
#
# Designed to be invoked WRAPPED in scripts/with_gpu_lock.sh (it queues behind the
# current lock holder via flock).
set -uo pipefail

ROOT=<USER_HOME>/src/wrf_gpu2/.wt-rc
INPUT=<DATA_ROOT>/wrf_downscale/nested_canary_training/ac1fit_20260614T220802Z/run_cpu
RR="$(cat /tmp/canary8h_rr.txt)"
GPU_OUT="$RR/gpu_output"
PROOFS="$RR/proofs"
SCRATCH="$RR/scratch"
LOG="$RR/canary_8h_gpu.log"
mkdir -p "$GPU_OUT" "$PROOFS" "$SCRATCH"

cat > "$RR/runinfo.txt" <<EOF
run_root=$RR
kind=v017_canary_8h_nested_9_3_1_gpu_benchmark
capability=standalone_live_nested_one_way (gpuwrf run --max-dom 3; execute_nested_pipeline; parent->child LBC live; IC from wrfinput_d0N; wrfbdy_d01 forces root; NO cpu wrfout dependency)
path=execute_nested_pipeline (segmented advance_chunk host loop; FAST-compile)
defaults=JAX_ENABLE_X64=true(fp64),GPUWRF_MYNN_BOULAC_ONZ=1,XLA_PYTHON_CLIENT_ALLOCATOR=platform(nested-OOM fix)
grid=9km(92x64) -> 3km(208x124) -> 1km(520x280 AC1_FIT)
init=2026-04-28_18:00:00
hours=8
input_dir=$INPUT
cpu_reference=$INPUT (12-rank CPU-WRF, hourly d01/d02/d03)
EOF
echo "RR=$RR"

cd "$ROOT"

# Background VRAM sampler (peak) for the duration of the GPU run.
VRAM_LOG="$RR/vram_samples.csv"
( while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null >> "$VRAM_LOG"
    sleep 5
  done ) &
VRAM_PID=$!
echo "vram_sampler_pid=$VRAM_PID"

START=$(date -u +%s)
echo "[canary8h] START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

env PYTHONPATH="$ROOT/src" \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=false \
    OMP_NUM_THREADS=4 \
    GPUWRF_MYNN_BOULAC_ONZ=1 \
    GPUWRF_SCRATCH="$SCRATCH" \
    nice -n 5 taskset -c 0-3 \
    python -m gpuwrf run \
      --input-dir "$INPUT" \
      --namelist "$INPUT/namelist.input" \
      --output-dir "$GPU_OUT" \
      --proof-dir "$PROOFS" \
      --scratch-dir "$SCRATCH" \
      --max-dom 3 \
      --hours 8 \
  >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)

kill "$VRAM_PID" 2>/dev/null || true

WALL=$((END - START))
echo "[canary8h] END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC wall_s=$WALL" | tee -a "$LOG"
echo "$RC" > "$RR/canary_8h_gpu.rc"
echo "$WALL" > "$RR/canary_8h_gpu.wall_s"

# Peak VRAM from samples (MiB).
PEAK_VRAM=$(awk -F',' '{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print m}' "$VRAM_LOG" 2>/dev/null)
echo "$PEAK_VRAM" > "$RR/peak_vram_mib.txt"
echo "[canary8h] peak_vram_mib=$PEAK_VRAM" | tee -a "$LOG"

echo "[canary8h] compile/timing alarms:" | tee -a "$LOG"
grep -E "Compiling module|The operation took|Compilation" "$LOG" | tail -20 | tee -a "$LOG" || echo "  (none)"
echo "[canary8h] GPU output files:" | tee -a "$LOG"
ls -la "$GPU_OUT" 2>/dev/null | tee -a "$LOG"

echo "[canary8h] GPU DONE rc=$RC wall_s=$WALL peak_vram_mib=$PEAK_VRAM"
exit "$RC"
