#!/usr/bin/env bash
# v0.17 canary FITTING GPU benchmark: d03 1km (all-islands AC1_FIT) STANDALONE/
# parent-driven, 8h, fp64. The full live-coupled 9/3/1 OOMs one RTX 5090
# (capability finding: ~30 GiB > 32; confirmed WITH BouLac-ONZ). The d03 1km
# layer (the dominant cost, ~88% of nested time) FITS standalone at 21.4 GiB.
#
# Mode: gpuwrf run --domain d03 --max-dom 1 against the canary run_cpu dir
# (has wrfout_d03 -> cpu_wrf_replay = the GPU recomputes d03 dynamics+physics
# each step, forced/compared vs CPU d03 hourly = per-checkpoint IDENTITY +
# a clean warm GPU d03 wallclock). GPUWRF_REPLAY_SEGMENTED=1 = the a67c239
# fast-compile fix (single-domain replay path), GPUWRF_MYNN_BOULAC_ONZ=1 = VRAM.
#
# Wrap in scripts/with_gpu_lock.sh. Launch AFTER the CPU nested run finishes 8h
# (all d03 forcing present).
set -uo pipefail

ROOT=<USER_HOME>/src/wrf_gpu2/.wt-rc
INPUT=<DATA_ROOT>/wrf_downscale/nested_canary_training/ac1fit_20260614T220802Z/run_cpu
HOURS="${HOURS:-8}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR=<DATA_ROOT>/wrf_gpu_validation/v017_canary_d03_${HOURS}h_gpu_replay_${STAMP}
GPU_OUT="$RR/gpu_output"; PROOFS="$RR/proofs"; SCRATCH="$RR/scratch"; LOG="$RR/canary_d03_8h_gpu.log"
mkdir -p "$GPU_OUT" "$PROOFS" "$SCRATCH"
echo "$RR" > /tmp/canary_d03_8h_rr.txt

cat > "$RR/runinfo.txt" <<EOF
run_root=$RR
kind=v017_canary_d03_1km_8h_gpu_replay_FITTING_BENCHMARK
capability=standalone/parent-driven d03 1km (full live 9/3/1 OOMs single-card fp64 -> this is the FITTING GPU mode)
path=gpuwrf run --domain d03 --max-dom 1 (cpu_wrf_replay vs CPU d03 wrfout); GPUWRF_REPLAY_SEGMENTED=1 fast-compile; GPUWRF_MYNN_BOULAC_ONZ=1
grid=d03 1km 520x280 (AC1_FIT, ~145k cols, all 7 islands)
init=2026-04-28_18:00:00 hours=$HOURS
cpu_reference=$INPUT (12-rank CPU-WRF full nested; d03 1km is the bottleneck domain)
EOF
echo "RR=$RR"
cd "$ROOT"

# d03 forcing present?
ND03=$(ls "$INPUT"/wrfout_d03_* 2>/dev/null | wc -l)
echo "[d03bench] CPU d03 wrfout available: $ND03 (need >=$((HOURS+1)) for ${HOURS}h)" | tee -a "$LOG"

VRAM_LOG="$RR/vram_samples.csv"
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null >> "$VRAM_LOG"; sleep 5; done ) &
VRAM_PID=$!

START=$(date -u +%s)
echo "[d03bench] START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
env PYTHONPATH="$ROOT/src" \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=false \
    OMP_NUM_THREADS=4 \
    GPUWRF_REPLAY_SEGMENTED=1 \
    GPUWRF_MYNN_BOULAC_ONZ=1 \
    GPUWRF_SCRATCH="$SCRATCH" \
    nice -n 5 taskset -c 0-3 \
    python -m gpuwrf run \
      --input-dir "$INPUT" \
      --namelist "$INPUT/namelist.input" \
      --output-dir "$GPU_OUT" \
      --proof-dir "$PROOFS" \
      --scratch-dir "$SCRATCH" \
      --domain d03 \
      --max-dom 1 \
      --hours "$HOURS" \
  >> "$LOG" 2>&1
RC=$?
END=$(date -u +%s)
kill "$VRAM_PID" 2>/dev/null || true
WALL=$((END - START))
echo "[d03bench] END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC wall_s=$WALL" | tee -a "$LOG"
echo "$RC" > "$RR/rc"; echo "$WALL" > "$RR/wall_s"
PEAK=$(awk -F',' '{gsub(/ MiB/,"",$2); if($2+0>m)m=$2+0} END{print m}' "$VRAM_LOG" 2>/dev/null)
echo "$PEAK" > "$RR/peak_vram_mib.txt"
echo "[d03bench] peak_vram_mib=$PEAK d03_out=$(ls "$GPU_OUT"/wrfout_d03_* 2>/dev/null | wc -l)" | tee -a "$LOG"
grep -E "Compiling module|The operation took|RESOURCE_EXHAUSTED|out of memory" "$LOG" | tail -10 | tee -a "$LOG"
echo "[d03bench] DONE rc=$RC wall_s=$WALL peak_vram_mib=$PEAK"
exit "$RC"
