#!/usr/bin/env bash
# v0.17 Switzerland d01 72h identity (FAST-compile path).
# Single-domain cpu_wrf_replay via the CLI, routed through the bounded segmented
# entry (GPUWRF_REPLAY_SEGMENTED=1 => run_forecast_operational_segmented advance_chunk;
# bit-identical to the while-loop per proofs/perf/segscan_equiv.json) so the cold
# compile is one small fixed-length segment reused across all hours -- no 10m+ tax.
# Identity = GPU-vs-CPU result equivalence within the FROZEN tolerance manifest, so
# XLA autotuning is NOT needed: autotune is left at default (the segmented compile is
# already small). Holds the shared GPU lock for the whole run. CPU postprocess after.
set -euo pipefail

ROOT=/home/enric/src/wrf_gpu2/.wt-rc
RR=/mnt/data/wrf_gpu_validation/v017_switzerland_d01_72h_identity_fast_$(date -u +%Y%m%dT%H%M%SZ)
INPUT=/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu
mkdir -p "$RR/gpu_output" "$RR/scratch" "$RR/proofs"
cat > "$RR/runinfo.txt" <<EOF
run_root=$RR
kind=v017_switzerland_d01_72h_identity_proof_fast_compile
fast_compile=GPUWRF_REPLAY_SEGMENTED=1 (run_forecast_operational_segmented, bit-identical seg-vs-production)
defaults=GPUWRF_MYNN_BOULAC_ONZ=1,fp64,JAX_ENABLE_X64=true
cpu_truth=$INPUT
domain=d01
hours=72
EOF
echo "RR=$RR"

cd "$ROOT"
env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=24 GPUWRF_MYNN_BOULAC_ONZ=1 \
    GPUWRF_REPLAY_SEGMENTED=1 \
    nice -n 10 taskset -c 0-23 python -m gpuwrf.cli run \
      --input-dir "$INPUT" \
      --output-dir "$RR/gpu_output" \
      --scratch-dir "$RR/scratch" \
      --domain d01 --hours 72 \
      --proof-dir "$RR/proofs" \
  > "$RR/switzerland_d01_72h_gpu.log" 2>&1
echo $? > "$RR/switzerland_d01_72h_gpu.rc"
echo "[switz] GPU rc=$(cat "$RR/switzerland_d01_72h_gpu.rc") wrfout=$(ls "$RR/gpu_output" | grep -c wrfout)"
echo "[switz] compile alarms:"; grep -E "Compiling module|The operation took" "$RR/switzerland_d01_72h_gpu.log" || echo "  (none)"

# CPU-side postprocess (no GPU lock needed) -- v0.15 finalgate recipe, v0.17 dirs.
echo "$RR" > /tmp/v017_switz_identity_rr.txt
echo "[switz] GPU DONE -> postprocess will run after lock release"
