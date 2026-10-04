#!/usr/bin/env bash
set -euo pipefail

mkdir -p proofs/v021/canary_gate/logs

scripts/with_gpu_lock.sh --timeout 7200 --label fused-aot-gpt-gate -- bash -lc '
set -euo pipefail

LOG1=proofs/v021/canary_gate/logs/fused_aot_capture.log
LOG2=proofs/v021/canary_gate/logs/fused_aot_warm_load.log
REF=proofs/v021/canary_gate/ts_proof_fused_aot_capture/warmK_state_digest.json

COMMON_ENV=(
  OMP_NUM_THREADS=4
  PYTHONPATH=src
  JAX_ENABLE_X64=true
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo
  CANARY_INPUT_DIR=<DATA_ROOT>/wrf_downscale/runs/20240901/cpu
  CANARY_MAXDOM=3
  MAXDOM=3
  CANARY_KWARM=12
  GPUWRF_NESTED_AOT=1
  GPUWRF_NESTED_PARALLEL_COMPILE=0
)

taskset -c 0-3 env "${COMMON_ENV[@]}" CANARY_TAG=fused_aot_capture \
  python proofs/v021/canary_gate/perstep_timing_driver.py 2>&1 | tee "$LOG1"
if rg -q "DRIVER RAISED" "$LOG1"; then
  exit 1
fi

test -f "$REF"

taskset -c 0-3 env "${COMMON_ENV[@]}" CANARY_TAG=fused_aot_warm \
  CANARY_REF_DIGEST="$REF" JAX_LOG_COMPILES=1 \
  python proofs/v021/canary_gate/perstep_timing_driver.py 2>&1 | tee "$LOG2"
if rg -q "DRIVER RAISED" "$LOG2"; then
  exit 1
fi
'
