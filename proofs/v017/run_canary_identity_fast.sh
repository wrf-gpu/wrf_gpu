#!/usr/bin/env bash
# v0.17 Canary L2 d02 72h identity (FAST-compile path).
# The L2 case is a max_dom=2 live-nest -> execute_nested_pipeline, which ALREADY
# drives the bounded segmented host loop (one output interval per compiled
# advance_chunk; v015 canary cold compile was ~2 min, NOT the 10m+ replay
# pathology). So no code change is needed for canary -- we replicate the v015
# proven launcher (run_one_case_v0120.py) on the v017-rc code. Identity scored
# vs CPU-WRF d02 truth with the FROZEN tolerance manifest.
set -euo pipefail

ROOT=/home/user/src/wrf_gpu2/.wt-rc
RUN_ID=20260501_18z_l2_72h_20260519T173026Z
RR=/mnt/data/wrf_gpu_validation/v017_canary_d02_72h_identity_fast_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RR/gpu_output" "$RR/proofs"
cat > "$RR/runinfo.txt" <<EOF
run_root=$RR
kind=v017_canary_l2_d02_72h_identity_proof_fast_compile
path=execute_nested_pipeline (segmented advance_chunk host loop; already compile-bounded)
defaults=GPUWRF_MYNN_BOULAC_ONZ=1,GPUWRF_MOIST_CQW=1,fp64
run_id=$RUN_ID
cpu_truth=/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/$RUN_ID
input_run_root=/mnt/data/canairy_meteo/runs/wrf_l2
domain=d02
hours=72
EOF
echo "RR=$RR"

cd "$ROOT"
env JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_ENABLE_COMPILATION_CACHE=false OMP_NUM_THREADS=24 \
    GPUWRF_MYNN_BOULAC_ONZ=1 GPUWRF_MOIST_CQW=1 \
    nice -n 10 taskset -c 0-23 python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
      --run-root /mnt/data/canairy_meteo/runs/wrf_l2 \
      --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
      --run-id "$RUN_ID" \
      --hours 72 \
      --output-root "$RR/gpu_output" \
      --proof-dir "$RR/proofs" \
  > "$RR/canary_d02_72h_gpu.log" 2>&1
echo $? > "$RR/canary_d02_72h_gpu.rc"
echo "[canary] GPU rc=$(cat "$RR/canary_d02_72h_gpu.rc")"
echo "[canary] compile alarms:"; grep -E "Compiling module|The operation took" "$RR/canary_d02_72h_gpu.log" || echo "  (none)"
ls "$RR/gpu_output" 2>/dev/null
echo "$RR" > /tmp/v017_canary_identity_rr.txt
echo "[canary] GPU DONE"
